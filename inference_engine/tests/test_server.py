"""HTTP-contract tests for the OpenAI-compatible server (no model/sampler)."""

import json

import pytest
from fastapi.testclient import TestClient

from tcpip_gen.engine import GenerationResult
from tcpip_gen.server import create_app, resolve_model_path


class FakeEngine:
    """An Engine that yields canned text deltas, recording the call args."""

    model_id = "fake-model"

    def __init__(self, deltas, finish_reason="stop", prompt_tokens=3):
        self.deltas = list(deltas)
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.last_call: dict | None = None

    def chat(self, messages, *, max_tokens, stop):
        self.last_call = {
            "messages": messages,
            "max_tokens": max_tokens,
            "stop": list(stop),
        }
        deltas = self.deltas
        result = GenerationResult(self.finish_reason, self.prompt_tokens, len(deltas))

        def _gen():
            yield from deltas
            return result

        return _gen()


def _sse_data_lines(raw: str) -> list[str]:
    return [
        line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")
    ]


def test_list_models():
    client = TestClient(create_app(FakeEngine([])))
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "fake-model"
    assert body["data"][0]["object"] == "model"
    assert "created" in body["data"][0]


def test_chat_completion_non_streaming():
    engine = FakeEngine(["Hello", " world"], finish_reason="stop", prompt_tokens=5)
    client = TestClient(create_app(engine))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "m"
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello world"}
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


def test_chat_completion_streaming():
    engine = FakeEngine(["A", "B"], finish_reason="length")
    client = TestClient(create_app(engine))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        raw = "".join(resp.iter_text())

    events = _sse_data_lines(raw)
    assert events[-1] == "[DONE]"

    first = json.loads(events[0])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}

    assert json.loads(events[1])["choices"][0]["delta"] == {"content": "A"}
    assert json.loads(events[2])["choices"][0]["delta"] == {"content": "B"}

    final = json.loads(events[3])
    assert final["choices"][0]["delta"] == {}
    assert final["choices"][0]["finish_reason"] == "length"


def test_request_params_passed_to_engine():
    engine = FakeEngine(["x"])
    client = TestClient(create_app(engine))
    client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 7,
            "stop": "\n",
            "temperature": 0.9,  # accepted but ignored downstream
        },
    )
    assert engine.last_call["max_tokens"] == 7
    assert engine.last_call["stop"] == ["\n"]
    assert engine.last_call["messages"] == [{"role": "user", "content": "hi"}]


# --- Ollama-compat health endpoints ----------------------------------------
#
# Each is registered at both the canonical Ollama path and a /v1/api/* alias so
# the same handler works regardless of the user's Ollama base URL. The
# parametrize hits both paths to keep them locked together.


@pytest.mark.parametrize("path", ["/api/version", "/v1/api/version"])
def test_ollama_version(path):
    client = TestClient(create_app(FakeEngine([])))
    resp = client.get(path)
    assert resp.status_code == 200
    parts = resp.json()["version"].split(".")
    # Open WebUI parses with tuple(map(int, …split("."))). Must be all-digit.
    assert len(parts) >= 2 and all(p.isdigit() for p in parts)


@pytest.mark.parametrize("path", ["/api/tags", "/v1/api/tags"])
def test_ollama_tags(path):
    client = TestClient(create_app(FakeEngine([])))
    resp = client.get(path)
    assert resp.status_code == 200
    body = resp.json()
    assert list(body) == ["models"]
    [model] = body["models"]
    assert model["model"] == "fake-model"
    assert model["name"] == "fake-model"
    assert {
        "parent_model",
        "format",
        "family",
        "families",
        "parameter_size",
        "quantization_level",
    } <= model["details"].keys()


@pytest.mark.parametrize("path", ["/api/ps", "/v1/api/ps"])
def test_ollama_ps(path):
    client = TestClient(create_app(FakeEngine([])))
    resp = client.get(path)
    assert resp.status_code == 200
    [model] = resp.json()["models"]
    assert model["model"] == "fake-model"
    # /api/ps adds runtime-only fields on top of the /api/tags entry shape.
    assert "expires_at" in model
    assert "size_vram" in model


# -- model path resolution ---------------------------------------------------


def test_resolve_model_path_passes_files_through(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    assert resolve_model_path(str(gguf)) == str(gguf)


def test_resolve_model_path_picks_the_only_gguf_in_a_directory(tmp_path):
    gguf = tmp_path / "SmolLM2.gguf"
    gguf.write_bytes(b"")
    (tmp_path / "README.md").write_text("not a model")
    assert resolve_model_path(str(tmp_path)) == str(gguf)


@pytest.mark.parametrize("names", [[], ["a.gguf", "b.gguf"]])
def test_resolve_model_path_rejects_ambiguous_directory(tmp_path, names):
    for name in names:
        (tmp_path / name).write_bytes(b"")
    with pytest.raises(SystemExit, match="exactly one"):
        resolve_model_path(str(tmp_path))
