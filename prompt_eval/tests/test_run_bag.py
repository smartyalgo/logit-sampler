"""Unit tests for the bag runner. The endpoint is faked via post_fn injection,
plus one test that exercises the real HTTP path against a local stub server."""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from run_bag import completion_text, main, post_chat_completion, run_bag, run_entry

OPENAI_RESPONSE = {
    "id": "chatcmpl-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "def add(a, b): return a + b"},
            "finish_reason": "stop",
        }
    ],
}


def fake_post(body: dict) -> dict:
    assert body["messages"][0]["role"] == "user"
    return OPENAI_RESPONSE


def test_completion_text_extracts_assistant_content() -> None:
    assert completion_text(OPENAI_RESPONSE) == "def add(a, b): return a + b"


def test_run_entry_success_has_uuid_prompt_response() -> None:
    entry = run_entry("Write add().", fake_post, max_tokens=32)
    assert set(entry) == {"id", "prompt", "response"}
    uuid.UUID(entry["id"])  # raises if not a valid UUID
    assert entry["prompt"] == "Write add()."
    assert entry["response"] == "def add(a, b): return a + b"


def test_run_entry_failure_records_error_and_continues() -> None:
    def broken_post(body: dict) -> dict:
        raise ConnectionError("connection refused")

    entry = run_entry("Write add().", broken_post, max_tokens=32)
    assert entry["response"] is None
    assert "connection refused" in entry["error"]


def test_run_bag_writes_one_jsonl_line_per_prompt(tmp_path: Path) -> None:
    prompts = [
        {"id": "T/0", "prompt": "task zero"},
        {"id": "T/1", "prompt": "task one"},
    ]
    out_path = tmp_path / "run.jsonl"
    error_count = run_bag(prompts, fake_post, out_path, max_tokens=32)

    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert error_count == 0
    assert [line["prompt"] for line in lines] == ["task zero", "task one"]
    assert all(line["response"] == "def add(a, b): return a + b" for line in lines)
    assert len({line["id"] for line in lines}) == 2  # UUIDs are unique


def test_post_chat_completion_speaks_http() -> None:
    class StubHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            request_body = json.loads(self.rfile.read(length))
            assert request_body["max_tokens"] == 32
            payload = json.dumps(OPENAI_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32}
        response = post_chat_completion(endpoint, 5.0, body)
        assert completion_text(response) == "def add(a, b): return a + b"
    finally:
        server.shutdown()
        server.server_close()


def test_main_end_to_end_with_fake_bag(tmp_path: Path, monkeypatch) -> None:
    bag = {
        "name": "tiny",
        "source": {"dataset": "d", "config": "c", "split": "s"},
        "prompts": [{"id": "T/0", "prompt": "task zero"}],
    }
    bag_path = tmp_path / "tiny.json"
    bag_path.write_text(json.dumps(bag))
    out_path = tmp_path / "out.jsonl"

    import run_bag as run_bag_module

    monkeypatch.setattr(
        run_bag_module,
        "post_chat_completion",
        lambda endpoint, timeout, body: OPENAI_RESPONSE,
    )
    exit_code = main([str(bag_path), "--out", str(out_path), "--limit", "1"])

    assert exit_code == 0
    lines = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["response"] == "def add(a, b): return a + b"
