"""OpenAI-compatible HTTP server in front of the remote-sampler engine.

Exposes ``GET /v1/models`` and ``POST /v1/chat/completions`` (streaming and
non-streaming) for OpenAI clients, plus three read-only Ollama-compat health
endpoints (``GET /api/version``, ``GET /api/tags``, ``GET /api/ps``) so Open
WebUI's Ollama-tab probes succeed without 404s. Chat itself stays on the OpenAI
path. The HTTP layer depends only on :class:`tcpip_gen.engine.Engine`, so it is
fully testable with a fake engine (no model or sampler needed).

Per-request sampling params (``temperature``, ``top_p``, ``seed``, ...) are
accepted for API compatibility but **ignored**: sampling is performed by the
remote Rust sampler with its own configured parameters. ``max_tokens``, ``stop``,
and ``stream`` are honored locally. Any ``Authorization`` header is accepted and
ignored.

Run with: ``python -m tcpip_gen.server --model <model.gguf>``.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .engine import Engine, GenerationResult
from .protocol import HOST, PORT

DEFAULT_MAX_TOKENS = 256


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    # Accept unknown fields; don't treat `model` as a pydantic-protected name.
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model: str = "tcpip_gen"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    # Accepted but ignored (sampling is owned by the remote sampler):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    n: int | None = None


def _ollama_model_entry(model_id: str, *, with_runtime: bool = False) -> dict:
    """One model entry in the Ollama ``/api/tags`` and ``/api/ps`` envelopes.

    Open WebUI doesn't validate the values beyond requiring the keys to exist;
    we return safe placeholders rather than introspecting the real GGUF.
    """
    entry = {
        "name": model_id,
        "model": model_id,
        "modified_at": "1970-01-01T00:00:00Z",
        "size": 0,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "llama",
            "families": ["llama"],
            "parameter_size": "unknown",
            "quantization_level": "unknown",
        },
    }
    if with_runtime:
        # /api/ps reports the in-memory copy of each model.
        entry["expires_at"] = "2099-01-01T00:00:00Z"
        entry["size_vram"] = 0
    return entry


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _chunk(cid: str, created: int, model: str, delta: dict, finish: str | None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _drain(gen: Iterator[str], parts: list[str]) -> GenerationResult:
    """Consume a chat stream, collecting text and returning its result."""
    while True:
        try:
            parts.append(next(gen))
        except StopIteration as done:
            return done.value


def _tag_result(gen: Iterator[str]) -> Iterator[tuple[str, object]]:
    """Re-yield deltas as ``("delta", text)`` and the result as ``("final", res)``.

    Lets the streaming path recover the :class:`GenerationResult` (the stream's
    ``StopIteration.value``) when iterated through a threadpool, which otherwise
    swallows it.
    """
    while True:
        try:
            yield ("delta", next(gen))
        except StopIteration as done:
            yield ("final", done.value)
            return


def create_app(engine: Engine, default_max_tokens: int = DEFAULT_MAX_TOKENS) -> FastAPI:
    """Build the FastAPI app around an :class:`Engine`."""
    app = FastAPI(title="tcpip_gen OpenAI-compatible server")

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tcpip_gen",
                }
            ],
        }

    # Ollama-compat health endpoints. Each is registered at the canonical
    # ``/api/...`` path AND a ``/v1/api/...`` alias so the same handler works
    # whether the user's Ollama base URL includes ``/v1`` or not.
    @app.get("/api/version")
    @app.get("/v1/api/version")
    def ollama_version() -> dict:
        # Open WebUI's version aggregator only needs a semver-parsable string.
        return {"version": "0.5.0"}

    @app.get("/api/tags")
    @app.get("/v1/api/tags")
    def ollama_tags() -> dict:
        return {"models": [_ollama_model_entry(engine.model_id)]}

    @app.get("/api/ps")
    @app.get("/v1/api/ps")
    def ollama_ps() -> dict:
        return {"models": [_ollama_model_entry(engine.model_id, with_runtime=True)]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        max_tokens = req.max_tokens or req.max_completion_tokens or default_max_tokens
        stops = _normalize_stop(req.stop)
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        model = req.model

        if req.stream:
            return EventSourceResponse(
                _stream_chat(engine, messages, max_tokens, stops, cid, created, model)
            )

        parts: list[str] = []
        result = await run_in_threadpool(
            _drain, engine.chat(messages, max_tokens=max_tokens, stop=stops), parts
        )
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(parts)},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.prompt_tokens + result.completion_tokens,
                },
            }
        )

    return app


async def _stream_chat(
    engine: Engine,
    messages: list[dict[str, str]],
    max_tokens: int,
    stops: Sequence[str],
    cid: str,
    created: int,
    model: str,
) -> AsyncIterator[dict]:
    """Yield SSE events for a streaming chat completion."""
    # First chunk announces the assistant role.
    yield {
        "data": _json(
            _chunk(cid, created, model, {"role": "assistant", "content": ""}, None)
        )
    }

    finish = "stop"
    tagged = _tag_result(engine.chat(messages, max_tokens=max_tokens, stop=stops))
    async for kind, payload in iterate_in_threadpool(tagged):
        if kind == "delta":
            if payload:
                yield {
                    "data": _json(
                        _chunk(cid, created, model, {"content": payload}, None)
                    )
                }
        else:  # ("final", GenerationResult)
            finish = payload.finish_reason

    yield {"data": _json(_chunk(cid, created, model, {}, finish))}
    yield {"data": "[DONE]"}


def _json(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcpip_gen.server",
        description="OpenAI-compatible HTTP server backed by the remote logit sampler.",
    )
    parser.add_argument("--model", required=True, help="path to a .gguf model")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    parser.add_argument(
        "--sampler-host", default=HOST, help=f"sampler host (default {HOST})"
    )
    parser.add_argument(
        "--sampler-port", type=int, default=PORT, help=f"sampler port (default {PORT})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="default max_tokens when a request omits it",
    )
    parser.add_argument(
        "--model-id", default=None, help="model id reported by /v1/models"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Heavy imports kept local so the module is importable without them.
    import uvicorn

    from .client import SamplerClient
    from .engine import LlamaSamplerEngine
    from .llama_backend import LlamaModel

    print(f"Loading model {args.model} ...", flush=True)
    model = LlamaModel(args.model)
    client = SamplerClient(args.sampler_host, args.sampler_port)
    client.connect()
    client.handshake(model.n_vocab, model.vocab_pieces())
    print(
        f"Connected to sampler at {args.sampler_host}:{args.sampler_port}", flush=True
    )

    model_id = args.model_id or os.path.basename(args.model)
    engine = LlamaSamplerEngine(model, client, model_id)
    app = create_app(engine, default_max_tokens=args.max_tokens)

    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        client.close()
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
