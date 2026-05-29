"""CLI entrypoint: ``python -m tcpip_gen <model.gguf> "<prompt>"``.

Implements the TCP/IP generation entry point for the Rust sampler wire protocol.

Each generation step runs `llama_decode` before sampling so the logits sent to
the sampler reflect a real forward pass.

Diagnostics go to stderr; generated text streams to stdout as raw bytes so
multi-byte UTF-8 tokens render correctly.
"""

from __future__ import annotations

import argparse
import sys

from .client import SamplerClient
from .generation import generate_tokens
from .protocol import HOST, PORT


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(piece: bytes) -> None:
    sys.stdout.buffer.write(piece)
    sys.stdout.buffer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcpip_gen",
        description=(
            "llama.cpp text generation with sampling delegated to a remote "
            "sampler over TCP"
        ),
    )
    parser.add_argument("model_path", help="path to a .gguf model")
    parser.add_argument("prompt", help="prompt text")
    parser.add_argument("--host", default=HOST, help=f"sampler host (default {HOST})")
    parser.add_argument(
        "--port", type=int, default=PORT, help=f"sampler port (default {PORT})"
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=256,
        help="maximum number of tokens to generate (default 256)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Imported lazily: pulls in llama_cpp (heavy) only when actually generating.
    from .llama_backend import LlamaModel

    _log("Starting...")
    model = LlamaModel(args.model_path)
    _log(f"Loaded model {args.model_path}")
    _log(f"n_token: {model.n_vocab}")

    client = SamplerClient(args.host, args.port)
    try:
        client.connect()
        _log(f"Connected to {args.host}:{args.port}")
        _log("Starting handshake...")
        client.handshake(model.n_vocab, model.vocab_pieces())
        _log(f"Finished handshake, sent {model.n_vocab} tokens")

        prompt_tokens = model.tokenize(args.prompt)
        n_prompt = len(prompt_tokens)
        model.make_context(n_prompt, args.n_predict)
        _log("Context loaded\n")

        for token in prompt_tokens:
            _emit(model.token_to_piece(token, special=False))

        tokens = generate_tokens(model, client, prompt_tokens, args.n_predict)
        try:
            while True:
                _emit(model.token_to_piece(next(tokens), special=False))
        except StopIteration as stop:
            if stop.value == "stop":
                # Hit an end-of-generation token: tell the sampler, then close.
                client.finish()

        _log("\ndone.")
    finally:
        client.close()
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
