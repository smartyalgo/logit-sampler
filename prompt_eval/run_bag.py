"""Run a prompt bag against the ``/v1/chat/completions`` endpoint.

Feeds each prompt in a bag (see ``fetch_prompts.py``) to the endpoint and
appends one JSON line per request to a per-run file::

    {"id": "<uuid4>", "prompt": "...", "response": "..."}

If a request fails, the entry gets ``"response": null`` plus an ``"error"``
key, and the run continues. Lines are flushed as they complete, so an
interrupted run keeps its finished prompts.

Run with: ``uv run python run_bag.py bags/mbpp_python.json``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 600.0

# Request body -> parsed response JSON. Injectable so tests need no server.
PostFn = Callable[[dict], dict]


def post_chat_completion(endpoint: str, timeout: float, body: dict) -> dict:
    """POST one chat-completion request and return the parsed JSON response."""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def completion_text(response: dict) -> str:
    """Extract the assistant text from an OpenAI chat-completion response."""
    return response["choices"][0]["message"]["content"]


def run_entry(prompt: str, post_fn: PostFn, max_tokens: int) -> dict:
    """Send one prompt and build its JSONL entry (never raises)."""
    entry: dict = {"id": str(uuid.uuid4()), "prompt": prompt}
    body = {
        "model": "tcpip_gen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    try:
        entry["response"] = completion_text(post_fn(body))
    except Exception as error:  # record the failure, keep the run going
        entry["response"] = None
        entry["error"] = str(error)
    return entry


def run_bag(
    prompts: list[dict],
    post_fn: PostFn,
    out_path: Path,
    max_tokens: int,
) -> int:
    """Run every prompt, append JSONL lines to ``out_path``, return error count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_count = 0
    with out_path.open("w", encoding="utf-8") as out_file:
        for index, prompt in enumerate(prompts, start=1):
            entry = run_entry(prompt["prompt"], post_fn, max_tokens)
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_file.flush()
            if "error" in entry:
                error_count += 1
                print(
                    f"[{index}/{len(prompts)}] {prompt['id']}: ERROR {entry['error']}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(prompts)}] {prompt['id']}: ok", flush=True)
    return error_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_bag",
        description="Feed a prompt bag to /v1/chat/completions, store JSONL responses.",
    )
    parser.add_argument("bag", type=Path, help="path to a bag JSON file")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"chat completions URL (default {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL path (default runs/<bag>-<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens per request (default {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N prompts (for smoke runs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bag = json.loads(args.bag.read_text())
    prompts = bag["prompts"][: args.limit]

    out_path = args.out
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = DEFAULT_RUNS_DIR / f"{bag['name']}-{timestamp}.jsonl"

    print(f"Running {len(prompts)} prompts from {bag['name']} against {args.endpoint}")
    post_fn = partial(post_chat_completion, args.endpoint, args.timeout)
    error_count = run_bag(prompts, post_fn, out_path, args.max_tokens)
    print(f"Wrote {out_path} ({error_count} errors)")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
