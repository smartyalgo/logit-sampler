# stateful-sampler-inference

`tcpip_gen` runs llama.cpp inference **locally** but delegates token **sampling**
to the Rust [`stateful_sampler`](../stateful_sampler) over TCP. It also ships an
OpenAI-compatible HTTP server (`python -m tcpip_gen.server`).

Full documentation (architecture, endpoints, SDK example, test layout):
[docs/components/inference-engine.md](../docs/components/inference-engine.md).

## How to run

Requires [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.12. Start the
sampler first (listening on `127.0.0.1:5146`), then:

```sh
uv sync                                   # creates .venv, installs deps (incl. llama-cpp-python)

# One-shot CLI generation
uv run python -m tcpip_gen <model.gguf> "<prompt>" --n-predict 64

# OpenAI-compatible server on :8000
uv run python -m tcpip_gen.server \
    --model <model.gguf> \
    --host 127.0.0.1 --port 8000 \
    --sampler-host 127.0.0.1 --sampler-port 5146 \
    --max-tokens 256
```

## How to test

```sh
uv run ruff format .     # format
uv run ruff check .      # lint
uv run pytest            # unit tests; run without llama_cpp
```
