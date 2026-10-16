# Manual quick start

Run each component from source. The sampler must be running **before** the
engine connects.

## 1. Start the Rust sampler

```bash
cd stateful_sampler
cargo run                      # listens on 0.0.0.0:5146 with default sampling params

# Override sampling parameters (defaults: temperature 1.0, top_k 1, top_p 1.0, min_p 0.05):
cargo run -- 0.0.0.0:5146 --temperature 0.8 --top-k 40 --top-p 0.95 --min-p 0.05

# Verbose logs (defaults to `info` when RUST_LOG is unset):
RUST_LOG=debug cargo run
```

See [Stateful sampler](../components/stateful-sampler.md) for logging and
Docker options.

## 2. Run the inference engine

```bash
cd inference_engine
uv sync                        # creates .venv, installs deps (incl. llama-cpp-python)

# One-shot CLI generation (delegates sampling to the sampler on :5146):
uv run python -m tcpip_gen <model.gguf> "The capital of France is" --n-predict 24

# Or expose an OpenAI-compatible HTTP API on :8000:
uv run python -m tcpip_gen.server \
    --model <model.gguf> \
    --host 127.0.0.1 --port 8000 \
    --sampler-host 127.0.0.1 --sampler-port 5146 \
    --max-tokens 256
```

Then call the server like any OpenAI endpoint:

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24}'
```

See [Inference engine](../components/inference-engine.md) for the CLI flags,
the server endpoints, and the `openai` SDK example.

!!! note "Sampling parameters are ignored"
    Sampling parameters sent by OpenAI clients (`temperature`, `top_p`, …) are
    **ignored**: the [wire protocol](../wire-protocol.md) only carries logits,
    so sampling is governed by the Rust sampler's own configuration. Set them
    via `cargo run -- …` flags.

## 3. Drive it from a chat UI

[Open WebUI](open-webui.md) can use the server as an OpenAI-compatible backend.
