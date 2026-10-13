# stateful_sampler

A Rust TCP server that receives logits from the
[inference engine](../inference_engine) and returns a sampled token id
(temperature / top-k / top-p / min-p). Listens on `0.0.0.0:5146`.

Full documentation (logging, design assumptions, TODO):
[docs/components/stateful-sampler.md](../docs/components/stateful-sampler.md).
Message layout: [docs/wire-protocol.md](../docs/wire-protocol.md).

## How to run

```bash
cargo run                      # listens on 0.0.0.0:5146 with default sampling params

# Override sampling parameters (defaults: temperature 1.0, top_k 1, top_p 1.0, min_p 0.05):
cargo run -- 0.0.0.0:5146 --temperature 0.8 --top_k 40 --top_p 0.95 --min_p 0.05

# Verbose logs (defaults to `info` when RUST_LOG is unset):
RUST_LOG=debug cargo run
```

## How to build and test

```bash
cargo build
cargo test
```

## How to run in Docker

```bash
docker build -t stateful-sampler .
docker run -p 5146:5146 stateful-sampler
# or, from the repo root, as part of the full stack:
docker compose up sampler
```
