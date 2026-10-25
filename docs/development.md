# Development

Each component is built and tested independently.

## Rust sampler

```bash
cd stateful_sampler
cargo fmt
cargo build
cargo test
```

## Python inference engine

```bash
cd inference_engine
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Prompt eval

```bash
cd prompt_eval
uv run ruff format .
uv run pytest
```

## Continuous integration

GitHub runs workflows only from the repo-root `.github/workflows/`.
`tests.yml` runs the lint and unit tests of all three components, with a
coverage report for the Python suites. `docs.yml` builds the docs site with
`--strict`, so a broken cross-page link or anchor fails the build.

## Test coverage

```bash
# Rust (one-time: cargo install cargo-llvm-cov)
cd stateful_sampler && cargo llvm-cov

# Python inference engine (no llama-cpp-python needed)
cd inference_engine && uv run --no-project --with pytest --with pytest-cov \
  --with numpy --with fastapi --with sse-starlette --with httpx \
  pytest --cov=tcpip_gen --cov-report=term-missing

# Prompt eval
cd prompt_eval && uv run --with pytest-cov pytest --cov=. --cov-report=term-missing
```

Line coverage on 2026-10-24 (SMA-415):

| Component | Line coverage | Not covered |
|-----------|---------------|-------------|
| `stateful_sampler` | 52% (was 34%; the suite did not compile before) | `main.rs`, the TCP connection loop in `sampler.rs` |
| `inference_engine` | 55% | `__main__.py`, `llama_backend.py` (need a GGUF model), the streaming paths in `engine.py` and `server.py` |
| `prompt_eval` | 93% (source files 82%–92%) | `main()` entry points, network fetch |

## Documentation

The docs live in `docs/` as plain Markdown and are built with
[MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The left panel
is the `nav` list in `mkdocs.yml`. Pages reference each other with relative
Markdown links, for example `[Wire protocol](../wire-protocol.md)`, and MkDocs
checks every link and anchor at build time. Mermaid fences render as diagrams.

```bash
uv sync                              # once, at the repo root; installs mkdocs-material
uv run mkdocs serve                  # live preview at http://127.0.0.1:8000
uv run mkdocs serve -a 127.0.0.1:8001   # if the inference server already holds :8000
uv run mkdocs build --strict         # one-shot build into ./site; fails on broken links
```

To add a page, create `docs/<name>.md` and add it to `nav` in `mkdocs.yml`.
