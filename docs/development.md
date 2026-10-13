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

The subprojects have GitHub Actions workflows
(`stateful_sampler/.github/workflows/rust.yml`,
`inference_engine/.github/workflows/ci.yml`). The docs site is built with
`--strict` by `.github/workflows/docs.yml`, so a broken cross-page link or
anchor fails the build.

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
