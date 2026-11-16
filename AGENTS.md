# Agent instructions

Read this before you change anything in this repository.

## Repository map

| Directory | What | Docs page |
|-----------|------|-----------|
| `inference_engine/` | Python. llama.cpp runner + OpenAI-compatible server | `docs/components/inference-engine.md` |
| `stateful_sampler/` | Rust. TCP sampler on `:5146` | `docs/components/stateful-sampler.md` |
| `prompt_eval/` | Python. Benchmark prompt bags + runner | `docs/components/prompt-eval.md` |
| `docs/` | The documentation. Markdown pages built by MkDocs Material | `docs/index.md` |

## Documentation system

- All documentation lives in `docs/` as Markdown. `mkdocs.yml` at the repo
  root builds it into a site. The `nav` list in `mkdocs.yml` is the left panel.
  A page that is not in `nav` is unreachable.
- READMEs (root and per component) are entry points only: title, one
  paragraph, How to run, How to test, link into `docs/`. Do not add sections
  to a README. Put the content in `docs/` and link to it.
- Pages reference each other with relative Markdown links:
  `[Wire protocol](../wire-protocol.md)` or
  `[ignored](../components/inference-engine.md#sampling-parameters-are-ignored)`.
  Link repo files with a GitHub URL. Mermaid fences render as diagrams.
  Callouts: `!!! note "Title"` with the body indented four spaces.
- The full recipe and the templates are in the `mkdocs-docs` plugin from the
  `smarty-agents` marketplace (`/plugin install mkdocs-docs@smarty-agents`).
  This file stands on its own without it.

### Where topics live

| Topic | Page |
|-------|------|
| Run the stack with Docker Compose, env vars, troubleshooting | `docs/getting-started/docker-compose.md` |
| Run sampler and engine from source | `docs/getting-started/manual.md` |
| Open WebUI | `docs/getting-started/open-webui.md` |
| Diagrams, component responsibilities, repo layout | `docs/architecture.md` |
| Handshake and LOGITS byte layout | `docs/wire-protocol.md` |
| Engine CLI, server endpoints, test files | `docs/components/inference-engine.md` |
| Sampler flags, logging, Docker, design assumptions | `docs/components/stateful-sampler.md` |
| Prompt bags, run format, adding a bag | `docs/components/prompt-eval.md` |
| Build, test, CI, docs commands | `docs/development.md` |

### Adding or changing a page

1. Write `docs/<section>/<name>.md`. Start with one `# Title`. How-to
   guides go in `docs/guides/` under a `Guides` nav section. Create the
   section on first use.
2. Add it to `nav` in `mkdocs.yml`.
3. Link it from at least one related page. Add a row to the README docs
   table only for a new top-level topic.
4. When you document a command or flag, run it with `--help` first and copy
   the spelling. The sampler flags are `--top-k`, `--top-p`, `--min-p`.
5. Run `uv run mkdocs build --strict` from the repo root. Zero warnings, or
   the change is not done. CI runs the same command.

### When code changes

- Changed a flag, port, env var, endpoint, protocol byte, or test file?
  Update the page from the table above in the same change.
- Found a code bug while writing docs? Report it in your summary. Fix it in a
  separate change unless asked.

## Commands

```bash
uv run mkdocs serve -a 127.0.0.1:8001   # preview; the engine uses :8000
uv run mkdocs build --strict            # required check before you finish
```

Component build and test commands: `docs/development.md`.
