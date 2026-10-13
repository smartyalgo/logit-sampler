# stateful-logit-sampler

This is the open source version of what we've built at printers.
This is a tool to inspect the model's bag of next-token prediction per next-token.
This is useful for diagnosing, fine-tuning, harness engineering, understanding what is the probability distribution of your desired output.

A two-part system that splits LLM **inference** from token **sampling** across a
TCP socket. The inference engine runs llama.cpp locally and produces logits each
decode step; a separate, stateful sampler process turns those logits into the
next token. Keeping sampling in its own process lets the sampling policy evolve
(and carry state across steps) independently of the model runtime.

## How to run (Docker Compose)

```bash
mkdir -p models
cp /path/to/your-model.gguf models/   # any llama.cpp-compatible GGUF
docker compose up --build              # sampler :5146, engine :8000, Open WebUI :3000
```

Open <http://localhost:3000> and pick the `local-llama` model. If `./models/`
holds several files, set `MODEL_FILE=your-model.gguf`. Environment variables and
troubleshooting: [Docker Compose](docs/getting-started/docker-compose.md).

## Documentation

The documentation is a set of Markdown pages under [`docs/`](docs/), built into
a site with a left-panel catalog by [MkDocs](https://www.mkdocs.org/) and the
Material theme. Pages link to each other with plain relative Markdown links.

```bash
uv sync                      # once, at the repo root; installs mkdocs-material
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
```

| Page | What it covers |
|------|----------------|
| [Overview](docs/index.md) | What the system is and where to start |
| [Docker Compose](docs/getting-started/docker-compose.md) | Full stack in one command, env vars, troubleshooting |
| [Manual quick start](docs/getting-started/manual.md) | Run the sampler and the engine from source |
| [Open WebUI](docs/getting-started/open-webui.md) | Drive the engine from a chat UI |
| [Architecture](docs/architecture.md) | Diagrams, component responsibilities, repository layout |
| [Wire protocol](docs/wire-protocol.md) | Handshake and per-token byte layout |
| [Inference engine](docs/components/inference-engine.md) | Python: llama.cpp runner + OpenAI-compatible server (`inference_engine/`) |
| [Stateful sampler](docs/components/stateful-sampler.md) | Rust: TCP sampler server on `:5146` (`stateful_sampler/`) |
| [Prompt eval](docs/components/prompt-eval.md) | Benchmark prompt bags and the JSONL runner (`prompt_eval/`) |
| [Development](docs/development.md) | Build, test, CI, and how to edit the docs |
