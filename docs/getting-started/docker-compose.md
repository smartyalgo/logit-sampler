# Docker Compose

One command brings up the sampler, the OpenAI-compatible engine, and Open WebUI.

**1. Get a GGUF model.** The engine needs a llama.cpp-compatible `.gguf` file.
The repo ships without one (`*.gguf` is gitignored), so copy one into
`./models/`:

```bash
mkdir -p models
cp /path/to/your-model.gguf models/
```

or point `MODELS_DIR` at a directory that already holds your models.

**2. Start the stack:**

```bash
docker compose up --build

# if the directory holds several models, pick one by file name:
MODEL_FILE=your-model.gguf docker compose up --build

# or with an external models directory:
MODELS_DIR=/path/to/llm-models MODEL_FILE=your-model.gguf docker compose up --build
```

**3. Chat.** Open <http://localhost:3000> and pick the model (default name
`local-llama`).

## Services

| Service | Port | What it is |
|---------|------|------------|
| `sampler` | 5146 | Rust [stateful sampler](../components/stateful-sampler.md) |
| `engine` | 8000 | OpenAI-compatible API (`/v1/chat/completions`), see [Inference engine](../components/inference-engine.md) |
| `open-webui` | 3000 | Chat UI at <http://localhost:3000>, see [Open WebUI](open-webui.md) |

## Environment variables

Set them inline or in a `.env` file.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODELS_DIR` | `./models` | Host directory mounted as `/models` |
| `MODEL_FILE` | unset | GGUF file name inside that directory. When unset, the engine uses the only `.gguf` there |
| `MODEL_ID` | `local-llama` | Model name shown to clients |
| `RUST_LOG` | `info` | Sampler log level |

Sampling parameters (temperature, top-k/p, min-p) are set on the `sampler`
service `command` in
[`docker-compose.yml`](https://github.com/smartyalgo/logit-sampler/blob/main/docker-compose.yml).
Client-side values are
[ignored](../components/inference-engine.md#sampling-parameters-are-ignored).

!!! warning "Troubleshooting"
    If the `engine` container restart-loops logging
    `--model /models/: expected exactly one .gguf, found ...`, the mounted
    directory holds zero or several models. Check `MODELS_DIR`, or set
    `MODEL_FILE` to the file you want.
