# stateful-logit-sampler

This is the open source version of what we've built at printers.
This is a tool to inspect the model's bag of next-token prediction per next-token.
This is useful for diagnosing, fine-tuning, harness engineering, understanding what is the probability distribution of your desired output.

A two-part system that splits LLM **inference** from token **sampling** across a
TCP socket. The inference engine runs llama.cpp locally and produces logits each
decode step; a separate, stateful sampler process turns those logits into the
next token. Keeping sampling in its own process lets the sampling policy evolve
(and carry state across steps) independently of the model runtime.

## Where to start

- [Docker Compose](getting-started/docker-compose.md) brings up the sampler, the engine, and Open WebUI in one command.
- [Manual quick start](getting-started/manual.md) runs the sampler and the engine from source.
- [Architecture](architecture.md) shows how the pieces fit together.
- [Development](development.md) covers build, test, CI, and how to edit these docs.

## Components

The repository is a monorepo of three cooperating components:

| Directory | Language | Role | Docs |
|-----------|----------|------|------|
| `inference_engine/` | Python | Loads a GGUF model with `llama-cpp-python`, runs the decode loop, streams raw logits to the sampler, and prints the chosen tokens. Also ships an OpenAI-compatible HTTP server. | [Inference engine](components/inference-engine.md) |
| `stateful_sampler/` | Rust | TCP server that receives logits and returns a sampled token id (temperature / top-k / top-p / min-p). Listens on `:5146`. | [Stateful sampler](components/stateful-sampler.md) |
| `prompt_eval/` | Python | Benchmark prompt "bags" (HumanEval, MBPP) plus a runner that feeds them to the chat endpoint and stores JSONL responses per run. | [Prompt eval](components/prompt-eval.md) |

The engine and the sampler talk over a small little-endian
[wire protocol](wire-protocol.md): a one-time **handshake** (vocabulary
exchange) followed by a **per-token** loop (`LOGITS` → token id).
