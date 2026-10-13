# Architecture

The inference engine owns the model and the decode loop. The sampler owns the
sampling policy. They meet over one TCP connection that carries the
[wire protocol](wire-protocol.md).

```mermaid
flowchart LR
    gguf[("GGUF model")]

    subgraph engine["inference_engine/ (Python)"]
        cli["CLI / OpenAI server"]
        lb["LlamaModel<br/>(llama-cpp-python)"]
        sc["SamplerClient"]
        cli --> lb
        cli --> sc
    end

    subgraph sampler["stateful_sampler/ (Rust)"]
        router["SamplerRouter<br/>TCP :5146"]
        samp["Sampler<br/>(per connection)"]
        logit["logit_manipulation<br/>(softmax, top-k/p, min-p)"]
        router --> samp --> logit
    end

    client["OpenAI client<br/>(SDK / curl / chat UI)"]

    lb -->|"ctypes"| gguf
    client -->|"/v1/chat/completions"| cli
    sc <-->|"HANDSHAKE then LOGITS → token id<br/>(little-endian, TCP :5146)"| router
```

Each component page has its own diagrams:
[Inference engine](components/inference-engine.md#architecture),
[Stateful sampler](components/stateful-sampler.md),
[Prompt eval](components/prompt-eval.md#architecture).

## Component responsibilities

```mermaid
classDiagram
    class SamplerRouter {
        +new(bind_address, params) SamplerRouter
        +run()
        -accept connections, spawn a Sampler per client
    }
    class Sampler {
        -SamplingParams params
        -TcpStream llm_conn_stream
        -Vec~String~ vocab
        +handle_new_connection() Result
        +handle_logits() Result~LlmConnectionState~
        -sample_logit(logits) usize
    }
    class SamplingParams {
        +Option~f32~ temperature
        +i64 top_k
        +f32 top_p
        +f32 min_p
    }
    class SamplerClient {
        +connect(timeout)
        +handshake(n_vocab, pieces) int
        +sample(logits) int
        +finish()
    }
    SamplerRouter "1" --> "*" Sampler : spawns per connection
    SamplerRouter --> SamplingParams : holds
    Sampler --> SamplingParams : holds
    SamplerClient ..> Sampler : drives over TCP
```

!!! note "Single instance only"
    The Rust sampler assumes **one** engine and **one** active chat completion at a
    time. There is no per-request context id, so concurrent chats / branched
    generations are not supported. See the sampler's
    [design assumptions](components/stateful-sampler.md#design-assumptions).

## Repository layout

```
.
├── docs/                 # This documentation site (Markdown pages, MkDocs)
├── inference_engine/     # Python: llama.cpp runner + OpenAI-compatible server
│   ├── tcpip_gen/        #   protocol, client, generation, engine, server, CLI
│   └── tests/
├── stateful_sampler/     # Rust: TCP sampler server
│   └── src/
│       ├── main.rs       #   CLI args + SamplerRouter bootstrap
│       └── sampler/      #   Sampler + logit_manipulation
├── prompt_eval/          # Python: benchmark prompt bags for the chat endpoint
│   ├── fetch_prompts.py  #   downloads bags from Hugging Face
│   ├── run_bag.py        #   feeds a bag to the endpoint, stores JSONL responses
│   └── bags/             #   stored prompt bags (JSON)
├── docker-compose.yml    # Full environment: sampler + engine + Open WebUI
├── mkdocs.yml            # Docs site: left-panel nav and Markdown extensions
└── README.md             # Entry point on GitHub; links into docs/
```
