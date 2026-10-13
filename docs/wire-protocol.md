# Wire protocol

The engine and the sampler exchange a one-time **handshake** (vocabulary
exchange) followed by a **per-token** loop. All integers and floats are
**little-endian**. The engine frames messages in `tcpip_gen/protocol.py`
([Inference engine](components/inference-engine.md)); the sampler parses them in
`src/sampler.rs` ([Stateful sampler](components/stateful-sampler.md)).

| Phase     | Engine → Sampler                                                       | Sampler → Engine                       |
|-----------|------------------------------------------------------------------------|----------------------------------------|
| Handshake | `"HANDSHAKE"` + `i32(n_vocab)` + `(piece + \x00) × n_vocab` + `\x00`    | `i32(n_vocab)` (4 bytes)               |
| Per token | `"LOGITS"` + `n_vocab × f32`                                            | `i32(count)` + `i32(token)` (8 bytes)  |
| End       | socket close on end-of-generation                                      | (read == 0 ⇒ "Completed")              |

The CLI (`python -m tcpip_gen`) sends a single `\x00` (`SamplerClient.finish()`)
before it closes the socket. The HTTP server keeps one connection open across
requests and never sends that byte, because it would desync the sampler's
fixed-length framing. See
[Sampling parameters are ignored](components/inference-engine.md#sampling-parameters-are-ignored).

## Sequence

```mermaid
sequenceDiagram
    participant inf as Inference Engine
    participant sampler as Sampler
    
    note over sampler: Startup: Expects sampling parameters

    %% Handshake with explicit type annotations
    rect rgba(255, 223, 191, 0.5)
        note over inf,sampler: Handshake Phase
    inf->>+sampler: "HANDSHAKE"|{num_of_tokens:i32 LE}|[token₁|0x00|token₂|0x00|...|tokenₙ|0x00]|0x00
    note right of inf: Request Format:<br/>String + Integer (LE) + Null-delimited Token Array + Null Byte
    sampler-->>-inf: {token_count:i32 LE}
    note left of sampler: Response:<br/>i32 (Little Endian)
    end
    
    %% Main loop with colored annotations
    rect rgba(191, 223, 255, 0.5)
        note over inf,sampler: Token Generation Loop
        loop While not end of text
            inf->>+sampler: "LOGITS"|[logits:float[vocab_size]]
            note right of inf: Request Format:<br/>String + Float Array (size = vocabulary size)
            sampler-->>-inf: {logits_size:i32}|{token_id:i32}
            note left of sampler: Response:<br/>Size + Token ID (i32)
        end
    end    
```
