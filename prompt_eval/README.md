# prompt_eval

Prompt "bags" for exercising the OpenAI-compatible `/v1/chat/completions`
server. `fetch_prompts.py` downloads a bag (HumanEval, MBPP) from Hugging Face
and `run_bag.py` feeds it to the endpoint, storing one JSONL file per run.
Both scripts use the standard library only.

Full documentation (bag and run formats, available bags, adding a bag):
[docs/components/prompt-eval.md](../docs/components/prompt-eval.md).

## How to run

The server must be up first (see
[docs/getting-started/manual.md](../docs/getting-started/manual.md)).

```bash
cd prompt_eval

# 1. Download all bags (writes ./bags/<name>.json):
uv run python fetch_prompts.py

# Or download one bag / write elsewhere:
uv run python fetch_prompts.py humaneval_python
uv run python fetch_prompts.py --out-dir /tmp/bags

# 2. Run a bag against the endpoint (writes ./runs/<bag>-<timestamp>.jsonl):
uv run python run_bag.py bags/mbpp_python.json

# Useful runner flags:
uv run python run_bag.py bags/humaneval_python.json \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --max-tokens 256 \
    --limit 5 \
    --timeout 600 \
    --out runs/smoke.jsonl
```

## How to test

```bash
cd prompt_eval
uv run pytest          # unit tests; no network, no model, no sampler needed
uv run ruff format .
```
