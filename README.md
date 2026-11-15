# APC Starter

A small, provider-neutral implementation of the Agentic Plan Caching (APC) loop. It keeps reusable **plan templates** in a local JSON cache, adapts a cached template for a new task, and only adds templates after an execution is judged successful.

This is the standalone APC stage. It does not implement wireless placement, shared edge storage, KV/prompt-state caching, or Minions chunk scheduling yet.

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible chat-completions endpoint and API key

The code uses only the Python standard library. It can work with a local endpoint as well as a hosted endpoint. Requests and context are sent to whichever endpoint you configure, so use an endpoint appropriate for your data.

## Paper model profile

The APC paper's main setup used GPT-4o as the large planner, Llama 3.1 8B as both the small planner and actor, and GPT-4o-mini for keyword extraction and cache generation. This prototype also has an evidence-sufficiency gate; it defaults to the local Llama model. The paper used GPT-4o as an offline evaluation model, not as an extra runtime gate.

## No hosted API bill: run every role locally

You can replace all paper models with a local open-weight model. Install Ollama, then choose one size:

```powershell
# Lower resource use (about 3.3-4.0 GB model download)
ollama pull qwen3.5:4b

# Stronger, larger option (about 6.6-7.6 GB model download)
# ollama pull qwen3.5:9b
```

In the same PowerShell terminal used to run APC, route every role to Ollama and use the chosen model for every role:

```powershell
$env:APC_BASE_URL = "http://localhost:11434/v1"
$env:APC_API_KEY = "ollama"
$env:APC_KEYWORD_MODEL = "qwen3.5:4b"
$env:APC_CACHE_MODEL = "qwen3.5:4b"
$env:APC_LARGE_MODEL = "qwen3.5:4b"
$env:APC_SMALL_MODEL = "qwen3.5:4b"
$env:APC_ACTOR_MODEL = "qwen3.5:4b"
$env:APC_JUDGE_MODEL = "qwen3.5:4b"
```

To use the 9B option, replace `qwen3.5:4b` in the commands with `qwen3.5:9b`. Ollama requires an API-key value for compatibility, but ignores `ollama`; no OpenAI or Together key is needed. Local inference has no per-token API charge, but uses your computer's memory, compute, and electricity. The results may differ from the paper's GPT-4o/Llama setup.

## Configure

PowerShell example:

```powershell
$env:APC_KEYWORD_MODEL = "gpt-4o-mini"
$env:APC_CACHE_MODEL = "gpt-4o-mini"
$env:APC_LARGE_MODEL = "gpt-4o"
$env:APC_JUDGE_MODEL = "llama3.1:8b"
$env:APC_SMALL_MODEL = "llama3.1:8b"
$env:APC_ACTOR_MODEL = "llama3.1:8b"

# OpenAI roles (the default provider); one key can serve all OpenAI roles.
$env:APC_API_KEY = "your-openai-api-key"

# Local Llama roles. Ollama requires an API-key field but ignores its value.
$env:APC_SMALL_BASE_URL = "http://localhost:11434/v1"
$env:APC_SMALL_API_KEY = "ollama"
$env:APC_ACTOR_BASE_URL = "http://localhost:11434/v1"
$env:APC_ACTOR_API_KEY = "ollama"
$env:APC_JUDGE_BASE_URL = "http://localhost:11434/v1"
$env:APC_JUDGE_API_KEY = "ollama"
```

Each configured endpoint must implement `POST /chat/completions`. For each role (`KEYWORD`, `CACHE`, `SMALL`, `LARGE`, `ACTOR`, `JUDGE`), set `APC_<ROLE>_BASE_URL` and `APC_<ROLE>_API_KEY` to override the shared `APC_BASE_URL` and `APC_API_KEY`. The client appends `/chat/completions`.

The OpenAI planner receives the task and actor responses, not the full context. The context is sent to the configured actor and judge endpoints (local in this example). Together AI removed its paper-era Llama model from serverless inference on March 6, 2026, so this example runs the Llama 3.1 8B Instruct model locally through Ollama instead. This uses the same model family but does not reproduce the paper's Together serving setup. Ollama lists `llama3.1:8b` at about 4.9 GB; inference needs additional memory for the model runtime and context.

## Run

```powershell
python -m apc_starter --query "What is the working capital ratio for Example Co in FY2024?" --context-file .\data\example.txt
```

The default cache is `apc_cache.json` in the current directory. Use `--cache` to choose another path. The cache is plain JSON, so it can be inspected and edited during development.

The initial implementation uses exact keyword matching, matching APC's primary design. Fuzzy retrieval, edge placement, plan scoring, and Minions integration are intentionally deferred until the core APC path is working.

After each run, the CLI prints input/output token usage by model role when the endpoint includes token counts in its response. OpenAI token counts are billable usage for the cloud calls; local Ollama token counts are local inference and are not OpenAI API usage. The number of turns and context length affect usage. If an endpoint omits usage metadata, the CLI says so rather than estimating.
