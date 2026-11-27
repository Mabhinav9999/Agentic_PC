# APC Starter

A small, provider-neutral implementation of the Agentic Plan Caching (APC) loop. It keeps reusable plan templates in a local JSON cache, adapts cached templates for new tasks, and adds templates after successful executions.

This is the standalone APC stage. It does not implement wireless placement, shared edge storage, KV or prompt-state caching, or Minions chunk scheduling yet.

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible chat-completions endpoint for each model role

The code uses only the Python standard library. By default, model roles use a local Ollama endpoint at `http://localhost:11434/v1`. You can configure roles to use other compatible endpoints. Requests and context are sent to the endpoints you configure; choose endpoints appropriate for your data.

## Model roles

APC uses separate configurable roles:

- `KEYWORD`: extracts a task keyword for cache lookup.
- `SMALL`: adapts and executes cached plans.
- `LARGE`: creates a plan when a cached plan is unavailable or fails.
- `ACTOR`: performs the task using the supplied context.
- `CACHE`: generates a reusable plan template after a successful execution.
- `JUDGE`: checks whether execution evidence is sufficient and, during evaluation, scores answers against reference answers.

The default local model name is `qwen3.5:4b` for local roles. The large planner's default model name is `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. A model name alone does not configure its provider: to use Cloudflare Workers AI for `LARGE`, configure that role's base URL and API key as described below.

The paper's model profile is described for reference; using other models or serving providers does not reproduce the paper's exact setup.

## Run all roles locally with Ollama

Install Ollama and download the model:

```powershell
ollama pull qwen3.5:4b
```

In the same PowerShell terminal used to run APC, configure the roles to use Ollama:

```powershell
$env:APC_BASE_URL = "http://localhost:11434/v1"
$env:APC_API_KEY = "ollama"

$env:APC_KEYWORD_MODEL = "qwen3.5:4b"
$env:APC_SMALL_MODEL = "qwen3.5:4b"
$env:APC_LARGE_MODEL = "qwen3.5:4b"
$env:APC_ACTOR_MODEL = "qwen3.5:4b"
$env:APC_CACHE_MODEL = "qwen3.5:4b"
$env:APC_JUDGE_MODEL = "qwen3.5:4b"
```

Ollama requires an API-key field for compatibility, but ignores the value `ollama`. Local inference has no per-token API charge, but uses your computer's memory, compute, and electricity.

## Configure a hosted large planner

Set the large planner's endpoint and secret in your terminal environment. Use the endpoint URL and API key provided by your model host:

```powershell
$env:APC_LARGE_BASE_URL = "<OpenAI-compatible endpoint base URL>"
$env:APC_LARGE_API_KEY = "<your API key>"
$env:APC_LARGE_MODEL = "<model name supported by that endpoint>"
```

Do not put real API keys in source code, README files, dataset files, or commits. If a key was exposed in a public repository or chat, revoke it and create a replacement.

Each role can have its own provider settings:

```text
APC_<ROLE>_BASE_URL
APC_<ROLE>_API_KEY
```

Supported role names are `KEYWORD`, `SMALL`, `LARGE`, `ACTOR`, `CACHE`, and `JUDGE`. Role-specific settings override the shared `APC_BASE_URL` and `APC_API_KEY`. The client appends `/chat/completions` to the configured base URL.

## Run one task

Create a text context file, then run:

```powershell
python -m apc_starter `
  --query "What is the working capital ratio for Example Co in FY2024?" `
  --context-file .\data\example.txt
```

The default cache file is `apc_cache.json` in the current directory. Choose another path with `--cache`.

## Evaluate a JSONL dataset

Each non-empty dataset line must be a JSON object containing string fields named `query`, `context`, and `reference_answer`. An optional `id` field is included in the output.

Example `data\examples.jsonl`:

```json
{"id":"example-1","query":"What is the working capital ratio?","context":"Current assets are 200. Current liabilities are 100. The ratio is current assets divided by current liabilities.","reference_answer":"The working capital ratio is 2.0."}
```

Run the evaluator:

```powershell
python -m apc_starter.evaluate `
  --dataset .\data\examples.jsonl `
  --out .\results.jsonl `
  --cache .\apc_cache.json
```

Optional: limit how many dataset rows are processed:

```powershell
python -m apc_starter.evaluate `
  --dataset .\data\examples.jsonl `
  --out .\results.jsonl `
  --cache .\apc_cache.json `
  --limit 10
```

The evaluator writes one JSON object per input row to the `--out` JSONL file. Each row includes the APC result, whole-run `latency_s`, and a `model_calls` list with metrics for the model calls made during that row.

It also writes a summary file beside the output. For `results.jsonl`, the summary is `results.summary.json`. It includes whole-run and model-call latency percentiles, total call and retry counts, cache hit rate, accuracy, and token totals when providers report token usage.

The summary's `model_calls_by_role` section breaks down call count, retries, latency percentiles, and reported input, output, and total tokens for each APC role: `KEYWORD`, `SMALL`, `LARGE`, `ACTOR`, `CACHE`, and `JUDGE`.

Token totals are unavailable when an endpoint does not report usage. The evaluator does not estimate token counts or API costs. Its accuracy score comes from the configured `JUDGE` model and is an automated evaluation, not a human-verified result.

It also writes a summary file beside the output. For `results.jsonl`, the summary is `results.summary.json`. It includes whole-run latency percentiles, model-call latency percentiles, call and retry counts, cache hit rate, accuracy, and token totals when the provider returns token usage.

Token totals are unavailable when an endpoint does not report usage. The evaluator does not estimate token counts or API costs. Its accuracy score comes from the configured `JUDGE` model and is an automated evaluation, not a human-verified result.
