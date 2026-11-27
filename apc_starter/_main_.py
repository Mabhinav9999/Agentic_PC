"""Command-line entry point for the APC starter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import APCAgent
from .cache import JsonPlanCache
from .llm import OpenAICompatibleClient


def model_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone APC Plan-Act agent"
    )
    parser.add_argument("--query", required=True, help="Task to solve")
    parser.add_argument(
        "--context-file",
        required=True,
        type=Path,
        help="Text context available to the actor",
    )
    parser.add_argument(
        "--cache",
        default="apc_cache.json",
        help="Local plan cache JSON path",
    )
    args = parser.parse_args()

    context = args.context_file.read_text(encoding="utf-8")

    roles = ("KEYWORD", "SMALL", "LARGE", "ACTOR", "CACHE", "JUDGE")
    clients = {
        role: OpenAICompatibleClient.from_env(role)
        for role in roles
    }

    local_model = model_env("APC_LOCAL_MODEL", "qwen3.5:4b")

    agent = APCAgent(
        clients,
        JsonPlanCache(args.cache),
        keyword_model=model_env("APC_KEYWORD_MODEL", local_model),
        small_model=model_env("APC_SMALL_MODEL", local_model),
        large_model=model_env(
            "APC_LARGE_MODEL",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        ),
        actor_model=model_env("APC_ACTOR_MODEL", local_model),
        cache_model=model_env("APC_CACHE_MODEL", local_model),
        judge_model=model_env("APC_JUDGE_MODEL", local_model),
    )

    result = agent.run(args.query, context)
    print(result.answer)
    print(
        f"\n[APC keyword={result.keyword!r} "
        f"cache_hit={result.cache_hit} "
        f"actor_turns={result.actor_turns} "
        f"evidence_sufficient={result.evidence_sufficient} "
        f"template_added={result.template_added}]"
    )


if __name__ == "__main__":
    main()
