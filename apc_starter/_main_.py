"""Command-line entry point for the APC starter."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import APCAgent
from .cache import JsonPlanCache
from .llm import OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standalone APC Plan-Act agent")
    parser.add_argument("--query", required=True, help="Task to solve")
    parser.add_argument("--context-file", required=True, type=Path, help="Text context available to the actor")
    parser.add_argument("--cache", default="apc_cache.json", help="Local plan cache JSON path")
    args = parser.parse_args()
    context = args.context_file.read_text(encoding="utf-8")

    def model_env(name: str, default: str) -> str:
        return os.environ.get(name, default)

    agent = APCAgent(
        OpenAICompatibleClient.from_env(),
        JsonPlanCache(args.cache),
        keyword_model=model_env("APC_KEYWORD_MODEL", "gpt-4o-mini"),
        small_model=model_env("APC_SMALL_MODEL", "gpt-4o-mini"),
        large_model=model_env("APC_LARGE_MODEL", "gpt-4o"),
        actor_model=model_env("APC_ACTOR_MODEL", "gpt-4o-mini"),
    )
    result = agent.run(args.query, context)
    print(result.answer)
    print(f"\n[APC keyword={result.keyword!r} cache_hit={result.cache_hit} actor_turns={result.actor_turns} evidence_sufficient={result.evidence_sufficient} template_added={result.template_added}]")


if __name__ == "__main__":
    main()
