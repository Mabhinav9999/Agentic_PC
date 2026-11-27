"""Evaluate APC on a JSONL dataset with reference answers."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .agent import APCAgent
from .cache import JsonPlanCache
from .llm import OpenAICompatibleClient
from .prompts import correctness_judge_prompt


def model_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def make_agent(cache_path: str) -> tuple[APCAgent, OpenAICompatibleClient, str]:
    roles = ("KEYWORD", "SMALL", "LARGE", "ACTOR", "CACHE", "JUDGE")
    clients = {
        role: OpenAICompatibleClient.from_env(role)
        for role in roles
    }

    local_model = model_env("APC_LOCAL_MODEL", "qwen3.5:4b")
    judge_model = model_env("APC_JUDGE_MODEL", local_model)

    agent = APCAgent(
        clients,
        JsonPlanCache(cache_path),
        keyword_model=model_env("APC_KEYWORD_MODEL", local_model),
        small_model=model_env("APC_SMALL_MODEL", local_model),
        large_model=model_env(
            "APC_LARGE_MODEL",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        ),
        actor_model=model_env("APC_ACTOR_MODEL", local_model),
        cache_model=model_env("APC_CACHE_MODEL", local_model),
        judge_model=judge_model,
    )
    return agent, clients["JUDGE"], judge_model


def nearest_rank_percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, (percentile * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate APC on a JSONL dataset"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", default="apc_cache.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    agent, judge_client, judge_model = make_agent(args.cache)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    scored: list[bool] = []

    with args.dataset.open("r", encoding="utf-8") as source:
        with args.out.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                if args.limit is not None and len(rows) >= args.limit:
                    break

                try:
                    item = json.loads(line)
                    query = item["query"]
                    context = item["context"]
                    reference_answer = item["reference_answer"]

                    if not all(
                        isinstance(value, str)
                        for value in (query, context, reference_answer)
                    ):
                        raise ValueError(
                            "query, context, and reference_answer must be strings"
                        )

                    started = time.perf_counter()
                    result = agent.run(query, context)
                    latency_s = time.perf_counter() - started
                    latencies.append(latency_s)

                    run_error = getattr(result, "error", None)
                    correct: bool | None = None
                    judge_error = None

                    if result.answer and not run_error:
                        try:
                            judgment = judge_client.complete(
                                judge_model,
                                correctness_judge_prompt(
                                    query,
                                    reference_answer,
                                    result.answer,
                                ),
                            )
                            grade = judgment.strip().strip("\"'` ")

                            if grade == "1":
                                correct = True
                            elif grade == "0":
                                correct = False
                            else:
                                raise ValueError(
                                    f"Judge returned unexpected grade: {grade!r}"
                                )

                            scored.append(correct)
                        except Exception as exc:
                            judge_error = f"{type(exc).__name__}: {exc}"

                    row = {
                        "line": line_number,
                        "id": item.get("id"),
                        "query": query,
                        "answer": result.answer,
                        "correct": correct,
                        "keyword": result.keyword,
                        "cache_hit": result.cache_hit,
                        "actor_turns": result.actor_turns,
                        "template_added": result.template_added,
                        "evidence_sufficient": result.evidence_sufficient,
                        "latency_s": latency_s,
                        "error": run_error or judge_error,
                    }

                except Exception as exc:
                    row = {
                        "line": line_number,
                        "correct": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    requests = len(rows)
    hit_count = sum(row.get("cache_hit") is True for row in rows)
    error_count = sum(bool(row.get("error")) for row in rows)

    summary = {
        "requests": requests,
        "scored": len(scored),
        "accuracy": (
            sum(scored) / len(scored)
            if scored
            else None
        ),
        "cache_hit_rate": hit_count / requests if requests else None,
        "error_rate": error_count / requests if requests else None,
        "latency_p50_s": nearest_rank_percentile(latencies, 50),
        "latency_p95_s": nearest_rank_percentile(latencies, 95),
    }

    summary_path = args.out.with_name(
        f"{args.out.stem}.summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
