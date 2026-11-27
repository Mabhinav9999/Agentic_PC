"""APC Plan-Act workflow with exact-key template retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .cache import JsonPlanCache, PlanTemplate
from .llm import OpenAICompatibleClient
from . import prompts


@dataclass
class AgentResult:
    answer: str
    keyword: str
    cache_hit: bool
    actor_turns: int
    template_added: bool
    evidence_sufficient: bool


class APCAgent:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        cache: JsonPlanCache,
        *,
        keyword_model: str,
        small_model: str,
        large_model: str,
        actor_model: str,
        max_turns: int = 6,
    ):
        self.client, self.cache = client, cache
        self.keyword_model, self.small_model = keyword_model, small_model
        self.large_model, self.actor_model = large_model, actor_model
        self.max_turns = max_turns

    def run(self, query: str, context: str) -> AgentResult:
        keyword = self.client.complete(self.keyword_model, prompts.keyword_prompt(query)).strip().strip('"\' .')
        if not keyword:
            raise RuntimeError("Keyword extraction returned an empty keyword")
        template = self.cache.get(keyword)
        cache_hit = template is not None
        history: list[dict[str, str]] = []
        trace: list[dict[str, str]] = []
        planner_model = self.small_model if cache_hit else self.large_model

        for _ in range(self.max_turns):
            planner_raw = self.client.complete(
                planner_model,
                prompts.planner_prompt(query, context, history, template.__dict__ if template else None),
                json_mode=True,
            )
            decision = self._json_object(planner_raw, "planner")
            status = decision.get("status")
            if status == "final":
                answer = str(decision.get("answer", "")).strip()
                if not answer:
                    raise RuntimeError("Planner returned final status without an answer")
                judged = self._json_object(
                    self.client.complete(
                        self.small_model,
                        prompts.judge_prompt(query, context, history, answer),
                        json_mode=True,
                    ),
                    "judge",
                )
                success = judged.get("success") is True
                added = False
                if success and not cache_hit:
                    workflow = self._generate_template(keyword, trace)
                    if workflow:
                        self.cache.put(PlanTemplate(
                            keyword=keyword,
                            workflow=workflow,
                            created_at=datetime.now(timezone.utc).isoformat(),
                        ))
                        added = True
                return AgentResult(answer, keyword, cache_hit, len(history), added, success)
            if status != "act":
                raise RuntimeError(f"Planner returned unsupported status: {status!r}")
            action = str(decision.get("message", "")).strip()
            if not action:
                raise RuntimeError("Planner requested an actor turn without a message")
            actor_response = self.client.complete(
                self.actor_model,
                prompts.actor_prompt(query, context, action, history),
            )
            history.append({"message": action, "response": actor_response})
            trace.extend([
                {"type": "message", "content": action},
                {"type": "output", "content": actor_response},
            ])
        raise RuntimeError(f"Planner did not finish within {self.max_turns} actor turns")

    def _generate_template(self, keyword: str, trace: list[dict[str, str]]) -> list[dict[str, str]]:
        if not trace:
            return []
        raw = self.client.complete(
            self.small_model,
            prompts.template_prompt(keyword, trace),
            json_mode=True,
        )
        data = self._json_object(raw, "template generator")
        workflow: Any = data.get("workflow")
        if not isinstance(workflow, list) or not workflow:
            return []
        normalized: list[dict[str, str]] = []
        for item in workflow:
            if not isinstance(item, dict) or item.get("type") not in {"message", "output", "answer"}:
                return []
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                return []
            normalized.append({"type": item["type"], "content": content.strip()})
        if normalized[0]["type"] != "message" or normalized[-1]["type"] != "answer":
            return []
        return normalized

    @staticmethod
    def _json_object(text: str, source: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise RuntimeError(f"{source} did not return valid JSON")
            try:
                value = json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{source} returned malformed JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"{source} response must be a JSON object")
        return value
