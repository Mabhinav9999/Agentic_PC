"""Prompt builders for APC's keyword, Plan-Act, adaptation, and template steps."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def keyword_prompt(query: str) -> str:
    return f"""Extract the higher-level task or intent from the user query.
Return only a short keyword or task phrase, independent of names, dates, numbers, and other instance-specific details.

Query: {query}
Keyword:"""


def planner_prompt(
    query: str,
    history: list[dict[str, str]],
    template: dict[str, Any] | None,
) -> str:
    template_text = _json(template) if template else "None"

    return f"""You are the planning component of a Plan-Act agent.
Return exactly one JSON object with these keys:
- status: "act" when you need the actor to retrieve or compute something; otherwise "final".
- message: a concise next instruction to the actor when status is act; otherwise an empty string.
- answer: the final user-facing answer when status is final; otherwise an empty string.

If a reference plan template is provided, adapt its method to the current task. Do not copy entity names, dates, or values from the template. Ask only for information not already present in prior actor responses. Do not claim success without sufficient evidence.

Task: {query}
Reference plan template: {template_text}
Previous actions and actor responses: {_json(history)}
JSON:"""


def actor_prompt(query: str, context: str, action: str, history: list[dict[str, str]]) -> str:
    return f"""You are the acting component of a Plan-Act agent. Follow the planner's requested action using only the supplied context and prior results.
If the context does not contain the requested information, say what is missing. Do not invent facts.

Task: {query}
Current action: {action}
Context: {context}
Prior actor responses: {_json(history)}
Actor response:"""


def judge_prompt(query: str, context: str, history: list[dict[str, str]], answer: str) -> str:
    return f"""Assess whether the proposed answer is sufficiently supported by the provided context and answers the task.
This is an evidence-sufficiency check, not a ground-truth correctness guarantee. Return exactly JSON: {{\"success\": true}} or {{\"success\": false}}. Be conservative: unsupported or invented facts are failures.

Task: {query}
Available context: {context}
Actor evidence: {_json(history)}
Proposed answer: {answer}
JSON:"""


def template_prompt(keyword: str, trace: list[dict[str, str]]) -> str:
    return f"""Generalize this successful agent execution into a reusable plan template.
Return only valid JSON with a \"workflow\" array of objects. Each object must have \"type\" (\"message\", \"output\", or \"answer\") and \"content\" (string).
Preserve the order of actions and expected information. Remove entity names, dates, numbers, and other task-specific values. Do not include the actual answer. Start with a message and end with an answer step.

Task keyword: {keyword}
Successful execution trace: {_json(trace)}
JSON:"""
