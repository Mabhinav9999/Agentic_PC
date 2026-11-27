"""Prompt builders for APC's keyword, Plan-Act, adaptation, and template steps."""

from __future__ import annotations

import json
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)

def keyword_prompt(query: str) -> str:
    return f"""Can you help me summarize what is the 'task' or 'keyword'
describing the higher-level goal or intent of this query? Please answer only
with the task / keyword, which must be independent from problem-specific
details.
{query}"""

def planner_prompt(
    query: str,
    history: list[dict[str, str]],
    template_step: dict[str, str] | None,
) -> str:
    step_text = _json(template_step) if template_step else "None"

    return f"""You are the planning component of a Plan-Act agent.
Return exactly one JSON object with these keys:
- status: "act" when you need the actor to retrieve or compute something; otherwise "final".
- message: a concise next instruction to the actor when status is act; otherwise an empty string.
- answer: the final user-facing answer when status is final; otherwise an empty string.

If a reference plan step is provided, adapt that step to the current task. Do not copy entity names, dates, or values from it. Ask only for information not already present in prior actor responses. Do not claim success without sufficient evidence.

Task: {query}
Current reference plan step: {step_text}
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

def template_prompt(
    keyword: str,
    trace: list[dict[str, str]],
) -> str:
    return f"""You will see a filtered JSON trace showing the workflow of a planner language model collaborating with an actor language model to solve a task. Clean up the workflow so it can be reused as a reference template for similar tasks.

Requirements:
1. Return a JSON object with a "workflow" array. Each workflow item must have a "type" and "content".
2. The type must be "message", "output", or "answer".
3. Remove problem-specific details, including entity names, dates, and numbers.
4. Preserve the workflow order: start with a "message", then maintain the output-to-message/answer sequence, and end with an "answer".
5. Return valid JSON that can be parsed by Python's json.loads().

Task keyword: {keyword}
Filtered JSON trace: {_json(trace)}
JSON:"""

def correctness_judge_prompt(
    query: str,
    reference_answer: str,
    response: str,
) -> str:
    return f"""You are a judge that grades numeric answers to data-intensive reasoning problems.

This is the question: {query}
This is the reference answer: {reference_answer}
This is the answer given by a language model: {response}

Please grade it. Requirements:
1. Allow minor deviations, including different units, percentages instead of decimals, and small rounding or numerical errors.
2. Mark incorrect answers wrong if the calculation is materially inaccurate, the direction is wrong, or the answer invents unsupported facts.
3. Reply only with "1" for correct or "0" for incorrect."""
