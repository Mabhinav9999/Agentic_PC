"""Small OpenAI-compatible chat-completions client using the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def __init__(
    self,
    clients: dict[str, OpenAICompatibleClient],
    cache: JsonPlanCache,
    *,
    keyword_model: str,
    small_model: str,
    large_model: str,
    actor_model: str,
    cache_model: str,
    judge_model: str,
    max_turns: int = 6,
):
    self.clients = {role.upper(): client for role, client in clients.items()}
    self.cache = cache
    self.keyword_model = keyword_model
    self.small_model = small_model
    self.large_model = large_model
    self.actor_model = actor_model
    self.cache_model = cache_model
    self.judge_model = judge_model
    self.max_turns = max_turns

    @classmethod
    @classmethod
def from_env(cls, role: str) -> "OpenAICompatibleClient":
    role = role.upper()

    base_url = (
        os.environ.get(f"APC_{role}_BASE_URL")
        or os.environ.get("APC_BASE_URL", "https://api.openai.com/v1")
    )
    api_key = (
        os.environ.get(f"APC_{role}_API_KEY")
        or os.environ.get("APC_API_KEY", "")
    )

    return cls(base_url, api_key)

    def complete(self, model: str, prompt: str, *, json_mode: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError("Set APC_API_KEY before making model requests")
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Model endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach model endpoint: {exc.reason}") from exc
        try:
            return result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Unexpected response from chat-completions endpoint") from exc
