"""Small OpenAI-compatible chat-completions client using the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        return cls(
            os.environ.get("APC_BASE_URL", "https://api.openai.com/v1"),
            os.environ.get("APC_API_KEY", ""),
        )

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
