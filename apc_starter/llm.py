"""OpenAI-compatible chat-completions client using the standard library."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelCallMetrics:
    role: str
    model: str
    actual_model: str | None
    elapsed_s: float
    request_bytes: int
    response_bytes: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    attempts: int
    success: bool
    error: str | None = None


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 120,
        *,
        role: str = "UNKNOWN",
        max_retries: int = 3,
        retry_base_s: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.role = role.upper()
        self.max_retries = max(0, max_retries)
        self.retry_base_s = max(0.0, retry_base_s)
        self.call_metrics: list[ModelCallMetrics] = []

    @classmethod
    def from_env(cls, role: str) -> "OpenAICompatibleClient":
        role = role.upper().strip()

        base_url = (
            os.environ.get(f"APC_{role}_BASE_URL")
            or os.environ.get("APC_BASE_URL")
            or "http://localhost:11434/v1"
        )
        api_key = (
            os.environ.get(f"APC_{role}_API_KEY")
            or os.environ.get("APC_API_KEY")
            or "ollama"
        )

        return cls(
            base_url=base_url,
            api_key=api_key,
            timeout=int(os.environ.get("APC_TIMEOUT_S", "120")),
            role=role,
            max_retries=int(os.environ.get("APC_MAX_RETRIES", "3")),
            retry_base_s=float(os.environ.get("APC_RETRY_BASE_S", "0.5")),
        )

    @staticmethod
    def _token_count(usage: dict[str, Any], key: str) -> int | None:
        value = usage.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        json_mode: bool = False,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Set APC_API_KEY or the role-specific APC_<ROLE>_API_KEY "
                "before making model requests"
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request_data = json.dumps(payload).encode("utf-8")
        request_bytes = 0
        response_bytes = 0
        attempts = 0
        started = time.perf_counter()

        try:
            while True:
                attempts += 1
                request_bytes += len(request_data)

                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=request_data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=self.timeout,
                    ) as response:
                        response_data = response.read()
                        response_bytes += len(response_data)

                    result = json.loads(response_data.decode("utf-8"))
                    break

                except urllib.error.HTTPError as exc:
                    error_data = exc.read()
                    response_bytes += len(error_data)
                    retryable = (
                        exc.code in {408, 425, 429}
                        or 500 <= exc.code <= 599
                    )

                    if retryable and attempts <= self.max_retries:
                        delay = min(
                            self.retry_base_s * (2 ** (attempts - 1)),
                            8.0,
                        )
                        time.sleep(delay)
                        continue

                    detail = error_data.decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"Model endpoint returned HTTP {exc.code}: {detail}"
                    ) from exc

                except urllib.error.URLError as exc:
                    if attempts <= self.max_retries:
                        delay = min(
                            self.retry_base_s * (2 ** (attempts - 1)),
                            8.0,
                        )
                        time.sleep(delay)
                        continue

                    raise RuntimeError(
                        f"Could not reach model endpoint: {exc.reason}"
                    ) from exc

            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "Unexpected response from chat-completions endpoint"
                ) from exc

            if not isinstance(content, str):
                raise RuntimeError(
                    "Model endpoint returned non-text message content"
                )

            usage = result.get("usage")
            if not isinstance(usage, dict):
                usage = {}

            prompt_tokens = self._token_count(usage, "prompt_tokens")
            completion_tokens = self._token_count(
                usage,
                "completion_tokens",
            )
            total_tokens = self._token_count(usage, "total_tokens")

            if (
                total_tokens is None
                and prompt_tokens is not None
                and completion_tokens is not None
            ):
                total_tokens = prompt_tokens + completion_tokens

            self.call_metrics.append(
                ModelCallMetrics(
                    role=self.role,
                    model=model,
                    actual_model=(
                        result.get("model")
                        if isinstance(result.get("model"), str)
                        else None
                    ),
                    elapsed_s=time.perf_counter() - started,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    attempts=attempts,
                    success=True,
                )
            )

            return content.strip()

        except Exception as exc:
            self.call_metrics.append(
                ModelCallMetrics(
                    role=self.role,
                    model=model,
                    actual_model=None,
                    elapsed_s=time.perf_counter() - started,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    attempts=attempts,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
