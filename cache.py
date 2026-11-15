"""Local JSON storage for reusable APC plan templates."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PlanTemplate:
    keyword: str
    workflow: list[dict[str, str]]
    created_at: str
    successes: int = 1
    version: int = 1


class JsonPlanCache:
    """Exact-key cache persisted as JSON, with atomic replacement on writes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, PlanTemplate] = {}
        self._load()

    @staticmethod
    def normalize(keyword: str) -> str:
        return " ".join(keyword.casefold().split()).strip(" .,:;!?\"'")

    def get(self, keyword: str) -> PlanTemplate | None:
        return self._entries.get(self.normalize(keyword))

    def put(self, template: PlanTemplate) -> None:
        key = self.normalize(template.keyword)
        if not key:
            raise ValueError("Cannot cache a template with an empty keyword")
        old = self._entries.get(key)
        if old:
            template.successes = old.successes + 1
        self._entries[key] = template
        self._save()

    def all(self) -> list[PlanTemplate]:
        return list(self._entries.values())

    def _load(self) -> None:
        if not self.path.exists():
            return
        data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in data.items():
            self._entries[key] = PlanTemplate(**value)

    def _save(self) -> None:
        payload = {key: asdict(value) for key, value in self._entries.items()}
        fd, temp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
