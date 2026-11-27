"""Redis-backed private device and shared edge plan caches."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import redis
from redis.exceptions import WatchError


@dataclass
class PlanTemplate:
    keyword: str
    workflow: list[dict[str, str]]
    created_at: str
    successes: int = 1
    version: int = 1


class RedisPlanCache:
    """Exact-key plan cache stored in Redis under a namespace."""

    def __init__(
        self,
        client: Any,
        namespace: str,
        *,
        cache_name: str,
    ) -> None:
        self.client = client
        self.namespace = namespace.strip(":")
        self.cache_name = cache_name

    @staticmethod
    def normalize(keyword: str) -> str:
        return " ".join(keyword.casefold().split()).strip(" .,:;!?\"'")

    def _redis_key(self, keyword: str) -> str:
        normalized = self.normalize(keyword)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{self.namespace}:{digest}"

    def get(self, keyword: str) -> PlanTemplate | None:
        key = self._redis_key(keyword)
        payload = self.client.get(key)
        if payload is None:
            return None

        try:
            value = json.loads(payload)
            return PlanTemplate(**value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid plan template stored in {self.cache_name} Redis cache"
            ) from exc

    def put(
        self,
        template: PlanTemplate,
        *,
        increment_success: bool = True,
    ) -> None:
        normalized = self.normalize(template.keyword)
        if not normalized:
            raise ValueError("Cannot cache a template with an empty keyword")

        key = self._redis_key(normalized)
        new_value = asdict(template)

        # Use WATCH/MULTI so simultaneous writers do not overwrite the
        # success count with a stale value.
        for _ in range(5):
            try:
                with self.client.pipeline() as pipe:
                    pipe.watch(key)
                    old_payload = pipe.get(key)

                    if old_payload is not None:
                        old_value = json.loads(old_payload)
                        old_successes = int(old_value.get("successes", 1))
                        if increment_success:
                            new_value["successes"] = old_successes + 1
                        else:
                            new_value["successes"] = old_successes

                    pipe.multi()
                    pipe.set(
                        key,
                        json.dumps(new_value, ensure_ascii=False),
                    )
                    pipe.execute()
                    return
            except WatchError:
                continue

        raise RuntimeError(
            f"Could not update {self.cache_name} Redis cache after concurrent writes"
        )

    def all(self) -> list[PlanTemplate]:
        pattern = f"{self.namespace}:*"
        templates: list[PlanTemplate] = []

        for key in self.client.scan_iter(match=pattern):
            payload = self.client.get(key)
            if payload is None:
                continue
            try:
                templates.append(PlanTemplate(**json.loads(payload)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid plan template stored in {self.cache_name} Redis cache"
                ) from exc

        return templates


class TieredPlanCache:
    """Private device cache backed by Redis, with a shared edge Redis cache."""

    def __init__(
        self,
        device_cache: RedisPlanCache,
        edge_cache: RedisPlanCache,
        *,
        share_new_plans_to_edge: bool = False,
    ) -> None:
        self.device_cache = device_cache
        self.edge_cache = edge_cache
        self.share_new_plans_to_edge = share_new_plans_to_edge

    @classmethod
    def from_env(cls) -> "TieredPlanCache":
        device_id = os.environ.get("APC_DEVICE_ID", "device-1").strip()
        if not device_id:
            raise ValueError("APC_DEVICE_ID cannot be empty")

        device_url = os.environ.get(
            "APC_DEVICE_REDIS_URL",
            "redis://localhost:6379/0",
        )
        edge_url = os.environ.get(
            "APC_EDGE_REDIS_URL",
            "redis://localhost:6379/1",
        )
        share_new_plans = os.environ.get(
            "APC_SHARE_NEW_PLANS_TO_EDGE",
            "false",
        ).strip().casefold() in {"1", "true", "yes", "on"}

        device_client = redis.Redis.from_url(
            device_url,
            decode_responses=True,
        )
        edge_client = redis.Redis.from_url(
            edge_url,
            decode_responses=True,
        )

        return cls(
            device_cache=RedisPlanCache(
                device_client,
                f"apc:device:{device_id}",
                cache_name=f"device {device_id}",
            ),
            edge_cache=RedisPlanCache(
                edge_client,
                "apc:edge",
                cache_name="shared edge",
            ),
            share_new_plans_to_edge=share_new_plans,
        )

    def get(self, keyword: str) -> PlanTemplate | None:
        template = self.device_cache.get(keyword)
        if template is not None:
            return template

        template = self.edge_cache.get(keyword)
        if template is not None:
            # Promote a shared edge hit into this device's private cache
            # without counting the promotion as a new successful execution.
            self.device_cache.put(template, increment_success=False)

        return template

    def put(self, template: PlanTemplate) -> None:
        # Successful new templates are always saved privately first.
        self.device_cache.put(template)

        # Edge sharing is opt-in because device plans may contain private
        # workflow details. Enable it only when sharing is intended.
        if self.share_new_plans_to_edge:
            self.edge_cache.put(template, increment_success=False)

    def publish_to_edge(self, template: PlanTemplate) -> None:
        """Explicitly publish a template to the shared edge cache."""
        self.edge_cache.put(template, increment_success=False)

    def all(self) -> list[PlanTemplate]:
        # Return a merged view, preferring the device's private copy.
        merged = {
            self.edge_cache.normalize(template.keyword): template
            for template in self.edge_cache.all()
        }
        merged.update({
            self.device_cache.normalize(template.keyword): template
            for template in self.device_cache.all()
        })
        return list(merged.values())
