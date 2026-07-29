import os
import json
import asyncio
import time
from typing import List, Dict, Optional

import httpx

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_POLL_INTERVAL,
    OPENROUTER_CACHE_FILE,
    MODEL_ROTATION_STRATEGY,
)

DEFAULT_MODELS_ENDPOINT = "https://api.openrouter.ai/v1/models"

class OpenRouterManager:
    def __init__(self):
        self._api_key = OPENROUTER_API_KEY
        self._models: List[Dict] = []
        self._index = 0
        self._lock = asyncio.Lock()
        self._last_refresh = 0
        self._ttl = OPENROUTER_POLL_INTERVAL
        self._cache_file = OPENROUTER_CACHE_FILE
        self._strategy = MODEL_ROTATION_STRATEGY
        self._unhealthy = set()

        # Try to load cache
        try:
            self._load_cache()
        except Exception:
            # ignore cache load errors
            pass

    def _load_cache(self):
        if os.path.exists(self._cache_file):
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._models = data.get("models", [])
                self._last_refresh = data.get("last_refresh", 0)

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump({"models": self._models, "last_refresh": self._last_refresh}, f)
        except Exception:
            pass

    async def _fetch_models(self) -> List[Dict]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = DEFAULT_MODELS_ENDPOINT
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        # OpenRouter models endpoint returns a list under 'models' or a raw list
        models = []
        if isinstance(payload, dict) and "models" in payload:
            models = payload["models"]
        elif isinstance(payload, list):
            models = payload
        else:
            # unknown shape, try to coerce
            models = payload.get("data") if isinstance(payload, dict) else []

        filtered = []
        for m in models:
            # Heuristic filter for "free/public" models. This may need adjusting.
            visibility = m.get("visibility") or m.get("public") or m.get("isPublic")
            name = m.get("name") or m.get("id") or ""
            desc = (m.get("description") or "").lower()
            tags = m.get("tags") or []

            is_public = False
            if isinstance(visibility, bool):
                is_public = visibility
            elif isinstance(visibility, str):
                is_public = visibility.lower() in ("public", "open")

            # Accept if clearly public OR description/tags mention free/open
            if is_public or "free" in desc or "open" in desc or any("free" in str(t).lower() for t in tags):
                filtered.append(m)
            else:
                # fallback: include models owned by openrouter or with short names
                owner = m.get("owner") or m.get("creator") or ""
                if owner and "openrouter" in str(owner).lower():
                    filtered.append(m)
                elif len(name) < 40:
                    filtered.append(m)

        return filtered

    async def refresh(self) -> None:
        try:
            models = await self._fetch_models()
            # Simplify model entries to dicts with id/name
            cleaned = []
            for m in models:
                mid = m.get("id") or m.get("name") or m.get("model_id")
                if not mid:
                    continue
                cleaned.append({
                    "id": mid,
                    "raw": m,
                })

            async with self._lock:
                self._models = cleaned
                self._last_refresh = int(time.time())
                self._unhealthy = set()
                # reset index to reduce favoring older models too much
                self._index = 0
                self._save_cache()
        except Exception:
            # Do not raise; keep existing models
            return

    async def ensure_fresh(self) -> None:
        now = time.time()
        if now - self._last_refresh > self._ttl:
            await self.refresh()

    async def get_model(self) -> Optional[Dict]:
        await self.ensure_fresh()
        async with self._lock:
            if not self._models:
                return None

            # filter healthy models
            healthy = [m for m in self._models if m["id"] not in self._unhealthy]
            if not healthy:
                # if all unhealthy, reset unhealthy set and use all
                self._unhealthy = set()
                healthy = self._models

            if self._strategy == "random":
                import random
                return random.choice(healthy)

            # round_robin
            model = healthy[self._index % len(healthy)]
            self._index = (self._index + 1) % len(healthy)
            return model

    async def mark_unhealthy(self, model_id: str) -> None:
        async with self._lock:
            self._unhealthy.add(model_id)


# Singleton for easy imports
_manager: Optional[OpenRouterManager] = None


def get_manager() -> OpenRouterManager:
    global _manager
    if _manager is None:
        _manager = OpenRouterManager()
    return _manager
