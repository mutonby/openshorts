"""Test session isolation.

The API reads configuration from the process environment at import time
(API_TOKEN, queue backend URLs, data directories). A developer who exports
OPEN_DISPATCH_API_TOKEN to talk to their running instance would otherwise
silently turn the whole suite red — 40 failures that look like code bugs but
are environment leakage.

Pin the variables the app reads at import time so the suite is deterministic
regardless of the developer's shell. Individual tests that need a specific
value still use monkeypatch.setenv and restore it themselves.
"""
from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

# Env vars read by api.app / api.queue / media.paths at import time.
_PINNED = {
    "OPEN_DISPATCH_API_TOKEN": "",
    "OPEN_DISPATCH_URL": "",
    "OPEN_DISPATCH_DATA": "",
    "OPEN_DISPATCH_MEDIA_DIR": "",
    "REDIS_URL": "",
    "DATABASE_URL": "",
    "WEBHOOK_SECRET": "",
    "AI_CAPTION_PROVIDER": "",
    "OPENROUTER_API_KEY": "",
    "OLLAMA_HOST": "",
}


@pytest.fixture(autouse=True, scope="session")
def _isolate_env() -> Generator[None, None, None]:
    saved = {k: os.environ.get(k) for k in _PINNED}
    for key in _PINNED:
        os.environ[key] = ""
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep queue/media writes inside the test's own tmp dir."""
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
