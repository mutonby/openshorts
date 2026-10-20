"""AI caption adapter tests — heuristic mode (no LLM calls)."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from opendispatch.ai.caption_adapter import (
    AdaptError,
    PLATFORM_SPECS,
    _heuristic_adapt,
    _heuristic_all,
    _enforce_limits,
    _parse_llm_json,
    _strip_code_fences,
    adapt_caption_async,
)


# ─── Heuristic per-platform ─────────────────────────────────────────────

def test_heuristic_twitter_short_returns_single_tweet():
    out = _heuristic_adapt("Hello world", "twitter")
    assert out == {"tweets": ["Hello world"]}


def test_heuristic_twitter_long_returns_thread():
    long_text = ("This is a sentence. " * 30).strip()
    out = _heuristic_adapt(long_text, "twitter")
    assert len(out["tweets"]) > 1
    for t in out["tweets"]:
        assert len(t) <= 280


def test_heuristic_bluesky_trims_to_300():
    out = _heuristic_adapt("x" * 1000, "bluesky")
    assert len(out["text"]) <= 300


def test_heuristic_telegram_passes_long_text():
    out = _heuristic_adapt("y" * 500, "telegram")
    assert out["text"] == "y" * 500


def test_heuristic_linkedin_strips_hashtags():
    out = _heuristic_adapt("Big launch today #ship #devtools", "linkedin")
    assert "#" not in out["text"]
    assert "Big launch today" in out["text"]


def test_heuristic_instagram_keeps_hashtags_at_end():
    out = _heuristic_adapt("New drop #fire #drop", "instagram")
    assert "#fire" in out["caption"]
    assert out["caption"].endswith("#drop") or "#drop" in out["caption"]


def test_heuristic_threads_caps_at_500():
    out = _heuristic_adapt("z" * 800, "threads")
    assert len(out["text"]) <= 500


def test_heuristic_all_uses_format_keys():
    out = _heuristic_all("hello", ["twitter", "bluesky", "linkedin"])
    assert set(out.keys()) == {"twitter_thread", "bluesky_post", "linkedin_post"}


# ─── Enforce limits (post-LLM safety net) ───────────────────────────────

def test_enforce_limits_trims_oversize_twitter():
    over = {"twitter_thread": {"tweets": ["x" * 400, "ok"]}}
    out = _enforce_limits(over)
    assert all(len(t) <= 280 for t in out["twitter_thread"]["tweets"])


def test_enforce_limits_trims_oversize_instagram():
    over = {"instagram_post": {"caption": "x" * 5000}}
    out = _enforce_limits(over)
    assert len(out["instagram_post"]["caption"]) <= 2200


def test_enforce_limits_ignores_unknown_keys():
    out = _enforce_limits({"twitter_thread": {"tweets": ["ok"]}, "garbage_key": {"text": "x"}})
    assert "garbage_key" not in out


# ─── LLM output parsing ─────────────────────────────────────────────────

def test_strip_code_fences_removes_json_block():
    assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_code_fences_removes_bare_block():
    assert _strip_code_fences('```\n{"a":1}\n```') == '{"a":1}'


def test_parse_llm_json_handles_clean_json():
    assert _parse_llm_json('{"x": 1}') == {"x": 1}


def test_parse_llm_json_handles_prefixed_text():
    # Some models prepend "Here is the JSON:" before the object
    raw = 'Here is the JSON:\n{"twitter_thread": {"tweets": ["hi"]}}'
    parsed = _parse_llm_json(raw)
    assert parsed["twitter_thread"]["tweets"] == ["hi"]


# ─── Public API ─────────────────────────────────────────────────────────

def test_adapt_caption_raises_on_empty_source():
    with pytest.raises(AdaptError):
        asyncio.run(adapt_caption_async("", ["twitter"]))


def test_adapt_caption_raises_on_no_valid_platforms():
    with pytest.raises(AdaptError):
        asyncio.run(adapt_caption_async("hello", ["myspace", "orkut"]))


def test_sync_adapt_caption_works_without_loop():
    from opendispatch.ai.caption_adapter import adapt_caption
    out = adapt_caption("hello world", ["twitter", "bluesky"])
    assert "twitter_thread" in out
    assert "bluesky_post" in out


def test_sync_adapt_caption_fails_loudly_inside_loop():
    """Old code crashed with a confusing run_until_complete error; now a clear one."""
    from opendispatch.ai.caption_adapter import adapt_caption
    import asyncio as _aio

    async def call_it():
        return adapt_caption("hello", ["twitter"])

    with pytest.raises(RuntimeError, match="adapt_caption_async"):
        _aio.run(call_it())


def test_adapt_caption_heuristic_explicit(monkeypatch):
    # Force heuristic so the test never makes a network call
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    out = asyncio.run(adapt_caption_async("Hello world", ["twitter", "linkedin"]))
    assert "twitter_thread" in out
    assert "linkedin_post" in out
    assert out["twitter_thread"]["tweets"] == ["Hello world"]


def test_adapt_caption_skips_unknown_platforms(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    out = asyncio.run(adapt_caption_async("hello", ["twitter", "myspace"]))
    assert "twitter_thread" in out
    # myspace isn't a known format_key
    assert all("myspace" not in k for k in out)


def test_platform_specs_have_expected_keys():
    for p, spec in PLATFORM_SPECS.items():
        assert "format_key" in spec
        assert "max_chars" in spec
        assert "style" in spec
        assert spec["max_chars"] > 0


# ─── API integration ────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    import opendispatch.api.queue as q
    importlib.reload(q)
    import opendispatch.api.app as appmod
    importlib.reload(appmod)
    from fastapi.testclient import TestClient
    return TestClient(appmod.app)


def test_ai_adapt_endpoint_heuristic(client):
    r = client.post("/ai/adapt", json={
        "text": "hello world",
        "platforms": ["twitter", "bluesky"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "twitter_thread" in body["formats"]
    assert "bluesky_post" in body["formats"]


def test_ai_adapt_rejects_empty_text(client):
    r = client.post("/ai/adapt", json={"text": "", "platforms": ["twitter"]})
    assert r.status_code == 400


def test_ai_adapt_rejects_empty_platforms(client):
    r = client.post("/ai/adapt", json={"text": "hi", "platforms": []})
    assert r.status_code == 400


def test_ai_adapt_rejects_non_object_body(client):
    r = client.post("/ai/adapt", content=b'"just a string"',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_compose_adapt_htmx_returns_html(client):
    r = client.post(
        "/_compose-adapt",
        data={"text": "hello world", "platforms": ["twitter", "bluesky"]},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    assert "Adapted" in r.text or "adapted" in r.text.lower() or "alert" in r.text.lower()


def test_compose_adapt_htmx_escapes_html_in_output(client):
    """XSS regression: adapted text must be HTML-escaped in the fragment."""
    r = client.post(
        "/_compose-adapt",
        data={"text": "<script>alert(1)</script> hello", "platforms": ["bluesky"]},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    # The raw tag must never appear verbatim — only its escaped form.
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text
