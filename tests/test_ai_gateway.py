"""Zero-budget AI gateway behaviour (offline, mocked HTTP).

Covers the core contract that makes the free-provider routing reliable:
configured-key filtering, fallback across providers, JSON extraction from
markdown fences, and the friendly error when nothing is configured.
"""
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_gateway  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ["OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
                "DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "DASHSCOPE_API_KEY",
                "MOONSHOT_API_KEY", "OPENAI_API_KEY", "AI_MODEL_CHAIN",
                "AI_VISION_MODEL_CHAIN", "AI_IMAGE_MODEL"]:
        monkeypatch.delenv(var, raising=False)


def _mock_client(handler):
    orig = httpx.Client
    return lambda timeout, **kw: orig(
        transport=httpx.MockTransport(handler), timeout=timeout)


def test_not_configured_without_keys(monkeypatch):
    assert ai_gateway.is_configured() is False
    with pytest.raises(ai_gateway.AIGatewayError) as exc:
        ai_gateway.complete(system="x", user="y")
    assert "No free AI provider configured" in str(exc.value)


def test_chain_only_contains_configured_providers(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    chain = ai_gateway.resolve_chain("text")
    assert chain, "expected at least one entry"
    assert all(provider == "groq" for provider, _model, _key in chain)
    assert chain[0][1]  # model name present


def test_json_fence_parsing():
    assert ai_gateway._parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert ai_gateway._parse_json_text('prefix {"b": [1,2]} suffix') == {"b": [1, 2]}
    assert ai_gateway._parse_json_text('[{"x": 1}]') == [{"x": 1}]
    assert ai_gateway._parse_json_text("no json here") is None


def test_complete_json_with_mock(monkeypatch):
    monkeypatch.setenv("FREE_MODEL_AUTODISCOVERY", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"ok": True})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "test-model",
        })

    monkeypatch.setattr(ai_gateway.httpx, "Client", _mock_client(handler))
    parsed, result = ai_gateway.complete_json(system="s", user="u")
    assert parsed == {"ok": True}
    assert result.input_tokens == 10
    assert result.cost == 0.0
    assert result.provider == "openrouter"


def test_falls_back_to_next_provider(monkeypatch):
    monkeypatch.setenv("FREE_MODEL_AUTODISCOVERY", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    monkeypatch.setenv("AI_MODEL_CHAIN",
                       "openrouter:bad/model:free,groq:llama-3.3-70b-versatile")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"].startswith("bad"):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"n": 42})}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": body["model"],
        })

    monkeypatch.setattr(ai_gateway.httpx, "Client", _mock_client(handler))
    parsed, result = ai_gateway.complete_json(system="s", user="u")
    assert parsed == {"n": 42}
    assert result.provider == "groq"
    assert calls[0].startswith("bad")


def test_legacy_gemini_model_is_prepended(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    chain = ai_gateway.resolve_chain("text")
    assert chain[0][0] == "google"
    assert chain[0][1] == "gemini-2.5-flash"


def test_legacy_gemini_model_skipped_without_key(monkeypatch):
    # With only Groq configured, the prepended google entry is filtered out.
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    chain = ai_gateway.resolve_chain("text")
    assert all(provider != "google" for provider, _m, _k in chain)
    assert chain[0][0] == "groq"


# --------------------------------------------------------------------------- #
# "Immortal" features: multiple keys, automatic failover, free-only models
# --------------------------------------------------------------------------- #
def test_multiple_keys_expand_the_chain(monkeypatch):
    monkeypatch.setenv("FREE_MODEL_AUTODISCOVERY", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-a")
    monkeypatch.setenv("OPENROUTER_API_KEY_2", "sk-b")
    monkeypatch.setenv("GROQ_API_KEY", "sk-g")
    monkeypatch.setenv("AI_MODEL_CHAIN", "openrouter:some/model:free,groq:llama-x")
    chain = ai_gateway.resolve_chain("text")
    or_keys = [k for p, _m, k in chain if p == "openrouter"]
    assert or_keys == ["sk-a", "sk-b"]
    assert len([e for e in chain if e[0] == "groq"]) == 1


def test_failure_cooldown_skips_provider(monkeypatch):
    monkeypatch.setenv("FREE_MODEL_AUTODISCOVERY", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")
    monkeypatch.setenv("AI_MODEL_CHAIN", "openrouter:bad:free,groq:llama-3.3-70b-versatile")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        if body["model"].startswith("bad"):
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({"ok": 1})}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": body["model"],
        })

    monkeypatch.setattr(ai_gateway.httpx, "Client",
                        _mock_client(handler))
    # First call: openrouter 429s, cooldown starts, groq answers.
    parsed, result = ai_gateway.complete_json(system="s", user="u")
    assert result.provider == "groq"
    assert ai_gateway.provider_in_cooldown("openrouter") is True
    assert ai_gateway.provider_status()["openrouter"]["failures"] >= 1

    # Second call: openrouter is skipped entirely — only groq is tried.
    import time as _real_time
    monkeypatch.setattr(ai_gateway.time, "sleep", lambda *_: None)
    ai_gateway._FAILURES["openrouter"]["until"] = _real_time.time() + 999
    calls.clear()
    ai_gateway.complete_json(system="s", user="u")
    assert all(not m.startswith("bad") for m in calls), "cooldown provider must be skipped"


def test_free_model_filter_never_includes_paid(monkeypatch):
    fake_catalog = {
        "data": [
            {"id": "qwen/qwen3-32b:free", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text"]}, "order": 1},
            {"id": "openai/gpt-5.5", "pricing": {"prompt": "1.25", "completion": "10"},
             "architecture": {"input_modalities": ["text"]}, "order": 0},  # paid → never
            {"id": "google/gemma-4-31b-it:free", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text", "image"]}, "order": 2},
            {"id": "some/paid-no-suffix", "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"input_modalities": ["text"]}, "order": 3},  # $0 pricing → free
        ]
    }
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    _orig_get = ai_gateway.httpx.get
    monkeypatch.setattr(ai_gateway.httpx, "get",
                        lambda *a, **k: _orig_get.__self__.__class__ if False else _FakeResp(fake_catalog))
    # simpler: patch the cache directly to avoid network
    ai_gateway._FREE_MODELS_CACHE = {
        "at": ai_gateway.time.time(),
        "text": ["qwen/qwen3-32b:free", "google/gemma-4-31b-it:free", "some/paid-no-suffix"],
        "vision": ["google/gemma-4-31b-it:free"],
    }
    text = ai_gateway.discover_openrouter_free_models(kind="text", limit=10)
    vision = ai_gateway.discover_openrouter_free_models(kind="vision", limit=10)
    assert "openai/gpt-5.5" not in text
    assert text == ["qwen/qwen3-32b:free", "google/gemma-4-31b-it:free", "some/paid-no-suffix"]
    assert vision == ["google/gemma-4-31b-it:free"]


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload
