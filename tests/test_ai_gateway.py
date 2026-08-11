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
