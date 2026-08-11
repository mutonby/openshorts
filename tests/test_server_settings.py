"""Server-side settings store: free AI keys + default caption theme."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

os.environ["BILLING_ENABLED"] = "0"

import ai_gateway  # noqa: E402
import app as app_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    """Point the settings file at a temp path and clear stored keys."""
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    monkeypatch.setattr(app_mod, "SETTINGS_DIR", str(tmp_path))
    app_mod._save_settings({"keys": {}, "caption_theme": "auto"})
    for env in app_mod.PROVIDER_KEY_ENV.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("CAPTION_THEME", raising=False)
    yield
    app_mod._apply_settings(app_mod._load_settings())


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_settings_get_reports_nothing_configured(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["configuredProviders"] == []
    assert data["captionTheme"] == "auto"
    assert "openrouter" in data["availableProviders"]


def test_settings_post_saves_key_and_applies_env(client):
    r = client.post("/api/settings", json={
        "keys": {"openrouter": "sk-or-123", "groq": "gsk-456"},
        "caption_theme": "neon",
    })
    assert r.status_code == 200
    data = r.json()
    assert set(data["configuredProviders"]) == {"openrouter", "groq"}
    assert data["aiConfigured"] is True
    # Applied live to the environment → the gateway sees it immediately.
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-or-123"
    assert os.environ.get("GROQ_API_KEY") == "gsk-456"
    assert os.environ.get("CAPTION_THEME") == "neon"
    # Persisted to disk (normalized to a list).
    stored = app_mod._load_settings()
    assert stored["keys"]["openrouter"] == ["sk-or-123"]
    assert stored["caption_theme"] == "neon"


def test_settings_post_clears_key(client):
    client.post("/api/settings", json={"keys": {"openrouter": "sk-or-123"}})
    r = client.post("/api/settings", json={"keys": {"openrouter": ""}})
    assert r.status_code == 200
    assert r.json()["configuredProviders"] == []
    assert os.environ.get("OPENROUTER_API_KEY") is None


def test_settings_post_ignores_unknown_provider(client):
    r = client.post("/api/settings", json={"keys": {"not-a-provider": "x"}})
    assert r.status_code == 200
    assert r.json()["configuredProviders"] == []


def test_caption_theme_resolution():
    import subtitles as subs

    # Unknown theme → None (caller falls back to default).
    assert subs.caption_theme("totally-made-up") is None

    beast = subs.caption_theme("beast")
    assert beast is not None
    assert beast["uppercase"] is True
    assert beast["font_name"] == "Impact"
    assert beast["effect"] == "pop"

    neon = subs.caption_theme("neon")
    assert neon["highlight_color"] == "#00FF88"
    assert neon["base_opacity"] == 0.55

    # Every theme merges over the base so all render fields exist.
    for name in subs.CAPTION_THEMES:
        style = subs.caption_theme(name)
        for field in ("alignment", "font_name", "font_size", "font_color",
                      "highlight_color", "border_color", "border_width",
                      "effect", "base_opacity", "uppercase", "max_chars",
                      "max_duration"):
            assert field in style, f"{name} missing {field}"


def test_resolve_caption_style_env_override(monkeypatch):
    import subtitles as subs

    monkeypatch.setenv("CAPTION_THEME", "tiktok")
    style = subs.resolve_caption_style()
    assert style["highlight_color"] == "#FE2C55"

    monkeypatch.setenv("CAPTION_THEME", "bogus")
    style = subs.resolve_caption_style()
    assert style == dict(subs.AUTO_CAPTION_STYLE)

    monkeypatch.delenv("CAPTION_THEME", raising=False)
    assert subs.resolve_caption_style() == dict(subs.AUTO_CAPTION_STYLE)


# --------------------------------------------------------------------------- #
# Multiple keys per provider + new endpoints
# --------------------------------------------------------------------------- #
def test_settings_post_multiple_keys_per_provider(client):
    r = client.post("/api/settings", json={
        "keys": {"openrouter": ["sk-a", "sk-b", "sk-c"]},
    })
    assert r.status_code == 200
    data = r.json()
    assert data["keyCounts"]["openrouter"] == 3
    # Env expansion: KEY, KEY_2, KEY_3
    assert os.environ.get("OPENROUTER_API_KEY") == "sk-a"
    assert os.environ.get("OPENROUTER_API_KEY_2") == "sk-b"
    assert os.environ.get("OPENROUTER_API_KEY_3") == "sk-c"
    # Gateway sees both keys in the chain.
    ai_gateway._FREE_MODELS_CACHE["at"] = 0  # force default chain path
    chain = [e for e in ai_gateway.resolve_chain("text") if e[0] == "openrouter"]
    or_keys = [k for _p, _m, k in chain]
    assert "sk-a" in or_keys and "sk-b" in or_keys and "sk-c" in or_keys


def test_settings_post_legacy_single_string(client):
    r = client.post("/api/settings", json={"keys": {"groq": "gsk-1"}})
    assert r.status_code == 200
    assert r.json()["keyCounts"]["groq"] == 1
    assert os.environ.get("GROQ_API_KEY") == "gsk-1"


def test_settings_post_replaces_keys(client):
    client.post("/api/settings", json={"keys": {"groq": ["gsk-old", "gsk-old2"]}})
    r = client.post("/api/settings", json={"keys": {"groq": ["gsk-new"]}})
    assert r.status_code == 200
    assert r.json()["keyCounts"]["groq"] == 1
    assert os.environ.get("GROQ_API_KEY") == "gsk-new"
    assert os.environ.get("GROQ_API_KEY_2") is None


def test_settings_post_clears_all_keys(client):
    client.post("/api/settings", json={"keys": {"groq": ["gsk-1", "gsk-2"]}})
    r = client.post("/api/settings", json={"keys": {"groq": []}})
    assert r.status_code == 200
    assert r.json()["configuredProviders"] == []
    assert os.environ.get("GROQ_API_KEY") is None
    assert os.environ.get("GROQ_API_KEY_2") is None


def test_summary_endpoint_missing_metadata(client):
    r = client.post("/api/summary", json={"job_id": "does-not-exist"})
    assert r.status_code in (400, 404, 500)  # job guard fires before AI


def test_platform_env_survives_settings_apply(monkeypatch, tmp_path):
    """_apply_settings must never clear a platform env key."""
    monkeypatch.setattr(app_mod, "SETTINGS_FILE", str(tmp_path / "s2.json"))
    monkeypatch.setattr(app_mod, "SETTINGS_DIR", str(tmp_path))
    app_mod._save_settings({"keys": {}, "caption_theme": "auto"})
    monkeypatch.setenv("OPENROUTER_API_KEY", "platform-key")
    app_mod._apply_settings(app_mod._load_settings())
    assert os.environ.get("OPENROUTER_API_KEY") == "platform-key"
