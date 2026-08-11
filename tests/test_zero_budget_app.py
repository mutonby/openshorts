"""Zero-budget edition wiring: the app boots in free mode and the gateway
integration is active. Requires fastapi/httpx; skipped elsewhere."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

os.environ["BILLING_ENABLED"] = "0"
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test")

import ai_gateway  # noqa: E402
import app as app_mod  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app)


def test_config_reports_free_mode(client):
    cfg = client.get("/api/config").json()
    assert cfg["billingEnabled"] is False
    assert cfg["aiConfigured"] is True
    assert "openrouter" in cfg["aiProviders"]


def test_process_accepts_no_key_header(client):
    # No X-Gemini-Key header must not 400 when the gateway is configured.
    r = client.post("/api/process", json={
        "url": "https://example.com/video",
        "acknowledged": True,
    })
    assert r.status_code != 400, r.text[:300]


def test_saasshorts_analyze_uses_gateway(client, monkeypatch):
    script = [{
        "title": "T", "style": "ugc", "duration_seconds": 20,
        "target_platform": "tiktok", "hook_text": "H",
        "segments": [
            {"type": "hook", "start": 0, "end": 5, "narration": "a",
             "visual": "actor_talking", "broll_prompt": None},
            {"type": "problem", "start": 5, "end": 9, "narration": "b",
             "visual": "broll", "broll_prompt": "x"},
            {"type": "solution", "start": 9, "end": 16, "narration": "c",
             "visual": "actor_talking", "broll_prompt": None},
            {"type": "demo", "start": 16, "end": 21, "narration": "d",
             "visual": "broll", "broll_prompt": "y"},
            {"type": "cta", "start": 21, "end": 20, "narration": "e",
             "visual": "actor_talking", "broll_prompt": None},
        ],
        "full_narration": "a b c d e", "actor_description": "a woman",
        "hashtags": ["#x"], "caption": "c",
    }]
    calls = []

    def fake_complete_json(system, user, **kw):
        calls.append(1)
        return script, ai_gateway.AIResult(
            text=json.dumps(script), model="m", provider="openrouter")

    monkeypatch.setattr(ai_gateway, "complete_json", fake_complete_json)
    r = client.post("/api/saasshorts/analyze", json={
        "description": "An AI invoicing tool",
        "num_scripts": 1, "style": "ugc", "language": "en",
    })
    assert r.status_code == 200, r.text[:500]
    assert r.json()["scripts"] == script
    assert calls


def test_saasshorts_generate_without_paid_keys(client):
    # No fal.ai / ElevenLabs keys: must be accepted as a free-mode job.
    r = client.post("/api/saasshorts/generate", json={
        "script": {"title": "T", "full_narration": "x", "segments": [],
                   "actor_description": "a woman"},
        "video_mode": "lowcost",
        "share_to_gallery": False,
    })
    assert r.status_code == 200, r.text[:500]
    assert r.json().get("job_id")
