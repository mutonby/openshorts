"""Publish kit: daily trending refresh + per-clip viral title/desc/hashtags."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

os.environ["BILLING_ENABLED"] = "0"

import ai_gateway  # noqa: E402
import publish_kit as pk  # noqa: E402


SAMPLE_METADATA = {
    "transcript": {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Hello there",
             "words": [{"word": "Hello", "start": 0.0, "end": 1.0},
                       {"word": "there", "start": 1.0, "end": 2.0}]},
            {"start": 4.0, "end": 8.0, "text": "world of AI",
             "words": [{"word": "world", "start": 4.0, "end": 5.0},
                       {"word": "of", "start": 5.0, "end": 5.5},
                       {"word": "AI", "start": 5.5, "end": 6.5}]},
        ],
    },
    "shorts": [
        {"start": 0.5, "end": 6.0, "title": "AI tip",
         "video_title_for_youtube_short": "The AI tip"},
        {"start": 6.0, "end": 8.0, "title": "Outro", "video_title_for_youtube_short": "Outro"},
    ],
}


@pytest.fixture(autouse=True)
def _trend_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(pk, "_cache_path",
                        lambda: str(tmp_path / "trending_cache.json"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield


def test_build_clip_context_picks_clip_words():
    clip, text = pk.build_clip_context(SAMPLE_METADATA, 0)
    assert clip["title"] == "AI tip"
    assert "Hello" in text and "world" in text


def test_build_clip_context_out_of_range():
    clip, text = pk.build_clip_context(SAMPLE_METADATA, 99)
    assert clip is None and text == ""


def test_fetch_trends24_parsing(monkeypatch):
    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"trends": [
                {"name": "#AI", "rank": 1, "tweet_count": 1000},
                "plain topic",
                {"name": ""},
                "Another Trend",
            ]}

    monkeypatch.setattr(pk.httpx, "get", lambda *a, **k: _FakeResp())
    out = pk._fetch_trends24("US")
    assert [t["topic"] for t in out] == ["#AI", "plain topic", "Another Trend"]


def test_fetch_trending_topics_caches_and_refreshes(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(region):
        calls["n"] += 1
        return [{"topic": f"trend-{calls['n']}"}]

    monkeypatch.setattr(pk, "_fetch_trends24", fake_fetch)
    topics, source = pk.fetch_trending_topics("US", "en")
    assert source == "trends24"
    assert topics == ["trend-1"]
    # Second call within TTL → served from cache, no network hit.
    topics2, source2 = pk.fetch_trending_topics("US", "en")
    assert source2 == "cache"
    assert topics2 == ["trend-1"]
    assert calls["n"] == 1


def test_fetch_trending_ai_fallback(monkeypatch):
    monkeypatch.setattr(pk, "_fetch_trends24", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))

    def fake_ai(region, language):
        return [{"topic": "ai-trend-1"}]

    monkeypatch.setattr(pk, "_fetch_trends_ai", fake_ai)
    topics, source = pk.fetch_trending_topics("US", "en")
    assert source == "ai"
    assert topics == ["ai-trend-1"]


def test_generate_publish_kit(monkeypatch):
    monkeypatch.setattr(pk, "_fetch_trends24", lambda *a: [{"topic": "viral thing"}])

    def fake_complete_json(system, user, **kw):
        assert "TODAY" in user or "TODAY'S DATE" in user or "TRENDING" in user
        return {
            "title": "This AI hack changed everything",
            "description": "Watch this. It's wild.",
            "hashtags": [
                {"tag": "#AI", "source": "trending", "why": "trending today"},
                {"tag": "#productivity", "source": "niche", "why": "fits content"},
                "#shorts",
            ],
        }, ai_gateway.AIResult(text="{}", model="m", provider="openrouter")

    monkeypatch.setattr(ai_gateway, "complete_json", fake_complete_json)
    kit = pk.generate_publish_kit(SAMPLE_METADATA, 0, language="en", region="US")
    assert kit["title"] == "This AI hack changed everything"
    assert kit["trend_source"] == "trends24"
    tags = [h["tag"] for h in kit["hashtags"]]
    assert "#AI" in tags and "#shorts" in tags
    assert len(tags) == len(set(t.lower() for t in tags))  # deduped


def test_generate_publish_kit_no_text_raises():
    meta = {"transcript": {"language": "en", "segments": []},
            "shorts": [{"start": 0, "end": 5}]}
    with pytest.raises(ValueError):
        pk.generate_publish_kit(meta, 0, language="en", region="US")


def test_format_kit_text():
    kit = {"title": "T", "description": "D", "hashtags": [
        {"tag": "#a", "source": "trending", "why": ""},
        {"tag": "#b", "source": "niche", "why": ""},
    ]}
    out = pk.format_kit_text(kit)
    assert "#a #b" in out
    assert "T" in out and "D" in out


# --------------------------------------------------------------------------- #
# Endpoint wiring
# --------------------------------------------------------------------------- #
def test_publish_kit_endpoint(monkeypatch):
    import app as app_mod
    from fastapi.testclient import TestClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = TestClient(app_mod.app)

    # Create a fake job with metadata on disk.
    job_id = "publishkit-test"
    job_dir = os.path.join(app_mod.OUTPUT_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    meta_path = os.path.join(job_dir, f"{job_id}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_METADATA, f)
    app_mod.jobs[job_id] = {
        "status": "completed",
        "output_dir": job_dir,
        "result": {"clips": [{"video_url": f"/videos/{job_id}/x.mp4"}]},
        "user_id": None,
    }

    def fake_generate(metadata, clip_index, language="auto", region="US"):
        return {
            "title": "T", "description": "D",
            "hashtags": [{"tag": "#AI", "source": "trending", "why": ""}],
            "trending_topics": ["AI"], "trend_source": "test",
            "language": "en", "region": region,
            "generated_at": "2026-08-11",
        }

    import publish_kit as pk_mod
    monkeypatch.setattr(pk_mod, "generate_publish_kit", fake_generate)

    r = client.post("/api/publish-kit", json={
        "job_id": job_id, "clip_index": 0, "region": "BD",
    })
    assert r.status_code == 200, r.text[:300]
    data = r.json()
    assert data["kit"]["title"] == "T"
    assert data["kit"]["region"] == "BD"
    assert data["kit"]["hashtags"][0]["tag"] == "#AI"
