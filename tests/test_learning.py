"""The learning loop: approvals and views refit a per-install ranking model."""
import math

import pytest

import hype_judge
import learning


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_DIR", str(tmp_path / "learn"))
    monkeypatch.delenv("BILLING_ENABLED", raising=False)
    monkeypatch.delenv("LEARNING", raising=False)
    learning._model_cache.update(mtime=None, model=None)


def _clip(start, *, chat=0, laughing=False, virality=50, hype=0.5, payoff="clear"):
    return {"start": start, "end": start + 30.0,
            "hype": {"score": hype, "signals": {
                "chat": {"peak_chatters_per_second": chat, "normal_chatters_per_second": 4,
                         "excitement": chat / 4.0, "clip_requests": 0},
                "streamer": {"excitement": 1.0, "laughing": laughing}}},
            "assessment": {"verdict": "supported", "virality": virality,
                           "ratings": {"hook": "clear", "visual_action": "clear", "reaction": "clear",
                                       "clarity": "clear", "payoff": payoff}}}


def test_features_are_finite_and_cover_every_name():
    f = learning.features(_clip(10, chat=40, laughing=True, virality=80))
    assert set(f) == set(learning.FEATURES)
    assert all(math.isfinite(v) for v in f.values())
    assert f["laughing"] == 1.0 and f["virality"] == 0.8 and f["duration"] == 30.0
    assert learning.features({})["virality"] == 0.5          # unknown is neutral, not zero


def test_no_model_until_enough_decisions():
    learning.record_review("job1", [_clip(i * 60) for i in range(5)])
    learning.record_approval("job1", [_clip(i * 60) for i in range(5)], [1, 2])
    assert learning.model() is None and learning.score(_clip(0)) is None
    assert learning.status()["decisions"] == 5


def test_approvals_teach_what_you_pick():
    # You always render the clips where chat exploded and skip the quiet ones.
    for job in range(6):
        shorts = [_clip(i * 60, chat=(40 if i < 2 else 3)) for i in range(5)]
        learning.record_review(f"job{job}", shorts)
        learning.record_approval(f"job{job}", shorts, [1, 2])
    assert learning.model()["kind"] == "approvals"
    loud, quiet = learning.score(_clip(0, chat=45)), learning.score(_clip(0, chat=2))
    assert loud > 0.7 > 0.3 > quiet


def test_views_outrank_approvals_once_there_are_enough():
    for i in range(learning.MIN_VIEWS):
        clip = _clip(i * 60, laughing=bool(i % 2))
        learning.record_post("jobv", clip, f"req{i}")
    rows = [{"request_id": f"req{i}", "platform": "tiktok", "post_id": f"p{i}",
             "post_metrics": {"views": 50000 if i % 2 else 800}} for i in range(learning.MIN_VIEWS)]
    rows += [{"request_id": "req1", "platform": "tiktok", "post_id": "p1",
              "post_metrics": {"views": 100}}]            # an older snapshot never lowers views
    assert learning.sync_views(rows) == learning.MIN_VIEWS
    m = learning.model()
    assert m["kind"] == "views" and m["examples"] == learning.MIN_VIEWS
    assert learning.score(_clip(0, laughing=True)) > learning.score(_clip(0, laughing=False))
    assert learning.status()["posts_with_views"] == learning.MIN_VIEWS


def test_the_ranking_blends_the_learned_score(monkeypatch):
    monkeypatch.setattr(learning, "score", lambda clip: 0.95 if clip["start"] == 300 else 0.05)
    a = {**_clip(0, virality=70), "hype": {"score": 0.6}}
    b = {**_clip(300, virality=60), "hype": {"score": 0.6}}
    ranked = hype_judge.rank([a, b])
    assert ranked[0]["start"] == 300
    assert "like your best clips" in ranked[0]["assessment"]["reasons"]


def test_nothing_is_learned_in_the_cloud(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "1")
    learning.record_review("job", [_clip(0)])
    assert learning.status() == {"enabled": False} and learning.score(_clip(0)) is None
