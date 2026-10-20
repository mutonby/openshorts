"""Posting a finished short: Instagram via a public URL, YouTube from disk."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from opendispatch.api.queue import JsonlQueue, bind_data_dir
from opendispatch.credentials import Credentials, InstagramCreds, YouTubeCreds
from opendispatch.local import create_posts
from opendispatch.shorts import DispatchError, ShortPost, publish_now, run_due, schedule


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers = {}

    def json(self):
        return self._payload


def _ig_creds():
    return Credentials(instagram=InstagramCreds("1789", "ig-token"))


def _yt_creds():
    return Credentials(youtube=YouTubeCreds("client", "secret", "refresh"))


def _store(tmp_path):
    bind_data_dir(tmp_path)
    return JsonlQueue()


def test_instagram_reel_uses_the_public_url_and_collaborators(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    staged = {}
    deleted = []

    def stage(path, key):
        staged["path"] = path
        staged["key"] = key
        return "https://pub-test.r2.dev/" + key

    monkeypatch.setattr("opendispatch.shorts.stage_instagram_video", stage)
    monkeypatch.setattr("opendispatch.shorts.unstage_instagram_video", lambda key: deleted.append(key))

    bodies = []

    def fake_post(url, data=None, headers=None, timeout=None):
        bodies.append({"url": url, "data": dict(data or {}), "headers": dict(headers or {})})
        if url.endswith("/media_publish"):
            return _Resp(200, {"id": "media-1"})
        return _Resp(200, {"id": "container-1"})

    def fake_get(url, headers=None, timeout=None):
        return _Resp(200, {"status_code": "FINISHED"})

    monkeypatch.setattr("opendispatch.adapters.instagram.httpx.post", fake_post)
    monkeypatch.setattr("opendispatch.adapters.instagram.httpx.get", fake_get)

    results = publish_now(
        ShortPost(
            video_path=str(video),
            platforms=["instagram"],
            caption="hello reel",
            collaborators=["@some.creator", "other_name"],
            job_id="job-1",
        ),
        _ig_creds(),
    )

    assert results[0].ok is True
    assert results[0].post_id == "media-1"
    assert staged["key"].startswith("dispatch/job-1/")
    assert staged["key"].endswith(".mp4")
    create = bodies[0]["data"]
    assert create["video_url"] == "https://pub-test.r2.dev/" + staged["key"]
    assert create["caption"] == "hello reel"
    assert create["collaborators"] == "some.creator,other_name"
    assert create["media_type"] == "REELS"
    assert create["share_to_feed"] == "true"
    assert "access_token" not in create
    assert bodies[0]["headers"]["Authorization"] == "Bearer ig-token"
    assert deleted == [staged["key"]]


def test_instagram_deletes_the_staged_object_when_publish_fails(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    deleted = []
    monkeypatch.setattr(
        "opendispatch.shorts.stage_instagram_video",
        lambda path, key: "https://pub-test.r2.dev/" + key,
    )
    monkeypatch.setattr("opendispatch.shorts.unstage_instagram_video", lambda key: deleted.append(key))

    def fake_post(url, data=None, headers=None, timeout=None):
        return _Resp(400, {"error": {"message": "bad media"}})

    monkeypatch.setattr("opendispatch.adapters.instagram.httpx.post", fake_post)

    results = publish_now(
        ShortPost(video_path=str(video), platforms=["instagram"], caption="x", job_id="job-2"),
        _ig_creds(),
    )
    assert results[0].ok is False
    assert deleted, "the public object must not stay up after a failed publish"


def test_a_fourth_collaborator_never_uploads(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    called = []
    monkeypatch.setattr(
        "opendispatch.shorts.stage_instagram_video",
        lambda path, key: called.append(key) or "https://pub-test.r2.dev/x.mp4",
    )
    results = publish_now(
        ShortPost(
            video_path=str(video),
            platforms=["instagram"],
            caption="x",
            collaborators=["a", "b", "c", "d"],
        ),
        _ig_creds(),
    )
    assert results[0].ok is False
    assert "at most 3" in results[0].error
    assert called == []


def test_youtube_gets_the_local_path_and_not_an_r2_url(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    seen = {}

    def fake_publish(unit, account=None, *, creds=None):
        seen["path"] = unit.formats["youtube_short"]["video_path"]
        seen["creds"] = creds
        assert "video_url" not in unit.formats["youtube_short"]
        return True, "yt-1", ""

    monkeypatch.setattr("opendispatch.adapters.youtube.publish", fake_publish)
    staged = []
    monkeypatch.setattr("opendispatch.shorts.stage_instagram_video", lambda *a, **k: staged.append(1))

    results = publish_now(
        ShortPost(video_path=str(video), platforms=["youtube"], title="A short", caption="desc"),
        _yt_creds(),
    )
    assert results[0].ok is True
    assert results[0].post_id == "yt-1"
    assert seen["path"] == str(video)
    assert seen["creds"].refresh_token == "refresh"
    assert staged == []


def test_an_explicit_title_over_100_characters_is_rejected(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    called = []
    monkeypatch.setattr(
        "opendispatch.adapters.youtube.publish",
        lambda *a, **k: called.append(1) or (True, "nope", ""),
    )
    results = publish_now(
        ShortPost(video_path=str(video), platforms=["youtube"], title="t" * 101, caption="ok"),
        _yt_creds(),
    )
    assert results[0].ok is False
    assert "100" in results[0].error
    assert called == []


def test_queue_keeps_the_credential_reference_and_not_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_TOKEN", "SUPERSECRET")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    store = _store(tmp_path / "q")
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    ids = schedule(
        ShortPost(
            video_path=str(video),
            platforms=["instagram"],
            caption="later",
            scheduled_for=future,
        ),
        store=store,
    )
    raw = json.dumps(store.get(ids[0]))
    assert "SUPERSECRET" not in raw
    assert store.get(ids[0])["unit"]["creds_ref"] == "env"


def test_a_future_row_is_not_claimed_and_a_due_row_publishes_once(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    store = _store(tmp_path / "q")
    calls = []

    def fake_dispatch(unit, platform, creds=None):
        calls.append(platform)
        return True, "posted", ""

    monkeypatch.setattr("opendispatch.shorts.dispatch_unit", fake_dispatch)
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    schedule(
        ShortPost(video_path=str(video), platforms=["youtube"], title="t", caption="c", scheduled_for=future),
        store=store,
    )
    assert run_due(store, lambda ref, platform: _yt_creds()) == 0
    assert calls == []

    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    schedule(
        ShortPost(video_path=str(video), platforms=["youtube"], title="t", caption="c", scheduled_for=past),
        store=store,
    )
    assert run_due(store, lambda ref, platform: _yt_creds()) == 1
    assert calls == ["youtube"]
    assert run_due(store, lambda ref, platform: _yt_creds()) == 0
    assert calls == ["youtube"]
    published = [row for row in store.list_all() if row["status"] == "published"]
    assert len(published) == 1
    assert published[0]["post_id"] == "posted"


def test_create_posts_rejects_a_missing_file_and_a_missing_token(tmp_path, monkeypatch):
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    payload = {
        "job_id": "job",
        "timezone": "UTC",
        "mode": "now",
        "clips": [{"index": 0, "platforms": ["instagram"], "title": "t", "caption": "c"}],
    }
    with pytest.raises(DispatchError, match="not found"):
        create_posts(payload, resolve_file=lambda index: str(tmp_path / "missing.mp4"), store=None, now=now)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    monkeypatch.delenv("IG_USER_ID", raising=False)
    monkeypatch.delenv("IG_TOKEN", raising=False)
    with pytest.raises(DispatchError, match="IG_TOKEN"):
        create_posts(payload, resolve_file=lambda index: str(video), store=None, now=now)


def test_create_posts_enqueues_one_row_per_platform(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    monkeypatch.setenv("IG_USER_ID", "1")
    monkeypatch.setenv("IG_TOKEN", "t")
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "c")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "s")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
    monkeypatch.setenv("R2_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_PUBLIC_BASE", "https://pub-test.r2.dev")
    store = _store(tmp_path / "q")
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    result = create_posts(
        {
            "job_id": "job-9",
            "timezone": "Asia/Kolkata",
            "mode": "at",
            "clips": [{
                "index": 0,
                "platforms": ["youtube", "instagram"],
                "title": "Hello",
                "caption": "Caption",
                "collaborators": ["friend"],
                "when": "2026-10-21T18:30",
            }],
        },
        resolve_file=lambda index: str(video),
        store=store,
        now=now,
    )
    assert len(result["row_ids"]) == 2
    assert {item["platform"] for item in result["planned"]} == {"youtube", "instagram"}
    assert result["planned"][0]["local_time"] == "2026-10-21 18:30"
    stored = store.list_all()
    assert len(stored) == 2
    blob = json.dumps(stored)
    assert "secret" not in blob.lower() or "R2" not in blob
    assert "ig-token" not in blob
    assert all(row["unit"]["creds_ref"] == "env" for row in stored)
    assert "SUPERSECRET" not in blob
    assert "https://pub-test.r2.dev" not in blob
