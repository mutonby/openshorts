"""youtube_direct: scheduling time conversion and the upload request sequence,
against a mocked YouTube API. Run: python tests/test_youtube_direct.py"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
import youtube_direct as yd

_DEFAULTS = yd.TOKEN_PATH, yd._transport


def teardown_function(_fn):
    """Other tests import the same module: leave it as it was found."""
    yd.TOKEN_PATH, yd._transport = _DEFAULTS
    yd._access.update(token=None, expires=0.0)


def _connect_fake(tmp):
    yd.TOKEN_PATH = os.path.join(tmp, "token.json")
    yd._save({"client_id": "cid", "client_secret": "sec", "refresh_token": "rt"})
    yd._access.update(token=None, expires=0.0)


def _mock(record, upload_status=200):
    def handler(req: httpx.Request):
        record.append(req)
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "at", "expires_in": 3600})
        if req.method == "POST":
            return httpx.Response(200, headers={"location": "https://upload.example/session1"})
        return httpx.Response(upload_status, json={"id": "vid123", "status": {"privacyStatus": "private"}})
    yd._transport = httpx.MockTransport(handler)


def test_publish_at_variants():
    future = datetime.now(timezone.utc) + timedelta(days=2)
    z = future.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert yd.publish_at(z) == future.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Naive local time + IANA zone (schedule-week modal): Istanbul is UTC+3.
    local = (future + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:00")
    assert yd.publish_at(local, "Europe/Istanbul") == future.strftime("%Y-%m-%dT%H:%M:00Z")
    assert yd.publish_at(None) is None
    try:
        yd.publish_at("2020-01-01T00:00:00Z")
        raise AssertionError("past time accepted")
    except yd.YouTubeDirectError:
        pass


def test_scheduled_upload_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        _connect_fake(tmp)
        assert yd.is_connected()
        clip = os.path.join(tmp, "clip.mp4")
        open(clip, "wb").write(b"\x00" * 1024)
        rec = []
        _mock(rec)
        when = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        out = yd.upload(clip, "Title", "Desc", when, "UTC")
        assert out["video_id"] == "vid123" and out["publish_at"]
        token, init, put = rec
        assert dict(httpx.QueryParams(token.content.decode()))["grant_type"] == "refresh_token"
        meta = json.loads(init.content)
        assert init.url.params["uploadType"] == "resumable"
        assert meta["status"]["privacyStatus"] == "private"
        assert meta["status"]["publishAt"] == out["publish_at"]
        assert init.headers["authorization"] == "Bearer at"
        assert put.url == "https://upload.example/session1"
        assert put.headers["content-length"] == "1024"


def test_immediate_upload_is_public_and_errors_surface():
    with tempfile.TemporaryDirectory() as tmp:
        _connect_fake(tmp)
        clip = os.path.join(tmp, "clip.mp4")
        open(clip, "wb").write(b"x")
        rec = []
        _mock(rec)
        yd.upload(clip, "T", "D")
        meta = json.loads(rec[1].content)
        assert meta["status"] == {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
        _mock([], upload_status=403)
        try:
            yd.upload(clip, "T", "D")
            raise AssertionError("403 not raised")
        except yd.YouTubeDirectError as e:
            assert "403" in str(e)


def test_not_connected_without_token():
    yd.TOKEN_PATH = "/nonexistent/token.json"
    assert not yd.is_connected()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
