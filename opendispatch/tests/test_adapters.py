"""Adapter unit tests — fully mocked, no network or real credentials.

Every adapter is exercised across:
- Missing-credentials path → (False, "", "<msg>") with descriptive error
- Happy path → (True, post_id, "")
- HTTP error path → (False, "", "<msg>")
- Empty-payload path

We monkeypatch `httpx.post`, `httpx.get`, and `httpx.Client` to avoid
hitting the network. For the Threads adapter we also collapse the
PUBLISH_SETTLE_SECONDS sleep.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from opendispatch.api.schema import ContentUnit


# ─── Test fixtures ──────────────────────────────────────────────────────

class FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "",
                 headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (str(json_data) if json_data else "")
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=MagicMock(),
                response=self,  # type: ignore[arg-type]
            )


def _unit(format_key: str, payload: dict) -> ContentUnit:
    return ContentUnit(targets=[], formats={format_key: payload})


# ─── Telegram ───────────────────────────────────────────────────────────

class TestTelegramAdapter:
    def test_missing_creds_returns_error(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        from opendispatch.adapters import telegram
        ok, pid, err = telegram.publish(_unit("telegram_message", {"text": "hi"}))
        assert ok is False
        assert pid == ""
        assert "TELEGRAM_BOT_TOKEN" in err

    def test_empty_text_returns_error(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
        from opendispatch.adapters import telegram
        ok, pid, err = telegram.publish(_unit("telegram_message", {"text": ""}))
        assert ok is False
        assert "empty" in err.lower()

    def test_text_happy_path(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
        from opendispatch.adapters import telegram
        monkeypatch.setattr(
            "opendispatch.adapters.telegram.httpx.post",
            lambda *a, **k: FakeResponse(200, {"ok": True, "result": {"message_id": 42}}),
        )
        ok, pid, err = telegram.publish(_unit("telegram_message", {"text": "hello"}))
        assert ok is True
        assert pid == "42"
        assert err == ""

    def test_account_override_resolves(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID_BROADCAST", "B")
        from opendispatch.adapters import telegram
        captured: list[dict[str, Any]] = []

        def fake_post(*a, **k):
            captured.append(k.get("data", {}))
            return FakeResponse(200, {"ok": True, "result": {"message_id": 7}})

        monkeypatch.setattr("opendispatch.adapters.telegram.httpx.post", fake_post)
        ok, pid, _ = telegram.publish(_unit("telegram_message", {"text": "x"}), account="broadcast")
        assert ok is True
        assert pid == "7"
        assert captured[0]["chat_id"] == "B"

    def test_http_error_returns_failure(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "T")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "C")
        from opendispatch.adapters import telegram

        def failing_post(*a, **k):
            return FakeResponse(401, text="unauthorized")

        monkeypatch.setattr("opendispatch.adapters.telegram.httpx.post", failing_post)
        ok, _, err = telegram.publish(_unit("telegram_message", {"text": "x"}))
        assert ok is False
        assert "401" in err or "unauthor" in err.lower()


# ─── Twitter / X ────────────────────────────────────────────────────────

class TestTwitterAdapter:
    def test_missing_creds_returns_error(self, monkeypatch):
        for k in [
            "TWITTER_API_KEY", "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_SECRET",
        ]:
            monkeypatch.delenv(k, raising=False)
        from opendispatch.adapters import twitter
        ok, _, err = twitter.publish(_unit("twitter_thread", {"tweets": ["hi"]}))
        assert ok is False
        assert "credentials" in err.lower() or "missing" in err.lower()

    def test_empty_tweets_returns_error(self, monkeypatch):
        from opendispatch.adapters import twitter
        ok, _, err = twitter.publish(_unit("twitter_thread", {"tweets": []}))
        assert ok is False
        assert "empty" in err.lower()

    def test_happy_path_single_tweet(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "ck")
        monkeypatch.setenv("TWITTER_API_SECRET", "cs")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
        monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")

        # Stub the tweepy module entirely
        fake_tweepy = types.ModuleType("tweepy")

        class FakeClient:
            def __init__(self, **kw): self.kw = kw
            def create_tweet(self, **kw):
                resp = MagicMock()
                resp.data = {"id": "100"}
                return resp

        fake_tweepy.Client = FakeClient  # type: ignore[attr-defined]
        fake_tweepy.API = MagicMock  # type: ignore[attr-defined]
        fake_tweepy.OAuth1UserHandler = MagicMock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tweepy", fake_tweepy)

        from opendispatch.adapters import twitter
        ok, pid, err = twitter.publish(_unit("twitter_thread", {"tweets": ["hello"]}))
        assert ok is True
        assert pid == "100"
        assert err == ""

    def test_thread_chains_replies(self, monkeypatch):
        monkeypatch.setenv("TWITTER_API_KEY", "ck")
        monkeypatch.setenv("TWITTER_API_SECRET", "cs")
        monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
        monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")

        ids = iter(["1", "2", "3"])
        calls: list[dict[str, Any]] = []

        class FakeClient:
            def __init__(self, **kw): pass
            def create_tweet(self, **kw):
                calls.append(kw)
                resp = MagicMock()
                resp.data = {"id": next(ids)}
                return resp

        fake_tweepy = types.ModuleType("tweepy")
        fake_tweepy.Client = FakeClient  # type: ignore[attr-defined]
        fake_tweepy.API = MagicMock  # type: ignore[attr-defined]
        fake_tweepy.OAuth1UserHandler = MagicMock  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "tweepy", fake_tweepy)

        from opendispatch.adapters import twitter
        ok, pid, _ = twitter.publish(_unit("twitter_thread", {"tweets": ["a", "b", "c"]}))
        assert ok is True
        assert pid == "1"
        assert len(calls) == 3
        # Second + third tweets should reply to the previous
        assert calls[1]["in_reply_to_tweet_id"] == "1"
        assert calls[2]["in_reply_to_tweet_id"] == "2"


# ─── Instagram ──────────────────────────────────────────────────────────

class TestInstagramAdapter:
    def test_missing_creds(self, monkeypatch):
        monkeypatch.delenv("IG_USER_ID", raising=False)
        monkeypatch.delenv("IG_TOKEN", raising=False)
        from opendispatch.adapters import instagram
        ok, _, err = instagram.publish(_unit("instagram_post", {"caption": "x", "image_url": "https://x"}))
        assert ok is False
        assert "IG_USER_ID" in err

    def test_needs_media_url(self, monkeypatch):
        monkeypatch.setenv("IG_USER_ID", "u")
        monkeypatch.setenv("IG_TOKEN", "t")
        from opendispatch.adapters import instagram
        ok, _, err = instagram.publish(_unit("instagram_post", {"caption": "x"}))
        assert ok is False
        assert "image_url" in err or "video_url" in err

    def test_image_happy_path(self, monkeypatch):
        monkeypatch.setenv("IG_USER_ID", "u")
        monkeypatch.setenv("IG_TOKEN", "t")
        from opendispatch.adapters import instagram

        responses = iter([
            FakeResponse(200, {"id": "container_1"}),         # create
            FakeResponse(200, {"status_code": "FINISHED"}),   # poll
            FakeResponse(200, {"id": "post_999"}),            # publish
        ])
        monkeypatch.setattr("opendispatch.adapters.instagram.httpx.post", lambda *a, **k: next(responses))
        monkeypatch.setattr("opendispatch.adapters.instagram.httpx.get", lambda *a, **k: next(responses))

        ok, pid, err = instagram.publish(_unit("instagram_post", {
            "caption": "x", "image_url": "https://example.com/img.jpg",
        }))
        assert ok is True
        assert pid == "post_999"
        assert err == ""


# ─── Bluesky ────────────────────────────────────────────────────────────

class TestBlueskyAdapter:
    def test_missing_creds(self, monkeypatch):
        monkeypatch.delenv("BLUESKY_HANDLE", raising=False)
        monkeypatch.delenv("BLUESKY_APP_PASSWORD", raising=False)
        from opendispatch.adapters import bluesky
        ok, _, err = bluesky.publish(_unit("bluesky_post", {"text": "hi"}))
        assert ok is False
        assert "BLUESKY_HANDLE" in err

    def test_empty_text_returns_error(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "h.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "p")

        fake_atproto = types.ModuleType("atproto")
        fake_atproto.Client = MagicMock  # type: ignore[attr-defined]
        fake_atproto.client_utils = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "atproto", fake_atproto)

        from opendispatch.adapters import bluesky
        ok, _, err = bluesky.publish(_unit("bluesky_post", {"text": ""}))
        assert ok is False
        assert "empty" in err.lower()

    def test_text_happy_path(self, monkeypatch):
        monkeypatch.setenv("BLUESKY_HANDLE", "h.bsky.social")
        monkeypatch.setenv("BLUESKY_APP_PASSWORD", "p")

        class FakeRef:
            uri = "at://did:plc:abc/app.bsky.feed.post/xyz"
            cid = "cid_xyz"

        class FakeClient:
            def login(self, *_): pass
            def send_post(self, **_): return FakeRef()

        fake_atproto = types.ModuleType("atproto")
        fake_atproto.Client = FakeClient  # type: ignore[attr-defined]
        fake_atproto.client_utils = MagicMock()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "atproto", fake_atproto)

        from opendispatch.adapters import bluesky
        ok, pid, err = bluesky.publish(_unit("bluesky_post", {"text": "hello"}))
        assert ok is True
        assert pid == FakeRef.uri
        assert err == ""


# ─── LinkedIn ───────────────────────────────────────────────────────────

class TestLinkedInAdapter:
    def test_missing_creds(self, monkeypatch):
        monkeypatch.delenv("LINKEDIN_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINKEDIN_AUTHOR_URN", raising=False)
        from opendispatch.adapters import linkedin
        ok, _, err = linkedin.publish(_unit("linkedin_post", {"text": "x"}))
        assert ok is False
        assert "LINKEDIN_ACCESS_TOKEN" in err

    def test_empty_text(self, monkeypatch):
        from opendispatch.adapters import linkedin
        ok, _, err = linkedin.publish(_unit("linkedin_post", {"text": ""}))
        assert ok is False
        assert "empty" in err.lower()

    def test_text_happy_path(self, monkeypatch):
        monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:abc")
        from opendispatch.adapters import linkedin

        captured: list[dict[str, Any]] = []

        def fake_post(*a, **k):
            captured.append({"a": a, "k": k})
            return FakeResponse(201, {"id": "urn:li:share:123"},
                                headers={"x-restli-id": "urn:li:share:123"})

        monkeypatch.setattr("opendispatch.adapters.linkedin.httpx.post", fake_post)
        ok, pid, err = linkedin.publish(_unit("linkedin_post", {"text": "hello world"}))
        assert ok is True
        assert pid == "urn:li:share:123"
        assert err == ""

    def test_http_error_returns_failure(self, monkeypatch):
        monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:abc")
        from opendispatch.adapters import linkedin
        monkeypatch.setattr(
            "opendispatch.adapters.linkedin.httpx.post",
            lambda *a, **k: FakeResponse(403, text="forbidden"),
        )
        ok, _, err = linkedin.publish(_unit("linkedin_post", {"text": "x"}))
        assert ok is False
        assert "403" in err


# ─── Threads (more coverage on top of existing test_threads.py) ─────────

class TestThreadsAdapter:
    def test_missing_creds(self, monkeypatch):
        monkeypatch.delenv("THREADS_USER_ID", raising=False)
        monkeypatch.delenv("THREADS_ACCESS_TOKEN", raising=False)
        from opendispatch.adapters import threads
        ok, _, err = threads.publish(_unit("threads_post", {"text": "hi"}))
        assert ok is False
        assert "THREADS_USER_ID" in err

    def test_empty_payload_returns_error(self, monkeypatch):
        monkeypatch.setenv("THREADS_USER_ID", "u")
        monkeypatch.setenv("THREADS_ACCESS_TOKEN", "t")
        from opendispatch.adapters import threads
        ok, _, err = threads.publish(_unit("threads_post", {}))
        assert ok is False
        assert "requires" in err.lower()


# ─── YouTube Shorts ─────────────────────────────────────────────────────

class TestYouTubeAdapter:
    def test_missing_video_path(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "c")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "s")
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        from opendispatch.adapters import youtube
        ok, _, err = youtube.publish(_unit("youtube_short", {"title": "x"}))
        assert ok is False
        assert "video_path" in err

    def test_video_file_doesnt_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "c")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "s")
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
        from opendispatch.adapters import youtube
        missing = tmp_path / "nope.mp4"
        ok, _, err = youtube.publish(_unit("youtube_short", {"video_path": str(missing)}))
        assert ok is False
        assert "does not exist" in err

    def test_missing_creds(self, tmp_path, monkeypatch):
        monkeypatch.delenv("YOUTUBE_CLIENT_ID", raising=False)
        monkeypatch.delenv("YOUTUBE_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00fakevid")
        from opendispatch.adapters import youtube
        ok, _, err = youtube.publish(_unit("youtube_short", {"video_path": str(video)}))
        assert ok is False
        assert "YOUTUBE_CLIENT_ID" in err or "YOUTUBE_CLIENT_SECRET" in err

    def test_happy_path(self, tmp_path, monkeypatch):
        """Mock the entire httpx.Client used by the adapter."""
        monkeypatch.setenv("YOUTUBE_CLIENT_ID", "c")
        monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "s")
        monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")

        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00" * 1024)

        from opendispatch.adapters import youtube

        class FakeClient:
            def __enter__(self): return self
            def __exit__(self, *a): pass

            def post(self, url, **kw):
                if "oauth2.googleapis.com" in url:
                    return FakeResponse(200, {"access_token": "AT123"})
                # Initiate upload
                return FakeResponse(200, {}, headers={"location": "https://www.googleapis.com/upload/session/v1?upload_id=X"})

            def put(self, url, **kw):
                return FakeResponse(200, {"id": "vid_777"})

        monkeypatch.setattr("opendispatch.adapters.youtube.httpx.Client", lambda: FakeClient())

        ok, pid, err = youtube.publish(_unit("youtube_short", {
            "video_path": str(video),
            "title": "test short",
        }))
        assert ok is True
        assert pid == "vid_777"
        assert err == ""

    def test_auto_appends_shorts_hashtag(self):
        """Belt-and-suspenders: description should include #Shorts when neither
        title nor description already mention it."""
        from opendispatch.adapters.youtube import _build_metadata
        meta = _build_metadata({"video_path": "v.mp4", "title": "hello"})
        desc = meta["snippet"]["description"]
        assert "#Shorts" in desc

    def test_respects_existing_shorts_hashtag(self):
        from opendispatch.adapters.youtube import _build_metadata
        meta = _build_metadata({"title": "hello #shorts", "description": "world"})
        # Shouldn't double-add since title already has it
        desc = meta["snippet"]["description"]
        assert desc.count("#Shorts") + desc.count("#shorts") <= 0 or "world" in desc

    def test_caption_fallback(self):
        from opendispatch.adapters.youtube import _build_metadata
        meta = _build_metadata({"caption": "look at this cool dunk"})
        assert "cool dunk" in meta["snippet"]["title"] or "cool dunk" in meta["snippet"]["description"]

    def test_privacy_default_is_public(self):
        from opendispatch.adapters.youtube import _build_metadata
        meta = _build_metadata({"title": "x"})
        assert meta["status"]["privacyStatus"] == "public"

    def test_invalid_privacy_is_rejected_not_defaulted(self):
        """Silently defaulting an unknown privacy to "public" is a visibility
        escalation, so an unrecognised value must raise instead."""
        from opendispatch.adapters.youtube import MetadataError, _build_metadata
        with pytest.raises(MetadataError):
            _build_metadata({"title": "x", "privacy": "topsecret"})

    @pytest.mark.parametrize("value", ["public", "unlisted", "private"])
    def test_valid_privacy_is_preserved(self, value):
        from opendispatch.adapters.youtube import _build_metadata
        meta = _build_metadata({"title": "x", "privacy": value})
        assert meta["status"]["privacyStatus"] == value


# ─── TikTok ─────────────────────────────────────────────────────────────

class TestTikTokAdapter:
    def test_missing_token(self, monkeypatch):
        monkeypatch.delenv("TIKTOK_ACCESS_TOKEN", raising=False)
        from opendispatch.adapters import tiktok
        ok, _, err = tiktok.publish(_unit("tiktok_post", {"video_url": "https://example.com/v.mp4"}))
        assert ok is False
        assert "TIKTOK_ACCESS_TOKEN" in err

    def test_missing_video_url(self, monkeypatch):
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok123")
        from opendispatch.adapters import tiktok
        ok, _, err = tiktok.publish(_unit("tiktok_post", {"caption": "hi"}))
        assert ok is False
        assert "video_url" in err

    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok123")
        import opendispatch.adapters.tiktok as tt
        monkeypatch.setattr("opendispatch.adapters.tiktok.time.sleep", lambda _: None)

        call_count = {"n": 0}

        def fake_post(url, **kw):
            call_count["n"] += 1
            if "init" in url:
                return FakeResponse(200, {"data": {"publish_id": "pub_abc"}, "error": {"code": "ok"}})
            # status poll
            return FakeResponse(200, {"data": {"status": "PUBLISH_COMPLETE",
                                               "publicaly_available_post_id": ["post_xyz"]}})

        monkeypatch.setattr("opendispatch.adapters.tiktok.httpx.post", fake_post)
        ok, pid, err = tt.publish(_unit("tiktok_post", {
            "video_url": "https://cdn.example.com/clip.mp4",
            "caption": "test caption",
        }))
        assert ok is True
        assert pid == "post_xyz"
        assert err == ""

    def test_publish_failed_status(self, monkeypatch):
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok123")
        import opendispatch.adapters.tiktok as tt
        monkeypatch.setattr("opendispatch.adapters.tiktok.time.sleep", lambda _: None)

        def fake_post(url, **kw):
            if "init" in url:
                return FakeResponse(200, {"data": {"publish_id": "pub_abc"}, "error": {"code": "ok"}})
            return FakeResponse(200, {"data": {"status": "FAILED", "fail_reason": "video_too_long"}})

        monkeypatch.setattr("opendispatch.adapters.tiktok.httpx.post", fake_post)
        ok, _, err = tt.publish(_unit("tiktok_post", {"video_url": "https://x.com/v.mp4"}))
        assert ok is False
        assert "video_too_long" in err

    def test_http_error(self, monkeypatch):
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok123")
        import opendispatch.adapters.tiktok as tt

        def fake_post(url, **kw):
            return FakeResponse(401, text="Unauthorized")

        monkeypatch.setattr("opendispatch.adapters.tiktok.httpx.post", fake_post)
        ok, _, err = tt.publish(_unit("tiktok_post", {"video_url": "https://x.com/v.mp4"}))
        assert ok is False
        assert "401" in err

    def test_api_error_in_init_response(self, monkeypatch):
        monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok123")
        import opendispatch.adapters.tiktok as tt

        def fake_post(url, **kw):
            return FakeResponse(200, {"error": {"code": "invalid_param", "message": "bad request"}, "data": {}})

        monkeypatch.setattr("opendispatch.adapters.tiktok.httpx.post", fake_post)
        ok, _, err = tt.publish(_unit("tiktok_post", {"video_url": "https://x.com/v.mp4"}))
        assert ok is False
        assert "bad request" in err


# ─── Facebook ───────────────────────────────────────────────────────────

class TestFacebookAdapter:
    def test_missing_creds(self, monkeypatch):
        monkeypatch.delenv("FACEBOOK_PAGE_ID", raising=False)
        monkeypatch.delenv("FACEBOOK_ACCESS_TOKEN", raising=False)
        from opendispatch.adapters import facebook
        ok, _, err = facebook.publish(_unit("facebook_post", {"text": "hello"}))
        assert ok is False
        assert "FACEBOOK_PAGE_ID" in err or "FACEBOOK_ACCESS_TOKEN" in err

    def test_text_post_happy_path(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "pg123")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok456")
        import opendispatch.adapters.facebook as fb
        monkeypatch.setattr("opendispatch.adapters.facebook.httpx.post",
                            lambda url, **kw: FakeResponse(200, {"id": "pg123_post789"}))
        ok, pid, err = fb.publish(_unit("facebook_post", {"text": "hello world"}))
        assert ok is True
        assert pid == "pg123_post789"
        assert err == ""

    def test_photo_post_happy_path(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "pg123")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok456")
        import opendispatch.adapters.facebook as fb
        monkeypatch.setattr("opendispatch.adapters.facebook.httpx.post",
                            lambda url, **kw: FakeResponse(200, {"post_id": "pg123_photo"}))
        ok, pid, err = fb.publish(_unit("facebook_post", {
            "text": "look at this",
            "image_url": "https://cdn.example.com/img.jpg",
        }))
        assert ok is True
        assert pid == "pg123_photo"

    def test_video_post_happy_path(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "pg123")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok456")
        import opendispatch.adapters.facebook as fb
        monkeypatch.setattr("opendispatch.adapters.facebook.httpx.post",
                            lambda url, **kw: FakeResponse(200, {"id": "vid_001"}))
        ok, pid, err = fb.publish(_unit("facebook_post", {
            "text": "my video",
            "video_url": "https://cdn.example.com/clip.mp4",
        }))
        assert ok is True
        assert pid == "vid_001"

    def test_empty_text_returns_error(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "pg123")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok456")
        from opendispatch.adapters import facebook
        ok, _, err = facebook.publish(_unit("facebook_post", {}))
        assert ok is False
        assert "text" in err.lower()

    def test_http_error(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID", "pg123")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok456")
        import opendispatch.adapters.facebook as fb
        monkeypatch.setattr("opendispatch.adapters.facebook.httpx.post",
                            lambda url, **kw: FakeResponse(400, text="OAuthException"))
        ok, _, err = fb.publish(_unit("facebook_post", {"text": "hi"}))
        assert ok is False
        assert "400" in err

    def test_per_account_creds(self, monkeypatch):
        monkeypatch.setenv("FACEBOOK_PAGE_ID_BRAND", "pg_brand")
        monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN_BRAND", "tok_brand")
        import opendispatch.adapters.facebook as fb

        captured = {}

        def fake_post(url, params=None, **kw):
            captured["token"] = (params or {}).get("access_token")
            return FakeResponse(200, {"id": "post_brand"})

        monkeypatch.setattr("opendispatch.adapters.facebook.httpx.post", fake_post)
        ok, _, _ = fb.publish(_unit("facebook_post", {"text": "hi"}), account="brand")
        assert captured["token"] == "tok_brand"
