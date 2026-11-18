"""livewatch free publishing: YouTube API, Instagram API, Telegram phone handoff, routing."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from livewatch import config, publish


class Resp:
    def __init__(self, status=200, data=None, headers=None, text=""):
        self.status_code, self._d, self.headers = status, data, headers or {}
        self.text = text or (json.dumps(data) if data is not None else "")
        self.content = self.text.encode()

    def json(self):
        if self._d is None:
            raise ValueError
        return self._d


class Fake:
    """Answers by (method, url prefix) in order; records every call."""

    def __init__(self, routes):
        self.routes = [(m, u, list(r) if isinstance(r, list) else [r]) for m, u, r in routes]
        self.calls = []

    def _answer(self, method, url, kw):
        self.calls.append((method, url, kw))
        for m, u, rs in self.routes:
            if m == method and url.startswith(u):
                return rs.pop(0) if len(rs) > 1 else rs[0]
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url, **kw):
        return self._answer("GET", url, kw)

    def post(self, url, **kw):
        return self._answer("POST", url, kw)

    def put(self, url, **kw):
        return self._answer("PUT", url, kw)

    def head(self, url, **kw):
        return self._answer("HEAD", url, kw)


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "short.mp4"
    v.write_bytes(b"\0" * 2048)
    return v


# ----------------------------------------------------------------------- youtube
def test_youtube_auth_url_uses_pkce_and_offline_access():
    url, state, verifier = publish.youtube_auth_url("cid", "http://127.0.0.1:8765/oauth/youtube")
    q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
    assert q["client_id"] == "cid" and q["access_type"] == "offline" and q["prompt"] == "consent"
    assert q["scope"] == publish.YT_SCOPE and q["code_challenge_method"] == "S256"
    assert q["state"] == state and len(verifier) > 40 and q["code_challenge"] != verifier


def _yt_fake(privacy="public"):
    return Fake([
        ("POST", publish.GOOGLE_TOKEN, Resp(200, {"access_token": "at"})),
        ("POST", publish.YT_UPLOAD, Resp(200, {}, headers={"Location": "https://upload.example/session"})),
        ("PUT", "https://upload.example/session", Resp(200, {"id": "abc123", "status": {"privacyStatus": privacy}})),
    ])


def test_youtube_upload_public_and_private(video):
    f = _yt_fake()
    res = publish.youtube_upload(video, "T" * 150, "desc", ["gta6", ""], client_id="c", client_secret="s",
                                 refresh_token="r", session=f)
    assert res == {"id": "abc123", "url": "https://youtube.com/shorts/abc123", "privacy": "public"}
    init = f.calls[1][2]
    assert init["json"]["snippet"]["title"] == "T" * 100 and init["json"]["snippet"]["tags"] == ["gta6"]
    assert init["json"]["status"] == {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
    assert init["headers"]["Authorization"] == "Bearer at"
    assert publish.youtube_upload(video, "t", "d", [], client_id="c", client_secret="s", refresh_token="r",
                                  session=_yt_fake("private"))["privacy"] == "private"
    with pytest.raises(publish.PublishError, match="Connect YouTube again"):
        publish.youtube_access_token("c", "s", "r", session=Fake([
            ("POST", publish.GOOGLE_TOKEN, Resp(400, {"error": "invalid_grant"}))]))


def test_youtube_exchange_code_needs_a_refresh_token():
    ok = Fake([("POST", publish.GOOGLE_TOKEN, Resp(200, {"refresh_token": "rt", "access_token": "a"}))])
    assert publish.youtube_exchange_code("c", "s", "code", "http://x", "v", session=ok) == "rt"
    assert ok.calls[0][2]["data"]["code_verifier"] == "v"
    with pytest.raises(publish.PublishError, match="no refresh token"):
        publish.youtube_exchange_code("c", "s", "code", "http://x", "v", session=Fake([
            ("POST", publish.GOOGLE_TOKEN, Resp(200, {"access_token": "a"}))]))


# --------------------------------------------------------------------- instagram
class FakeTunnel:
    def __init__(self, path):
        self.url = f"https://quiet-river.trycloudflare.com/s3cret/{Path(path).name}"
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True


IG = f"{publish.IG_API}/{publish.IG_VERSION}"


def _ig_fake(statuses=("IN_PROGRESS", "FINISHED")):
    return Fake([
        ("GET", f"{IG}/me", Resp(200, {"user_id": "1789", "username": "gtaclips"})),
        ("GET", "https://cloudflare-dns.com/dns-query", Resp(200, {"Status": 0, "Answer": [{"data": "1.2.3.4"}]})),
        ("POST", f"{IG}/1789/media_publish", Resp(200, {"id": "media9"})),
        ("POST", f"{IG}/1789/media", Resp(200, {"id": "cont5"})),
        ("GET", f"{IG}/cont5", [Resp(200, {"status_code": s}) for s in statuses]),
    ])


def test_instagram_publish_flow(video, monkeypatch):
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    f = _ig_fake()
    res = publish.instagram_publish(video, "caption #gta6", token="tok", session=f,
                                    tunnel_factory=FakeTunnel, poll_seconds=0)
    assert res == {"id": "media9", "username": "gtaclips"}
    create = next(c for c in f.calls if c[1] == f"{IG}/1789/media")[2]["data"]
    assert create["media_type"] == "REELS" and create["video_url"].endswith("/s3cret/short.mp4")
    assert create["caption"] == "caption #gta6"
    publish_call = next(c for c in f.calls if c[1].endswith("media_publish"))[2]["data"]
    assert publish_call["creation_id"] == "cont5"


def test_instagram_facebook_type_token(video, monkeypatch):
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    FB = f"{publish.FB_API}/{publish.IG_VERSION}"
    # no account id and no linked Page: a clear message, not a crash
    with pytest.raises(publish.PublishError, match="Instagram account ID"):
        publish.instagram_me("EAAxyz", session=Fake([("GET", f"{FB}/me/accounts", Resp(200, {"data": []}))]))
    # a Page-linked account is found by itself
    me = publish.instagram_me("EAAxyz", session=Fake([("GET", f"{FB}/me/accounts", Resp(200, {"data": [
        {"instagram_business_account": {"id": "1784", "username": "gtaclips"}}]}))]))
    assert me == {"id": "1784", "username": "gtaclips"}
    # with the account id, the whole Reel flow runs on graph.facebook.com
    f = Fake([
        ("GET", f"{FB}/1784", Resp(200, {"id": "1784", "username": "gtaclips"})),
        ("GET", "https://cloudflare-dns.com/dns-query", Resp(200, {"Status": 0, "Answer": [{"data": "1.2.3.4"}]})),
        ("POST", f"{FB}/1784/media_publish", Resp(200, {"id": "media1"})),
        ("POST", f"{FB}/1784/media", Resp(200, {"id": "c1"})),
        ("GET", f"{FB}/c1", Resp(200, {"status_code": "FINISHED"})),
    ])
    res = publish.instagram_publish(video, "cap", token="EAAxyz", account_id="1784", session=f,
                                    tunnel_factory=FakeTunnel, poll_seconds=0)
    assert res == {"id": "media1", "username": "gtaclips"}
    assert not any(publish.IG_API in c[1] for c in f.calls)
    assert publish.instagram_refresh("EAAxyz", session=Fake([])) is None  # system-user tokens do not expire


def test_instagram_processing_error_is_retried_then_reported(video, monkeypatch):
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    # one failed fetch, then success on the second try: published
    f = _ig_fake(("ERROR", "FINISHED"))
    assert publish.instagram_publish(video, "c", token="t", session=f, tunnel_factory=FakeTunnel,
                                     poll_seconds=0)["id"] == "media9"
    assert sum(1 for c in f.calls if c[1] == f"{IG}/1789/media") == 2
    # failing twice: the error carries Instagram's own explanation
    f = Fake([
        ("GET", f"{IG}/me", Resp(200, {"user_id": "1789", "username": "g"})),
        ("GET", "https://cloudflare-dns.com/dns-query", Resp(200, {"Status": 0, "Answer": [{}]})),
        ("POST", f"{IG}/1789/media", Resp(200, {"id": "cont5"})),
        ("GET", f"{IG}/cont5", Resp(200, {"status_code": "ERROR", "status": "Error: video download failed"})),
    ])
    with pytest.raises(publish.PublishError, match="video download failed"):
        publish.instagram_publish(video, "c", token="t", session=f, tunnel_factory=FakeTunnel, poll_seconds=0)


def test_one_file_tunnel_serves_only_its_file(video, monkeypatch):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    t = publish.OneFileTunnel(video)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), t._handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/{t.secret}/short.mp4") as r:
            assert r.status == 200 and len(r.read()) == 2048
        for bad in ("/", "/short.mp4", f"/{t.secret}/other.mp4", f"/x{t.secret}/short.mp4"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(base + bad)
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------- telegram
TG = f"{publish.TG_API}/bot123:ABC"


def test_telegram_find_chat_and_handoff(video):
    f = Fake([("GET", f"{TG}/getUpdates", Resp(200, {"result": [
        {"message": {"chat": {"id": 42, "first_name": "Yas"}}}]}))])
    assert publish.telegram_find_chat("123:ABC", session=f) == {"chat_id": "42", "name": "Yas"}
    with pytest.raises(publish.PublishError, match="press Start"):
        publish.telegram_find_chat("123:ABC", session=Fake([("GET", f"{TG}/getUpdates", Resp(200, {"result": []}))]))
    f = Fake([("POST", f"{TG}/sendVideo", Resp(200, {"result": {"message_id": 7}})),
              ("POST", f"{TG}/sendMessage", Resp(200, {"ok": True}))])
    out = publish.telegram_handoff(video, {"tiktok": "caption #gta6"}, token="123:ABC", chat_id="42",
                                   title="Big moment", session=f)
    assert out == {"message_id": 7}
    assert f.calls[0][2]["data"]["chat_id"] == "42" and "video" in f.calls[0][2]["files"]
    assert "TIKTOK caption" in f.calls[1][2]["data"]["text"] and "#gta6" in f.calls[1][2]["data"]["text"]


# ----------------------------------------------------------------------- routing
def test_route_and_missing():
    r = publish.route(["youtube", "instagram", "tiktok", "x"], "api", "api", "phone")
    assert r == {"youtube": "api", "instagram": "api", "tiktok": "phone", "x": "phone"}
    assert publish.route(["tiktok"], "api", "api", "off") == {}
    assert publish.route(["youtube"], "phone", "api", "phone") == {"youtube": "phone"}
    none = {k: "" for k in ("yt_id", "yt_secret", "yt_refresh", "ig_token", "tg_token", "tg_chat", "up_key")}
    msgs = publish.missing(r, none)
    assert len(msgs) == 3 and any("YouTube" in m for m in msgs) and any("Telegram" in m for m in msgs)
    assert publish.missing(r, dict(none, yt_id="a", yt_secret="b", yt_refresh="c", ig_token="d",
                                   tg_token="e", tg_chat="f")) == []


def test_publish_one_network_failing_never_stops_the_others(video, monkeypatch):
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "c")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "s")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "r")
    monkeypatch.setenv("INSTAGRAM_ACCESS_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(publish, "youtube_upload",
                        lambda *a, **k: {"id": "v", "url": "https://youtube.com/shorts/v", "privacy": "private"})

    def ig_fail(*a, **k):
        raise publish.PublishError("token expired")
    monkeypatch.setattr(publish, "instagram_publish", ig_fail)
    sent = {}
    monkeypatch.setattr(publish, "telegram_handoff", lambda v, caps, **k: sent.update(caps) or {"message_id": 1})
    posts = {"youtube": {"title": "T", "description": "D"}, "instagram": {"title": "T", "description": "IG"},
             "tiktok": {"title": "T", "description": "TT #gta6"}}
    res = publish.publish(video, posts, {"youtube": "api", "instagram": "api", "tiktok": "phone"},
                          tags=["gta6"], log=lambda m: None)
    assert res["status"] == "partial"
    assert res["networks"]["youtube"]["status"] == "private" and "audit" in res["networks"]["youtube"]["why"]
    assert res["networks"]["instagram"] == {"status": "failed", "error": "token expired"}
    assert res["networks"]["tiktok"] == {"status": "sent to phone"} and sent == {"tiktok": "TT #gta6"}


# ------------------------------------------------------------- service + queue
def _clip(tmp_path, s):
    d = s.data_dir / "clips" / "twitch_abc" / "20261119-210000"
    d.mkdir(parents=True)
    (d / "short.mp4").write_bytes(b"\0" * 100)
    (d / "meta.json").write_text(json.dumps({
        "text": {"hashtags": ["gta6"]}, "posted": None,
        "posts": {"youtube": {"title": "T", "description": "D"}, "tiktok": {"title": "T", "description": "C"}}}),
        encoding="utf-8")
    return d


def test_deliver_skips_networks_that_are_not_set_up(tmp_path, monkeypatch):
    from livewatch import service
    for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    s = config.from_dict({"general": {"data_dir": str(tmp_path / "data")}})
    d = _clip(tmp_path, s)
    calls = []
    monkeypatch.setattr(service.publish, "publish",
                        lambda v, posts, routes, **k: calls.append(routes) or
                        {"status": "posted", "networks": {n: {"status": "sent to phone"} for n in routes}})
    res = service.deliver(d, s, lambda m: None)
    assert calls == [{"tiktok": "phone"}]
    assert res["status"] == "partial" and res["networks"]["youtube"]["status"] == "skipped"
    assert service.deliver(d, s, lambda m: None, only=["youtube"])["status"] == "held"


def test_posts_are_spaced_through_the_queue(tmp_path, monkeypatch):
    from livewatch import service, viral
    s = config.from_dict({"general": {"data_dir": str(tmp_path / "data")},
                          "post": {"enabled": True, "min_gap_minutes": 30},
                          "channel": [{"platform": "twitch", "name": "abc"}]})
    svc = service.Service(s)
    delivered = []
    monkeypatch.setattr(service, "deliver", lambda d, s, log, only=None: delivered.append(d) or {"status": "posted"})
    text = viral.ClipText("h", "t", "d", source="offline")
    d1, d2 = _clip(tmp_path, s), tmp_path / "x"
    assert svc.maybe_post(d1, text)["status"] == "posted"
    queued = svc.maybe_post(d2, text)
    assert queued["status"] == "queued"
    when = datetime.fromisoformat(queued["at"])
    assert timedelta(minutes=29) < when - datetime.now(timezone.utc) <= timedelta(minutes=30)
    assert svc._post_queue == [(when, d2)]
    # a restart picks the queued clip up again from its meta.json
    meta = json.loads((d1 / "meta.json").read_text(encoding="utf-8"))
    meta["posted"] = queued
    (d1 / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    svc2 = service.Service(s)
    svc2._requeue_from_disk()
    assert [p for _, p in svc2._post_queue] == [d1]
    assert svc.maybe_post(d1, viral.ClipText("h", "t", "d", spoiler=True))["why"] == "spoiler"


def test_youtube_connect_callback(tmp_path, monkeypatch):
    from livewatch.webui import ApiError, Controller
    for k in ("YOUTUBE_REFRESH_TOKEN",):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "sec")
    (tmp_path / "livewatch.toml").write_text('[general]\ndata_dir = "d"\n', encoding="utf-8")
    ctl = Controller(tmp_path / "livewatch.toml")
    url = ctl.youtube_connect("http://127.0.0.1:8765/oauth/youtube")["url"]
    state = parse_qs(urlparse(url).query)["state"][0]
    seen = {}
    monkeypatch.setattr(publish, "youtube_exchange_code",
                        lambda cid, sec, code, redirect, verifier: seen.update(code=code, redirect=redirect) or "rt-1")
    with pytest.raises(ApiError):
        ctl.youtube_callback("wrong-state", "code-1")
    assert "connected" in ctl.youtube_callback(state, "code-1")
    assert seen == {"code": "code-1", "redirect": "http://127.0.0.1:8765/oauth/youtube"}
    assert "YOUTUBE_REFRESH_TOKEN=rt-1" in (tmp_path / ".env").read_text(encoding="utf-8")
    with pytest.raises(ApiError):  # a state is single-use
        ctl.youtube_callback(state, "code-1")
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
