"""livewatch web UI, music library, replay helpers and the TOML writer."""
import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from livewatch import config, music, tomlwrite
from livewatch.chat import parse_youtube_replay_line
from livewatch.replay import find_peaks, video_id
from livewatch.settings_schema import schema
from livewatch.webui import ApiError, Controller, make_server

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# --------------------------------------------------------------------------- toml
def test_tomlwrite_round_trip():
    data = {"general": {"data_dir": "C:\\clips\\\"x\"", "max_live_channels": 2, "keep_raw_clips": False},
            "detect": {"threshold": 3.5},
            "post": {"platforms": ["youtube", "x"], "upload_post_user": "me\nyou"},
            "channel": [{"platform": "twitch", "name": "a", "facecam": [0.0, 0.6, 0.3, 0.4]},
                        {"platform": "youtube", "name": "@b", "tags": []}]}
    text = tomlwrite.dumps(data)
    assert tomllib.loads(text) == data
    with pytest.raises(TypeError):
        tomlwrite.dumps({"general": {"x": object()}})
    with pytest.raises(ValueError):
        tomlwrite.dumps({"detect": {"threshold": float("nan")}})


def test_schema_matches_settings():
    sc = schema()
    keys = {(s["id"], f["key"]) for s in sc["sections"] for f in s["fields"]}
    assert ("style", "description_style") in keys and ("music", "mode") in keys
    raw = {s["id"]: {f["key"]: f["default"] for f in s["fields"]} for s in sc["sections"]}
    raw["channel"] = [{f["key"]: f["default"] for f in sc["channel"] if f["default"] is not None}
                      | {"name": "streamer"}]
    config.from_dict(raw)  # every default the page shows is a valid config


# -------------------------------------------------------------------------- music
def test_music_safe_names_and_modes():
    assert music.safe_name("../../evil track.MP3") == "evil track.mp3"
    assert music.safe_name("x.exe") == "" and music.safe_name(".mp3") == ""
    assert music.wants_music("always", 100, 10) and not music.wants_music("off", 0, 10)
    assert music.wants_music("quiet", 5, 30) and not music.wants_music("quiet", 60, 30)


def test_music_audio_graph():
    assert music.audio_graph(True, 20, None).startswith("[0:a]loudnorm")
    assert music.audio_graph(False, 20, None) == ""
    mix = music.Mix(Path("t.mp3"), 3.0, -20)
    g = music.audio_graph(True, 20, mix)
    assert "sidechaincompress" in g and "amix" in g and "volume=-20.0dB" in g and g.endswith("[aout]")
    assert "afade=t=out:st=18.00" in g
    assert "sidechain" not in music.audio_graph(False, 20, mix)


def _tone(path: Path, seconds: int):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=330:duration={seconds}", "-c:a", "libmp3lame", str(path)], check=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_music_library_rotation(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _tone(src / "a.mp3", 60)
    lib = music.Library(tmp_path / "music")
    assert lib.pick(20, -20) is None
    a = lib.add("first track.mp3", (src / "a.mp3").read_bytes())
    b = lib.add("first track.mp3", (src / "a.mp3").read_bytes())
    assert (a, b) == ("first track.mp3", "first track (2).mp3")
    with pytest.raises(ValueError):
        lib.add("fake.mp3", b"not audio at all")
    picks = [lib.pick(20, -20, seed=str(i)).path.name for i in range(4)]
    assert picks[0] != picks[1] and picks[0] == picks[2]  # least recently used first
    lib.set_enabled(b, False)
    assert {lib.pick(20, -20).path.name for _ in range(3)} == {a}
    m = lib.pick(20, -18, seed="s")
    assert 0 <= m.offset <= 60 - 20 and m.volume_db == -18
    lib.remove(a)
    assert [t["name"] for t in lib.describe()] == [b]


# -------------------------------------------------------------------- upload-post
class _Resp:
    def __init__(self, status, data=None, text=""):
        self.status_code, self._d = status, data
        self.text = text or (json.dumps(data) if data is not None else "")
        self.content = self.text.encode()

    def json(self):
        if self._d is None:
            raise ValueError
        return self._d


class _UP:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0)

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0)


def test_upload_post_profiles():
    from livewatch import poster
    sess = _UP(_Resp(200, {"success": True, "profiles": [
        {"username": "gta6clips", "social_accounts": {"tiktok": {"handle": "x"}, "youtube": "",
                                                      "instagram": {"id": 1}}},
        {"username": "empty", "social_accounts": {}}, {"nope": 1}]}))
    profs = poster.list_profiles("k", session=sess)
    assert profs == [{"username": "gta6clips", "connected": ["tiktok", "instagram"]},
                     {"username": "empty", "connected": []}]
    assert sess.calls[0][2]["headers"]["Authorization"] == "Apikey k"
    with pytest.raises(poster.UploadPostError, match="401"):
        poster.list_profiles("bad", session=_UP(_Resp(401, {"message": "Invalid API key"})))


def test_upload_post_create_and_connect():
    from livewatch import poster
    assert poster.create_profile("k", "gta6clips", session=_UP(_Resp(201, {}))) == "gta6clips"
    assert poster.create_profile("k", "gta6clips", session=_UP(_Resp(409, text="Username already in use")))
    with pytest.raises(poster.UploadPostError):
        poster.create_profile("k", "me@mail.com", session=_UP())  # an email is not a profile name
    with pytest.raises(poster.UploadPostError, match="no free profile"):
        poster.create_profile("k", "another", session=_UP(_Resp(403, text="profile_limit_reached")))
    sess = _UP(_Resp(200, {"access_url": "https://connect.upload-post.com/abc"}))
    assert poster.connect_url("k", "gta6clips", "http://127.0.0.1:8765/", session=sess) \
        == "https://connect.upload-post.com/abc"
    assert sess.calls[0][2]["json"]["redirectUrl"] == "http://127.0.0.1:8765/"
    with pytest.raises(poster.UploadPostError):
        poster.connect_url("k", "gta6clips", session=_UP(_Resp(200, {"access_url": "javascript:x"})))


def test_controller_upload_post_needs_a_key(ctl):
    with pytest.raises(ApiError) as e:
        ctl.upload_profiles()
    assert e.value.status == 400 and "API key" in e.value.message


# -------------------------------------------------------------------------- procs
def test_terminate_kills_the_whole_tree_on_windows(monkeypatch):
    from livewatch import procs
    calls = []
    monkeypatch.setattr(procs, "IS_WINDOWS", True)
    monkeypatch.setattr(procs.subprocess, "run", lambda cmd, **kw: calls.append(cmd))

    class P:
        pid, rc = 4242, None

        def poll(self):
            return self.rc

        def wait(self, timeout=None):
            self.rc = 1

    procs.terminate(P())
    assert calls == [["taskkill", "/T", "/F", "/PID", "4242"]]


# ------------------------------------------------------------------------- replay
def test_video_id():
    assert video_id("https://www.youtube.com/watch?v=02veXbcADjw&t=10") == "02veXbcADjw"
    assert video_id("https://youtu.be/02veXbcADjw") == "02veXbcADjw"
    assert video_id("https://www.youtube.com/live/02veXbcADjw?si=x") == "02veXbcADjw"
    for bad in ("https://evil.example/watch?v=02veXbcADjw", "file:///etc/passwd",
                "https://www.youtube.com/watch?v=short", "not a url"):
        assert video_id(bad) is None


def _replay_line(offset_ms, text, user="u1", ts_usec=1794954481602076):
    item = {"liveChatTextMessageRenderer": {"message": {"runs": [{"text": text}]},
                                            "timestampUsec": str(ts_usec),
                                            "authorExternalChannelId": user}}
    return json.dumps({"replayChatItemAction": {"actions": [{"addChatItemAction": {"item": item}}],
                                                "videoOffsetTimeMsec": str(offset_ms)}})


def test_replay_line_and_peaks():
    [(off, msg)] = parse_youtube_replay_line(_replay_line(65_500, "NO WAY"))
    assert off == 65.5 and msg.text == "NO WAY"
    assert parse_youtube_replay_line("{}") == [] and parse_youtube_replay_line("junk") == []
    import random
    rng = random.Random(4)
    msgs = []
    for s in range(1200):
        rate = 25 if 700 <= s < 708 else 2
        for _ in range(rng.randint(rate - 1, rate + 1)):
            msgs.append((s + rng.random(), _msg(f"u{rng.randrange(3000)}", "KEKW" if rate > 2 else "hi")))
    peaks = find_peaks(msgs, config.from_dict({}), 0, 1200, max_clips=3)
    assert len(peaks) == 1 and 690 <= peaks[0].ts <= 708


def _msg(user, text):
    from livewatch.chat import ChatMessage
    return ChatMessage(0, user, text)


# ---------------------------------------------------------------------- controller
@pytest.fixture
def ctl(tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "UPLOADPOST_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "livewatch.toml").write_text(
        '[general]\ndata_dir = "data"\n[[channel]]\nplatform = "twitch"\nname = "abc"\n', encoding="utf-8")
    return Controller(tmp_path / "livewatch.toml")


def test_controller_config_save_and_validation(ctl):
    raw = ctl.raw_config()
    raw["detect"] = {"threshold": 4.5}
    assert ctl.save_config(raw)["ok"]
    assert ctl.settings().threshold == 4.5
    with pytest.raises(ApiError) as e:
        ctl.save_config({"clip": {"pre_seconds": 70, "post_seconds": 20}})
    assert e.value.status == 400 and "max_seconds" in e.value.message
    assert ctl.settings().threshold == 4.5  # a refused save leaves the file alone


def test_a_stale_page_cannot_overwrite_newer_settings(ctl):
    seen = ctl.config_version()
    raw = ctl.raw_config()
    raw["detect"] = {"threshold": 6.0}
    out = ctl.save_config(raw, seen)          # the page's own version: accepted
    assert out["version"] != seen
    old_form = ctl.raw_config()
    old_form["general"]["data_dir"] = "replaydata"
    with pytest.raises(ApiError) as e:        # loaded before that save: refused
        ctl.save_config(old_form, seen)
    assert e.value.status == 409
    assert ctl.settings().data_dir.name == "data"


def test_every_save_keeps_a_backup_and_can_be_restored(ctl):
    original = ctl.raw_config()
    hand_edit = dict(original, channel=original["channel"] + [{"platform": "kick", "name": "xyz"}])
    ctl.save_config(hand_edit)
    ctl.save_config(dict(original, channel=[{"platform": "twitch", "name": "only"}]))
    names = [c for b in ctl.backups() for c in b["channels"]]
    assert "kick:xyz" in names  # the version that was overwritten is kept
    newest_with_xyz = next(b for b in ctl.backups() if "kick:xyz" in b["channels"])
    ctl.restore_backup(newest_with_xyz["name"])
    assert [c.key for c in ctl.settings().channels] == ["twitch:abc", "kick:xyz"]
    with pytest.raises(ApiError):
        ctl.restore_backup("../livewatch.toml")
    for i in range(25):
        ctl.save_config(original)
    assert len(ctl.backups()) == ctl.BACKUPS_KEPT


def test_controller_secrets_go_to_env_file(ctl, monkeypatch):
    out = ctl.save_secrets({"GEMINI_API_KEY": "abcd1234efgh5678"})
    assert out["GEMINI_API_KEY"]["set"] and out["GEMINI_API_KEY"]["hint"].endswith("5678")
    assert "abcd1234" not in json.dumps(out)
    assert "GEMINI_API_KEY=abcd1234efgh5678" in ctl.env_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY" not in ctl.config_path.read_text(encoding="utf-8")
    ctl.save_secrets({"GEMINI_API_KEY": ""})
    assert not ctl.secrets()["GEMINI_API_KEY"]["set"]
    for bad in ({"PATH": "x"}, {"GEMINI_API_KEY": "a\nEVIL=1"}):
        with pytest.raises(ApiError):
            ctl.save_secrets(bad)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def _fake_clip(ctl, cid="twitch_abc/20261119-210000"):
    d = ctl.clips_root() / cid
    d.mkdir(parents=True)
    (d / "short.mp4").write_bytes(b"\0" * 5000)
    (d / "meta.json").write_text(json.dumps({
        "channel": "twitch:abc", "streamer": "abc", "duration": 30, "score": 5,
        "text": {"hook": "<img src=x onerror=alert(1)>", "title": "t", "style": "ooc"},
        "posts": {"youtube": {"title": "T", "description": "D"}}, "posted": {"status": "held"}}),
        encoding="utf-8")
    return cid, d


def test_controller_clips(ctl):
    cid, d = _fake_clip(ctl)
    [c] = ctl.clips()
    assert c["id"] == cid and c["has_video"] and c["text"]["hook"].startswith("<img")  # escaped by the page
    ctl.update_posts(cid, {"youtube": {"title": "New", "description": "D2"}})
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["posts"]["youtube"]["title"] == "New"
    for bad in ({"myspace": {"title": "x"}}, {"youtube": {"title": 5}}, {"youtube": {"title": "x" * 400}}):
        with pytest.raises(ApiError):
            ctl.update_posts(cid, bad)
    for bad_id in ("../../etc", "twitch_abc/../../x", "a/b/c", ""):
        with pytest.raises(ApiError):
            ctl.clip_dir(bad_id)
    with pytest.raises(ApiError) as e:
        ctl.post_clip(cid)  # no Upload-Post key
    assert e.value.status == 400
    ctl.delete_clip(cid)
    assert not d.exists() and ctl.clips() == []


def test_controller_start_needs_valid_config(tmp_path):
    (tmp_path / "livewatch.toml").write_text('[general]\ndata_dir = "d"\n', encoding="utf-8")
    c = Controller(tmp_path / "livewatch.toml")
    with pytest.raises(ApiError) as e:
        c.start()
    assert "channel" in e.value.message
    st = c.status()
    assert st["running"] is False and st["config_error"] == "" and st["channels"] == []


# ---------------------------------------------------------------------------- HTTP
@pytest.fixture
def server(ctl):
    srv = make_server(ctl, port=0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield ctl, port
    srv.shutdown()
    srv.server_close()


def _req(port, path, method="GET", body=None, headers=None, host=None):
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_http_pages_and_guards(server):
    ctl, port = server
    st, body, _ = _req(port, "/")
    assert st == 200 and b"livewatch" in body
    st, body, _ = _req(port, "/api/status")
    assert st == 200 and json.loads(body)["channels"][0]["key"] == "twitch:abc"
    assert _req(port, "/api/status", host="evil.example")[0] == 403          # DNS rebinding
    assert _req(port, "/api/stop", "POST", body={})[0] == 403                # no X-Livewatch
    assert _req(port, "/api/stop", "POST", body={},
                headers={"X-Livewatch": "1", "Origin": "https://evil.example"})[0] == 403
    assert _req(port, "/api/stop", "POST", body={}, headers={"X-Livewatch": "1"})[0] == 200
    assert _req(port, "/nope")[0] == 404


def test_http_config_and_clip_media(server):
    ctl, port = server
    st, body, _ = _req(port, "/api/config")
    data = json.loads(body)
    assert st == 200 and data["raw"]["channel"][0]["name"] == "abc" and data["schema"]["sections"]
    raw, ver = data["raw"], data["version"]
    raw["detect"] = {"threshold": 5.0}
    st, body, _ = _req(port, "/api/config", "PUT", {"raw": raw, "version": ver}, {"X-Livewatch": "1"})
    assert st == 200
    ver = json.loads(body)["version"]
    st, body, _ = _req(port, "/api/config", "PUT", {"raw": {"detect": {"threshold": 0}}, "version": ver},
                       {"X-Livewatch": "1"})
    assert st == 400 and "threshold" in json.loads(body)["error"]
    assert _req(port, "/api/config", "PUT", {"raw": raw, "version": "stale"}, {"X-Livewatch": "1"})[0] == 409
    assert _req(port, "/api/config", "PUT", raw, {"X-Livewatch": "1"})[0] == 400  # old page format
    cid, _ = _fake_clip(ctl)
    st, body, hdr = _req(port, f"/media/{cid}/short.mp4", headers={"Range": "bytes=100-199"})
    assert st == 206 and len(body) == 100 and hdr["Content-Range"] == "bytes 100-199/5000"
    assert _req(port, "/media/..%2F..%2Flivewatch.toml/short.mp4")[0] == 404
    st, body, _ = _req(port, "/api/clips")
    assert json.loads(body)["clips"][0]["id"] == cid
