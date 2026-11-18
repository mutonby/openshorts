"""livewatch: config, chat parsing, peak detection, text, captions, posting, rendering."""
import json
import random
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from livewatch import captions, config, poster, render, viral
from livewatch.chat import ChatMessage, parse_twitch_irc, parse_youtube_chat_line
from livewatch.detector import PeakDetector, is_hype, normalise_message, robust_z
from livewatch.recorder import (Segment, SegmentBuffer, parse_rms_lines,
                                parse_segment_list_line)
from livewatch.transcribe import Word, clean_words

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# --------------------------------------------------------------------------- config
def test_config_defaults_and_channels(tmp_path):
    s = config.from_dict({
        "general": {"data_dir": "data"},
        "style": {"caption_style": "rotate"},
        "post": {"platforms": ["youtube", "x"], "base_hashtags": ["#gta6"]},
        "channel": [
            {"platform": "twitch", "name": "SomeStreamer", "game_filter": "Grand Theft Auto"},
            {"platform": "kick", "name": "kicker", "facecam": [0, 0.6, 0.3, 0.4]},
            {"platform": "youtube", "name": "@Handle", "caption_style": "brainrot"},
        ],
    }, base_dir=tmp_path)
    assert s.data_dir == (tmp_path / "data").resolve()
    assert s.rotate_styles and s.post_platforms == ["youtube", "x"]
    assert s.base_hashtags == ["gta6"]
    tw, kk, yt = s.channels
    assert tw.url == "https://www.twitch.tv/somestreamer" and tw.game_filter == ["Grand Theft Auto"]
    assert kk.url == "https://kick.com/kicker" and kk.facecam == [0.0, 0.6, 0.3, 0.4]
    assert yt.url == "https://www.youtube.com/@Handle/live"
    assert s.style_for(yt) == "brainrot"
    assert [s.style_for(tw, n) for n in range(3)] == list(config.CAPTION_STYLES)
    assert s.description_for(5) == "normal"


def test_description_style_and_removed_chinese_style(tmp_path):
    s = config.from_dict({"style": {"description_style": "rotate"}}, base_dir=tmp_path)
    assert [s.description_for(n) for n in range(3)] == ["normal", "weird", "normal"]
    # a config written when "chinese" existed still loads, as plain English
    old = config.from_dict({"style": {"caption_style": "chinese", "description_style": "chinese"},
                            "channel": [{"platform": "twitch", "name": "a",
                                         "caption_style": "chinese"}]}, base_dir=tmp_path)
    assert old.caption_style == "ooc" and old.description_style == "normal"
    assert old.style_for(old.channels[0]) == "ooc"
    with pytest.raises(config.ConfigError):
        config.from_dict({"style": {"description_style": "klingon"}}, base_dir=tmp_path)


@pytest.mark.parametrize("bad", [
    {"channel": [{"platform": "facebook", "name": "x"}]},
    {"channel": [{"platform": "twitch", "name": "bad name!"}]},
    {"channel": [{"platform": "twitch", "name": "a", "facecam": [0.8, 0, 0.5, 0.5]}]},
    {"channel": [{"platform": "twitch", "name": "a"}, {"platform": "twitch", "name": "A"}]},
    {"channel": [{"platform": "youtube", "name": "https://evil.example/live"}]},
    {"style": {"caption_style": "comic"}},
    {"post": {"platforms": ["myspace"]}},
    {"clip": {"pre_seconds": 70, "post_seconds": 20}},
    {"detect": {"threshold": 0}},
    {"general": {"quality": "best; rm -rf"}},
])
def test_config_rejects(bad, tmp_path):
    with pytest.raises(config.ConfigError):
        config.from_dict(bad, base_dir=tmp_path)


def test_config_load_reads_toml_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / "livewatch.toml").write_text(
        '[[channel]]\nplatform = "twitch"\nname = "abc"\n', encoding="utf-8")
    (tmp_path / ".env").write_text('GEMINI_API_KEY="k-123"\n# comment\n', encoding="utf-8")
    s = config.load(tmp_path / "livewatch.toml")
    assert s.gemini_api_key == "k-123" and s.channels[0].name == "abc"
    with pytest.raises(config.ConfigError):
        config.load(tmp_path / "missing.toml")


def test_matches_game():
    assert config.matches_game("Grand Theft Auto VI", ["grand theft auto"])
    assert not config.matches_game("Just Chatting", ["Grand Theft Auto"])
    assert not config.matches_game("", ["Grand Theft Auto"])
    assert config.matches_game("", [])


# --------------------------------------------------------------------------- chat
def test_parse_twitch_irc():
    line = ("@badge-info=;display-name=Some\\sUser;tmi-sent-ts=1794954739607;user-id=1 "
            ":someuser!someuser@someuser.tmi.twitch.tv PRIVMSG #chan :NO WAY KEKW")
    m = parse_twitch_irc(line)
    assert m == ChatMessage(ts=1794954739.607, user="some user", text="NO WAY KEKW")
    action = ":u!u@u.tmi.twitch.tv PRIVMSG #chan :\x01ACTION dances\x01"
    assert parse_twitch_irc(action, now=5.0) == ChatMessage(5.0, "u", "dances")
    assert parse_twitch_irc(":tmi.twitch.tv 001 justinfan1 :Welcome") is None
    assert parse_twitch_irc("PING :tmi.twitch.tv") is None
    assert parse_twitch_irc("") is None


def test_parse_youtube_chat_line():
    item = {"liveChatTextMessageRenderer": {
        "message": {"runs": [{"text": "this is insane "},
                             {"emoji": {"emojiId": "😂"}},
                             {"emoji": {"emojiId": "UCx/abc", "isCustomEmoji": True,
                                        "shortcuts": [":yt:"]}}]},
        "timestampUsec": "1794954481602076", "authorExternalChannelId": "UCabc"}}
    gift = {"giftMessageViewModel": {"text": {"content": "sent Star"}}}
    line = json.dumps({"replayChatItemAction": {"actions": [
        {"addChatItemAction": {"item": item}}, {"addChatItemAction": {"item": gift}}]}})
    msgs = parse_youtube_chat_line(line)
    assert len(msgs) == 1
    assert msgs[0].ts == pytest.approx(1794954481.602076)
    assert msgs[0].user == "ucabc" and msgs[0].text.startswith("this is insane 😂")
    assert parse_youtube_chat_line("not json") == []
    assert parse_youtube_chat_line("[1, 2]") == []


# ----------------------------------------------------------------------- detector
def test_is_hype_and_normalise():
    for t in ("KEKW", "no way bro", "???", "💀💀", "CLIP IT", "WHAT IS THIS", "omegalul"):
        assert is_hype(t), t
    for t in ("hello everyone", "what time is it", "ok"):
        assert not is_hype(t), t
    assert normalise_message("  NOOOOOOO   way ") == "nooo way"
    assert robust_z(10, [1, 1, 1, 1], floor=1.0) == 9.0


def _feed_chat(det, t0, seconds, rate, rng, users=300, words=("hello", "nice", "gg", "true")):
    for s in range(seconds):
        n = rng.poisson(rate) if hasattr(rng, "poisson") else _poisson(rng, rate)
        for _ in range(n):
            det.add_message(ChatMessage(t0 + s + rng.random(), f"u{rng.randrange(users)}",
                                        rng.choice(words)))


def _poisson(rng, lam):
    # Knuth; fine for small rates
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _run(det, t_from, t_to):
    peaks = []
    for now in range(int(t_from), int(t_to)):
        p = det.evaluate(now)
        if p:
            peaks.append(p)
    return peaks


def test_detector_fires_once_on_a_chat_burst():
    rng = random.Random(7)
    det = PeakDetector(threshold=3.5, warmup=120, cooldown=60, reaction_lag=4)
    t0 = 1_000_000
    _feed_chat(det, t0, 400, 2.0, rng)
    # the moment at t0+400, chat explodes 2 s later for 8 s
    burst = ("KEKW", "NO WAY", "???", "LMAO", "CLIP IT", "💀")
    _feed_chat(det, t0 + 402, 8, 18.0, rng, users=2000, words=burst)
    _feed_chat(det, t0 + 410, 100, 2.0, rng)
    peaks = _run(det, t0 + 130, t0 + 500)
    assert len(peaks) == 1, [p.ts - t0 for p in peaks]
    p = peaks[0]
    assert 395 <= p.ts - t0 <= 408
    assert p.chat_z > 3.5 and p.hype_share > 0.5
    assert any(m in ("KEKW", "NO WAY", "???", "LMAO", "CLIP IT", "💀") for m in p.messages)


def test_detector_quiet_stream_has_no_false_peaks():
    rng = random.Random(3)
    det = PeakDetector(threshold=3.5, warmup=120, cooldown=60)
    t0 = 2_000_000
    _feed_chat(det, t0, 1800, 3.0, rng)
    assert _run(det, t0 + 130, t0 + 1800) == []


def test_detector_ignores_warmup_and_single_spammer():
    rng = random.Random(5)
    det = PeakDetector(threshold=3.5, warmup=120, cooldown=60)
    t0 = 3_000_000
    _feed_chat(det, t0, 60, 2.0, rng)
    _feed_chat(det, t0 + 60, 8, 20.0, rng, words=("KEKW",))  # during warmup
    assert _run(det, t0 + 1, t0 + 75) == []
    _feed_chat(det, t0 + 75, 300, 2.0, rng)
    for s in range(8):  # one user spamming 15 msg/s
        for i in range(15):
            det.add_message(ChatMessage(t0 + 380 + s + i / 15, "spammer", "KEKW"))
    assert _run(det, t0 + 130, t0 + 420) == []


def test_detector_audio_only_peak_and_cooldown():
    det = PeakDetector(threshold=3.0, warmup=60, cooldown=200, audio_weight=1.0)
    rng = random.Random(1)
    t0 = 4_000_000
    for s in range(600):
        db = -30 + rng.uniform(-2, 2)
        if 300 <= s < 305 or 380 <= s < 385:
            db = -8
        det.add_loudness(t0 + s, db)
    peaks = _run(det, t0 + 70, t0 + 600)
    assert len(peaks) == 1  # the second spike is inside the cooldown
    assert 298 <= peaks[0].ts - t0 <= 306 and peaks[0].audio_z > 3
    assert peaks[0].chat_z == 0


def test_detector_hourly_cap():
    det = PeakDetector(threshold=3.0, warmup=0, cooldown=10, max_per_hour=2)
    t0 = 5_000_000
    for s in range(1200):
        loud = -8 if s % 200 == 150 or s % 200 == 151 or s % 200 == 152 else -30 + (s % 3)
        det.add_loudness(t0 + s, loud)
    assert len(_run(det, t0 + 40, t0 + 1200)) == 2


# ----------------------------------------------------------------------- recorder
def test_segment_list_and_rms_parsing():
    assert parse_segment_list_line("seg_00012.ts,120.000000,129.999667") == ("seg_00012.ts", 120.0, 129.999667)
    assert parse_segment_list_line("../x.ts,0,1") is None
    assert parse_segment_list_line("seg_00001.ts,a,b") is None
    out = parse_rms_lines("frame:0    pts:0       pts_time:0\nlavfi.astats.Overall.RMS_level=-23.3\n"
                          "frame:1    pts:16000   pts_time:1\nlavfi.astats.Overall.RMS_level=-inf\n")
    assert out == [(0.0, -23.3), (1.0, -90.0)]


def test_quality_steps_go_down_only():
    from livewatch.recorder import quality_steps
    assert quality_steps("720p60,720p,best") == ["720p60,720p,best", "720p,480p,best",
                                                 "480p,360p,best", "360p,worst"]
    assert quality_steps("480p,best") == ["480p,best", "360p,worst"]
    assert quality_steps("1080p60,best")[1] == "720p60,720p,best"


def test_recorder_notices_it_is_falling_behind(tmp_path):
    from livewatch.recorder import BEHIND, Recorder
    log = tmp_path / "streamlink.log"
    rec = Recorder("https://www.twitch.tv/x", tmp_path, SegmentBuffer(60), "best")
    log.write_bytes(b"[stream.segmented][warning] Sequence gap of 5 segments\n")
    gaps = []
    pos = rec._falling_behind(log, gaps, 0)
    assert not rec.stop_event.is_set() and len(gaps) == 1
    with open(log, "ab") as fh:
        fh.write(b"Sequence gap of 3\nSequence gap of 9\n")
    rec._falling_behind(log, gaps, pos)
    assert rec.stop_event.is_set() and rec.exit_reason == BEHIND


def test_segment_buffer_prunes_and_selects(tmp_path):
    buf = SegmentBuffer(keep_seconds=30)
    dropped = []
    for i in range(6):
        dropped += buf.add(Segment(tmp_path / f"seg_{i:05d}.ts", 100 + i * 10, 110 + i * 10))
    assert [d.path.name for d in dropped] == ["seg_00000.ts", "seg_00001.ts"]
    assert len(buf) == 4 and buf.latest_end == 160 and buf.earliest_start == 120
    segs, offset, real = buf.select(125, 145)
    assert [s.path.name for s in segs] == ["seg_00002.ts", "seg_00003.ts", "seg_00004.ts"]
    assert offset == 5 and real == 20
    segs, offset, real = buf.select(100, 200)  # clamped to what is held
    assert offset == 0 and real == 40
    assert buf.select(0, 50) == ([], 0.0, 0.0)


# --------------------------------------------------------------------------- text
CTX = viral.ClipContext(streamer="Streamer", platform="twitch", category="Grand Theft Auto VI",
                        transcript="bro what was that", chat=["KEKW KEKW KEKW", "NO WAY 💀",
                                                              "https://spam.example"], seed="s1")


@pytest.mark.parametrize("style", config.CAPTION_STYLES)
def test_offline_text_every_style(style):
    t = viral.write_text(CTX, style, base_tags=["gta6"], channel_tags=["streamer"])
    assert t.source == "offline" and t.style == style
    assert t.hook and t.title and t.description
    assert t.hashtags[0] == "gta6" and "Streamer".lower() in [h.lower() for h in t.hashtags]
    assert len(t.hashtags) == len({h.lower() for h in t.hashtags})
    assert not viral.has_cjk(t.hook + t.title)  # on-video text is always English
    assert t.to_dict() == viral.write_text(CTX, style, base_tags=["gta6"],
                                           channel_tags=["streamer"]).to_dict()


@pytest.mark.parametrize("desc", viral.DESCRIPTION_STYLES)
def test_description_styles_only_change_the_social_caption(desc):
    t = viral.write_text(CTX, "ooc", desc_style=desc, base_tags=["gta6"])
    posts = viral.platform_posts(t, ["youtube", "instagram", "tiktok", "x"])
    assert not viral.has_cjk(t.hook + t.title + t.description)
    assert not viral.has_cjk(posts["youtube"]["title"] + posts["youtube"]["description"]
                                + posts["x"]["title"])
    if desc == "normal":
        assert t.social_description == ""
        assert posts["instagram"]["description"].startswith(t.description.split("\n")[0])
    else:
        assert t.social_description and t.description_style == desc
        assert posts["instagram"]["description"].startswith(t.social_description)
        assert posts["tiktok"]["description"].startswith(t.social_description)
        assert not viral.has_cjk(t.social_description)


def test_non_english_ai_text_is_rejected():
    zh = dict(GOOD, hook="这一幕我直接看傻了")
    assert viral.parse_gemini_response(_gemini_payload(zh), "ooc") is None
    odd = dict(GOOD, social_description="主播当场破防了")
    t = viral.parse_gemini_response(_gemini_payload(odd), "ooc", "weird")
    assert t is not None and t.social_description == ""  # the offline weird line is used instead
    tags = dict(GOOD, hashtags=["gta6", "游戏"])
    assert viral.parse_gemini_response(_gemini_payload(tags), "ooc").hashtags == ["gta6"]


@pytest.mark.parametrize("line,ok", [
    ("xxxxxxx", False), ("KEKW", False), ("monka", False), ("KEKW KEKW KEKW", False),
    ("STREAMER AWRADS", True), ("oh no he did it", True), ("@someone mods are useless bruh", True),
    ("https://x.example go here", False), ("ww", False), ("asdfghjkl qwrtp", False),
])
def test_usable_chat_line(line, ok):
    assert (viral._usable_chat_line(line) is not None) == ok


def test_game_name_and_tags():
    assert viral.game_name("Grand Theft Auto VI") == "GTA 6"
    assert viral.game_name("Grand Theft Auto V") == "GTA 5"
    assert viral.game_name("Survival") == "Survival"
    assert viral.clean_tag("#GTA 6!") == "GTA6"
    assert viral.merge_tags(["a", "A", "#b"], ["c"], limit=2) == ["a", "b"]
    assert viral.strip_emoji("no way 💀😭 bro") == "no way bro"


def test_platform_posts_respect_limits():
    t = viral.ClipText(hook="h", title="x" * 120, description="d" * 3000,
                       hashtags=[f"tag{i}" for i in range(20)], style="ooc")
    posts = viral.platform_posts(t, ["youtube", "tiktok", "instagram", "x", "threads", "bluesky"])
    assert len(posts["youtube"]["title"]) <= 100 and "#shorts" in posts["youtube"]["description"]
    assert len(posts["x"]["title"]) <= 280 and len(posts["bluesky"]["title"]) <= 300
    assert posts["instagram"]["description"].count("#") <= 5
    assert posts["threads"]["title"].count("#") <= 1
    assert len(posts["tiktok"]["description"]) <= 2200
    short = viral.platform_posts(viral.ClipText("h", "T", "line one\nline  two", ["gta6"]), ["tiktok"])
    assert short["tiktok"]["description"] == "line one\nline two\n\n#gta6"


def _gemini_payload(obj):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]}


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        return self.responses.pop(0)


GOOD = {"hook": "he really did that 💀", "title": "GTA 6 chaos", "description": "no context",
        "hashtags": ["#gta6", "chaos", 5], "spoiler": False, "postworthy": 8}


def test_gemini_text_parses_and_validates(monkeypatch):
    monkeypatch.setattr(viral.time, "sleep", lambda s: None)
    sess = FakeSession([FakeResp(429), FakeResp(200, _gemini_payload(GOOD))])
    t = viral.gemini_text(CTX, "ooc", "key", "m", session=sess)
    assert t.source == "gemini" and t.postworthy == 8 and t.hashtags == ["gta6", "chaos"]
    url, kw = sess.calls[0]
    assert url.endswith("/models/m:generateContent") and kw["headers"]["x-goog-api-key"] == "key"
    assert "KEKW" in kw["json"]["contents"][0]["parts"][0]["text"]
    assert viral.gemini_text(CTX, "ooc", "key", "m", session=FakeSession([FakeResp(400)])) is None
    bad = _gemini_payload({"hook": "", "title": "t", "description": "d"})
    assert viral.parse_gemini_response(bad, "ooc") is None
    assert viral.parse_gemini_response({"candidates": []}, "ooc") is None


def test_write_text_falls_back_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(viral.time, "sleep", lambda s: None)
    t = viral.write_text(CTX, "brainrot", "key", "m",
                         session=FakeSession([FakeResp(500)] * 3))
    assert t.source == "offline" and t.style == "brainrot"


# ----------------------------------------------------------------------- captions
def test_clean_words_drops_hallucinations():
    assert clean_words([Word(0, 1, " Thanks for watching!")]) == []
    assert clean_words([Word(0, 1, " hey"), Word(1, 1, "x"), Word(1, 2, " ")]) == [Word(0, 1, "hey")]


def test_build_ass_hook_and_karaoke():
    words = [Word(0.2 + i * 0.3, 0.45 + i * 0.3, w) for i, w in
             enumerate(["no", "way", "he", "{did}", "that", "😭"])]
    ass = captions.build_ass(words, "he really thought nobody saw that 😭", 10, style="ooc")
    assert "Style: Hook,Montserrat ExtraBold" in ass and "nobody saw that" in ass
    assert "😭" not in ass and "{did}" not in ass and "(DID)" in ass
    dialogue = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogue) == 1 + 5  # hook + one event per spoken word
    assert dialogue[0].startswith("Dialogue: 1,0:00:00.00,0:00:10.00,Hook")
    assert captions.ass_time(3725.5) == "1:02:05.50"
    # caption events never overlap, and the last word lingers through a short pause
    spans = [(l.split(",")[1], l.split(",")[2]) for l in dialogue[1:]]
    secs = [tuple(sum(float(x) * m for x, m in zip(t.split(":"), (3600, 60, 1))) for t in s) for s in spans]
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(secs, secs[1:]))
    lone = captions.build_ass([Word(1.0, 1.3, "hey"), Word(5.0, 5.3, "you")], "", 10, show_hook=False)
    assert "0:00:01.00,0:00:01.90,Cap" in lone
    plain = captions.build_ass(words, "x", 10, style="ooc", show_captions=False, show_hook=False)
    assert "Dialogue:" not in plain


def test_shift_words_and_wrap():
    ws = captions.shift_words([Word(1, 2, "a"), Word(9, 12, "b"), Word(20, 21, "c")], 5, 6)
    assert ws == [Word(4, 6, "b")]
    assert captions.wrap("one two three four five six", 9) == ["one two", "three", "four five", "six"]


# ------------------------------------------------------------------------ posting
def test_upload_fields_per_network():
    posts = viral.platform_posts(viral.ClipText("h", "Title", "Desc", ["gta6"]),
                                 ["youtube", "tiktok", "instagram", "facebook", "x"])
    f = poster.upload_fields(posts)
    assert f["youtube_title"].startswith("Title") and f["privacyStatus"] == "public"
    assert f["tiktok_title"].startswith("Desc") and f["media_type"] == "REELS"
    assert f["facebook_media_type"] == "REELS" and f["x_title"].startswith("Title")
    assert f["title"]


def test_post_ledger_gap_cap_and_persistence(tmp_path):
    led = poster.PostLedger(tmp_path / "l.json")
    now = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
    gap = timedelta(minutes=20)
    assert led.slot(now, gap, 3) == now
    led.record(now)
    assert led.slot(now + timedelta(minutes=5), gap, 3) == now + gap
    led.record(now + gap)
    led.record(now + 2 * gap)
    assert led.slot(now + timedelta(hours=2), gap, 3) is None
    assert poster.PostLedger(tmp_path / "l.json").times == led.times


def test_post_video_sends_platforms_and_schedule(tmp_path):
    video = tmp_path / "short.mp4"
    video.write_bytes(b"\0" * 100)
    posts = viral.platform_posts(viral.ClipText("h", "T", "D", ["gta6"]), ["youtube", "x"])
    sess = FakeSession([FakeResp(200, {"request_id": "r1"})])
    later = datetime.now(timezone.utc) + timedelta(minutes=30)
    assert poster.post_video(video, posts, api_key="k", user="me", when=later, session=sess) == {"request_id": "r1"}
    _, kw = sess.calls[0]
    fields = kw["data"]
    assert ("platform[]", "youtube") in fields and ("platform[]", "x") in fields
    assert ("user", "me") in fields and any(k == "scheduled_date" for k, _ in fields)
    assert kw["headers"]["Authorization"] == "Apikey k"
    with pytest.raises(RuntimeError):
        poster.post_video(video, posts, api_key="k", user="me",
                          session=FakeSession([FakeResp(401, text="bad key")]))


# ------------------------------------------------------------------------- render
def test_build_filter_layouts():
    g = render.build_filter("blur", 30, None, "captions.ass", "../fonts", "x264")
    assert "boxblur" in g and "ass=captions.ass:fontsdir=../fonts" in g and g.endswith("format=yuv420p[vout]")
    g = render.build_filter("facecam", 30, [0.1, 0.6, 0.3, 0.4], None, None, "qsv")
    assert "vstack" in g and "crop=iw*0.3000:ih*0.4000:iw*0.1000:ih*0.6000" in g and "ass=" not in g
    assert g.endswith("format=nv12[vout]")
    assert render._filter_path(Path("C:/a b/c.ass"), Path("D:/x")).count(r"\:") <= 1


def _make_segments(d: Path, n=3, seconds=4):
    segs = []
    for i in range(n):
        p = d / f"seg_{i:05d}.ts"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc2=size=640x360:rate=30:duration={seconds}", "-f", "lavfi", "-i",
                        f"sine=frequency={300 + 100 * i}:duration={seconds}", "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(p)], check=True)
        segs.append(Segment(p, 1000 + i * seconds, 1000 + (i + 1) * seconds))
    return segs


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_end_to_end(tmp_path):
    segs = _make_segments(tmp_path)
    raw = render.extract_raw(segs, tmp_path / "job" / "raw.ts")
    assert render.probe(raw)["duration"] == pytest.approx(12, abs=0.6)
    ass = tmp_path / "job" / "captions.ass"
    ass.write_text(captions.build_ass([Word(0.5, 1.0, "hello"), Word(1.0, 1.6, "chat")],
                                      "no context needed", 6, "ooc"), encoding="utf-8")
    out = render.render_short(raw, tmp_path / "job" / "short.mp4", 3.0, 6.0,
                              ass_path=ass, encoder="x264")
    info = render.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["audio"]
    assert info["duration"] == pytest.approx(6, abs=0.3)
    out2 = render.render_short(raw, tmp_path / "job" / "fc.mp4", 0, 3, layout="facecam",
                               facecam=[0, 0.5, 0.4, 0.5], encoder="x264")
    assert render.probe(out2)["height"] == 1920


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_service_processes_a_moment(tmp_path, monkeypatch):
    from livewatch.detector import Peak
    from livewatch.service import ChannelWatch, ClipJob, LiveInfo, Service
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = config.from_dict({"general": {"data_dir": str(tmp_path / "data")},
                          "clip": {"pre_seconds": 5, "post_seconds": 2, "min_seconds": 5},
                          "style": {"whisper_model": "off", "encoder": "x264"},
                          "channel": [{"platform": "twitch", "name": "abc", "tags": ["abc"]}]})
    svc = Service(s)
    ch = s.channels[0]
    w = ChannelWatch.__new__(ChannelWatch)  # no recorder/chat threads in a unit test
    w.channel, w.info, w.pending = ch, LiveInfo(True, "title", "Grand Theft Auto VI"), []
    w.buffer = SegmentBuffer(600)
    for seg in _make_segments(tmp_path):
        w.buffer.add(seg)
    peak = Peak(ts=1006.0, score=5.0, chat_z=5.0, audio_z=0, hype_share=0.6,
                messages=["NO WAY", "KEKW"], reason="chat x5.0")
    svc._on_peak(w, peak)
    assert len(w.pending) == 1
    svc._flush_pending(w)
    assert w.pending == [] and svc.jobs.qsize() == 1
    job = svc.jobs.get()
    assert isinstance(job, ClipJob) and job.duration == pytest.approx(7, abs=0.01)
    out = svc.process(job)
    meta = json.loads((job.dir / "meta.json").read_text(encoding="utf-8"))
    assert out.exists() and not job.raw.exists()  # raw cut removed after render
    assert meta["posted"]["status"] == "held" and meta["text"]["source"] == "offline"
    assert set(meta["posts"]) == {"youtube", "tiktok", "instagram"}
    assert (s.data_dir / "clips.csv").read_text(encoding="utf-8").count("\n") == 2
