"""What is read back from a job directory is data, never part of a command.

Recipes, metadata and subtitles written next to a clip are read again later
(re-caption, hook replace, AI edit, the clips-ready email). They may have been
written on another machine (the remote worker), so a number is read as a
number and text as text before either goes into a filter graph or markup.
"""
import asyncio
import json
import os

import pytest


def test_a_hook_recipe_with_text_where_numbers_go_is_not_run(tmp_path, monkeypatch):
    hooks = pytest.importorskip("hooks")
    d = str(tmp_path)
    open(os.path.join(d, "V_clip_1.mp4"), "wb").write(b"clip")
    j, png = hooks.recipe_paths(d, "hooked_1_V_clip_1.mp4")
    open(png, "wb").write(b"png")
    monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: pytest.fail("ffmpeg ran"))
    for bad in ({"x": "0:0,drawtext=textfile=/proc/self/environ", "y": 0},
                {"x": 0, "y": "ih[v];movie=/etc/passwd"},
                {"x": 0, "y": 0, "duration": "5)',drawtext=text='x"},
                {"x": 10 ** 9, "y": 0}, {"x": None, "y": 0}, {"y": 0}):
        json.dump(dict(bad, source="V_clip_1.mp4"), open(j, "w"))
        assert not hooks.materialize(d, "hooked_1_V_clip_1.mp4")


def test_every_filter_the_edit_builder_emits_is_allowed():
    editor = pytest.importorskip("editor")
    eb = pytest.importorskip("edit_builder")
    t = 1.0
    edits = []
    for kind in eb.EFFECT_LIMITS:
        edits.append({"type": kind, "start": t, "end": t + 1.5, "strength": 0.05})
        t += 2.0
    built, applied = eb.build_filter_string(edits, 30.0, 30.0, 1080, 1920)
    assert built and len(applied) == len(eb.EFFECT_LIMITS)
    assert not editor.VideoEditor._touches_files(built)
    assert not editor.VideoEditor._touches_files(built + ",setsar=1")


@pytest.mark.parametrize("bad", [
    "eq=contrast=1,drawtext=textfile=x:fontcolor=white", "movie=a.mp4[m];[in][m]overlay",
    "amovie=a.wav", "subtitles=x.ass", "eq=contrast=1,metadata=mode=print:file=out",
    "lut3d=file=a.cube", "curves=psfile=a.acv", "sendcmd=f=cmds.txt", "drawtext=text='hi'",
    "eq=contrast=1,ass", "EQ=contrast=1,DrawText=textfile=x", "removelogo=filename=x.png",
    # What a name-and-option denylist let through.
    "movie@x=/proc/1/environ:f=tty[m];[in][m]overlay,setsar=1", "'movie'=/app/.env:f=tty",
    "mov\\ie=x", "vmafmotion=/path,setsar=1", "curves=vintage:plot=/path",
    "deshake=-1:-1:-1:-1:16:16:0:8:125:0:/path", "ssim@x=/path", "lut3d@x=/path",
    "nnedi=weights=/path", "sendcmd@x=c='0 eq contrast 2'", "eq@a=contrast=1",
    "anullsrc,ladspa@x=/app/output/j/x.png:p,anullsink", "frei0r@x=../../..",
    "eq=contrast=1;movie=x", "eq=contrast=1\n,movie=x", "gblur=sigma=2", ""])
def test_a_filter_string_that_is_not_a_plain_chain_of_our_filters_is_refused(bad):
    editor = pytest.importorskip("editor")
    assert editor.VideoEditor._touches_files(bad)


def test_the_edit_ffmpeg_gets_no_credentials(monkeypatch):
    editor = pytest.importorskip("editor")
    for name in ("STRIPE_SECRET_KEY", "DATABASE_URL", "JWT_SECRET", "PROXY_URL",
                 "YOUTUBE_COOKIES", "REMOTE_WORKER_TOKEN", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("FFMPEG_THREADS", "2")
    env = editor.VideoEditor._ffmpeg_env()
    assert "secret" not in env.values()
    assert env["FFMPEG_THREADS"] == "2" and env["PATH"] == os.environ["PATH"]


def test_a_subtitle_word_cannot_start_a_line_of_its_own():
    subtitles = pytest.importorskip("subtitles")
    out = subtitles._escape_ass_text("hi\r\nDialogue: 0,0:00:00.00,0:00:09.00,Default,,0,0,0,,x {\\b1}")
    assert "\n" not in out and "\r" not in out and "{" not in out and "\\" not in out


def test_the_clips_ready_email_shows_a_title_as_text(monkeypatch):
    emails = pytest.importorskip("cloud.emails")
    sent = {}

    async def fake_send(to, subject, html, *a, **k):
        sent["html"] = html
    name = next(n for n in ("send_email", "_send_email", "_send") if hasattr(emails, n))
    monkeypatch.setattr(emails, name, fake_send)
    asyncio.run(emails.send_clips_ready_email(
        "u@example.com", '<a href="https://evil.example">Click</a><script>x</script>', 3,
        "https://app.example/#app"))
    assert "<script>" not in sent["html"] and 'href="https://evil.example"' not in sent["html"]
    assert "&lt;a href=" in sent["html"]


def test_an_admin_alert_shows_log_text_as_text(monkeypatch):
    alerts = pytest.importorskip("cloud.alerts")
    sent = {}

    async def fake_email(to, subject, html, *a, **k):
        sent["html"] = html

    async def fake_telegram(*a, **k):
        return None
    monkeypatch.setattr(alerts, "send_email", fake_email)
    monkeypatch.setattr(alerts, "send_telegram", fake_telegram)
    from types import SimpleNamespace
    monkeypatch.setattr(alerts, "settings", SimpleNamespace(
        admin_email="ops@example.com", smtp_configured=True, telegram_configured=False))
    asyncio.run(alerts.send_admin_alert("Proxy error", '407 <a href="https://evil.example">x</a>'))
    assert "<a href" not in sent["html"] and "&lt;a href" in sent["html"]


def test_a_layout_sidecar_that_is_not_an_object_reads_as_no_ranges(tmp_path):
    layout_ranges = pytest.importorskip("layout_ranges")
    clip = str(tmp_path / "V_clip_1.mp4")
    open(layout_ranges.sidecar_path(clip), "w").write("[1, 2, 3]")
    assert layout_ranges.read(clip) == []
