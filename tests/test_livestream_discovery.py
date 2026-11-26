import asyncio
from io import BytesIO
import shutil
import subprocess
import uuid

import pytest

import candidate_review
import livestream
import range_media


def test_review_only_prepares_shortlist_and_does_not_fetch_production(monkeypatch, tmp_path):
    source = object()
    source_data = {"id": "12345678901", "title": "Fixture gameplay", "height": 1080}
    class Fixture:
        url = "https://www.youtube.com/watch?v=12345678901"
        duration = 120
        summary = source_data
        def section(self, directory, ident, start, end, *, preview):
            assert preview and (ident, start, end) == (1, 25, 45)
            return source, 700

    monkeypatch.setattr(livestream, "inspect_source", lambda _url, **_kwargs: Fixture())
    monkeypatch.setattr(livestream, "fetch_replay", lambda *_: None)
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(livestream, "download_audio", lambda *_: audio)
    monkeypatch.setattr("transcribe_backends.media_seconds", lambda _p: 120)
    monkeypatch.setattr(livestream, "collect", lambda *a, **k: {"chat_status": "unavailable", "source": source_data})
    monkeypatch.setattr(livestream, "make_preview", lambda span, path, *_: path.write_bytes(b"preview"))
    monkeypatch.setattr(candidate_review, "assess_clip", lambda _path, clip, *_a, **_k: {
        **clip, "assessment": {"verdict": "supported", "ratings": {}, "evidence": "visible"},
    })
    plan = livestream.discover(
        Fixture.url, tmp_path, transcribe=lambda _p: {"segments": [{"text": "speech"}]},
        analyze=lambda *_a, **_k: {"shorts": [{"start": 25, "end": 45}]},
        load_transcript=lambda *_: None, save_transcript=lambda *_: None,
        require_speech=lambda *_: True, output_format="vertical", framing_method="gameplay")
    result = candidate_review.public_result(plan)
    assert result["source_duration"] == 120
    assert result["performance"]["preview_transfer_bytes"] == 700
    assert result["candidates"][0]["preview"] is True
    assert not list(tmp_path.rglob("*source-*.mkv"))


def test_staged_preview_and_approval_are_keyed_by_candidate_not_file_path(tmp_path, monkeypatch):
    app = pytest.importorskip("app")
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", False)
    job_id = str(uuid.uuid4())
    output = tmp_path / job_id
    folder = livestream.workspace(output)
    url = "https://www.youtube.com/watch?v=12345678901"
    cmd = ["python", "-u", "main.py", "-u", url, "-o", str(output),
           "--review-candidates", "--audio-first"]
    (folder / "preview-3.mp4").write_bytes(b"private preview")
    (folder / "totally-different.mp4").write_bytes(b"wrong preview")
    candidate_review.write(output, {"source_video": ".audio-first", "source_duration": 120,
                                    "deferred": {"id": "12345678901"}, "shorts": [
        {"start": 10, "end": 25, "preview_slot": 2},
        {"start": 41, "end": 65, "preview_slot": 3},
    ]})
    app._write_resume_manifest(job_id, cmd, 2, None, None, False)
    app.jobs[job_id] = {"cmd": cmd, "logs": app._TimedLog()}
    app._finish_job(job_id, 0, str(output))
    app.jobs.pop(job_id)
    monkeypatch.setattr(app, "_enqueue_job", lambda _job_id, _priority: None)

    async def call():
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            base = f"/api/jobs/{job_id}"
            assert (await client.get(base + "/candidate-preview/1")).status_code == 404
            assert (await client.get(base + "/candidate-preview/3")).status_code == 404
            assert (await client.get(base + "/candidate-preview/2")).content == b"private preview"
            result = await client.post(base + "/render-candidates", json={"candidate_ids": [2]})
            assert result.status_code == 202, result.text

    try:
        asyncio.run(call())
        actual = app.jobs[job_id]["cmd"]
        assert actual[-2:] == ["--render-candidates", "2"]
        assert url in actual
        assert "--audio-first" not in actual
    finally:
        app.jobs.pop(job_id, None)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Needs FFmpeg")
def test_packet_range_preserves_original_clock_and_decoded_frame(tmp_path):
    src = tmp_path / "source.mp4"
    dest = tmp_path / "range.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30",
        "-f", "lavfi", "-i", "sine=frequency=550", "-t", "24", "-g", "60",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1", "-c:a", "aac", str(src),
    ], check=True, timeout=20)
    blob = src.read_bytes()

    class Response(BytesIO):
        status = 206

    def opener(_url, requested):
        start, end = requested[6:].split("-")
        start, end = int(start), int(end) if end else len(blob) - 1
        body = Response(blob[start:end + 1])
        body.headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
        return body

    with range_media.byte_ranges(["fixture"], len(blob) * 4, opener=opener) as (inputs, stats):
        range_media.copy_packets(inputs, dest, 17, 20, http=True)
        span = range_media.inspect(dest, 17, 20, height=180)
    assert 12 <= span.offset < 17
    assert 20 <= span.end < 22
    assert stats["bytes"] > 0

    def frame(path, when):
        return subprocess.check_output([
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{when:.4f}", "-i", str(path),
            "-map", "0:v:0", "-frames:v", "1", "-f", "md5", "-",
        ], timeout=15)

    assert frame(src, 18.5) == frame(dest, span.local(18.5))


def test_pause_and_resume_preserve_checkpoint_and_do_not_dispatch_twice(tmp_path, monkeypatch):
    app = pytest.importorskip("app")
    from httpx import ASGITransport, AsyncClient

    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", False)
    monkeypatch.setenv("GEMINI_API_KEY", "fixture-key")
    enqueued = []
    monkeypatch.setattr(app, "_enqueue_job", lambda *args: enqueued.append(args))
    job_id = str(uuid.uuid4())
    output = tmp_path / job_id
    output.mkdir()
    checkpoint = livestream.workspace(output) / "signals.json"
    checkpoint.write_text('{"chat_status":"unavailable"}')
    cmd = ["python", "-u", "main.py", "-u", "https://www.youtube.com/watch?v=12345678901",
           "-o", str(output), "--review-candidates", "--audio-first"]
    app._write_resume_manifest(job_id, cmd, 2, None, None, False)
    app.jobs[job_id] = {"status": "processing", "cmd": cmd, "logs": app._TimedLog(), "user_id": None}
    app._running_jobs.add(job_id)

    async def call():
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            base = f"/api/jobs/{job_id}"
            assert (await client.post(base + "/pause-discovery")).status_code == 202
            assert app.jobs[job_id].pop("pause_requested")
            assert app._pause_discovery(job_id, app.jobs[job_id])
            app._running_jobs.discard(job_id)
            app.jobs.pop(job_id)
            status = (await client.get(f"/api/status/{job_id}")).json()
            assert status["status"] == "paused"
            assert (await client.post(base + "/resume-discovery")).status_code == 202
            assert (await client.post(base + "/resume-discovery")).status_code == 409

    try:
        asyncio.run(call())
        assert enqueued == [(job_id, 2)]
        assert checkpoint.exists()
        assert app.jobs[job_id]["cmd"] == cmd
    finally:
        app._running_jobs.discard(job_id)
        app.jobs.pop(job_id, None)


def test_a_typo_in_the_chat_settings_does_not_crash_discovery(tmp_path, monkeypatch):
    import livestream
    monkeypatch.setenv("HIGHLIGHT_CHAT_WINDOW", "ten")
    monkeypatch.setenv("HIGHLIGHT_CHAT_DELAY", "")
    monkeypatch.setenv("HIGHLIGHT_CHAT_OFFSET", "nan")
    seen = {}
    monkeypatch.setattr(livestream.highlight_signals, "chat_evidence",
                        lambda path, duration, **kw: seen.update(kw) or {"chat_status": "unavailable", "chat": []})
    result = livestream.collect(tmp_path, None, 600.0, None, audio=False)
    assert result["chat_status"] == "unavailable"
    assert (seen["window"], seen["reaction_delay"], seen["offset"]) == (10, 7.0, 0.0)


def test_chat_settings_stay_clamped(tmp_path, monkeypatch):
    import livestream
    monkeypatch.setenv("HIGHLIGHT_CHAT_WINDOW", "500")
    monkeypatch.setenv("HIGHLIGHT_CHAT_DELAY", "-4")
    monkeypatch.setenv("HIGHLIGHT_CHAT_OFFSET", "12.5")
    seen = {}
    monkeypatch.setattr(livestream.highlight_signals, "chat_evidence",
                        lambda path, duration, **kw: seen.update(kw) or {"chat_status": "unavailable", "chat": []})
    livestream.collect(tmp_path, None, 600.0, None, start=2.5, audio=False)
    assert (seen["window"], seen["reaction_delay"], seen["offset"]) == (30, 0.0, 10.0)


def _fake_scan(candidates):
    return {"candidates": candidates, "lag": 6.0, "lag_confidence": 0.3, "chat_status": "available",
            "voice": True, "markers": {"voice": [{"kind": "voice", "time": 120, "start": 118, "end": 124}]},
            "curve": {"step": 5, "values": [0.1, 0.9]}}


def test_hype_review_runs_the_stages_in_order(tmp_path, monkeypatch):
    import hype_engine
    import hype_judge
    import livestream
    monkeypatch.setenv("CLIP_MIN_SECONDS", "30")
    monkeypatch.setenv("CLIP_MAX_SECONDS", "60")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cands = [{"start": 100.0, "end": 140.0, "peak": 120, "hype": 0.9, "area": 9, "signals": {}},
             {"start": 400.0, "end": 430.0, "peak": 420, "hype": 0.6, "area": 5, "signals": {}}]
    monkeypatch.setattr(hype_engine, "scan", lambda *a, **k: _fake_scan(cands))
    asked = []
    transcript = {"language": "en", "segments": [], "coverage": [[92, 148], [392, 438]]}
    out = livestream.hype_review(
        tmp_path, "audio.m4a", 3600.0, None,
        transcribe_ranges=lambda media, ranges: asked.append(ranges) or transcript,
        proxy=lambda i, c: None, gameplay=True)
    assert asked == [[(100.0, 140.0), (400.0, 430.0)]]
    assert [s["start"] for s in out["shorts"]] == [100.0, 400.0]      # no key: ranked by hype
    assert out["transcript"] is transcript
    assert out["signals"]["engine"] == "hype" and out["signals"]["chat_lag"] == 6.0
    assert out["signals"]["voice"][0]["time"] == 120
    assert (tmp_path / ".highlight" / "signals.json").is_file()


def test_hype_review_falls_back_when_nothing_stands_out(tmp_path, monkeypatch):
    import hype_engine
    import livestream
    monkeypatch.setattr(hype_engine, "scan", lambda *a, **k: _fake_scan([]))
    assert livestream.hype_review(tmp_path, "a.m4a", 600.0, None, transcribe_ranges=None,
                                  proxy=None, gameplay=False) is None


def test_replay_reach_reads_the_last_offset(tmp_path):
    import livestream
    path = tmp_path / "replay.live_chat.json.part"
    path.write_text('{"replayChatItemAction": {"videoOffsetTimeMsec": "1500"}}\n'
                    '{"replayChatItemAction": {"videoOffsetTimeMsec": "5785346"}}\n{"truncated')
    assert livestream._replay_reach(path) == pytest.approx(5785.346)
    assert livestream._replay_reach(tmp_path / "missing") == 0.0


def test_replay_limits_default_far_above_an_upload(monkeypatch):
    import chat_hype
    import livestream
    monkeypatch.delenv("HIGHLIGHT_REPLAY_MAX_MB", raising=False)
    monkeypatch.delenv("HIGHLIGHT_REPLAY_MINUTES", raising=False)
    size, seconds = livestream.replay_limits()
    assert size > 20 * chat_hype.MAX_UPLOAD_BYTES and seconds == 3600
