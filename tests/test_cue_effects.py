"""Timed cues follow the source cut, can be cleared, and render bounded effects."""
import asyncio
import json
import shutil
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

import cue_effects
import export_quality


def test_source_anchor_reappears_in_repeated_cuts_and_cannot_cross_a_cut():
    cue = {"kind": "boom", "source_time": 4.5, "duration": 1.2}
    segments = [
        {"start": 7, "end": 9}, {"start": 4, "end": 5}, {"start": 4, "end": 6},
    ]
    rendered = cue_effects.placements([cue], segments)
    assert [(item["start"], item["duration"]) for item in rendered] == [(2.5, 0.5), (3.5, 1.2)]
    assert cue_effects.placements([cue], segments[:1]) == []
    assert cue_effects.placements([cue], [{"start": 4, "end": 4.6}]) == []


def test_invalid_cues_and_unknown_assets_cannot_enter_the_filter_graph():
    cue = {"kind": "hit", "source_time": 1.23456, "duration": 99, "volume": 0.29}
    assert cue_effects.normalize([
        cue, {**cue, "kind": "movie=/path"}, {**cue, "source_time": float("nan")},
        {**cue, "volume": "0.5;movie=bad"}, {**cue, "duration": float("inf")},
    ]) == [{"kind": "hit", "source_time": 1.235, "duration": 0.3, "volume": 0.29}]


@pytest.mark.parametrize("original_audio", [True, False])
def test_ffmpeg_timings_sound_volume_and_video_dip(tmp_path, monkeypatch, original_audio):
    np = pytest.importorskip("numpy")
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    monkeypatch.setenv("SFX_CACHE_DIR", str(tmp_path / "sfx"))
    path = tmp_path / "body.mp4"
    command = ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x284:r=24"]
    if original_audio:
        command.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-c:a", "aac"])
    subprocess.run([*command, "-t", "2", "-c:v", "libx264", str(path)], check=True)
    cues = cue_effects.placements([
        {"kind": "hit", "source_time": 0.5, "duration": 0.3, "volume": 0.7},
        {"kind": "whoosh", "source_time": 1.5, "duration": 0.3, "volume": 0},
        {"kind": "dip_to_black", "source_time": 1, "duration": 0.5},
    ], [{"start": 0, "end": 2}])
    cue_effects.render(path, cues)

    def red_at(second):
        pixel = subprocess.check_output([
            "ffmpeg", "-hide_banner", "-v", "error", "-ss", str(second), "-i", str(path),
            "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ])
        return pixel[0]

    info = export_quality.inspect(path)
    assert float(info["format"]["duration"]) == pytest.approx(2, abs=0.08)
    assert red_at(0.3) > 200 and red_at(1.25) < 40 and red_at(1.8) > 200
    samples = np.frombuffer(subprocess.check_output([
        "ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ar", "48000", "-ac", "1", "-f", "f32le", "pipe:1",
    ]), dtype="<f4")

    def rms(start, end):
        section = samples[round(start * 48000):round(end * 48000)]
        return np.sqrt(np.mean(section ** 2))

    before = rms(0.2, 0.35)
    assert (before > 0.04) if original_audio else (before < 0.003)
    assert rms(0.51, 0.55) > before + 0.05
    assert rms(1.55, 1.65) == pytest.approx(before, abs=0.015)
    assert not list(tmp_path.glob("temp_cue_*"))


def test_sound_mix_preserves_delayed_source_audio(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    monkeypatch.setenv("SFX_CACHE_DIR", str(tmp_path / "sfx"))
    path = tmp_path / "delayed.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x284:r=24",
        "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "2", "-c:v", "libx264", "-c:a", "aac", str(path),
    ], check=True)
    source_audio = next(s for s in export_quality.inspect(path)["streams"] if s["codec_type"] == "audio")
    assert float(source_audio["start_time"]) == pytest.approx(0.5, abs=0.05)

    cue_effects.render(path, [{"kind": "hit", "start": 1.4, "duration": 0.3, "volume": 0.7}])
    samples = np.frombuffer(subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ar", "48000",
        "-ac", "1", "-f", "f32le", "pipe:1",
    ]), dtype="<f4")
    assert np.max(np.abs(samples[4800:14400])) < 0.003
    assert np.max(np.abs(samples[28800:38400])) > 0.04


def test_api_saves_and_reapplies_cues_until_explicitly_cleared(tmp_path, monkeypatch):
    import app as api

    ident = str(uuid4())
    directory = tmp_path / ident
    directory.mkdir()
    metadata = directory / "video_metadata.json"
    filename = "video_clip_1.mp4"
    (directory / filename).touch()
    (directory / "source.mp4").touch()
    clip = {"start": 0, "end": 2, "video_url": f"/videos/{ident}/{filename}"}
    metadata.write_text(json.dumps({"shorts": [clip], "source_video": "source.mp4", "output_format": "vertical"}))
    job = {"status": "completed", "logs": [], "result": {"clips": [clip.copy()]}}
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(api, "BILLING_ENABLED", False)
    monkeypatch.setattr(api, "jobs", {ident: job})
    monkeypatch.setattr(api, "_source_duration_seconds", lambda _path: 2)
    monkeypatch.setenv("SFX_CACHE_DIR", str(tmp_path / "sfx"))
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return filename, filename

    async def quality(*_args, **_kw):
        return None

    monkeypatch.setattr(api.recut, "perform_recut", render)
    monkeypatch.setattr(api, "_check_edited_export", quality)
    cue = {"kind": "hit", "source_time": 0.6, "duration": 0.3, "volume": 0.4}
    body = {"job_id": ident, "clip_index": 0, "segments": [{"start": 0, "end": 2}], "reapply_captions": False}

    async def run():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/clip/rerender", json={**body, "cue_effects": [cue]})
            assert response.status_code == 200, response.text
            assert response.json()["cue_effects"] == [cue]
            edl = (await client.get(f"/api/clip/{ident}/0/edl")).json()
            assert edl["cue_effects"] == [cue]
            assert {item["kind"] for item in edl["cue_effects_catalog"]} == set(cue_effects.CATALOG)
            recut = await client.post("/api/clip/rerender", json=body)
            assert recut.status_code == 200
            assert recut.json()["cue_effects"] == [cue]
            reframe = await api._reframe_locked(
                SimpleNamespace(job_id=ident, clip_index=0, reapply_captions=False), None, job, {})
            assert reframe["cue_effects"] == [cue]
            assert all(call["picture_layers"][-1].func is cue_effects.render for call in calls)

            cleared = await client.post("/api/clip/rerender", json={**body, "cue_effects": []})
            assert cleared.status_code == 200 and not calls[-1]["picture_layers"]
            assert json.loads(metadata.read_text())["shorts"][0]["cue_effects"] == []
            assert (await client.post("/api/clip/rerender", json={
                **body, "cue_effects": [{**cue, "kind": "arbitrary-filter"}],
            })).status_code == 422
            sound = await client.get("/api/clip-cues/sounds/hit")
            assert sound.status_code == 200 and sound.content.startswith(b"RIFF")
            assert (await client.get("/api/clip-cues/sounds/unknown")).status_code == 404

    asyncio.run(run())
