"""Looped secondary footage stays below the main clip throughout the export."""

import json
import shutil
import subprocess
import uuid

import pytest

import broll
import export_quality
import layout_ranges
import recut
import video_looks


@pytest.fixture
def videos(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and FFprobe are required")
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    primary = tmp_path / "primary.mp4"
    secondary = tmp_path / "secondary.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=160x284:r=24",
        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2.5", "-c:v", "libx264",
        "-threads", "1", "-c:a", "aac", str(primary),
    ], check=True, timeout=15)
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x96:r=24:d=0.5",
        "-f", "lavfi", "-i", "color=lime:s=160x96:r=24:d=0.5",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[video]", "-map", "[video]",
        "-c:v", "libx264", "-threads", "1", str(secondary),
    ], check=True, timeout=15)
    return primary, secondary


def _audio_hash(path):
    return subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-",
    ], timeout=15)


def _color_at(path, time, y):
    rgb = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", str(time), "-i", str(path), "-frames:v", "1",
        "-vf", f"crop=2:2:80:{y},scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], timeout=15)
    assert len(rgb) == 3
    return max(range(3), key=rgb.__getitem__)


def test_bottom_video_loops_and_keeps_audio_geometry_and_caption_seam(videos):
    primary, secondary = videos
    audio = _audio_hash(primary)
    asset = broll.prepare_asset(secondary, "secondary.mp4", primary.parent)
    broll.render_bottom_video(primary, asset["id"])

    assert broll._probe(primary) == pytest.approx((160, 284, 2.5, 0), abs=0.05)
    assert _audio_hash(primary) == audio
    for time in (0.2, 0.7, 1.2, 1.7, 2.2):
        assert _color_at(primary, time, y=70) == 2
        expected_bottom = 0 if time % 1 < 0.5 else 1
        assert _color_at(primary, time, y=210) == expected_bottom
    assert layout_ranges.caption_anchors(layout_ranges.read(str(primary))) == [(0.0, 2.5, 0.5)]


@pytest.mark.parametrize("mode", ["broll", "bottom"])
def test_video_overlay_preserves_a_delayed_main_picture(videos, mode):
    primary, secondary = videos
    delayed = primary.with_name("delayed.mp4")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-itsoffset", "0.5", "-i", str(primary), "-i", str(primary),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(delayed),
    ], check=True, timeout=15)
    audio = _audio_hash(delayed)
    asset = broll.prepare_asset(secondary, secondary.name, primary.parent)
    if mode == "bottom":
        broll.render_bottom_video(delayed, asset["id"])
        assert layout_ranges.caption_anchors(layout_ranges.read(str(delayed))) == [(0.5, 3.0, 0.5)]
        assert _color_at(delayed, 0.7, y=210) == 0
        assert _color_at(delayed, 1.2, y=210) == 1
    else:
        broll.render(delayed, [{"asset_id": asset["id"], "offset": 0, "start": 1, "duration": 0.35}])
        assert _color_at(delayed, 1.15, y=210) == 0
    streams = {stream["codec_type"]: stream for stream in export_quality.inspect(delayed)["streams"]}
    assert float(streams["video"]["start_time"]) == pytest.approx(0.5, abs=0.05)
    assert float(streams["video"]["duration"]) == pytest.approx(2.5, abs=0.05)
    assert float(streams["audio"]["start_time"]) == pytest.approx(0, abs=0.05)
    assert _audio_hash(delayed) == audio


def test_bottom_video_edit_round_trips_and_keeps_asset_until_disabled(videos, monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    api = pytest.importorskip("app")
    primary, secondary = videos
    job = str(uuid.uuid4())
    directory = tmp_path / job
    directory.mkdir()
    shutil.copy(primary, directory / "video_clip_1.mp4")
    asset = broll.prepare_asset(secondary, "secondary.mp4", directory)
    metadata = directory / "video_metadata.json"
    clip = {"start": 0, "end": 2.5, "video_url": f"/videos/{job}/video_clip_1.mp4"}
    metadata.write_text(json.dumps({"shorts": [clip], "output_format": "vertical"}))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(api, "jobs", {job: {"status": "completed", "result": {"clips": [clip.copy()]}}})
    monkeypatch.setattr(api, "_archive_clip_edit_bg", lambda *_args: None)
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return "video_clip_1.mp4", "video_clip_1.mp4"

    monkeypatch.setattr(recut, "perform_recut", render)
    client = TestClient(api.app)
    base = {"job_id": job, "clip_index": 0, "segments": [{"start": 0, "end": 2.5}],
            "reapply_captions": False}
    save = client.post("/api/clip/rerender", json={**base, "bottom_video_asset_id": asset["id"],
                       "color_filter": "warm", "broll": [{"asset_id": asset["id"], "start": 0.5, "duration": 1}]})
    assert save.status_code == 200, save.text
    assert save.json()["bottom_video_asset_id"] == asset["id"]
    assert [layer.func for layer in calls[-1]["picture_layers"]] == [
        video_looks.render, broll.render, broll.render_bottom_video,
    ]

    rerender = client.post("/api/clip/rerender", json={**base, "broll": []})
    assert rerender.json()["bottom_video_asset_id"] == asset["id"]
    edl = client.get(f"/api/clip/{job}/0/edl").json()
    assert edl["bottom_video_asset_id"] == asset["id"]
    delete = f"/api/clip/{job}/0/assets/broll/{asset['id']}"
    assert client.delete(delete).status_code == 409

    missing = client.post("/api/clip/rerender", json={**base, "bottom_video_asset_id": uuid.uuid4().hex})
    assert missing.status_code == 400
    off = client.post("/api/clip/rerender", json={**base, "bottom_video_asset_id": None})
    assert off.status_code == 200, off.text
    assert off.json()["bottom_video_asset_id"] is None
    assert [layer.func for layer in calls[-1]["picture_layers"]] == [video_looks.render]
    assert client.delete(delete).status_code == 200
