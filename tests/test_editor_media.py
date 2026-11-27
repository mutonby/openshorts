"""The editor's uploaded media must survive saves and leave timing and audio intact."""
from array import array
import json
import shutil
import subprocess
import uuid

import pytest
from starlette.testclient import TestClient

api = pytest.importorskip("app")
import broll
import export_quality
import music
import recut
import video_looks


@pytest.fixture
def media(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    source = tmp_path / "source.mp4"
    insert = tmp_path / "insert.mp4"
    track = tmp_path / "track.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=160x284:r=24",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=160x284:r=24",
                    "-t", "1", "-c:v", "libx264", str(insert)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=880",
                    "-t", "1", str(track)], check=True)
    return source, insert, track


def test_picture_and_music_render_keep_audio_and_timing(media):
    source, insert, track = media
    asset = broll.prepare_asset(insert, "insert.mp4", source.parent)
    bed = music.prepare(track, "track.wav", source.parent)
    video_looks.render(source, "monochrome")
    layers = [{"asset_id": asset["id"], "start": 0.5, "duration": 1, "offset": 0}]
    broll.render(source, layers)
    music.render(source, {"asset_id": bed["id"], "volume": 0.15, "offset": 0.1, "ducking": True})
    check = export_quality.validate(source, expected_duration=2, output_format="vertical", audio_expected=True, decode=False)
    assert check["valid"], check
    assert check["video"]["duration"] == pytest.approx(2, abs=0.08)
    # Grading precedes B-roll so the inserted red footage retains its color.
    pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "0.8", "-i", str(source),
                                      "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    assert pixels[0] > pixels[2] + 100


@pytest.mark.parametrize("ducking", [False, True])
def test_music_keeps_playing_after_source_audio_ends(media, ducking):
    source, _insert, track = media
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=160x284:r=24:d=3",
        "-f", "lavfi", "-i", "sine=frequency=440:d=0.75", "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac", str(source),
    ], check=True, timeout=15)
    bed = music.prepare(track, track.name, source.parent)
    music.render(source, {"asset_id": bed["id"], "volume": 0.3, "offset": 0, "ducking": ducking})
    result = export_quality.inspect(source)
    assert all(float(stream["duration"]) == pytest.approx(3, abs=0.05) for stream in result["streams"])
    tail = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", "2", "-i", str(source), "-t", "0.4", "-map", "0:a:0",
        "-ac", "1", "-f", "s16le", "-",
    ], timeout=15)
    assert max(abs(sample) for sample in array("h", tail)) > 400


def test_broll_anchor_follows_reordered_cut_and_is_removed_when_cut_away():
    layer = {"asset_id": "x", "start": 1, "duration": 2, "offset": 0, "source_time": 12}
    segments = [{"start": 20, "end": 22}, {"start": 11, "end": 15}]
    assert list(broll.active_layers([layer], segments))[0]["start"] == 3
    assert not list(broll.active_layers([layer], [{"start": 20, "end": 22}]))


def test_editor_media_upload_persist_and_remove(media, monkeypatch, tmp_path):
    source, insert, track = media
    ident = str(uuid.uuid4())
    directory = tmp_path / "output" / ident
    directory.mkdir(parents=True)
    shutil.copy(source, directory / "video_clip_1.mp4")
    shutil.copy(source, directory / "source.mp4")
    metadata = directory / "video_metadata.json"
    clip = {"start": 0, "end": 2, "video_url": f"/videos/{ident}/video_clip_1.mp4"}
    metadata.write_text(json.dumps({"shorts": [clip], "source_video": "source.mp4", "output_format": "vertical"}))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(directory.parent))
    monkeypatch.setattr(api, "jobs", {ident: {"status": "completed", "result": {"clips": [clip.copy()]}}})
    calls = []

    def render(**kwargs):
        calls.append(kwargs)
        return "video_clip_1.mp4", "video_clip_1.mp4"

    monkeypatch.setattr(recut, "perform_recut", render)
    client = TestClient(api.app)
    response = client.post(f"/api/clip/{ident}/0/broll/assets", files={"file": ("insert.mp4", insert.read_bytes(), "video/mp4")})
    assert response.status_code == 201, response.text
    asset = response.json()
    response = client.post(f"/api/clip/{ident}/0/music/assets", files={"file": ("track.wav", track.read_bytes(), "audio/wav")})
    assert response.status_code == 201, response.text
    track_asset = response.json()
    settings = {"job_id": ident, "clip_index": 0, "segments": [{"start": 0, "end": 2}],
                "reapply_captions": False, "color_filter": "warm",
                "broll": [{"asset_id": asset["id"], "start": 0.5, "duration": 1}],
                "music": {"asset_id": track_asset["id"], "volume": 0.2}}
    saved = client.post("/api/clip/rerender", json=settings)
    assert saved.status_code == 200, saved.text
    data = json.loads(metadata.read_text())["shorts"][0]
    assert data["color_filter"] == "warm" and data["music"]["volume"] == 0.2
    assert data["broll"][0]["asset_id"] == asset["id"]
    assert len(calls[-1]["picture_layers"]) == 3
    edl = client.get(f"/api/clip/{ident}/0/edl").json()
    assert edl["music_assets"][0]["id"] == track_asset["id"]
    delete_url = f"/api/clip/{ident}/0/assets/broll/{asset['id']}"
    assert client.delete(delete_url).status_code == 409
    reset = client.post("/api/clip/rerender", json={**settings, "color_filter": "original", "broll": [], "music": None})
    assert reset.status_code == 200, reset.text
    assert not calls[-1]["picture_layers"]
    assert client.delete(delete_url).status_code == 200
    assert client.delete(delete_url).status_code == 404


def test_project_credit_is_explicit_and_reaches_manual_and_scheduled_posts(monkeypatch, tmp_path):
    import httpx

    ident = str(uuid.uuid4())
    directory = tmp_path / ident
    directory.mkdir()
    metadata = directory / "video_metadata.json"
    metadata.write_text(json.dumps({"shorts": [{"start": 0, "end": 2}]}))
    (directory / "video_clip_1.mp4").write_bytes(b"fixture")
    clip = {"clip_index": 3, "video_url": f"/videos/{ident}/video_clip_1.mp4"}
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(api, "jobs", {ident: {"status": "completed", "result": {"clips": [clip]}}})
    sent = []

    def post(_url, _headers, data, *_args):
        sent.append(data)
        return httpx.Response(202, json={"request_id": "fixture-post"})

    monkeypatch.setattr(api, "_post_video_blocking", post)
    client = TestClient(api.app)
    response = client.put(f"/api/projects/{ident}/publishing", json={"source_credit_url": "https://example.com/source"})
    assert response.status_code == 200, response.text
    body = {"job_id": ident, "clip_index": 3, "api_key": "fixture", "user_id": "fixture",
            "platforms": ["youtube"], "description": "Example description"}
    for extra in ({}, {"scheduled_date": "2027-01-01T12:00:00Z"}):
        response = client.post("/api/social/post", json={**body, **extra})
        assert response.status_code == 200, response.text
        assert "https://example.com/source" in str(sent[-1])
    response = client.post("/api/social/post", json={**body, "source_credit_url": None})
    assert response.status_code == 200, response.text
    assert "https://example.com/source" not in str(sent[-1])
    assert client.put(f"/api/projects/{ident}/publishing", json={"source_credit_url": "javascript:alert(1)"}).status_code == 422
