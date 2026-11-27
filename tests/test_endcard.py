"""End cards stay outside the editable body and are appended once after edits."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest
from starlette.testclient import TestClient

import endcard
import export_quality

api = pytest.importorskip("app")


@pytest.fixture
def body(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required")
    monkeypatch.setenv("FFMPEG_ENCODER", "x264")
    monkeypatch.setenv("AUDIO_NORMALIZE", "1")
    path = tmp_path / "body.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=160x284:r=24",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "2", "-c:v", "libx264", "-c:a", "aac", str(path)], check=True)
    return path


@pytest.mark.parametrize("sound", ["silent", "chime"])
def test_card_preserves_body_and_export_timing(body, sound):
    digest = hashlib.sha256(body.read_bytes()).digest()
    output = endcard.append(body, {"text": "Follow for more", "handle": "@example", "duration": 1, "sound": sound})
    result = export_quality.validate(output, expected_duration=3, output_format="vertical", audio_expected=True)
    assert result["valid"], result
    assert result["video"]["duration"] == pytest.approx(3, abs=0.08)
    assert hashlib.sha256(body.read_bytes()).digest() == digest
    assert endcard.body_name(output.name) == body.name
    with pytest.raises(endcard.EndCardError, match="body"):
        endcard.append(output, {"duration": 1})


def test_image_asset_retains_animation_and_card_render(body):
    from PIL import Image
    gif = body.with_name("animation.gif")
    Image.new("RGB", (60, 60), "red").save(gif, save_all=True,
        append_images=[Image.new("RGB", (60, 60), "green")], duration=[100, 100], loop=0)
    asset = endcard.prepare_asset(gif, "animation.gif", body.parent)
    assert asset["animated"]
    output = endcard.append(body, {"text": "Thanks", "duration": 1, "sound": "silent", "asset_id": asset["id"]})
    assert export_quality.validate(output, expected_duration=3, output_format="vertical", audio_expected=True)["valid"]


def test_nonzero_video_start_keeps_audio_aligned(body):
    import numpy as np
    shifted = body.with_name("shifted.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=blue:s=160x284:r=24:d=2",
        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*440*t)*between(t\\,0.65\\,0.85):s=48000:d=2.5",
        "-vf", "setpts=PTS+0.5/TB", "-c:v", "libx264", "-c:a", "aac", "-fps_mode", "passthrough", str(shifted)], check=True)
    start = endcard._probe(shifted)[-1]
    assert start == pytest.approx(0.5, abs=0.05)
    output = endcard.append(shifted, {"duration": 1, "sound": "silent"})
    samples = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
        "-map", "0:a", "-f", "f32le", "-ac", "1", "-ar", "48000", "pipe:1"]), dtype="<f4")
    def energy(a, b):
        return float(np.mean(samples[round(a * 48000):round(b * 48000)] ** 2))
    assert energy(0.20, 0.30) > 0.1
    assert energy(0.65, 0.75) < 0.001
    assert export_quality.validate(output, expected_duration=3, output_format="vertical", audio_expected=True)["valid"]


def test_audio_entirely_after_body_becomes_silence(body):
    import numpy as np
    shifted = body.with_name("late_audio.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(body), "-itsoffset", "3", "-i", str(body),
        "-map", "0:v", "-map", "1:a", "-c", "copy", str(shifted)], check=True)
    output = endcard.append(shifted, {"duration": 1, "sound": "silent"})
    samples = np.frombuffer(subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
        "-map", "0:a", "-f", "f32le", "-ac", "1", "pipe:1"]), dtype="<f4")
    assert float(np.max(np.abs(samples))) < 0.001
    assert export_quality.validate(output, expected_duration=3, output_format="vertical", audio_expected=True)["valid"]


def test_generation_setting_is_bounded_text_only_and_worker_safe():
    import remote_worker
    config = {"text": "Follow 👋", "handle": "@creator", "duration": 3, "sound": "chime"}
    raw = json.dumps(config, ensure_ascii=False)
    assert endcard.generation_config(raw) == config
    assert remote_worker.allowed_option("CLIP_ENDCARD_JSON", raw)
    assert endcard.generation_config("null") is None
    for invalid in ({"duration": True}, {"duration": 4}, {"text": "x" * 97}, {"asset_id": "a" * 32}, {"sound": "external"}):
        with pytest.raises(endcard.EndCardError):
            endcard.generation_config(invalid)
        assert not remote_worker.allowed_option("CLIP_ENDCARD_JSON", json.dumps(invalid))


def test_committed_metadata_wins_over_failed_end_card_artifact(tmp_path):
    clean = "video_clip_1.mp4"
    (tmp_path / clean).write_bytes(b"body")
    (tmp_path / f"endcard_deadbeef_{clean}").write_bytes(b"failed candidate")
    (tmp_path / "video_metadata.json").write_text(json.dumps({"shorts": [{"video_url": f"/videos/job/{clean}"}]}))
    assert api._canonical_clip_file(tmp_path, "video", 0) == clean


def test_card_survives_caption_and_hook_replacements_once(body, tmp_path, monkeypatch):
    import recut

    ident = str(uuid.uuid4())
    directory = tmp_path / "output" / ident
    directory.mkdir(parents=True)
    canonical = directory / "video_clip_1.mp4"
    shutil.copy(body, canonical)
    clip = {"start": 0, "end": 2, "video_url": f"/videos/{ident}/{canonical.name}"}
    words = [{"word": " Hello", "start": 0.1, "end": 0.5}, {"word": " world", "start": 0.6, "end": 1}]
    metadata = directory / "video_metadata.json"
    metadata.write_text(json.dumps({"shorts": [clip], "output_format": "vertical", "transcript": {"language": "en", "segments": [{"words": words}]}}))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(directory.parent))
    monkeypatch.setattr(api, "jobs", {ident: {"status": "completed", "result": {"clips": [clip.copy()]}}})

    def recut_body(**kwargs):
        name = "recut_1_video_clip_1.mp4"
        shutil.copy(body, directory / name)
        return name, name

    def captions(input_path, output_path, *_args):
        assert endcard.body_name(Path(input_path).name) == Path(input_path).name
        shutil.copy(input_path, output_path)
        return True

    def hook(input_path, *_args, **_kwargs):
        assert endcard.body_name(Path(input_path).name) == Path(input_path).name
        target = directory / f"hooked_{uuid.uuid4().int}_video_clip_1.mp4"
        shutil.copy(input_path, target)
        return str(target)

    monkeypatch.setattr(recut, "perform_recut", recut_body)
    monkeypatch.setattr(api, "_burn_caption_style", captions)
    monkeypatch.setattr(api, "_burn_clip_text", hook)
    client = TestClient(api.app)
    req = {"job_id": ident, "clip_index": 0, "segments": [{"start": 0, "end": 2}], "reapply_captions": False,
           "endcard": {"text": "Follow", "duration": 1, "sound": "silent"}}
    response = client.post("/api/clip/rerender", json=req)
    assert response.status_code == 200, response.text
    assert response.json()["export_duration"] == 3
    for endpoint, extra in [("/api/subtitle", {}), ("/api/hook", {"text": "Opening"}), ("/api/subtitle/remove", {})]:
        response = client.post(endpoint, json={"job_id": ident, "clip_index": 0, **extra})
        assert response.status_code == 200, response.text
        filename = response.json()["new_video_url"].rsplit("/", 1)[-1]
        assert endcard.body_name(filename) != filename
        assert export_quality.validate(directory / filename, expected_duration=3, output_format="vertical", audio_expected=True, decode=False)["valid"]
    response = client.post("/api/clip/rerender", json={**req, "endcard": None})
    assert response.status_code == 200, response.text
    assert response.json()["export_duration"] == 2


def test_archive_keeps_the_body_under_an_end_card():
    from cloud.videos import _twins_to_archive, _superseded_names
    body = "subtitled_1_hooked_2_video_clip_1.mp4"
    served = f"endcard_deadbeef_{body}"
    twins = _twins_to_archive(f"wm_{served}", "video", 0)
    assert served in twins and body in twins and "hooked_2_video_clip_1.mp4" in twins
    assert body not in _superseded_names(body, "video_clip_1.mp4", served, "video", 0)
