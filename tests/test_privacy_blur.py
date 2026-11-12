"""Opt-in privacy blur.

The choice is per job and off unless the request asks. Detection and the
pixel pass are local. These tests cover the decision, the text rules and,
when OpenCV and ffmpeg are installed, the plate finder and the encode.
"""
import asyncio
import hashlib
import os
import shutil
import subprocess

import pytest

import privacy_blur


def test_job_env_value_accepts_only_an_explicit_yes():
    assert privacy_blur.job_env_value(None) is None
    assert privacy_blur.job_env_value(False) is None
    assert privacy_blur.job_env_value(True) == "1"
    for raw in ("1", " true ", "YES", "Yes", 1):
        assert privacy_blur.job_env_value(raw) == "1"
    for raw in ("0", "false", "no", "", "on", "2", 0, 1.0):
        assert privacy_blur.job_env_value(raw) is None


def test_wanted_is_explicit_or_the_exact_env_value(monkeypatch):
    monkeypatch.setenv(privacy_blur.ENV_NAME, "1")
    assert privacy_blur.wanted(None) is True
    assert privacy_blur.wanted(False) is False
    monkeypatch.setenv(privacy_blur.ENV_NAME, "yes")
    assert privacy_blur.wanted(None) is False
    assert privacy_blur.wanted(True) is True
    monkeypatch.delenv(privacy_blur.ENV_NAME, raising=False)
    assert privacy_blur.wanted(None) is False


def test_from_job_trusts_the_env_and_only_a_boolean_in_metadata():
    assert privacy_blur.from_job({privacy_blur.ENV_NAME: "1"}, {"blur_sensitive": False}) is True
    assert privacy_blur.from_job({privacy_blur.ENV_NAME: "0"}, {"blur_sensitive": True}) is False
    assert privacy_blur.from_job({}, {"blur_sensitive": True}) is True
    assert privacy_blur.from_job({}, {"blur_sensitive": "true"}) is False
    assert privacy_blur.from_job({}, {"blur_sensitive": 1}) is False
    assert privacy_blur.from_job(None, None) is False


def test_pii_spans_cover_personal_details_and_skip_ordinary_numbers():
    spans = privacy_blur.pii_spans(
        "mail ada@example.com or Ada.Bee+tag@example.co.uk, "
        "call +44 7700 900123, 555-123-4567 and 1-800-555-1212, "
        "ssn 123-45-6789, iban gb82west12345698765432, "
        "card 4242424242424242")
    text = ("mail ada@example.com or Ada.Bee+tag@example.co.uk, "
            "call +44 7700 900123, 555-123-4567 and 1-800-555-1212, "
            "ssn 123-45-6789, iban gb82west12345698765432, "
            "card 4242424242424242")
    found = [text[s:e] for s, e in spans]
    for piece in (
        "ada@example.com", "Ada.Bee+tag@example.co.uk", "+44 7700 900123",
        "555-123-4567", "1-800-555-1212", "123-45-6789",
        "gb82west12345698765432", "4242424242424242",
    ):
        assert piece in found

    assert privacy_blur.pii_spans("shot on 2024-11-12 at 192.168.0.1") == []
    assert privacy_blur.pii_spans("card 4242424242424243") == []
    assert privacy_blur.pii_spans("0000000000000000") == []
    assert privacy_blur.pii_spans("order 1234567890") == []
    assert privacy_blur.pii_spans("") == []


def test_boxes_for_line_keep_only_the_words_that_overlap():
    words = [
        {"text": "email", "x": 0, "y": 0, "w": 20, "h": 10},
        {"text": "ada@example.com", "x": 30, "y": 0, "w": 40, "h": 10},
        {"text": "today", "x": 80, "y": 0, "w": 20, "h": 10},
    ]
    boxes = privacy_blur.boxes_for_line(words)
    assert len(boxes) == 1
    assert boxes[0][0] == 30 and boxes[0][4] == "text"

    split = [
        {"text": "call", "x": 0, "y": 0, "w": 20, "h": 10},
        {"text": "555", "x": 30, "y": 0, "w": 15, "h": 10},
        {"text": "123-4567", "x": 50, "y": 0, "w": 30, "h": 10},
        {"text": "please", "x": 200, "y": 0, "w": 20, "h": 10},
    ]
    boxes = privacy_blur.boxes_for_line(split)
    assert len(boxes) == 1
    assert boxes[0][0] == 30
    assert boxes[0][2] == 50  # 30+15 through 50+30


def test_expand_box_clamps_and_rejects_a_frame_filling_glitch():
    box = privacy_blur.expand_box(-20, -10, 40, 30, "face", 200, 200)
    assert box is not None and box[0] == 0 and box[1] == 0 and box[4] == "face"
    assert privacy_blur.expand_box(0, 0, 90, 90, "plate", 100, 100) is None
    assert privacy_blur.expand_box(10, 10, 30, 20, "plate", 0, 100) is None
    kept = privacy_blur.expand_box(40, 40, 20, 12, "text", 320, 240)
    assert kept is not None and kept[2] >= 20 and kept[4] == "text"


def test_update_tracks_holds_unsearched_kinds():
    tracks = [{"box": (0, 0, 10, 10), "kind": "face", "missed": 1}]
    held = privacy_blur.update_tracks(tracks, None, ("face",))
    assert held == tracks
    assert held[0] is not tracks[0]

    mixed = privacy_blur.update_tracks(
        tracks, [(5, 5, 8, 8, "text")], ("text",))
    face = next(t for t in mixed if t["kind"] == "face")
    assert face["missed"] == 1 and face["box"] == (0, 0, 10, 10)
    assert any(t["kind"] == "text" and t["missed"] == 0 for t in mixed)

    aged = privacy_blur.update_tracks(tracks, [], ("face",))
    assert aged[0]["missed"] == 2
    expired = [{"box": (0, 0, 10, 10), "kind": "text", "missed": 3}]
    assert privacy_blur.update_tracks(expired, [], ("text",)) == []


def test_parse_tsv_drops_low_confidence_and_scales():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
        "left\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t80\tada@x.co\n"
        "5\t1\t1\t1\t1\t2\t50\t20\t10\t12\t10\tnope\n"
        "4\t1\t1\t1\t1\t0\t0\t0\t0\t0\t-1\tline\n"
        "5\t1\t2\t1\t1\t1\t0\t0\t8\t8\t90\thello\tthere\n"
    )
    words = privacy_blur.parse_tsv(tsv, scale=2)
    assert [w["text"] for w in words] == ["ada@x.co", "hello\tthere"]
    assert words[0]["x"] == 20 and words[0]["w"] == 60
    boxes = privacy_blur.pii_boxes(words)
    assert len(boxes) == 1 and boxes[0][4] == "text"


def test_merge_and_quads():
    merged = privacy_blur.merge_boxes([
        (0, 0, 10, 10, "face"),
        (2, 2, 10, 10, "face"),
        (0, 0, 10, 10, "plate"),
    ])
    kinds = sorted(b[4] for b in merged)
    assert kinds == ["face", "plate"]
    assert len([b for b in merged if b[4] == "face"]) == 1

    boxes = privacy_blur.quads_to_boxes([[[0, 0], [20, 1], [19, 18], [1, 20]]])
    assert boxes == [(0, 0, 20, 20, "qr")]
    assert privacy_blur.quads_to_boxes(None) == []
    assert privacy_blur.quads_to_boxes([]) == []


def test_autopilot_does_not_ask_for_privacy_blur():
    # Autopilot submits on the user's behalf. It must not opt them in.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "cloud", "autopilot.py"), encoding="utf-8").read()
    assert "blur_sensitive" not in text
    assert "BLUR_SENSITIVE" not in text


def test_cli_sends_blur_only_when_the_flag_is_passed(monkeypatch):
    import cli.openshorts_cli as cli
    seen = {}

    def fake(method, path, body=None):
        seen["body"] = body
        return 200, {"job_id": "j"}

    monkeypatch.setattr(cli, "_request", fake)
    cli.main(["process", "https://youtu.be/abcdefghijk"])
    assert "blur_sensitive" not in seen["body"]
    cli.main(["--json", "process", "https://youtu.be/abcdefghijk", "--blur-sensitive"])
    assert seen["body"]["blur_sensitive"] is True


def test_resume_keeps_an_opt_in_and_ignores_a_host_export(tmp_path, monkeypatch):
    app = pytest.importorskip("app")
    monkeypatch.setenv(privacy_blur.ENV_NAME, "1")
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(app, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app, "jobs", {})
    monkeypatch.setattr(app, "job_queue", asyncio.PriorityQueue())

    def resume(job_id, blur):
        job_dir = out / job_id
        job_dir.mkdir()
        manifest = {
            "cmd": ["python", "-u", "main.py"],
            "priority": 2,
            "attempts": 0,
            "submitted_at": 1,
        }
        if blur is not None:
            manifest["blur_sensitive"] = blur
        app._register_resumed([(job_id, manifest)])
        return app.jobs[job_id]["env"]

    assert resume("11111111-1111-1111-1111-111111111111", True).get("BLUR_SENSITIVE") == "1"
    # A string in a tampered or old manifest is not a yes, even when the
    # host itself exported the variable.
    assert "BLUR_SENSITIVE" not in resume("22222222-2222-2222-2222-222222222222", "1")
    assert "BLUR_SENSITIVE" not in resume("33333333-3333-3333-3333-333333333333", None)
    assert "BLUR_SENSITIVE" not in resume("44444444-4444-4444-4444-444444444444", False)


def _plate_frame():
    import numpy as np
    img = np.full((240, 320, 3), 25, np.uint8)
    x, y, w, h = 40, 150, 200, 40
    img[y:y + h, x:x + w] = (235, 235, 235)
    img[y:y + 3, x:x + w] = 0
    img[y + h - 3:y + h, x:x + w] = 0
    img[y:y + h, x:x + 4] = 0
    img[y:y + h, x + w - 4:x + w] = 0
    for i in range(10, w - 10, 9):
        img[y + 8:y + h - 8, x + i:x + i + 3] = (15, 15, 15)
    return img


def test_find_plates_hits_a_glyph_plate_and_ignores_a_blank():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    found = privacy_blur.find_plates(_plate_frame())
    assert found, "a high-contrast plate should be found"
    assert any(b[4] == "plate" and b[0] < 80 and b[1] > 100 for b in found)

    blank = np.full((240, 320, 3), 40, np.uint8)
    blank[80:140, 40:200] = (230, 230, 230)
    assert privacy_blur.find_plates(blank) == []


def test_plates_found_on_the_small_frame_are_scaled_back(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    frame = np.zeros((720, 1280, 3), np.uint8)
    seen = {}

    def vehicles(image, yolo, lock):
        return [(200, 300, 800, 300)]

    def plates(bgr, vehicles=None):
        seen["width"] = bgr.shape[1]
        seen["vehicles"] = list(vehicles or [])
        return [(20, 30, 80, 20, "plate")]

    monkeypatch.setattr(privacy_blur, "_faces", lambda image: [])
    monkeypatch.setattr(privacy_blur, "_qrs", lambda image: [])
    monkeypatch.setattr(privacy_blur, "_text", lambda image: [])
    monkeypatch.setattr(privacy_blur, "_vehicles", vehicles)
    monkeypatch.setattr(privacy_blur, "find_plates", plates)

    class Cap:
        def __init__(self):
            self.done = False

        def read(self):
            if self.done:
                return False, None
            self.done = True
            return True, frame

    regions, hits = privacy_blur._scan_clip(Cap(), cv2, 30, 1280, 720, object(), None, None)
    assert seen["width"] == 640
    assert seen["vehicles"][0][0] == 100  # 200 / (1280/640)
    assert hits == 1
    x, y, w, h, kind = regions[0][0]
    assert kind == "plate" and w >= 200


def _clip(path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=0.4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True)


def _region_std(path, box):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    x, y, w, h = box
    return float(frame[y:y + h, x:x + w].std()), frame


def _has_audio(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return "audio" in probe.stdout


def test_apply_blurs_the_region_and_keeps_the_audio(tmp_path):
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg")
    path = tmp_path / "clip.mp4"
    _clip(path)
    box = (40, 40, 80, 80)
    before, _ = _region_std(path, box)
    far_before, _ = _region_std(path, (220, 40, 80, 80))

    def detect(frame, index):
        return [(*box, "face")]

    assert privacy_blur.apply(str(path), detect=detect) is True
    after, _ = _region_std(path, box)
    far_after, _ = _region_std(path, (220, 40, 80, 80))
    assert after < before * 0.75
    assert abs(far_after - far_before) / max(far_before, 1) < 0.2
    assert _has_audio(path)
    assert not os.path.exists(str(path) + ".privacy.partial")


def test_apply_leaves_the_file_when_nothing_matches_or_detection_fails(tmp_path):
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg")
    path = tmp_path / "clip.mp4"
    _clip(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert privacy_blur.apply(str(path), detect=lambda frame, index: []) is True
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    def boom(frame, index):
        raise RuntimeError("detector down")

    assert privacy_blur.apply(str(path), detect=boom) is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_apply_fails_closed_when_the_face_detector_is_missing(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg")
    monkeypatch.setattr(privacy_blur, "_FACE", None)
    monkeypatch.setattr(privacy_blur, "_FACE_ERR", "face detector unavailable (test)")
    path = tmp_path / "clip.mp4"
    _clip(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert privacy_blur.apply(str(path)) is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
