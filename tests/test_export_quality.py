import shutil
import subprocess

import pytest

from export_quality import validate, validate_metadata


def metadata(sar="1:1", width=1080, height=1920):
    return {"format": {"duration": 24}, "streams": [
        {"codec_type": "video", "width": width, "height": height, "sample_aspect_ratio": sar,
         "start_time": "0", "duration": "24"},
        {"codec_type": "audio", "start_time": "0", "duration": "24"},
    ]}


def test_distorted_sar_is_rejected_before_publication():
    issues, sample = validate_metadata(metadata(sar="64:5"), expected_duration=24, output_format="vertical",
                                       audio_expected=True)
    assert "non-square pixels" in " ".join(issues)
    assert sample["sample_aspect_ratio"] == "64:5"
    valid, _ = validate_metadata(metadata(), expected_duration=24, output_format="vertical", audio_expected=True)
    assert not valid


def test_checks_duration_and_audio_track():
    data = metadata()
    data["streams"].pop()
    issues, _sample = validate_metadata(data, expected_duration=20, output_format="vertical", audio_expected=True)
    assert "duration" in issues[0]
    assert "audio" in issues[1]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg is optional in unit CI")
def test_real_wrong_sar_and_blank_frames(tmp_path):
    def file(name, source, sar):
        path = tmp_path / name
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
                        "-f", "lavfi", "-i", source, "-t", "2", "-vf", f"setsar={sar}",
                        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path)],
                       check=True, timeout=15)
        return path
    good = file("good.mp4", "testsrc2=s=144x256:r=15", "1")
    stretched = file("stretched.mp4", "testsrc2=s=144x256:r=15", "64/5")
    blank = file("blank.mp4", "color=black:s=144x256:r=15", "1")
    tests = {"expected_duration": 2, "output_format": "vertical", "audio_expected": False}
    assert validate(good, **tests)["valid"]
    assert not validate(stretched, **tests)["valid"]
    assert not validate(blank, **tests)["valid"]


def test_has_audio_is_false_when_the_source_cannot_be_probed(tmp_path):
    """Whether the source has audio only sets an expectation: a probe that
    fails must not fail (or 500) the clip that is being checked."""
    from export_quality import has_audio
    assert has_audio(tmp_path / "missing.mp4") is False
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    assert has_audio(tmp_path / "junk.mp4") is False


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_has_audio_reads_real_streams(tmp_path):
    from export_quality import has_audio
    silent, voiced = tmp_path / "silent.mp4", tmp_path / "voiced.mp4"
    base = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", "color=black:s=64x64:d=1"]
    subprocess.run([*base, str(silent)], check=True)
    subprocess.run([*base, "-f", "lavfi", "-i", "sine=d=1", "-shortest", str(voiced)], check=True)
    assert has_audio(silent) is False and has_audio(voiced) is True
