"""Verify clean segment extraction/concat on a real MP4.

Usage:
    python scripts/verify_pts_concat.py demo-openshorts.mp4

The script checks: output duration, A/V duration agreement, monotonic packet
DTS/PTS, and that the joined output is approximately the sum of its inputs.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from brutal_truth import extract_clean_segment, concat_clean_segments


def probe(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,duration,start_time",
            "-of", "json", path,
        ], capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def packet_timestamps_monotonic(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,dts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    pts = []
    dts = []
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if len(fields) >= 2:
            if fields[0] not in ("N/A", ""):
                pts.append(float(fields[0]))
            if fields[1] not in ("N/A", ""):
                dts.append(float(fields[1]))
    return all(b >= a for a, b in zip(pts, pts[1:])) and all(
        b >= a for a, b in zip(dts, dts[1:])
    )


def main():
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-openshorts.mp4").resolve()
    if not source.exists():
        raise SystemExit(f"Missing source video: {source}")
    with tempfile.TemporaryDirectory(prefix="truelife_pts_") as tmp:
        a = os.path.join(tmp, "segment_a.mp4")
        b = os.path.join(tmp, "segment_b.mp4")
        joined = os.path.join(tmp, "joined.mp4")
        assert extract_clean_segment(str(source), 2, 7, a)
        assert extract_clean_segment(str(source), 20, 25, b)
        assert concat_clean_segments([a, b], joined)

        a_duration = float(probe(a)["format"]["duration"])
        b_duration = float(probe(b)["format"]["duration"])
        joined_probe = probe(joined)
        joined_duration = float(joined_probe["format"]["duration"])
        streams = {
            s["codec_type"]: float(s["duration"])
            for s in joined_probe.get("streams", [])
            if s.get("duration")
        }
        assert abs(joined_duration - (a_duration + b_duration)) < 0.35
        assert abs(streams["video"] - streams["audio"]) < 0.12
        assert packet_timestamps_monotonic(joined)
        print(json.dumps({
            "status": "PASS",
            "joined_duration": joined_duration,
            "video_duration": streams["video"],
            "audio_duration": streams["audio"],
            "pts_dts_monotonic": True,
        }, indent=2))


if __name__ == "__main__":
    main()
