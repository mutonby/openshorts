import os
import sys
import time
import types

import live_clipper


def test_read_segments_parses_completed_parts(tmp_path):
    p = tmp_path / "list_1.csv"
    p.write_text("part_00000.mp4,0.000000,1200.016667\n"
                 "part_00001.mp4,1200.016667,1800.5\n"
                 "garbage\n")
    assert live_clipper.read_segments(str(p)) == [
        ("part_00000.mp4", 0.0, 1200.016667),
        ("part_00001.mp4", 1200.016667, 1800.5),
    ]


def test_read_segments_missing_file_is_empty(tmp_path):
    assert live_clipper.read_segments(str(tmp_path / "nope.csv")) == []


def test_recorder_cmd_writes_fragmented_mp4_chunks(tmp_path):
    formats = [{"url": "https://v.example/v.m3u8", "http_headers": {"User-Agent": "UA"}},
               {"url": "https://v.example/a.m3u8"}]
    cmd = live_clipper.recorder_cmd(formats, "chunks", str(tmp_path), 2, 7, 1200)
    assert cmd[cmd.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in cmd
    assert cmd[cmd.index("-segment_time") + 1] == "1200"
    assert cmd[cmd.index("-segment_start_number") + 1] == "7"
    assert cmd[cmd.index("-segment_format") + 1] == "mp4"
    assert "frag_keyframe" in cmd[cmd.index("-segment_format_options") + 1]
    assert cmd[cmd.index("-segment_list") + 1] == os.path.join(str(tmp_path), "list_2.csv")
    assert cmd[-1] == os.path.join("chunks", "part_%05d.mp4")
    assert cmd[cmd.index("-user_agent") + 1] == "UA"


def test_recorder_cmd_single_muxed_input():
    cmd = live_clipper.recorder_cmd([{"url": "https://v.example/av.m3u8"}], "c", "s", 1, 0, 600)
    assert "0:a:0" in cmd


def test_slug_is_filename_safe():
    assert live_clipper.slug('Why "this" works? / no:way') == "Why-this-works-no-way"
    assert live_clipper.slug("") == "clip"
    assert live_clipper.slug("जब कोई हद पार कर दे! 😂 #shorts") == "shorts"


def test_romanize_keeps_ascii_titles_without_a_model_call():
    assert live_clipper.romanize("Already English") == "Already English"
    assert len(live_clipper.slug("x" * 200)) == 60


def test_upcoming_errors_are_recognised():
    assert live_clipper.is_upcoming_error("This live event will begin in 3 hours.")
    assert live_clipper.is_upcoming_error("Premieres in 10 minutes")
    assert not live_clipper.is_upcoming_error("Video unavailable")


def test_future_start_time_schedules_the_session(tmp_path):
    s = live_clipper.LiveSession("u", str(tmp_path), 20, start_at=time.time() + 600)
    assert s.state == "scheduled"
    assert live_clipper.LiveSession("u", str(tmp_path), 20).state == "waiting"


def _fake_core(monkeypatch, jobs=None, durations=None):
    core = types.SimpleNamespace(
        jobs=jobs or {},
        _media_duration_seconds=lambda p: (durations or {}).get(os.path.basename(p), 0.0))
    monkeypatch.setitem(sys.modules, "app", core)
    return core


def test_collect_numbers_chunks_and_continues_after_restart(tmp_path, monkeypatch):
    _fake_core(monkeypatch)
    s = live_clipper.LiveSession("u", str(tmp_path), 20)
    os.makedirs(s.state_dir)
    s.run = 2
    with open(os.path.join(s.state_dir, "list_1.csv"), "w") as f:
        f.write("part_00000.mp4,0,1200\npart_00001.mp4,1200,1300\n")
    with open(os.path.join(s.state_dir, "list_2.csv"), "w") as f:
        f.write("part_00002.mp4,0,1200\n")
    s.collect()
    s.collect()
    assert [c["index"] for c in s.chunks] == [0, 1, 2]
    assert [c["duration"] for c in s.chunks] == [1200.0, 100.0, 1200.0]
    assert s.next_index == 3
    assert s.public()["chunkCounts"] == {"recorded": 3}


def test_restore_resumes_and_recovers_chunks_left_on_disk(tmp_path, monkeypatch):
    _fake_core(monkeypatch, jobs={"j-alive": {"status": "processing"}},
               durations={"part_00003.mp4": 412.0})
    s = live_clipper.LiveSession("u", str(tmp_path), 20, language="hi")
    os.makedirs(s.chunk_dir)
    for n in range(4):
        open(os.path.join(s.chunk_dir, f"part_{n:05d}.mp4"), "wb").close()
    s.chunks = [
        {"index": 0, "file": os.path.join(s.chunk_dir, "part_00000.mp4"), "duration": 1200,
         "status": "clipping", "jobId": "j-alive", "clips": 0, "error": None, "recordedAt": None},
        {"index": 1, "file": os.path.join(s.chunk_dir, "part_00001.mp4"), "duration": 1200,
         "status": "clipping", "jobId": "j-gone", "clips": 0, "error": None, "recordedAt": None},
        {"index": 2, "file": os.path.join(s.chunk_dir, "part_00002.mp4"), "duration": 1200,
         "status": "submitting", "jobId": None, "clips": 0, "error": None, "recordedAt": None},
    ]
    s.state = "recording"
    s.next_index = 3
    s.save()

    import json
    with open(os.path.join(s.dir, "session.json"), encoding="utf-8") as f:
        r = live_clipper.LiveSession.restore(json.load(f))

    assert r.id == s.id and r.dir == s.dir and r.language == "hi"
    assert r.state == "waiting"
    assert [c["status"] for c in r.chunks] == ["clipping", "recorded", "recorded", "recorded"]
    assert r.chunks[3]["index"] == 3 and r.chunks[3]["duration"] == 412.0
    assert r.next_index == 4


def test_restore_keeps_a_paused_session_paused(tmp_path, monkeypatch):
    _fake_core(monkeypatch)
    s = live_clipper.LiveSession("u", str(tmp_path), 20)
    os.makedirs(s.chunk_dir)
    s.state = "paused"
    s.save()
    import json
    with open(os.path.join(s.dir, "session.json"), encoding="utf-8") as f:
        r = live_clipper.LiveSession.restore(json.load(f))
    assert r.state == "paused" and r.want_recording is False
