"""A clip that never rendered must not be reported as delivered.

The metadata JSON is written BEFORE the clips render, and main.py's worker pool
swallows a per-clip exception (it only prints "❌ Clip N failed") and still
exits 0. app.py then rebuilt the job result from that metadata and named every
clip through _canonical_clip_file, which returns a name whether or not the file
exists. A job whose renders all failed was therefore reported as completed with
N clips: the minutes were committed, ClipsDelivered fired, archive_job skipped
every missing file and uploaded the metadata alone (returning before it writes
the Project row, so the job could not even be restored), and the dashboard
offered clips whose only existence was a title — "download all" then answered
404 "No clip files found for this job".

Reported as ticket #4711 (job 89d76bbc, 6-sep-2026): two titles, two viral
scores, blank players, and 404 on download. Measured over the R2 archive on
7-sep-2026: 195 of 1687 archived jobs held a metadata file and not one clip,
across 184 users.
"""
import os

import pytest

app = pytest.importorskip("app")


def _clip(title):
    return {"video_title_for_youtube_short": title, "start": 0.0, "end": 20.0}


def _write(job_dir, name, size=1024):
    path = os.path.join(job_dir, name)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    return path


class TestPhantomClips:
    """_clips_actually_rendered is the gate between 'promised' and 'delivered'."""

    def test_saved_edit_wins_when_another_clip_is_repaired(self, tmp_path, monkeypatch):
        job = "repair-existing-edit"
        original = "MKG_clip_1.mp4"
        edited = "edited_deadbeef_MKG_clip_1.mp4"
        _write(str(tmp_path), original)
        _write(str(tmp_path), edited)
        monkeypatch.setitem(app.jobs, job, {"ready_files": {0: original}})
        clip = {**_clip("saved"), "video_url": f"/videos/{job}/{edited}"}
        kept, missing = app._clips_actually_rendered(job, str(tmp_path), "MKG", [clip])
        assert missing == 0
        assert kept[0]["video_url"].endswith("/" + edited)

    def test_drops_clips_whose_file_was_never_written(self, tmp_path):
        job_dir = str(tmp_path)
        _write(job_dir, "MKG_clip_1.mp4")  # only clip 1 rendered
        kept, missing = app._clips_actually_rendered(
            "job-1", job_dir, "MKG", [_clip("one"), _clip("two")])

        assert [c["video_title_for_youtube_short"] for c in kept] == ["one"]
        assert missing == 1
        assert kept[0]["video_url"] == "/videos/job-1/MKG_clip_1.mp4"

    def test_a_job_that_rendered_nothing_keeps_nothing(self, tmp_path):
        """The ticket #4711 shape: metadata on disk, not a single mp4."""
        kept, missing = app._clips_actually_rendered(
            "89d76bbc", str(tmp_path), "89d76bbc_MKG ep 3 draft 4_6",
            [_clip("This Car Design Detail Is Perfection"),
             _clip("Why This Car Is In A League Of Its Own")])

        assert kept == []
        assert missing == 2

    def test_a_zero_byte_clip_is_not_a_clip(self, tmp_path):
        """ffmpeg can leave an empty file behind when it dies mid-write."""
        _write(str(tmp_path), "MKG_clip_1.mp4", size=0)
        kept, missing = app._clips_actually_rendered(
            "job-2", str(tmp_path), "MKG", [_clip("one")])

        assert kept == []
        assert missing == 1

    def test_a_fully_rendered_job_is_untouched(self, tmp_path):
        job_dir = str(tmp_path)
        _write(job_dir, "MKG_clip_1.mp4")
        _write(job_dir, "MKG_clip_2.mp4")
        kept, missing = app._clips_actually_rendered(
            "job-3", job_dir, "MKG", [_clip("one"), _clip("two")])

        assert len(kept) == 2
        assert missing == 0

    def test_the_derived_file_still_wins(self, tmp_path):
        """Captions/hooks are served from the derived file, as before the fix."""
        job_dir = str(tmp_path)
        _write(job_dir, "MKG_clip_1.mp4")
        _write(job_dir, "subtitled_2_hooked_1_MKG_clip_1.mp4")
        kept, _ = app._clips_actually_rendered(
            "job-4", job_dir, "MKG", [_clip("one")])

        assert kept[0]["video_url"] == (
            "/videos/job-4/subtitled_2_hooked_1_MKG_clip_1.mp4")

    def test_a_name_with_spaces_and_parens_resolves(self, tmp_path):
        """Uploads keep the user's own filename as the stem (app.py:2434)."""
        stem = "89d76bbc_MK Garage Season 1 episode 1 (Pilot)_2"
        _write(str(tmp_path), f"{stem}_clip_1.mp4")
        kept, missing = app._clips_actually_rendered(
            "89d76bbc", str(tmp_path), stem, [_clip("one")])

        assert missing == 0
        assert kept[0]["video_url"] == f"/videos/89d76bbc/{stem}_clip_1.mp4"

    def test_the_clip_ready_marker_wins_over_the_guessed_name(self, tmp_path):
        """main.py names the finished file; the convention is only a fallback."""
        job_dir = str(tmp_path)
        _write(job_dir, "dubbed_9_MKG_clip_1.mp4")  # a prefix no glob covers
        app.jobs["job-5"] = {"ready_files": {0: "dubbed_9_MKG_clip_1.mp4"}}
        try:
            kept, missing = app._clips_actually_rendered(
                "job-5", job_dir, "MKG", [_clip("one")])
        finally:
            app.jobs.pop("job-5", None)

        assert missing == 0
        assert kept[0]["video_url"] == "/videos/job-5/dubbed_9_MKG_clip_1.mp4"

    def test_a_marker_for_a_file_that_vanished_is_still_dropped(self, tmp_path):
        app.jobs["job-6"] = {"ready_files": {0: "gone_MKG_clip_1.mp4"}}
        try:
            kept, missing = app._clips_actually_rendered(
                "job-6", str(tmp_path), "MKG", [_clip("one")])
        finally:
            app.jobs.pop("job-6", None)

        assert kept == []
        assert missing == 1
