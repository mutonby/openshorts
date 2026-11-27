"""/videos must hand out deliverables, not the whole working directory.

The mount was a plain StaticFiles over OUTPUT_DIR, so anyone holding a job id
(it travels in every clip URL) could also fetch `.owner` (a user uuid),
`.resume.json` (the customer's own webhook_secret), `.transcript_checkpoint.json`
and `<title>_metadata.json`, which carries the full transcript of their video.
All four were verified served, unauthenticated, against production on
7-sep-2026. media_auth shipped the allowlist in 96f0139 but nothing imported it.

This is the path half only: the clips and the untouched source video in the
same directory are still public to whoever has the job id.
"""
import pytest

media_auth = pytest.importorskip("media_auth")
restoring_static = pytest.importorskip("restoring_static")


class TestTheGuardIsWiredIn:
    def test_the_videos_mount_carries_the_allowlist(self):
        app = pytest.importorskip("app")
        mount = next(r for r in app.app.routes
                     if getattr(r, "name", None) == "videos")
        # app._media_guard wraps the allowlist (and adds the free-plan rule
        # that refuses a marked job's clean twins, tests/test_watermark_delivery).
        assert mount.app.guard is app._media_guard, \
            "/videos is serving the working directory with no allowlist"
        assert not app._media_guard("job/.resume.json")
        assert not app._media_guard("job/x_metadata.json")
        assert app._media_guard("job/x_clip_1.mp4")


class TestWhatMayLeave:
    @pytest.mark.parametrize("rel", [
        "job/clip_1.mp4", "job/subtitled_1_clip_1.mp4", "job/a.webm",
        "job/autosubs_1.ass", "job/subs.srt", "job/thumb.jpg", "job/clips.zip",
    ])
    def test_deliverables_pass(self, rel):
        assert media_auth.is_servable(rel)

    @pytest.mark.parametrize("rel", [
        "job/.owner",
        "job/.resume.json",
        "job/.transcript_checkpoint.json",
        "job/My_Video_metadata.json",
        "job/clip_1.mp4.layout.json",
        "job/.instance",
        "../app.py",
        "job/../../app.py",
        "/etc/passwd",
        "",
    ])
    def test_everything_else_is_refused(self, rel):
        assert not media_auth.is_servable(rel)


def test_thumbnails_mount_serves_images_only(tmp_path, monkeypatch):
    """/thumbnails was a plain StaticFiles mount: any file a session left in
    its directory was public. It now shares the /videos allowlist.

    Served from a temporary directory: the real output/thumbnails can belong
    to another user (a Docker volume), and a test must not write there."""
    import os
    import uuid
    app_module = pytest.importorskip("app")
    from fastapi.testclient import TestClient

    mount = next(r for r in app_module.app.routes if getattr(r, "path", "") == "/thumbnails")
    monkeypatch.setattr(mount.app, "all_directories", [str(tmp_path)])
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(tmp_path))

    sid = str(uuid.uuid4())
    d = tmp_path / sid
    d.mkdir()
    for name, body in (("thumb.jpg", b"\xff\xd8jpeg"), ("session.json", b"{}"), (".secret", b"x")):
        (d / name).write_bytes(body)
    c = TestClient(app_module.app)
    assert c.get(f"/thumbnails/{sid}/thumb.jpg").status_code == 200
    assert c.get(f"/thumbnails/{sid}/session.json").status_code == 404
    assert c.get(f"/thumbnails/{sid}/.secret").status_code == 404
    assert os.path.isdir(d)
