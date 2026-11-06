"""Client-supplied ids that become directory names.

Thumbnail session ids, SaaSShorts retry ids and thumbnail URLs are joined onto
OUTPUT_DIR. A "../<other>" value must never plant ownership of, attach, or
later delete another tenant's folder, and a job whose in-memory record is gone
must still be checked against its on-disk owner.
"""
import asyncio
import os
import uuid

import httpx
import pytest

app_module = pytest.importorskip("app")

OWNER = "11111111-1111-4111-8111-111111111111"
ATTACKER = "22222222-2222-4222-8222-222222222222"


class _User:
    def __init__(self, uid):
        self.id = uid
        self.plan = "pro"


def _call(method, path, **kwargs):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await getattr(client, method)(path, **kwargs)
    return asyncio.run(_do())


def _auth(uid):
    return {"Authorization": f"Bearer {uid}"}


@pytest.fixture()
def cloud(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "thumbnails").mkdir(parents=True)
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(out / "thumbnails"))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "saas_jobs", {})
    monkeypatch.setattr(app_module, "thumbnail_sessions", {})

    async def fake_user(request):
        auth = request.headers.get("authorization", "")
        return _User(auth.split()[-1]) if auth else None
    monkeypatch.setattr(app_module, "_user_from_request", fake_user)

    async def fake_gemini(request):
        return "test-key"
    monkeypatch.setattr(app_module, "resolve_gemini", fake_gemini)

    async def no_project(job_id):
        return None, None
    monkeypatch.setattr(app_module, "_media_project", no_project)
    return out


def _victim_job(out):
    job_id = str(uuid.uuid4())
    job_dir = out / job_id
    job_dir.mkdir()
    (job_dir / ".owner").write_text(OWNER)
    return job_id, job_dir


class TestOwnerMarker:
    def test_refuses_a_directory_outside_the_output_dirs(self, cloud, tmp_path):
        app_module._write_owner_marker(str(tmp_path / "elsewhere"), ATTACKER)
        assert not (tmp_path / "elsewhere" / ".owner").exists()

    def test_refuses_a_traversal_out_of_thumbnails(self, cloud):
        job_id, job_dir = _victim_job(cloud)
        app_module._write_owner_marker(
            os.path.join(app_module.THUMBNAILS_DIR, f"../{job_id}"), ATTACKER)
        assert (job_dir / ".owner").read_text() == OWNER

    def test_never_changes_an_existing_owner(self, cloud):
        job_id, job_dir = _victim_job(cloud)
        app_module._write_owner_marker(str(job_dir), ATTACKER)
        assert (job_dir / ".owner").read_text() == OWNER

    def test_still_writes_a_fresh_directory(self, cloud):
        d = cloud / f"saas_{uuid.uuid4()}"
        app_module._write_owner_marker(str(d), OWNER)
        assert (d / ".owner").read_text() == OWNER


class TestThumbnailSessionIds:
    def test_manual_title_with_a_traversal_id_gets_a_fresh_session(self, cloud):
        job_id, job_dir = _victim_job(cloud)
        resp = _call("post", "/api/thumbnail/titles", headers=_auth(ATTACKER),
                     json={"title": "x", "session_id": f"../{job_id}"})
        assert resp.status_code == 200
        new_id = resp.json()["session_id"]
        assert app_module._is_uuid_id(new_id) and new_id != f"../{job_id}"
        assert (job_dir / ".owner").read_text() == OWNER

    def test_manual_title_with_a_client_uuid_still_works(self, cloud):
        sid = str(uuid.uuid4())
        resp = _call("post", "/api/thumbnail/titles", headers=_auth(OWNER),
                     json={"title": "x", "session_id": sid})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid
        assert (cloud / "thumbnails" / sid / ".owner").read_text() == OWNER

    def test_someone_elses_session_on_disk_is_not_found(self, cloud):
        sid = str(uuid.uuid4())
        (cloud / "thumbnails" / sid).mkdir()
        (cloud / "thumbnails" / sid / ".owner").write_text(OWNER)
        resp = _call("post", "/api/thumbnail/titles", headers=_auth(ATTACKER),
                     json={"title": "x", "session_id": sid})
        assert resp.status_code == 404
        assert sid not in app_module.thumbnail_sessions


class TestThumbnailPublishPath:
    def _session(self, cloud, monkeypatch):
        sid = str(uuid.uuid4())
        video = cloud / "v.mp4"
        video.write_bytes(b"v")
        app_module.thumbnail_sessions[sid] = {"user_id": OWNER, "video_path": str(video)}

        async def fake_upload_post(request, api_key):
            return "k", "os_owner"
        monkeypatch.setattr(app_module, "resolve_upload_post", fake_upload_post)

        async def no_rate(request):
            return None
        monkeypatch.setattr(app_module, "_check_social_post_rate", no_rate)
        return sid

    def _publish(self, sid, url):
        return _call("post", "/api/thumbnail/publish", headers=_auth(OWNER), data={
            "session_id": sid, "title": "t", "description": "d", "thumbnail_url": url})

    def test_another_folders_image_is_refused(self, cloud, monkeypatch):
        sid = self._session(cloud, monkeypatch)
        other, other_dir = _victim_job(cloud)
        (other_dir / "x.jpg").write_bytes(b"img")
        for url in (f"/thumbnails/../{other}/x.jpg", f"/thumbnails/{sid}/../../{other}/x.jpg",
                    f"/thumbnails/{other}/x.jpg"):
            assert self._publish(sid, url).status_code == 400, url

    def test_a_non_image_in_the_session_is_refused(self, cloud, monkeypatch):
        sid = self._session(cloud, monkeypatch)
        (cloud / "thumbnails" / sid).mkdir()
        (cloud / "thumbnails" / sid / ".owner").write_text(OWNER)
        assert self._publish(sid, f"/thumbnails/{sid}/.owner").status_code == 400


class TestSaasRetry:
    def test_a_non_uuid_retry_id_is_rejected(self, cloud, monkeypatch):
        async def entitled(request):
            return None
        monkeypatch.setattr(app_module, "require_managed_entitlement", entitled)
        resp = _call("post", "/api/saasshorts/generate",
                     headers={**_auth(ATTACKER), "X-Fal-Key": "f", "X-ElevenLabs-Key": "e"},
                     json={"script": {"hook": "h"}, "retry_job_id": "x/../thumbnails"})
        assert resp.status_code == 400
        assert not (cloud / "thumbnails" / ".owner").exists()
        assert not app_module.saas_jobs

    def test_purge_never_deletes_a_shared_folder(self, cloud):
        (cloud / "thumbnails" / "keep.jpg").write_bytes(b"k")
        app_module.saas_jobs["evil"] = {"user_id": ATTACKER,
                                        "output_dir": str(cloud / "thumbnails")}
        app_module._purge_local_jobs_for_user(ATTACKER)
        assert (cloud / "thumbnails" / "keep.jpg").exists()


class TestDiskOwnerFallback:
    def test_download_all_checks_the_disk_owner(self, cloud):
        job_id, job_dir = _victim_job(cloud)
        (job_dir / "x_metadata.json").write_text('{"shorts": []}')
        for headers in (_auth(ATTACKER), {}):
            resp = _call("get", f"/api/jobs/{job_id}/download-all", headers=headers)
            assert resp.status_code == 404 and resp.json()["detail"] == "Not found"

    def test_download_all_owner_is_not_refused(self, cloud):
        job_id, job_dir = _victim_job(cloud)
        (job_dir / "x_metadata.json").write_text('{"shorts": []}')
        resp = _call("get", f"/api/jobs/{job_id}/download-all", headers=_auth(OWNER))
        # Past the owner check: this fixture simply has no clip files.
        assert resp.json()["detail"] == "No clip files found for this job"

    def test_source_upload_without_job_folder_checks_the_owner(self, cloud, monkeypatch):
        job_id = str(uuid.uuid4())
        up = os.path.join(app_module.UPLOAD_DIR, f"{job_id}_src.mp4")
        with open(up, "wb") as f:
            f.write(b"src")

        async def project(jid):
            return (OWNER, None) if jid == job_id else (None, None)
        monkeypatch.setattr(app_module, "_media_project", project)
        monkeypatch.setattr(app_module, "_job_record", lambda jid: None)
        assert _call("get", f"/api/source/{job_id}", headers=_auth(ATTACKER)).status_code == 404
        assert _call("get", f"/api/source/{job_id}").status_code == 404
        assert _call("get", f"/api/source/{job_id}", headers=_auth(OWNER)).status_code == 200
