"""An archived clip is fetched from storage, not through this host.

/videos is one process that also runs the queue; a finished clip is in R2
behind an edge cache. So the owner's browser is sent there, and only there:
for exactly the file storage holds, never a free job's clean twin, never in
exchange for a signed link, and never for a client that could not follow.
"""
import asyncio

import httpx
import pytest

app_module = pytest.importorskip("app")
import media_auth  # noqa: E402
from cloud import media_redirect  # noqa: E402

SECRET = "test-media-secret"
OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
JOB = "33333333-3333-4333-8333-333333333333"
BASE = "https://cdn.example.test"
CLIP_NAME = "video_clip_1.mp4"
MARKED_NAME = "wm_video_clip_1.mp4"
CLIP = f"/videos/{JOB}/{CLIP_NAME}"
BODY = b"clip bytes"
PLAYER = {"sec-fetch-mode": "no-cors", "sec-fetch-dest": "video"}


class _User:
    def __init__(self, uid):
        self.id = uid
        self.plan = "pro"


def _request(method, path, headers=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, headers=headers or {})
    return asyncio.run(_do())


def _get(path, headers=None):
    return _request("GET", path, headers)


def _key(name):
    return f"users/{OWNER}/{JOB}/{name}"


@pytest.fixture()
def cloud(tmp_path, monkeypatch):
    out = tmp_path / "output"
    job_dir = out / JOB
    job_dir.mkdir(parents=True)
    (job_dir / CLIP_NAME).write_bytes(BODY)
    (job_dir / ".owner").write_text(OWNER)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    for route in app_module.app.routes:
        if getattr(route, "path", None) == "/videos":
            monkeypatch.setattr(route.app, "directory", str(out))
            monkeypatch.setattr(route.app, "all_directories", [str(out)])
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_media_secret", lambda: SECRET)

    async def fake_user(request):
        auth = request.headers.get("authorization", "")
        return _User(auth.split()[-1]) if auth else None
    monkeypatch.setattr(app_module, "_user_from_request", fake_user)

    async def fake_project(job_id):
        return OWNER, None
    monkeypatch.setattr(app_module, "_media_project", fake_project)

    restores = []

    async def fake_restore(job_id):
        restores.append(job_id)
        return False
    monkeypatch.setattr(app_module, "_restore_for_public_path", fake_restore)

    for name, value in (("R2_ENDPOINT", "https://r2.example.test"), ("R2_BUCKET", "b"),
                        ("R2_ACCESS_KEY_ID", "k"), ("R2_SECRET_ACCESS_KEY", "s"),
                        ("R2_PUBLIC_BASE", BASE)):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MEDIA_REDIRECT", raising=False)

    # What the history rows say storage holds for this job: {filename: size}.
    archived = {CLIP_NAME: len(BODY)}

    async def fake_served(owner, job_id):
        if (owner, job_id) != (OWNER, JOB):
            return {}
        return {name: (_key(name), size) for name, size in archived.items()}
    monkeypatch.setattr(media_redirect, "_served_files", fake_served)
    media_redirect.forget(JOB)
    yield {"archived": archived, "restores": restores, "dir": job_dir}
    media_redirect.forget(JOB)
    app_module.jobs.pop(JOB, None)


def _owner_url(path=CLIP):
    return f"{path}?mt={media_auth.mint_user_token(OWNER, SECRET)}"


class TestThePlayerGoesToStorage:
    def test_the_owners_player_is_sent_to_the_archived_copy(self, cloud):
        resp = _get(_owner_url(), PLAYER)
        assert resp.status_code == 307
        target = resp.headers["location"]
        assert target.startswith(f"{BASE}/{_key(CLIP_NAME)}?v={len(BODY)}-")
        # The answer depends on who asks and how: it must not be reused.
        assert resp.headers["cache-control"] == "no-store"

    def test_the_signed_links_of_the_dashboard_go_there_with_the_owners_cookie(self, cloud):
        """A fresh job is played through the signed links of its status."""
        cookie = f"{media_auth.MEDIA_COOKIE_NAME}={media_auth.mint_user_token(OWNER, SECRET)}"
        resp = _get(media_auth.sign_media_url(CLIP, SECRET), {**PLAYER, "cookie": cookie})
        assert resp.status_code == 307
        assert resp.headers["location"].startswith(f"{BASE}/{_key(CLIP_NAME)}?v=")

    def test_a_link_the_browser_follows_by_itself_goes_there_too(self, cloud):
        resp = _get(_owner_url(), {"sec-fetch-mode": "navigate", "sec-fetch-dest": "document"})
        assert resp.status_code == 307

    def test_a_purged_job_is_not_pulled_back_for_a_player(self, cloud):
        """The whole job used to be restored to this host to serve one file."""
        (cloud["dir"] / CLIP_NAME).unlink()
        resp = _get(_owner_url(), PLAYER)
        assert resp.status_code == 307
        assert resp.headers["location"] == f"{BASE}/{_key(CLIP_NAME)}?v={len(BODY)}"
        assert cloud["restores"] == []


class TestWhoIsStillServedHere:
    def test_an_api_client_gets_the_bytes(self, cloud):
        """No browser headers: curl, an agent, a server fetching a clip."""
        resp = _get(CLIP, {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 200 and resp.content == BODY

    def test_a_scripts_fetch_gets_the_bytes(self, cloud):
        """Its redirect would reach storage with no origin and be refused."""
        resp = _get(_owner_url(), {"sec-fetch-mode": "cors", "sec-fetch-dest": "empty"})
        assert resp.status_code == 200 and resp.content == BODY

    def test_a_head_is_answered_here(self, cloud):
        assert _request("HEAD", _owner_url(), PLAYER).status_code == 200

    def test_a_signed_link_by_itself_is_never_traded_for_a_storage_link(self, cloud):
        """The signature expires; the storage link would not."""
        resp = _get(media_auth.sign_media_url(CLIP, SECRET), PLAYER)
        assert resp.status_code == 200 and resp.content == BODY

    def test_a_signed_link_with_someone_elses_cookie_is_served_here(self, cloud):
        cookie = f"{media_auth.MEDIA_COOKIE_NAME}={media_auth.mint_user_token(OTHER, SECRET)}"
        resp = _get(media_auth.sign_media_url(CLIP, SECRET), {**PLAYER, "cookie": cookie})
        assert resp.status_code == 200 and resp.content == BODY

    def test_a_stranger_is_refused_before_anything_else(self, cloud):
        token = media_auth.mint_user_token(OTHER, SECRET)
        assert _get(f"{CLIP}?mt={token}", PLAYER).status_code == 403

    def test_a_file_storage_does_not_hold_is_served_here(self, cloud):
        cloud["archived"].clear()
        resp = _get(_owner_url(), PLAYER)
        assert resp.status_code == 200 and resp.content == BODY

    def test_a_local_file_newer_than_the_archive_is_served_here(self, cloud):
        """An edit that reuses the file name: storage still has the old one."""
        (cloud["dir"] / CLIP_NAME).write_bytes(BODY + b" edited")
        resp = _get(_owner_url(), PLAYER)
        assert resp.status_code == 200 and resp.content == BODY + b" edited"

    def test_nobody_is_sent_to_storage_while_the_job_is_being_archived_again(self, cloud):
        media_redirect.hold(JOB)
        assert _get(_owner_url(), PLAYER).status_code == 200
        media_redirect.forget(JOB)
        assert _get(_owner_url(), PLAYER).status_code == 307

    def test_the_switch_turns_it_off(self, cloud, monkeypatch):
        monkeypatch.setenv("MEDIA_REDIRECT", "0")
        assert _get(_owner_url(), PLAYER).status_code == 200

    def test_without_a_public_domain_nothing_changes(self, cloud, monkeypatch):
        """The signed S3 endpoint is not one a browser can be sent to."""
        monkeypatch.delenv("R2_PUBLIC_BASE")
        assert _get(_owner_url(), PLAYER).status_code == 200

    def test_self_host_is_untouched(self, cloud, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
        monkeypatch.setattr(app_module, "_media_secret", lambda: "")
        resp = _get(CLIP, PLAYER)
        assert resp.status_code == 200 and resp.content == BODY


class TestTheFreePlansCleanTwin:
    """Storage keeps the clean twin next to the marked copy. Only the marked
    one is a history row, so only the marked one is ever linked."""

    def _free_job(self, cloud, local=True):
        cloud["archived"].clear()
        cloud["archived"][MARKED_NAME] = len(BODY)
        if local:
            (cloud["dir"] / MARKED_NAME).write_bytes(BODY)
            (cloud["dir"] / ".marked").write_text("")
        else:
            for name in (CLIP_NAME, ".marked"):
                path = cloud["dir"] / name
                if path.exists():
                    path.unlink()

    def test_the_marked_copy_is_linked(self, cloud):
        self._free_job(cloud)
        resp = _get(_owner_url(f"/videos/{JOB}/{MARKED_NAME}"), PLAYER)
        assert resp.status_code == 307
        assert f"/{MARKED_NAME}?v=" in resp.headers["location"]

    def test_the_clean_twin_is_refused(self, cloud):
        self._free_job(cloud)
        assert _get(_owner_url(), PLAYER).status_code == 404

    def test_the_clean_twin_of_a_purged_job_is_not_linked_either(self, cloud):
        """No marker on disk to read: the guard lets the name through, and the
        rows are what keeps the twin where it is."""
        self._free_job(cloud, local=False)
        resp = _get(_owner_url(), PLAYER)
        assert resp.status_code == 404
        assert "location" not in resp.headers

    def test_the_link_endpoint_refuses_the_twin_as_well(self, cloud):
        self._free_job(cloud)
        resp = _get(f"/api/media-url?job_id={JOB}&filename={CLIP_NAME}",
                    {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 404
        self._free_job(cloud, local=False)
        resp = _get(f"/api/media-url?job_id={JOB}&filename={CLIP_NAME}",
                    {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 200 and resp.json() == {"url": None}


class TestTheLinkEndpoint:
    URL = f"/api/media-url?job_id={JOB}&filename={CLIP_NAME}"

    def test_the_owner_gets_the_storage_link(self, cloud):
        resp = _get(self.URL, {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 200
        assert resp.json()["url"].startswith(f"{BASE}/{_key(CLIP_NAME)}?v=")

    def test_null_when_only_this_host_has_the_file(self, cloud):
        cloud["archived"].clear()
        resp = _get(self.URL, {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 200 and resp.json() == {"url": None}

    def test_someone_else_learns_nothing(self, cloud):
        assert _get(self.URL, {"authorization": f"Bearer {OTHER}"}).status_code == 404

    def test_it_needs_a_session(self, cloud):
        assert _get(self.URL).status_code == 401

    def test_a_file_that_is_not_a_deliverable_is_not_found(self, cloud):
        resp = _get(f"/api/media-url?job_id={JOB}&filename=.owner",
                    {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code == 404

    def test_a_path_is_not_a_file_name(self, cloud):
        resp = _get(f"/api/media-url?job_id={JOB}&filename=../{JOB}/{CLIP_NAME}",
                    {"authorization": f"Bearer {OWNER}"})
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert resp.json() == {"url": None}


class TestTheRowsAreReadCarefully:
    def test_only_objects_of_that_owners_job_count(self, monkeypatch):
        rows = [(_key(CLIP_NAME), 10),
                (f"users/{OTHER}/{JOB}/{CLIP_NAME}", 10),
                (f"users/{OWNER}/{JOB}/deeper/{CLIP_NAME}", 10),
                (None, 10)]

        class _Result:
            def all(self):
                return rows

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, _query):
                return _Result()

        monkeypatch.setattr(media_redirect.database, "session", lambda: _Session())
        media_redirect.forget(JOB)
        files = asyncio.run(media_redirect._served_files(OWNER, JOB))
        media_redirect.forget(JOB)
        assert files == {CLIP_NAME: (_key(CLIP_NAME), 10)}

    def test_a_failed_lookup_serves_from_here(self, monkeypatch):
        for name, value in (("R2_ENDPOINT", "https://r2.example.test"), ("R2_BUCKET", "b"),
                            ("R2_ACCESS_KEY_ID", "k"), ("R2_SECRET_ACCESS_KEY", "s"),
                            ("R2_PUBLIC_BASE", BASE)):
            monkeypatch.setenv(name, value)

        async def broken(owner, job_id):
            raise RuntimeError("pool exhausted")
        monkeypatch.setattr(media_redirect, "_served_files", broken)
        assert asyncio.run(media_redirect.storage_url(OWNER, JOB, CLIP_NAME)) is None
