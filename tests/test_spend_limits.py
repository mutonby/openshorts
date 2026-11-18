"""Limits on the paths that spend managed resources (Gemini, download proxies,
the managed Upload-Post account, disk).

Each block pins one rule at the function or HTTP surface:

- the per-GB proxy's daily budget reaches the env of a NEW job;
- parallel /api/process submissions count against the per-plan job limit;
- a refused /api/process deletes the source it was handed (an agent upload
  goes back to its slot);
- Thumbnail Studio validates, probes, caps and reserves a link before anything
  downloads it, bills transcription by length, and bounds its text calls;
- social profiles are created only by the connect page, posting is capped per
  hour, and a free account posts a few times in total before the plans modal;
- agent upload slots, OAuth registrations and SaaSShorts inputs are bounded;
- the free AI-edit allowance is counted in the ledger, so it survives restarts.
"""
import asyncio
import io
import os
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import cloud
import cloud.social_profiles  # noqa: F401  (submodule app.py reaches as cloud.social_profiles)
from cloud import config

app_module = pytest.importorskip("app")


def _call(method, path, **kw):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await getattr(c, method)(path, **kw)
    return asyncio.run(_do())


def _req():
    return SimpleNamespace(state=SimpleNamespace(), client=None, headers={})


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "pending_uploads", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    return SimpleNamespace(out=out_root, up=up_root)


@pytest.fixture()
def cloud_user(monkeypatch):
    """Cloud mode with one signed-in, entitled user (plan editable)."""
    user = SimpleNamespace(id="u1", plan="free", email=None, entitled=True,
                           upload_post_profile=None)

    async def _user(request):
        return user

    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "managed_keys", SimpleNamespace(
        has_active_entitlement=lambda u: u is not None and u.entitled,
        gemini_key=lambda: "managed-gemini", upload_post_key=lambda: "managed-up"))
    monkeypatch.setattr(app_module, "_user_from_request", _user)
    return user


# --------------------------------------------------------------------------- #
# Paid proxy budget on new jobs
# --------------------------------------------------------------------------- #
def _fake_reserve(paid_allowed):
    async def _reserve(request, url, input_path, job_id, max_minutes=None, **_kw):
        request.state.paid_allowed = paid_allowed
        return None, 2, None, None, None
    return _reserve


async def _no_gate(url, force_low):
    return None


async def _allow(url):
    return None


@pytest.mark.parametrize("paid_allowed, has_proxy", [(False, False), (True, True)])
def test_new_job_drops_the_paid_proxy_once_the_budget_is_spent(dirs, monkeypatch,
                                                               paid_allowed, has_proxy):
    monkeypatch.setenv("PROXY_URL", "http://paid.invalid:1")
    monkeypatch.setattr(app_module, "reserve_process_minutes", _fake_reserve(paid_allowed))
    monkeypatch.setattr(app_module, "_quality_gate", _no_gate)
    monkeypatch.setattr(app_module, "_validate_source_url", _allow)
    resp = _call("post", "/api/process", headers={"X-Gemini-Key": "k"},
                 json={"url": "https://www.youtube.com/watch?v=ok", "acknowledged": True})
    assert resp.status_code == 200, resp.text
    env = app_module.jobs[resp.json()["job_id"]]["env"]
    assert ("PROXY_URL" in env) is has_proxy


def test_budget_check_is_off_in_self_host(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    assert asyncio.run(app_module._paid_proxy_budget_exceeded()) is False


# --------------------------------------------------------------------------- #
# Per-plan job limit under parallel submissions
# --------------------------------------------------------------------------- #
def test_parallel_submissions_count_against_the_job_limit(cloud_user, monkeypatch):
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "_submissions", {})
    gate = asyncio.Event()

    class _M:
        @staticmethod
        async def get_balance(uid):
            await gate.wait()
            return {"remaining": 0, "plan": "free"}

    monkeypatch.setattr(app_module, "_metering", _M)

    async def run():
        first = asyncio.create_task(app_module.reserve_process_minutes(
            _req(), "https://youtu.be/a", None, "job-1"))
        await asyncio.sleep(0)  # the first submission is now inside its balance check
        with pytest.raises(HTTPException) as e:
            await app_module.reserve_process_minutes(_req(), "https://youtu.be/b", None, "job-2")
        assert e.value.status_code == 429  # free plan: one job at a time
        gate.set()
        with pytest.raises(HTTPException) as e:
            await first
        assert e.value.status_code == 402
        # A refused submission frees its place right away.
        assert app_module._submissions_in_flight("u1") == 0

    asyncio.run(run())


def test_stale_submission_entries_expire(monkeypatch):
    monkeypatch.setattr(app_module, "_submissions", {})
    app_module._begin_submission("j", "u1")
    assert app_module._submissions_in_flight("u1") == 1
    owner, t = app_module._submissions["j"]
    app_module._submissions["j"] = (owner, t - app_module._SUBMISSION_MAX_AGE - 1)
    assert app_module._submissions_in_flight("u1") == 0


# --------------------------------------------------------------------------- #
# A refused /api/process leaves nothing on disk
# --------------------------------------------------------------------------- #
async def _refuse(*a, **k):
    raise HTTPException(status_code=402, detail="quota")


def test_refused_file_upload_is_deleted(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "reserve_process_minutes", _refuse)
    resp = _call("post", "/api/process", headers={"X-Gemini-Key": "k"},
                 files={"file": ("big.mp4", io.BytesIO(b"x" * 4096), "video/mp4")},
                 data={"acknowledged": "true"})
    assert resp.status_code == 402
    assert os.listdir(dirs.up) == []
    assert os.listdir(dirs.out) == []


def test_refused_agent_upload_goes_back_to_its_slot(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "reserve_process_minutes", _refuse)
    path = dirs.up / "pending_abc_v.mp4"
    path.write_bytes(b"video")
    slot = {"user_id": None, "filename": "v.mp4", "path": str(path),
            "created": 1e18, "bytes": 5, "complete": True}
    app_module.pending_uploads["abc"] = slot
    resp = _call("post", "/api/process", headers={"X-Gemini-Key": "k"},
                 json={"upload_id": "abc", "acknowledged": True})
    assert resp.status_code == 402
    assert app_module.pending_uploads["abc"] is slot
    assert os.listdir(dirs.up) == ["pending_abc_v.mp4"]
    assert os.listdir(dirs.out) == []


# --------------------------------------------------------------------------- #
# Thumbnail Studio: links are checked and reserved before any download
# --------------------------------------------------------------------------- #
class TestThumbnailBilling:
    @pytest.mark.parametrize("minutes, cost", [(0, 2), (4, 2), (20, 2), (21, 3), (60, 6), (180, 18)])
    def test_transcription_is_billed_by_length(self, minutes, cost):
        assert app_module.thumbnail_transcribe_minutes(minutes) == cost

    def test_length_cap(self):
        app_module._check_thumbnail_length(config.THUMBNAIL_MAX_SOURCE_MINUTES)
        with pytest.raises(HTTPException) as e:
            app_module._check_thumbnail_length(config.THUMBNAIL_MAX_SOURCE_MINUTES + 30)
        assert e.value.status_code == 413


@pytest.fixture()
def thumb_cloud(cloud_user, dirs, monkeypatch):
    """Cloud Thumbnail Studio with a fake ledger, probe and downloader."""
    log = SimpleNamespace(order=[], reserved=[], downloads=[], probe_minutes=60.0,
                          remaining=20.0, budget_spent=False)

    class _M:
        @staticmethod
        async def get_balance(uid):
            return {"remaining": log.remaining, "plan": "free"}

        @staticmethod
        def probe_url_minutes(url, allow_paid=True):
            log.order.append(("probe", allow_paid))
            return log.probe_minutes

        @staticmethod
        async def run_probe(url, allow_paid=True):
            return _M.probe_url_minutes(url, allow_paid=allow_paid)

        @staticmethod
        async def commit_reservation(rid):
            log.order.append("commit")

        @staticmethod
        async def release_reservation(rid):
            log.order.append("release")

    async def _reserve(request, minutes, job_id, job_type):
        log.order.append("reserve")
        log.reserved.append((minutes, job_type))
        return "res-1"

    async def _download(url, session_id, paid_allowed=True, cap_minutes=None):
        log.order.append("download")
        log.downloads.append((paid_allowed, cap_minutes))
        raise RuntimeError("stop here")  # the test is about what came before

    async def _budget():
        return log.budget_spent

    monkeypatch.setattr(app_module, "_metering", _M)
    monkeypatch.setattr(app_module, "reserve_managed_action", _reserve)
    monkeypatch.setattr(app_module, "_download_thumbnail_source", _download)
    monkeypatch.setattr(app_module, "_paid_proxy_budget_exceeded", _budget)
    monkeypatch.setattr(app_module, "_validate_source_url", _allow)
    monkeypatch.setattr(app_module, "_probe_times", {})
    monkeypatch.setattr(app_module, "thumbnail_sessions", {})
    return log


URL = "https://www.youtube.com/watch?v=abc"


def test_thumbnail_link_is_probed_and_reserved_before_download(thumb_cloud):
    thumb_cloud.budget_spent = True
    resp = _call("post", "/api/thumbnail/upload", data={"url": URL})
    assert resp.status_code == 200, resp.text
    asyncio.run(asyncio.sleep(0))
    assert thumb_cloud.order[:2] == [("probe", False), "reserve"]
    assert thumb_cloud.reserved == [(6, "thumbnail_transcribe")]  # 60 min source


def test_thumbnail_analyze_link_reserves_before_download(thumb_cloud):
    resp = _call("post", "/api/thumbnail/analyze", data={"url": URL})
    assert resp.status_code == 500  # the fake download stops it
    assert thumb_cloud.order == [("probe", True), "reserve", "download", "release"]
    # analysis + the transcription it runs itself
    assert thumb_cloud.reserved == [(config.MANAGED_ANALYSIS_MINUTES + 6, "thumbnail_analyze")]
    assert thumb_cloud.downloads == [(True, 60.0)]


def test_thumbnail_link_too_long_is_refused_without_download(thumb_cloud):
    thumb_cloud.probe_minutes = config.THUMBNAIL_MAX_SOURCE_MINUTES + 60
    resp = _call("post", "/api/thumbnail/upload", data={"url": URL})
    assert resp.status_code == 413
    assert "reserve" not in thumb_cloud.order and "download" not in thumb_cloud.order


def test_thumbnail_link_with_no_minutes_is_refused_before_the_probe(thumb_cloud):
    thumb_cloud.remaining = 0
    resp = _call("post", "/api/thumbnail/analyze", data={"url": URL})
    assert resp.status_code == 402
    assert thumb_cloud.order == []


def test_thumbnail_link_is_validated_in_self_host_too(dirs, monkeypatch):
    monkeypatch.setattr(app_module, "thumbnail_sessions", {})
    resp = _call("post", "/api/thumbnail/upload", data={"url": "http://127.0.0.1:8000/health"})
    assert resp.status_code == 400
    assert app_module.thumbnail_sessions == {}


# --------------------------------------------------------------------------- #
# Thumbnail Studio: text calls
# --------------------------------------------------------------------------- #
@pytest.fixture()
def titles(monkeypatch):
    calls = []

    def _refine(api_key, context, message, conversation):
        calls.append(len(conversation))
        return {"titles": ["t"], "thumbnail_texts": []}

    monkeypatch.setattr(app_module, "refine_titles", _refine)
    monkeypatch.setattr(app_module, "thumbnail_sessions", {
        "s1": {"user_id": None, "context": "c", "titles": [], "language": "en",
               "conversation": []}})
    monkeypatch.setattr(app_module, "_thumbnail_text_times", {})
    return calls


def _titles(body):
    return _call("post", "/api/thumbnail/titles", json=body, headers={"X-Gemini-Key": "k"})


def test_titles_input_lengths_are_capped(titles):
    r = _titles({"session_id": "s1", "message": "x" * (config.THUMBNAIL_MESSAGE_MAX_CHARS + 1)})
    assert r.status_code == 413
    r = _titles({"title": "x" * (config.THUMBNAIL_TITLE_MAX_CHARS + 1)})
    assert r.status_code == 413
    assert titles == []


def test_titles_conversation_sent_to_the_model_is_bounded(titles):
    for _ in range(30):
        assert _titles({"session_id": "s1", "message": "shorter"}).status_code == 200
    assert max(titles) <= config.THUMBNAIL_CONVERSATION_MAX_TURNS
    assert len(app_module.thumbnail_sessions["s1"]["conversation"]) <= \
        config.THUMBNAIL_CONVERSATION_MAX_TURNS


def test_titles_refinements_per_session_are_capped(titles, monkeypatch):
    app_module.thumbnail_sessions["s1"]["refines"] = config.THUMBNAIL_REFINES_PER_SESSION
    assert _titles({"session_id": "s1", "message": "more"}).status_code == 429
    assert titles == []


def test_free_user_at_zero_minutes_needs_a_paid_session(titles, cloud_user, monkeypatch):
    class _M:
        @staticmethod
        async def get_balance(uid):
            return {"remaining": 0, "plan": "free"}
    monkeypatch.setattr(app_module, "_metering", _M)
    # A manual-title session never paid for anything.
    assert _titles({"session_id": "s1", "message": "more"}).status_code == 402
    # One that went through a metered analysis keeps working.
    app_module.thumbnail_sessions["s1"]["metered"] = True
    assert _titles({"session_id": "s1", "message": "more"}).status_code == 200


def test_text_calls_are_capped_per_hour(titles, cloud_user, monkeypatch):
    app_module.thumbnail_sessions["s1"]["metered"] = True
    monkeypatch.setattr(config, "THUMBNAIL_TEXT_CALLS_PER_HOUR", 3)
    codes = [_titles({"session_id": "s1", "message": "m"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_describe_title_is_capped(monkeypatch):
    monkeypatch.setattr(app_module, "thumbnail_sessions", {})
    r = _call("post", "/api/thumbnail/describe", headers={"X-Gemini-Key": "k"},
              json={"session_id": "s1", "title": "x" * (config.THUMBNAIL_TITLE_MAX_CHARS + 1)})
    assert r.status_code == 413


def test_description_prompt_is_bounded(monkeypatch):
    import thumbnail
    sent = {}

    class _Models:
        def generate_content(self, model, contents, **kw):
            sent["prompt"] = contents if isinstance(contents, str) else str(contents)
            return SimpleNamespace(text='{"description": "d"}')

    monkeypatch.setattr(thumbnail.genai, "Client", lambda api_key: SimpleNamespace(models=_Models()))
    segs = [{"start": i, "end": i + 1, "text": "word " * 50} for i in range(2000)]
    thumbnail.generate_youtube_description("k", "t", segs, "en", 2000, max_chars=5000)
    assert len(sent["prompt"]) < 5000 + 5000  # transcript capped; the rest is the template


# --------------------------------------------------------------------------- #
# Social: profiles and posting
# --------------------------------------------------------------------------- #
def test_reading_social_state_never_creates_a_vendor_profile(cloud_user, monkeypatch):
    async def _boom(user):
        raise AssertionError("profile created on a read")
    monkeypatch.setattr(app_module, "cloud", cloud)
    monkeypatch.setattr(cloud.social_profiles, "ensure_profile", _boom)

    key, profile = asyncio.run(app_module.resolve_upload_post(_req()))
    assert (key, profile) == ("managed-up", None)

    cloud_user.id = __import__("uuid").UUID(int=1)
    resp = _call("get", "/api/social/user")
    assert resp.status_code == 200
    body = resp.json()
    assert body["profiles"][0]["connected"] == []
    assert body["profiles"][0]["username"].startswith("os_")


def test_stored_profile_is_still_used(cloud_user):
    cloud_user.upload_post_profile = "os_mine"
    key, profile = asyncio.run(app_module.resolve_upload_post(_req()))
    assert profile == "os_mine"


def test_thumbnail_publish_never_falls_back_to_the_client_profile(cloud_user, monkeypatch):
    monkeypatch.setattr(app_module, "thumbnail_sessions", {"s1": {"user_id": None}})
    resp = _call("post", "/api/thumbnail/publish", data={
        "session_id": "s1", "title": "t", "description": "d",
        "thumbnail_url": "/x.png", "user_id": "os_someone_else"})
    assert resp.status_code == 409


class TestSocialPostRate:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        monkeypatch.setattr(app_module, "_social_post_times", {})

    def test_every_plan_has_an_hourly_cap(self, cloud_user, monkeypatch):
        cloud_user.plan = "starter"
        monkeypatch.setattr(config, "SOCIAL_POSTS_PER_HOUR", 4)
        for _ in range(4):
            asyncio.run(app_module._check_social_post_rate(_req()))
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_module._check_social_post_rate(_req()))
        assert e.value.status_code == 429

    def test_self_host_is_not_limited(self, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
        for _ in range(200):
            asyncio.run(app_module._check_social_post_rate(_req()))


class _Ledger:
    """metering's free-action rows, in memory."""
    def __init__(self):
        self.rows = {}
        self.recorded = 0

    async def free_actions_total(self, uid, job_type):
        return sum(1 for r in self.rows.values() if r == (uid, job_type))

    async def record_free_action(self, uid, job_id, job_type):
        self.recorded += 1
        rid = f"r{self.recorded}"
        self.rows[rid] = (uid, job_type)
        return rid

    async def undo_free_action(self, rid):
        self.rows.pop(rid, None)


class TestFreePostsArePaid:
    """Posting is part of the paid plans: a free account tries it a few times,
    then every post opens the plans modal (402 post_requires_plan)."""

    @pytest.fixture(autouse=True)
    def ledger(self, monkeypatch):
        led = _Ledger()
        monkeypatch.setattr(app_module, "_metering", led)
        monkeypatch.setattr(app_module, "_free_post_locks", {})
        monkeypatch.setattr(config, "FREE_SOCIAL_POSTS", 3)
        return led

    def test_a_free_account_posts_three_times_then_meets_the_plans(self, cloud_user, ledger):
        left = [asyncio.run(app_module._claim_free_post(_req(), "j"))[1] for _ in range(3)]
        assert left == [2, 1, 0]
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_module._claim_free_post(_req(), "j"))
        assert e.value.status_code == 402
        assert e.value.detail["error"] == "post_requires_plan"
        assert "upgrade_url" in e.value.detail

    def test_the_count_lives_in_the_ledger_so_a_restart_keeps_it(self, cloud_user, ledger, monkeypatch):
        for _ in range(3):
            asyncio.run(app_module._claim_free_post(_req(), "j"))
        monkeypatch.setattr(app_module, "_free_post_locks", {})      # a restart
        with pytest.raises(HTTPException):
            asyncio.run(app_module._claim_free_post(_req(), "j"))

    def test_paid_plans_and_self_host_are_not_counted(self, cloud_user, ledger, monkeypatch):
        cloud_user.plan = "creator"
        assert asyncio.run(app_module._claim_free_post(_req(), "j")) == (None, None)
        monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
        assert asyncio.run(app_module._claim_free_post(_req(), "j")) == (None, None)
        assert ledger.rows == {}

    def test_a_post_that_fails_gives_its_try_back(self, cloud_user, ledger, monkeypatch, tmp_path):
        monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
        (tmp_path / "j1").mkdir()
        (tmp_path / "j1" / "c.mp4").write_bytes(b"v")
        cloud_user.upload_post_profile = "os_mine"
        monkeypatch.setattr(app_module, "_social_post_times", {})
        monkeypatch.setattr(app_module, "jobs", {"j1": {
            "user_id": "u1", "result": {"clips": [{"video_url": "/videos/j1/c.mp4"}]}}})

        async def _noop(*a, **k):
            return None

        async def _none_stale(*a, **k):
            return []
        monkeypatch.setattr(app_module, "_ensure_job_files", _noop)
        monkeypatch.setattr(app_module, "_assert_job_owner", _noop)
        monkeypatch.setattr(app_module, "_platforms_needing_reconnect", _none_stale)

        def _vendor_down(*a, **k):
            raise RuntimeError("The write operation timed out")
        monkeypatch.setattr(app_module, "_post_video_blocking", _vendor_down)
        body = {"job_id": "j1", "clip_index": 0, "platforms": ["tiktok"]}
        assert _call("post", "/api/social/post", json=body).status_code == 500
        assert ledger.recorded == 1 and ledger.rows == {}

        monkeypatch.setattr(app_module, "_post_video_blocking",
                            lambda *a, **k: SimpleNamespace(status_code=200,
                                                            json=lambda: {"request_id": "r"}))
        r = _call("post", "/api/social/post", json=body)
        assert r.status_code == 200 and r.json()["free_posts_left"] == 2


# --------------------------------------------------------------------------- #
# Agent upload slots, OAuth registration, SaaSShorts inputs
# --------------------------------------------------------------------------- #
def test_open_upload_slots_per_user_are_capped(cloud_user, dirs, monkeypatch):
    monkeypatch.setattr(config, "MAX_PENDING_UPLOADS_PER_USER", 2)

    async def _owner(request):
        return cloud_user.id
    monkeypatch.setattr(app_module, "_owner_id", _owner)
    codes = [_call("post", "/api/uploads", json={}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    cloud_user.id = "u2"
    assert _call("post", "/api/uploads", json={}).status_code == 200


def test_oauth_registration_is_rate_limited_per_ip(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cloud import mcp_oauth
    monkeypatch.setattr(mcp_oauth, "_register_times", {})
    monkeypatch.setattr(config, "OAUTH_REGISTRATIONS_PER_HOUR", 2)
    app = FastAPI()
    app.include_router(mcp_oauth.router)
    c = TestClient(app)
    bad = {"redirect_uris": ["http://evil.com/cb"]}  # refused before any DB write
    codes = [c.post("/oauth/register", json=bad).status_code for _ in range(3)]
    assert codes == [400, 400, 429]
    assert c.post("/oauth/register", json=bad).json()["error"] == "slow_down"


@pytest.mark.parametrize("body, field", [
    ({"description": "x", "num_scripts": 500}, "num_scripts"),
    ({"description": "x", "num_scripts": 0}, "num_scripts"),
    ({"description": "x" * (config.SAAS_DESCRIPTION_MAX_CHARS + 1)}, "description"),
])
def test_saasshorts_inputs_are_bounded(body, field):
    r = _call("post", "/api/saasshorts/analyze", json=body, headers={"X-Gemini-Key": "k"})
    assert r.status_code in (400, 413)
    assert field in r.json()["detail"]


# --------------------------------------------------------------------------- #
# Free AI edits survive a restart
# --------------------------------------------------------------------------- #
def test_free_ai_edit_allowance_is_read_from_the_ledger(monkeypatch):
    rows = []

    class _M:
        @staticmethod
        async def free_actions_today(uid, job_type):
            return sum(1 for r in rows if r == (uid, job_type))

        @staticmethod
        async def record_free_action(uid, job_id, job_type):
            rows.append((uid, job_type))

    async def _owner(request):
        return "u1"

    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_metering", _M)
    monkeypatch.setattr(app_module, "_owner_id", _owner)
    for _ in range(config.AI_EDITS_FREE_PER_DAY):
        assert asyncio.run(app_module._ai_edit_minutes(None, "j")) == 0
    monkeypatch.setattr(app_module, "_ai_edit_counts", {})  # a restart
    assert asyncio.run(app_module._ai_edit_minutes(None, "j")) == config.MANAGED_ANALYSIS_MINUTES
