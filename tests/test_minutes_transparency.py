"""Minutes users can see and steer before a job runs.

- The cost line (/api/process/estimate + ``estimate_outcome``) says what a
  source will cost before Generate.
- A stretch picked in the dashboard (``source_start`` / ``source_end``) is
  billed alone and main.py cuts the source to it.
- Minutes are rounded down (``config.billable_minutes``), and editing an
  existing clip is free, bounded by an hourly cap instead.
- Posting follows the upload to its real outcome and refuses accounts whose
  login expired; YouTube titles are cut to 100 characters.
"""
import asyncio
import json
import os
import shutil
import subprocess

import httpx
import pytest
from fastapi import HTTPException

app_module = pytest.importorskip("app")
from cloud import config  # noqa: E402


class TestBillableMinutes:
    def test_rounds_down_with_a_floor_of_one(self):
        assert config.billable_minutes(4.6) == 4
        assert config.billable_minutes(20.0) == 20
        assert config.billable_minutes(0.3) == 1
        assert config.billable_minutes("lots") == 1

    def test_editor_actions_are_free(self):
        assert config.EDITOR_ACTION_MINUTES == 0


class TestPlanSourceRange:
    def test_no_range_or_the_whole_video_is_an_ordinary_job(self):
        assert app_module.plan_source_range(45, None) is None
        assert app_module.plan_source_range(45, (0, None)) is None
        assert app_module.plan_source_range(45, (None, 45 * 60)) is None
        # Past the end is clamped to it: still the whole video.
        assert app_module.plan_source_range(45, (0, 99999)) is None

    def test_a_stretch_is_kept_in_seconds(self):
        assert app_module.plan_source_range(45, (120, 420)) == {
            "start_seconds": 120.0, "end_seconds": 420.0}
        assert app_module.plan_source_range(45, ("600", None)) == {
            "start_seconds": 600.0, "end_seconds": 2700.0}

    @pytest.mark.parametrize("rng", [(10, 40), (500, 100), ("a", 100)])
    def test_too_short_backwards_or_garbage_is_a_400(self, rng):
        with pytest.raises(HTTPException) as e:
            app_module.plan_source_range(45, rng)
        assert e.value.status_code == 400


class TestEstimateOutcome:
    def test_covered_by_the_balance(self):
        out = app_module.estimate_outcome("starter", 57.4, 100, True)
        assert out["outcome"] == "whole" and out["cost_minutes"] == 57
        # 20.7 minutes cost 20: fits a 20-minute balance whole.
        assert app_module.estimate_outcome("free", 20.7, 20, True)["outcome"] == "whole"

    def test_free_first_video_is_clipped_whole(self):
        out = app_module.estimate_outcome("free", 45.5, 20, False)
        assert out["outcome"] == "first_video_free" and out["cost_minutes"] == 20

    def test_free_later_video_is_clipped_to_the_balance(self):
        out = app_module.estimate_outcome("free", 50, 20, True)
        assert out["outcome"] == "first_part" and out["processed_minutes"] == 20

    def test_paid_plan_past_the_balance_meets_the_wall(self):
        out = app_module.estimate_outcome("starter", 120, 30, True)
        assert out["outcome"] == "needs_more" and out["partial_minutes"] == 30


class TestApplySourceCut:
    def test_picked_stretch_is_cut_to_its_exact_length(self):
        env = {"SOURCE_START_SECONDS": "stale"}
        app_module._apply_source_cut(env, {
            "processed_minutes": 5, "total_minutes": 45,
            "start_seconds": 130.0, "end_seconds": 460.0, "chosen": True})
        assert env["MAX_SOURCE_MINUTES"] == "5.5000"
        assert env["SOURCE_START_SECONDS"] == "130.000"

    def test_quota_wall_cut_is_unchanged(self):
        env = {"SOURCE_START_SECONDS": "9"}
        app_module._apply_source_cut(env, {"processed_minutes": 20, "total_minutes": 45})
        assert env == {"MAX_SOURCE_MINUTES": "20"}

    def test_no_partial_clears_both(self):
        env = {"MAX_SOURCE_MINUTES": "3", "SOURCE_START_SECONDS": "9"}
        app_module._apply_source_cut(env, None)
        assert env == {}


# --------------------------------------------------------------------------- #
# Endpoint: the picked stretch reaches the reservation and the job
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())

    async def _probe(url):
        return {"max_height": 1080, "duration": 45 * 60}

    async def _allow_url(url):
        return None

    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
    monkeypatch.setattr(app_module, "_validate_source_url", _allow_url)
    return out_root, up_root


def test_source_range_reaches_the_reservation_and_the_job(dirs, monkeypatch):
    out_root, _ = dirs
    partial = {"processed_minutes": 5, "total_minutes": 45, "start_seconds": 130.0,
               "end_seconds": 460.0, "chosen": True}
    seen = {}

    async def _reserve(request, url, input_path, job_id, max_minutes=None, source_range=None):
        seen["source_range"] = source_range
        return 7, 1, "res-1", "starter", partial
    monkeypatch.setattr(app_module, "reserve_process_minutes", _reserve)

    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process", json={
                "url": "https://www.youtube.com/watch?v=long", "acknowledged": True,
                "source_start": 130, "source_end": 460}, headers={"X-Gemini-Key": "k"})
    resp = asyncio.run(_do())
    assert resp.status_code == 200, resp.text
    assert seen["source_range"] == (130, 460)
    job = app_module.jobs[resp.json()["job_id"]]
    assert job["env"]["SOURCE_START_SECONDS"] == "130.000"
    assert job["env"]["MAX_SOURCE_MINUTES"] == "5.5000"
    manifest = json.load(open(os.path.join(out_root, resp.json()["job_id"], app_module._RESUME_FILE)))
    assert manifest["partial"] == partial


def test_self_host_range_trims_job_and_reuses_the_quality_probe(dirs, monkeypatch):
    probes = []

    async def probe(url):
        probes.append(url)
        return {"max_height": 1080, "duration": 600}

    monkeypatch.setattr(app_module, "_probe_youtube_quality", probe)
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 720)
    url = "https://www.youtube.com/watch?v=long"

    async def do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process", json={
                "url": url, "acknowledged": True,
                "source_start": 90, "source_end": 270,
            }, headers={"X-Gemini-Key": "k"})

    response = asyncio.run(do())
    assert response.status_code == 200, response.text
    assert probes == [url]
    job = app_module.jobs[response.json()["job_id"]]
    assert job["env"]["SOURCE_START_SECONDS"] == "90.000"
    assert job["env"]["MAX_SOURCE_MINUTES"] == "3.0000"
    assert job["partial"]["chosen"] is True


def test_estimate_without_billing_says_so():
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process/estimate", json={"seconds": 600})
    resp = asyncio.run(_do())
    assert resp.status_code == 200
    assert resp.json() == {"billing": False}


def test_probe_cache_expires(monkeypatch):
    monkeypatch.setattr(app_module, "_probe_cache", {})
    app_module._store_probe("u1", "https://youtu.be/x", 12.5)
    assert app_module._cached_probe("u1", "https://youtu.be/x") == 12.5
    assert app_module._cached_probe("u2", "https://youtu.be/x") is None
    key = ("u1", "https://youtu.be/x")
    ts, m = app_module._probe_cache[key]
    app_module._probe_cache[key] = (ts - app_module._PROBE_CACHE_TTL - 1, m)
    assert app_module._cached_probe("u1", "https://youtu.be/x") is None


def test_ai_edits_are_free_only_up_to_the_daily_allowance(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_ai_edit_counts", {})
    owner = {"id": "u1"}

    async def _owner(request):
        return owner["id"]
    monkeypatch.setattr(app_module, "_owner_id", _owner)

    def cost():
        return asyncio.run(app_module._ai_edit_minutes(None))
    assert [cost() for _ in range(config.AI_EDITS_FREE_PER_DAY)] == [0] * config.AI_EDITS_FREE_PER_DAY
    assert cost() == config.MANAGED_ANALYSIS_MINUTES  # Gemini spend is metered again
    owner["id"] = "u2"
    assert cost() == 0  # per user


def test_free_editor_actions_are_capped_per_hour(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    monkeypatch.setattr(app_module, "_editor_action_times", {})
    for _ in range(config.EDITOR_ACTIONS_PER_HOUR):
        app_module._check_editor_rate("u1")
    with pytest.raises(HTTPException) as e:
        app_module._check_editor_rate("u1")
    assert e.value.status_code == 429
    app_module._check_editor_rate("u2")  # per user


# --------------------------------------------------------------------------- #
# Posting
# --------------------------------------------------------------------------- #
class TestYoutubeTitle:
    def test_short_title_is_untouched(self):
        assert app_module.youtube_title("  How I  doubled my views ") == "How I doubled my views"

    def test_long_title_is_cut_on_a_word(self):
        long = "word " * 40
        out = app_module.youtube_title(long)
        assert len(out) <= app_module.YOUTUBE_TITLE_MAX
        assert out.endswith("word")


def test_post_status_only_answers_the_account_that_posted(monkeypatch):
    monkeypatch.setattr(app_module, "_post_request_owners", {"req-1": "someone-else"})

    async def _owner(request):
        return "me"
    monkeypatch.setattr(app_module, "_owner_id", _owner)

    async def _get(path):
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path)
    assert asyncio.run(_get("/api/social/post-status/req-1")).status_code == 404
    unknown = asyncio.run(_get("/api/social/post-status/never-seen"))
    assert unknown.status_code == 200 and unknown.json() == {"status": "unknown"}


# --------------------------------------------------------------------------- #
# main.cap_source_duration with a start offset
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg")
def test_cut_from_a_start_offset_happens_once(tmp_path):
    main = pytest.importorskip("main")
    path = str(tmp_path / "src.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-t", "150", "-c:v", "libx264", "-preset", "ultrafast", "-g", "25",
        "-c:a", "aac", "-shortest", path], check=True)

    def seconds():
        return float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path]).decode().strip())

    main.cap_source_duration(path, 1, start_seconds=30)
    assert 59 <= seconds() <= 62
    assert os.path.exists(f"{path}.range")
    # A resumed job runs the same cut again on the already-cut file: no-op.
    main.cap_source_duration(path, 1, start_seconds=30)
    assert 59 <= seconds() <= 62


def test_failed_posts_are_logged_once_per_request(monkeypatch, capsys):
    monkeypatch.setattr(app_module, "_logged_post_failures", set())
    results = [{"platform": "tiktok", "success": False, "message": "Account  needs\nreauth"},
               {"platform": "youtube", "success": True, "message": "Published"}]
    app_module._log_post_failures("req-9", "completed", results)
    app_module._log_post_failures("req-9", "completed", results)  # the next poll
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["[social-post-failed] request=req-9 platform=tiktok status=completed "
                   "message='Account needs reauth'"]
    app_module._log_post_failures("req-ok", "completed", results[1:])
    app_module._log_post_failures("req-x", "failed", [])
    out = capsys.readouterr().out
    assert "req-ok" not in out and "platform=all status=failed" in out


def _saturate_editor_budget(monkeypatch, key):
    import time
    from types import SimpleNamespace
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_cloud_config", config)
    now = time.monotonic()
    monkeypatch.setattr(app_module, "_editor_action_times",
                        {key: [now] * config.EDITOR_ACTIONS_PER_HOUR})

    async def _noop(*a, **k):
        return True

    async def _owner(request):
        return "u1"

    async def _user(request):
        return SimpleNamespace(id="u1")
    monkeypatch.setattr(app_module, "require_managed_entitlement", _noop)
    monkeypatch.setattr(app_module, "_ensure_job_files", _noop)
    monkeypatch.setattr(app_module, "_owner_id", _owner)
    monkeypatch.setattr(app_module, "_user_from_request", _user)
    monkeypatch.setitem(app_module.jobs, "job-rl", {"user_id": "u1", "status": "completed"})


def test_scene_listing_is_capped_per_hour(monkeypatch):
    """Listing scenes cuts the clip from source and runs scene detection."""
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "scenes:u1")
    r = TestClient(app_module.app).get("/api/clip/job-rl/0/scenes")
    assert r.status_code == 429


def test_scene_listing_has_its_own_budget(monkeypatch):
    """Opening the framing editor must not eat the budget of the edits."""
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "u1")
    r = TestClient(app_module.app).get("/api/clip/job-rl/0/scenes")
    assert r.status_code != 429


def test_caption_removal_is_capped_per_hour(monkeypatch):
    from fastapi.testclient import TestClient
    _saturate_editor_budget(monkeypatch, "u1")
    r = TestClient(app_module.app).post("/api/subtitle/remove",
                                        json={"job_id": "job-rl", "clip_index": 0})
    assert r.status_code == 429
