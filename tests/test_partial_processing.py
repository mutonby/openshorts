"""The quota wall's "clip the first N minutes" offer.

Most walls open on an account with its free minutes untouched that pasted a
21-90 min video (93 of 99 sampled to 16-sep-2026): the user is asked to pay
before seeing a single clip, and most leave Stripe without paying. Instead of
a bare 402 the server now offers to clip the first N minutes (N = the balance),
the client resubmits with ``max_minutes``, only N are reserved and main.py cuts
the source to N before anything reads it.
"""
import asyncio
import json
import os
import shutil
import subprocess

import httpx
import pytest

app_module = pytest.importorskip("app")
from cloud import config  # noqa: E402


class TestPartialOffer:
    def test_offers_the_floored_balance(self):
        assert app_module.partial_offer(45, 20.0) == 20
        assert app_module.partial_offer(45, 12.7) == 12

    def test_no_offer_below_the_floor(self, monkeypatch):
        monkeypatch.setattr(config, "PARTIAL_MIN_MINUTES", 5)
        assert app_module.partial_offer(45, 4.9) == 0
        assert app_module.partial_offer(45, 0) == 0
        assert app_module.partial_offer(45, None) == 0

    def test_no_offer_when_the_balance_covers_the_video(self):
        # Then the 402 came from something else (concurrency, rounding): don't
        # offer to cut a video that fits.
        assert app_module.partial_offer(20, 20.0) == 0
        assert app_module.partial_offer(15, 20.0) == 0


class TestPlanPartialMinutes:
    def test_whole_video_when_not_asked(self):
        assert app_module.plan_partial_minutes(45, 20.0, None) == (45, None)

    def test_slice_capped_by_request_and_balance(self):
        reserve, partial = app_module.plan_partial_minutes(45, 20.0, 20)
        assert reserve == 20
        assert partial == {"processed_minutes": 20, "total_minutes": 45}
        # A client cannot name a bigger cut than it can pay for.
        reserve, partial = app_module.plan_partial_minutes(45, 8.4, 20)
        assert (reserve, partial["processed_minutes"]) == (8, 8)
        # Nor does asking for less than the balance reserve more than asked.
        reserve, _ = app_module.plan_partial_minutes(45, 20.0, 10)
        assert reserve == 10

    def test_fits_whole_means_no_partial(self):
        assert app_module.plan_partial_minutes(15, 20.0, 20) == (15, None)
        assert app_module.plan_partial_minutes(20, 20.0, 20) == (20, None)

    def test_too_small_slice_falls_through_to_the_402(self, monkeypatch):
        monkeypatch.setattr(config, "PARTIAL_MIN_MINUTES", 5)
        # Reserving the whole 45 then raises QuotaExceeded as before.
        assert app_module.plan_partial_minutes(45, 3.0, 20) == (45, None)

    def test_garbage_request_is_ignored(self):
        assert app_module.plan_partial_minutes(45, 20.0, "lots") == (45, None)


class TestFreeOverflow:
    """A free source past the balance never meets the wall: the first video
    (<= FIRST_VIDEO_MAX_MINUTES) is clipped whole for the balance, any other
    is clipped to the first N minutes."""

    @pytest.fixture(autouse=True)
    def _cap(self, monkeypatch):
        monkeypatch.setattr(config, "FIRST_VIDEO_MAX_MINUTES", 60)

    def test_first_video_up_to_the_cap_is_granted_whole(self):
        assert app_module.free_overflow("free", 50, 20.0, None, False) == (20, None)
        assert app_module.free_overflow("free", 60, 20.0, None, False) == (20, None)
        # The grant charges the floored balance, which lands at zero.
        assert app_module.free_overflow("free", 45, 17.6, None, False) == (17, None)

    def test_first_video_past_the_cap_is_clipped_to_the_balance(self):
        assert app_module.free_overflow("free", 61, 20.0, None, False) == (None, 20.0)

    def test_later_videos_are_clipped_to_the_balance(self):
        grant, max_minutes = app_module.free_overflow("free", 50, 20.0, None, True)
        assert grant is None and max_minutes == 20.0
        # ...which plan_partial_minutes turns into the usual first-N cut.
        assert app_module.plan_partial_minutes(50, 20.0, max_minutes) == (
            20, {"processed_minutes": 20, "total_minutes": 50})

    def test_untouched_cases(self):
        # Fits the balance, a paid plan, or the client already named a cut.
        assert app_module.free_overflow("free", 15, 20.0, None, False) == (None, None)
        assert app_module.free_overflow("starter", 50, 20.0, None, False) == (None, None)
        assert app_module.free_overflow("free", 50, 20.0, 10, False) == (None, 10)

    def test_grant_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "FIRST_VIDEO_MAX_MINUTES", 0)
        assert app_module.free_overflow("free", 50, 20.0, None, False) == (None, 20.0)


# --------------------------------------------------------------------------- #
# Endpoint: max_minutes reaches the reservation, the cut reaches the job
# --------------------------------------------------------------------------- #
def _post_process(json_body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process", json=json_body,
                                     headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


def _get_status(job_id):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/api/status/{job_id}")
    return asyncio.run(_do())


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
    monkeypatch.setattr(app_module, "_probe_youtube_quality", _probe)
    return out_root, up_root


def _stub_reservation(monkeypatch, partial):
    seen = {}

    async def _reserve(request, url, input_path, job_id, max_minutes=None, **_kw):
        seen["max_minutes"] = max_minutes
        return 7, 1, "res-1", "free", partial
    monkeypatch.setattr(app_module, "reserve_process_minutes", _reserve)
    return seen


def test_max_minutes_reaches_the_reservation_and_the_job(dirs, monkeypatch):
    out_root, _ = dirs
    partial = {"processed_minutes": 20, "total_minutes": 45}
    seen = _stub_reservation(monkeypatch, partial)
    resp = _post_process({"url": "https://www.youtube.com/watch?v=long",
                          "acknowledged": True, "max_minutes": 20})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert seen["max_minutes"] == 20
    assert body["partial"] == partial
    job = app_module.jobs[body["job_id"]]
    assert job["env"]["MAX_SOURCE_MINUTES"] == "20"
    assert job["partial"] == partial
    # The status poll carries it, so the dashboard can label the clips.
    assert _get_status(body["job_id"]).json()["partial"] == partial
    # And the resume manifest, so a redeploy mid-job does not re-run the
    # whole 45 minutes on a 20-minute reservation.
    manifest = json.load(open(os.path.join(out_root, body["job_id"], app_module._RESUME_FILE)))
    assert manifest["partial"] == partial


def test_generation_endcard_is_saved_for_resume_and_invalid_card_is_rejected(dirs, monkeypatch):
    out_root, _ = dirs
    _stub_reservation(monkeypatch, None)
    card = {"text": "Follow 👋", "handle": "@creator", "duration": 3, "sound": "silent"}
    response = _post_process({"url": "https://www.youtube.com/watch?v=long", "acknowledged": True,
        "endcard_settings": card})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    assert json.loads(app_module.jobs[job_id]["env"]["CLIP_ENDCARD_JSON"]) == card
    manifest = json.loads((out_root / job_id / app_module._RESUME_FILE).read_text())
    assert json.loads(manifest["generation_options"]["CLIP_ENDCARD_JSON"]) == card
    bad = _post_process({"url": "https://www.youtube.com/watch?v=long", "acknowledged": True,
        "endcard_settings": {"duration": 10}})
    assert bad.status_code == 400


def test_whole_video_job_is_untouched(dirs, monkeypatch):
    _stub_reservation(monkeypatch, None)
    resp = _post_process({"url": "https://www.youtube.com/watch?v=ok", "acknowledged": True})
    assert resp.status_code == 200
    job = app_module.jobs[resp.json()["job_id"]]
    assert "MAX_SOURCE_MINUTES" not in job["env"]
    assert resp.json()["partial"] is None
    assert job["partial"] is None


def test_exact_range_queues_short_fractional_selection_without_ai(dirs, monkeypatch):
    out_root, _ = dirs
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def allow_source(_url):
        pass

    monkeypatch.setattr(app_module, "_validate_source_url", allow_source)
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 0)
    monkeypatch.setattr(app_module, "MIN_SOURCE_SECONDS", 0)

    async def submit():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/api/process", json={
                "url": "https://www.youtube.com/watch?v=test", "acknowledged": True,
                "exact_range": True, "source_start": 0.375, "source_end": 3.625,
            })

    resp = asyncio.run(submit())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["partial"]["start_seconds"] == 0.375
    assert body["partial"]["end_seconds"] == 3.625
    job_id = body["job_id"]
    env = app_module.jobs[job_id]["env"]
    assert env["CLIP_EXACT_RANGE"] == "1"
    assert env["SOURCE_START_SECONDS"] == "0.375"
    assert float(env["MAX_SOURCE_MINUTES"]) * 60 == pytest.approx(3.25)
    manifest = json.loads((out_root / job_id / app_module._RESUME_FILE).read_text())
    assert manifest["generation_options"]["CLIP_EXACT_RANGE"] == "1"


def test_resumed_partial_job_keeps_its_cut(dirs, monkeypatch):
    out_root, _ = dirs
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    d = out_root / "cut"
    d.mkdir()
    (d / app_module._RESUME_FILE).write_text(json.dumps({
        "cmd": ["python", "main.py"], "priority": 2, "user_id": None,
        "reservation_id": "res-cut", "watermark": False, "attempts": 0,
        "partial": {"processed_minutes": 20, "total_minutes": 45},
    }))
    (out_root / "whole").mkdir()
    (out_root / "whole" / app_module._RESUME_FILE).write_text(json.dumps({
        "cmd": ["python", "main.py"], "priority": 2, "user_id": None,
        "reservation_id": "res-whole", "watermark": False, "attempts": 0,
    }))
    monkeypatch.setenv("MAX_SOURCE_MINUTES", "3")  # a stale value must not leak in
    app_module._resume_interrupted_jobs()
    assert app_module.jobs["cut"]["env"]["MAX_SOURCE_MINUTES"] == "20"
    assert app_module.jobs["cut"]["partial"]["processed_minutes"] == 20
    assert "MAX_SOURCE_MINUTES" not in app_module.jobs["whole"]["env"]
    assert app_module.jobs["whole"]["partial"] is None


# --------------------------------------------------------------------------- #
# main.cap_source_duration: the cut itself
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="needs ffmpeg")
class TestCapSourceDuration:
    @pytest.fixture()
    def source(self, tmp_path):
        main = pytest.importorskip("main")
        path = tmp_path / "src.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-t", "150", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest",
            str(path)], check=True)
        return main, str(path)

    @staticmethod
    def _seconds(path):
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", path]).decode().strip()
        return float(out)

    def test_cuts_in_place_to_the_first_minutes(self, source):
        main, path = source
        assert main.cap_source_duration(path, 2) == path
        assert 119 <= self._seconds(path) <= 122
        assert not os.path.exists(path.replace(".mp4", ".capped.mp4"))

    def test_shorter_source_is_left_alone(self, source):
        main, path = source
        before = os.path.getsize(path)
        main.cap_source_duration(path, 3)
        assert os.path.getsize(path) == before
        assert 149 <= self._seconds(path) <= 151


class TestFirstVideoIpFingerprint:
    """The per-network limit keys on an HMAC of the IP, never the IP."""

    def test_stable_and_not_the_ip(self, monkeypatch):
        from cloud import metering
        monkeypatch.setenv("JWT_SECRET", "s3cret")
        a = metering.ip_fingerprint("203.0.113.7")
        assert a == metering.ip_fingerprint("203.0.113.7")
        assert "203.0.113.7" not in a and len(a) == 64
        assert a != metering.ip_fingerprint("203.0.113.8")

    def test_unknown_ip_has_no_fingerprint(self):
        from cloud import metering
        assert metering.ip_fingerprint("") == ""
        assert metering.ip_fingerprint("unknown") == ""
        assert metering.ip_fingerprint(None) == ""
