"""Saturation mode of the free queue: lanes, newest first under a jam, brake.

Codes ``XA``/``XB`` are ISO user-assigned placeholders, not real countries.
Fake clock where it matters; nothing leaves the process.
"""
import asyncio
import itertools
import json
import os
import time
from types import SimpleNamespace

import pytest

app_module = pytest.importorskip("app")
qm = pytest.importorskip("cloud.queue_mode")
qw = pytest.importorskip("cloud.queue_watch")

H = 3600
FREE, SLOW = qm.FREE_PRIORITY, qm.SLOW_PRIORITY


@pytest.fixture
def env(monkeypatch):
    for k in ("QUEUE_SATURATION_DISABLED", "QUEUE_RECENT_FIRST_MINUTES",
              "QUEUE_RECENT_FIRST_RELEASE_MINUTES", "SLOW_LANE_BRAKE_HOURS",
              "SLOW_LANE_BRAKE_RELEASE_HOURS", "SLOW_LANE_BRAKE_CLIPS",
              "QUEUE_COUNTRY_HEADER", "FREE_SLOW_LANE_COUNTRIES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BILLING_ENABLED", "1")
    qm.reset()
    yield monkeypatch
    qm.reset()


@pytest.fixture
def sat(env):
    """Saturation mode configured: slow lane XA/XB, newest first past 60 min."""
    env.setenv("FREE_SLOW_LANE_COUNTRIES", "xa, XB")
    env.setenv("QUEUE_RECENT_FIRST_MINUTES", "60")
    return env


@pytest.fixture
def out(tmp_path, env):
    root = tmp_path / "output"
    root.mkdir()
    env.setattr(app_module, "OUTPUT_DIR", str(root))
    env.setattr(app_module, "INSTANCE_ID", "me")
    env.setattr(app_module, "BILLING_ENABLED", True)
    env.setattr(app_module, "_draining", False)
    env.setattr(app_module, "_running_jobs", set())
    env.setattr(app_module, "_admitting", set())
    env.setattr(app_module, "jobs", {})
    env.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    env.setattr(app_module, "_job_seq", itertools.count())
    env.setattr(app_module, "_seen_priority_gen", app_module._UNSEEN)
    env.setattr(app_module, "_owner_plan_pending", set())
    from cloud import config as cloud_config
    env.setattr(app_module, "_cloud_config", cloud_config)
    return root


def _job(out, jid, prio, submitted, user="u", manifest=True):
    d = out / jid
    d.mkdir(exist_ok=True)
    if manifest:
        (d / app_module._RESUME_FILE).write_text(json.dumps(
            {"cmd": ["python", "main.py"], "priority": prio, "user_id": user,
             "reservation_id": f"r-{jid}", "submitted_at": submitted,
             "lane": qm.lane_of_priority(prio)}))
    app_module.jobs[jid] = {"status": "queued", "user_id": user, "logs": [],
                            "submitted_at": submitted, "env": {},
                            "reservation_id": f"r-{jid}", "output_dir": str(d)}
    app_module._enqueue_job(jid, prio)


def _order():
    q, out = app_module.job_queue, []
    while not q.empty():
        out.append(q.get_nowait()[2])
    return out


def _peek_order():
    return [e[2] for e in sorted(app_module.job_queue._queue)]


def _assert_heap():
    heap = list(app_module.job_queue._queue)
    for i, e in enumerate(heap):
        for c in (2 * i + 1, 2 * i + 2):
            if c < len(heap):
                assert e <= heap[c]


def _set_recent(normal=None, slow=None):
    s = qm.current()
    if normal is not None:
        s["lanes"]["normal"]["recent_first"] = normal
    if slow is not None:
        s["lanes"]["slow"]["recent_first"] = slow
    qm.adopt(s)


def _set_brake(on, now=None):
    s = qm.current()
    s["brake"]["active"] = on
    s["brake"]["since"] = now or time.time()
    qm.adopt(s)


# --- Lanes ------------------------------------------------------------------------

class TestLaneAssignment:
    def test_empty_list_means_no_slow_lane(self, env):
        assert qm.slow_countries() == frozenset()
        assert qm.lane_for("free", "XA") == qm.NORMAL
        assert not qm.enabled()

    def test_listed_country_on_free_plan_is_slow(self, sat):
        assert qm.slow_countries() == {"XA", "XB"}
        assert qm.lane_for("free", "xa") == qm.SLOW
        assert qm.lane_for("free", "XC") == qm.NORMAL

    def test_unknown_country_is_normal(self, sat):
        sat.setenv("FREE_SLOW_LANE_COUNTRIES", "XA,XX,T1")
        assert qm.slow_countries() == {"XA"}          # CDN "unknown" codes never listed
        assert qm.lane_for("free", "") == qm.NORMAL
        assert qm.lane_for("free", None) == qm.NORMAL
        for raw in ("", "XX", "T1", "zzz", "1A", None):
            assert qm.request_country({"cf-ipcountry": raw} if raw is not None else {}) == ""

    def test_paid_plans_never_slow(self, sat):
        for plan in ("starter", "creator", "pro", None):
            assert qm.lane_for(plan, "XA") == qm.NORMAL
        assert qm.priority_for(1, qm.SLOW) == 1
        assert qm.priority_for(0, qm.SLOW) == 0
        assert qm.priority_for(FREE, qm.SLOW) == SLOW
        assert qm.priority_for(FREE, qm.NORMAL) == FREE

    def test_header_is_configurable(self, sat):
        assert qm.request_country({"cf-ipcountry": "xa"}) == "XA"
        sat.setenv("QUEUE_COUNTRY_HEADER", "X-Geo-Country")
        assert qm.request_country({"x-geo-country": "XB"}) == "XB"

    def test_kill_switch(self, sat):
        sat.setenv("QUEUE_SATURATION_DISABLED", "1")
        assert qm.lane_for("free", "XA") == qm.NORMAL
        assert not qm.enabled()


def _request(country=None, api_key=False):
    headers = {}
    if country:
        headers["cf-ipcountry"] = country
    if api_key:
        headers["X-API-Key"] = "osk_x"
    return SimpleNamespace(headers=headers, state=SimpleNamespace(),
                           client=SimpleNamespace(host="client"))


class FakeMetering:
    def __init__(self, remaining=20.0, topup=0.0):
        self.balance = {"plan": "free", "remaining": remaining, "topup_remaining": topup}
        self.reserved, self.first_video, self.probed = [], [], 0

    async def get_balance(self, uid):
        return dict(self.balance)

    async def reserve_minutes(self, uid, minutes, job_id, job_type="process"):
        self.reserved.append((uid, minutes, job_id))
        return f"res-{job_id}"

    async def has_processed_before(self, uid):
        return False

    def ip_fingerprint(self, ip):
        return "h"

    async def ip_had_first_video(self, h):
        return False

    async def record_first_video(self, h, job_id):
        self.first_video.append(job_id)

    def probe_file_minutes(self, path):
        self.probed += 1
        return 5.0

    def pop_statics_bot_checked(self, url):
        return False


@pytest.fixture
def managed(out, sat):
    met = FakeMetering()
    user = SimpleNamespace(id="u-free", plan="free")
    sat.setattr(app_module, "_metering", met)
    sat.setattr(app_module, "_submissions", {})
    sat.setattr(app_module, "_maybe_send_quota_email", lambda u: None)
    sat.setattr(app_module, "managed_keys",
                SimpleNamespace(has_active_entitlement=lambda u: True))

    async def from_request(request):
        return user

    async def no_budget():
        return False
    sat.setattr(app_module, "_user_from_request", from_request)
    sat.setattr(app_module, "_paid_proxy_budget_exceeded", no_budget)
    return SimpleNamespace(met=met, user=user)


def _reserve(request, job_id="j1"):
    try:
        return asyncio.run(app_module.reserve_process_minutes(request, None, "/tmp/x.mp4", job_id))
    finally:
        app_module._end_submission(job_id)  # the job would be registered here


class TestSubmit:
    def test_slow_country_gets_slow_priority(self, managed):
        req = _request("XA")
        uid, prio, rid, plan, partial = _reserve(req)
        assert prio == SLOW and req.state.queue_lane == qm.SLOW
        assert managed.met.reserved == [("u-free", 5, "j1")]

    def test_unknown_country_gets_normal_lane(self, managed):
        req = _request(None)
        assert _reserve(req)[1] == FREE and req.state.queue_lane == qm.NORMAL

    def test_paid_user_never_slow(self, managed):
        managed.user.plan = "starter"
        managed.met.balance["plan"] = "starter"
        req = _request("XA")
        assert _reserve(req)[1] == 1 and req.state.queue_lane == qm.NORMAL

    def test_topup_credit_is_not_free(self, managed):
        managed.met.balance["topup_remaining"] = 30.0
        req = _request("XA")
        assert _reserve(req)[1] == FREE

    def test_brake_refuses_without_spending_anything(self, managed):
        _set_brake(True)
        req = _request("XA", api_key=True)
        with pytest.raises(app_module.HTTPException) as e:
            _reserve(req)
        assert e.value.status_code == 402
        d = e.value.detail
        assert d["error"] == "free_queue_full"
        assert d["upgrade_url"] in d["message"]
        assert "Come back later" in d["message"]
        assert "XA" not in json.dumps(d) and "region" not in d["message"].lower()
        # No probe, no reservation, no first-video grant, no submission slot left.
        assert managed.met.reserved == [] and managed.met.first_video == []
        assert managed.met.probed == 0
        assert app_module._submissions_in_flight("u-free") == 0
        assert qm.current()["brake"]["rejected"] == 1

    def test_brake_lets_other_lanes_and_paid_through(self, managed):
        _set_brake(True)
        assert _reserve(_request("XC"), "j2")[1] == FREE
        managed.user.plan = "pro"
        managed.met.balance["plan"] = "pro"
        assert _reserve(_request("XA"), "j3")[1] == 0

    def test_out_of_minutes_still_meets_the_quota_wall_first(self, managed):
        _set_brake(True)
        managed.met.balance["remaining"] = 0
        with pytest.raises(app_module.HTTPException) as e:
            _reserve(_request("XA"))
        assert e.value.detail["error"] == "quota_exceeded"

    def test_nothing_configured_is_todays_queue(self, managed, monkeypatch):
        monkeypatch.delenv("FREE_SLOW_LANE_COUNTRIES")
        monkeypatch.delenv("QUEUE_RECENT_FIRST_MINUTES")
        _set_brake(True)
        req = _request("XA")
        assert _reserve(req)[1] == FREE


# --- Order --------------------------------------------------------------------------

class TestOrder:
    def _fill(self, out):
        t = 1_000_000.0
        _job(out, "slow1", SLOW, t + 1)
        _job(out, "free1", FREE, t + 2)
        _job(out, "starter1", 1, t + 3)
        _job(out, "slow2", SLOW, t + 4)
        _job(out, "free2", FREE, t + 5)
        _job(out, "pro1", 0, t + 6)
        _job(out, "free3", FREE, t + 7)

    def test_fifo_without_jam(self, out, sat):
        self._fill(out)
        assert _order() == ["pro1", "starter1", "free1", "free2", "free3", "slow1", "slow2"]

    def test_newest_first_inside_each_free_lane(self, out, sat):
        self._fill(out)
        _set_recent(normal=True, slow=True)
        _assert_heap()
        assert _order() == ["pro1", "starter1", "free3", "free2", "free1", "slow2", "slow1"]

    def test_new_arrival_while_jammed_goes_first_in_its_lane(self, out, sat):
        self._fill(out)
        _set_recent(normal=True)
        _job(out, "free4", FREE, 1_000_100.0)
        _job(out, "slow3", SLOW, 1_000_101.0)
        order = _order()
        assert order[:3] == ["pro1", "starter1", "free4"]
        assert order[-3:] == ["slow1", "slow2", "slow3"]   # slow lane still FIFO

    def test_back_to_fifo_old_ones_first(self, out, sat):
        self._fill(out)
        _set_recent(normal=True, slow=True)
        _set_recent(normal=False, slow=False)
        _assert_heap()
        assert _order() == ["pro1", "starter1", "free1", "free2", "free3", "slow1", "slow2"]

    def test_slow_lane_never_passes_anything(self, out, sat):
        # The oldest slow job against the newest normal job, in every mode.
        for normal, slow in ((False, False), (True, False), (False, True), (True, True)):
            app_module.job_queue = asyncio.PriorityQueue()
            app_module.jobs.clear()
            _set_recent(normal=normal, slow=slow)
            _job(out, "slow_old", SLOW, 1.0)
            _job(out, "free_new", FREE, 9e9)
            _job(out, "paid_new", 1, 9e9 + 1)
            assert _order() == ["paid_new", "free_new", "slow_old"]

    def test_paying_moves_a_slow_job_into_the_paid_line(self, out, sat):
        self._fill(out)
        _set_recent(normal=True, slow=True)
        assert asyncio.run(app_module.reprioritize_user_jobs("u", 1)) == 5  # pro1, starter1 stay
        _assert_heap()
        # Every job is u's: all paid now, in arrival order, pro first.
        assert _order() == ["pro1", "slow1", "free1", "starter1", "slow2", "free2", "free3"]
        m = json.loads((out / "slow1" / app_module._RESUME_FILE).read_text())
        assert m["priority"] == 1

    def test_owner_plan_lookup_never_pulls_free_rows(self, out, sat, monkeypatch):
        _job(out, "slow1", SLOW, 1.0, user="a11ce000-0000-0000-0000-000000000000")

        async def lookup(owners):
            return {u: "free" for u in owners}
        monkeypatch.setattr(app_module, "_owner_live_plans", lookup)
        assert asyncio.run(app_module._raise_to_owner_plans(["slow1"])) == 0
        assert app_module.job_queue._queue[0][0] == SLOW

    def test_heap_invariant_under_random_churn(self, out, sat):
        import random
        rng = random.Random(7)
        for i in range(400):
            _job(out, f"j{i}", rng.choice([0, 1, FREE, FREE, SLOW, SLOW]), rng.uniform(0, 1e6),
                 user=f"u{i % 9}", manifest=False)
            if i % 37 == 0:
                _set_recent(normal=rng.random() < 0.5, slow=rng.random() < 0.5)
                _assert_heap()
            if i % 53 == 0:
                asyncio.run(app_module.reprioritize_user_jobs(f"u{i % 9}", 1))
                _assert_heap()
        popped = [app_module.job_queue.get_nowait() for _ in range(app_module.job_queue.qsize())]
        assert popped == sorted(popped)


class TestStatus:
    def test_position_follows_the_dispatch_order(self, out, sat):
        TestOrder()._fill(out)
        _set_recent(normal=True)
        expected = _peek_order()
        for i, jid in enumerate(expected):
            snap = app_module.queue_snapshot(jid)
            assert snap["position"] == i + 1 and snap["ahead"] == i
        assert app_module.queue_snapshot("pro1")["eta_is_minimum"] is False
        assert app_module.queue_snapshot("free1")["eta_is_minimum"] is True   # newest first
        assert app_module.queue_snapshot("slow1")["eta_is_minimum"] is True   # behind the normal lane

    def test_position_updates_after_mode_switch_and_arrival(self, out, sat):
        TestOrder()._fill(out)
        assert app_module.queue_snapshot("free1")["position"] == 3
        _set_recent(normal=True)
        assert app_module.queue_snapshot("free1")["position"] == 5
        _job(out, "free9", FREE, 2_000_000.0)
        assert app_module.queue_snapshot("free9")["position"] == 3
        assert app_module.queue_snapshot("free1")["position"] == 6

    def test_fifo_eta_is_an_estimate(self, out, env):
        _job(out, "a", FREE, 1.0)
        assert app_module.queue_snapshot("a")["eta_is_minimum"] is False


# --- Brake ---------------------------------------------------------------------------

def _fig(normal_wait, slow_wait, slow_queued=10):
    return {"normal": {"queued": 5, "wait": normal_wait},
            "slow": {"queued": slow_queued, "wait": slow_wait}}


class TestDecide:
    def test_brake_enters_and_leaves_with_hysteresis(self, sat):
        s = qm.default_state()
        s, ev = qm.decide(s, _fig(0, 11 * H), 0)
        assert ev == ["recent_on:slow"] and not s["brake"]["active"]
        s, ev = qm.decide(s, _fig(0, 12 * H), 1)
        assert ev == ["brake_on"] and s["brake"]["active"]
        s, ev = qm.decide(s, _fig(0, 7 * H), 2)            # under 12 h, over 6 h
        assert ev == [] and s["brake"]["active"]
        s, ev = qm.decide(s, _fig(0, 11.9 * H), 3)
        assert ev == []
        s, ev = qm.decide(s, _fig(0, 5.9 * H), 4)
        assert "brake_off" in ev and not s["brake"]["active"] and s["brake"]["off_at"] == 4

    def test_recent_first_hysteresis_per_lane(self, sat):
        s = qm.default_state()
        s, ev = qm.decide(s, _fig(61 * 60, 61 * 60), 0)
        assert sorted(ev) == ["recent_on:normal", "recent_on:slow"]
        s, ev = qm.decide(s, _fig(31 * 60, 31 * 60), 1)     # release at half: 30 min
        assert ev == []
        s, ev = qm.decide(s, _fig(29 * 60, 45 * 60), 2)
        assert ev == ["recent_off:normal"]
        assert s["lanes"]["slow"]["recent_first"] and not s["lanes"]["normal"]["recent_first"]

    def test_thresholds_are_configurable(self, sat):
        sat.setenv("SLOW_LANE_BRAKE_HOURS", "2")
        sat.setenv("SLOW_LANE_BRAKE_RELEASE_HOURS", "1.5")
        s, ev = qm.decide(qm.default_state(), _fig(0, 2 * H), 0)
        assert "brake_on" in ev
        s, ev = qm.decide(s, _fig(0, 1.6 * H), 1)
        assert "brake_off" not in ev
        s, ev = qm.decide(s, _fig(0, 1.4 * H), 2)
        assert "brake_off" in ev

    def test_no_slow_lane_no_brake(self, env):
        env.setenv("QUEUE_RECENT_FIRST_MINUTES", "60")
        s, ev = qm.decide(qm.default_state(), _fig(0, 48 * H), 0)
        assert "brake_on" not in ev and not s["brake"]["active"]

    def test_defaults_do_nothing(self, env):
        s, ev = qm.decide(qm.default_state(), _fig(99 * H, 99 * H), 0)
        assert ev == []

    def test_kill_switch_releases_everything(self, sat):
        s, _ = qm.decide(qm.default_state(), _fig(99 * H, 99 * H), 0)
        assert s["brake"]["active"]
        sat.setenv("QUEUE_SATURATION_DISABLED", "1")
        s, ev = qm.decide(s, _fig(99 * H, 99 * H), 1)
        assert not s["brake"]["active"] and "brake_off" in ev
        assert not any(s["lanes"][lane]["recent_first"] for lane in qm.LANES)


class TestBrakeClips:
    def test_cap_respects_the_floor_and_user_target(self, sat):
        job = {"env": {}, "logs": []}
        app_module._apply_brake_clips(job)
        assert job["env"]["CLIP_TARGET_MIN"] == job["env"]["CLIP_TARGET_MAX"] == "3"
        assert "3 best clips" in job["logs"][-1]
        job = {"env": {"CLIP_TARGET_MIN": "2", "CLIP_TARGET_MAX": "2"}, "logs": []}
        app_module._apply_brake_clips(job)
        assert job["env"]["CLIP_TARGET_MAX"] == "2" and job["env"]["CLIP_TARGET_MIN"] == "2"
        sat.setenv("SLOW_LANE_BRAKE_CLIPS", "4")
        job = {"env": {"CLIP_TARGET_MIN": "8", "CLIP_TARGET_MAX": "8"}, "logs": []}
        app_module._apply_brake_clips(job)
        assert job["env"]["CLIP_TARGET_MAX"] == "4" and job["env"]["CLIP_TARGET_MIN"] == "4"

    def test_clip_targets_reach_the_selection(self, sat, monkeypatch):
        from clip_selection import clip_count_targets
        job = {"env": {}, "logs": []}
        app_module._apply_brake_clips(job)
        for k, v in job["env"].items():
            monkeypatch.setenv(k, v)
        assert clip_count_targets(10) == (3, 3)

    def test_log_line_is_visible_in_cloud(self):
        from log_view import friendly_logs
        line = app_module.BRAKE_CLIPS_LOG.format(n=3)
        assert friendly_logs([line]) == [line]
        assert friendly_logs([app_module.QUEUE_EXPIRED_LOG]) == [app_module.QUEUE_EXPIRED_LOG]

    def _dispatch(self, out, monkeypatch):
        async def no_wait():
            return None
        started = []

        async def fake_run(job_id):
            started.append(job_id)
        monkeypatch.setattr(app_module, "_wait_for_shared_gpu", no_wait)
        monkeypatch.setattr(app_module, "run_job_wrapper", fake_run)
        monkeypatch.setattr(app_module, "concurrency_semaphore", asyncio.Semaphore(10))

        async def run():
            worker = asyncio.create_task(app_module.process_queue())
            for _ in range(50):
                await asyncio.sleep(0.01)
                if app_module.job_queue.empty():
                    break
            await asyncio.sleep(0.02)
            worker.cancel()
        asyncio.run(run())
        return started

    def test_dispatch_caps_only_slow_jobs_while_braked(self, out, sat, monkeypatch):
        _job(out, "slow1", SLOW, 1.0)
        _job(out, "free1", FREE, 2.0)
        _set_brake(True)
        assert self._dispatch(out, monkeypatch) == ["free1", "slow1"]
        assert app_module.jobs["slow1"]["env"]["CLIP_TARGET_MAX"] == "3"
        assert "CLIP_TARGET_MAX" not in app_module.jobs["free1"]["env"]

    def test_no_cap_once_the_brake_let_go(self, out, sat, monkeypatch):
        _job(out, "slow1", SLOW, 1.0)
        assert self._dispatch(out, monkeypatch) == ["slow1"]
        assert "CLIP_TARGET_MAX" not in app_module.jobs["slow1"]["env"]


# --- Starved jobs ---------------------------------------------------------------------

class TestExpiry:
    @pytest.fixture
    def hooks(self, out, sat, monkeypatch):
        calls = {"settle": [], "webhook": [], "autopilot": []}

        async def settle(jid, job=None):
            calls["settle"].append((jid, job.get("status")))

        async def webhook(jid):
            calls["webhook"].append(jid)

        async def autopilot(jid, job):
            calls["autopilot"].append(jid)
        monkeypatch.setattr(app_module, "_settle_reservation", settle)
        monkeypatch.setattr(app_module, "_notify_job_webhook", webhook)
        monkeypatch.setattr(app_module, "_autopilot_job_finished", autopilot)
        return calls

    def test_old_free_jobs_end_cleanly(self, out, hooks):
        now = 10 * 86400.0
        limit = app_module.ACTIVE_JOB_MAX_SECONDS
        _job(out, "old_slow", SLOW, now - limit - 60)
        _job(out, "old_free", FREE, now - limit - 1)
        _job(out, "old_paid", 1, now - 3 * limit)
        _job(out, "fresh", FREE, now - 60)
        gone = asyncio.run(app_module._expire_starved_jobs(now))
        assert sorted(gone) == ["old_free", "old_slow"]
        assert _peek_order() == ["old_paid", "fresh"]
        _assert_heap()
        for jid in gone:
            job = app_module.jobs[jid]
            assert job["status"] == "failed" and job["queue_expired"]
            assert job["logs"][-1] == app_module.QUEUE_EXPIRED_LOG
            assert not (out / jid / app_module._RESUME_FILE).exists()
        assert sorted(hooks["settle"]) == [("old_free", "failed"), ("old_slow", "failed")]
        assert sorted(hooks["webhook"]) == ["old_free", "old_slow"]
        assert (out / "old_paid" / app_module._RESUME_FILE).exists()
        assert app_module.queue_snapshot("old_free") is None
        # Not resumed again by this or the next instance.
        assert app_module._resume_interrupted_jobs() == {"r-old_paid", "r-fresh"}

    def test_not_without_saturation_mode(self, out, hooks, monkeypatch):
        monkeypatch.delenv("FREE_SLOW_LANE_COUNTRIES")
        monkeypatch.delenv("QUEUE_RECENT_FIRST_MINUTES")
        _job(out, "old", FREE, 0.0)
        assert asyncio.run(app_module._expire_starved_jobs(10 * 86400.0)) == []
        assert app_module.jobs["old"]["status"] == "queued"

    def test_draining_instance_leaves_them(self, out, hooks, monkeypatch):
        monkeypatch.setattr(app_module, "_draining", True)
        _job(out, "old", FREE, 0.0)
        assert asyncio.run(app_module._expire_starved_jobs(10 * 86400.0)) == []

    def test_expired_jobs_are_not_output(self, out, hooks):
        _job(out, "old", FREE, 0.0)
        app_module.jobs["old"]["logs"] = app_module._TimedLog(["queued"])
        asyncio.run(app_module._expire_starved_jobs(time.time() + 10 * 86400))
        assert app_module._queue_watch_snapshot()["done_last_hour"] == 0


# --- Deploys ------------------------------------------------------------------------

class TestResumeOrder:
    def _manifest(self, out, jid, **m):
        d = out / jid
        d.mkdir()
        base = {"cmd": ["python", "main.py"], "priority": FREE, "user_id": None,
                "reservation_id": f"r-{jid}", "attempts": 0}
        base.update(m)
        (d / app_module._RESUME_FILE).write_text(json.dumps(base))
        return d

    def test_requeue_keeps_submission_order_not_listdir(self, out, sat, monkeypatch):
        # Names sort opposite to submission; listdir order is whatever the fs says.
        names = [f"z{i:02d}" for i in range(20)]
        for i, n in enumerate(names):
            self._manifest(out, n, submitted_at=1000.0 + (20 - i),
                           priority=SLOW if i % 3 == 0 else FREE,
                           lane=qm.SLOW if i % 3 == 0 else qm.NORMAL)
        monkeypatch.setattr(app_module.os, "listdir",
                            lambda p, _real=os.listdir: sorted(_real(p)))
        app_module._resume_interrupted_jobs()
        order = _order()
        normal = [n for i, n in enumerate(names) if i % 3]
        slow = [n for i, n in enumerate(names) if i % 3 == 0]
        assert order == list(reversed(normal)) + list(reversed(slow))
        assert app_module.jobs["z00"]["lane"] == qm.SLOW

    def test_requeue_under_jam_is_newest_first(self, out, sat):
        for i in range(5):
            self._manifest(out, f"j{i}", submitted_at=1000.0 + i)
        _set_recent(normal=True)
        app_module._resume_interrupted_jobs()
        assert _order() == ["j4", "j3", "j2", "j1", "j0"]

    def test_legacy_manifest_uses_owner_file_and_is_stamped(self, out, sat):
        d1 = self._manifest(out, "old_a")
        d2 = self._manifest(out, "old_b")
        (d1 / ".owner").write_text("u")
        (d2 / ".owner").write_text("u")
        os.utime(d1 / ".owner", (5000, 5000))
        os.utime(d2 / ".owner", (4000, 4000))
        app_module._resume_interrupted_jobs()
        assert _order() == ["old_b", "old_a"]
        m = json.loads((d1 / app_module._RESUME_FILE).read_text())
        assert m["submitted_at"] == 5000
        assert app_module.jobs["old_a"]["lane"] == qm.NORMAL   # no lane known: normal

    def test_submit_manifest_carries_lane_and_time(self, out, sat):
        d = out / "s1"
        d.mkdir()
        app_module._write_resume_manifest("s1", ["python"], SLOW, "u", "r", False,
                                          lane=qm.SLOW, submitted_at=123.5)
        m = json.loads((d / app_module._RESUME_FILE).read_text())
        assert m["lane"] == qm.SLOW and m["submitted_at"] == 123.5 and m["priority"] == SLOW


# --- Queue watch: measuring, sharing, alerts -------------------------------------------

class TestWatch:
    @pytest.fixture
    def world(self, sat, tmp_path, monkeypatch):
        for k in ("QUEUE_ALERTS_DISABLED", "ADMIN_EMAIL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            monkeypatch.delenv(k, raising=False)
        w = SimpleNamespace(active=True, snap=None, sent=[], dir=str(tmp_path))
        w.snap = {"waiting": [], "running": 5, "slots": 5, "done_last_hour": 40,
                  "eta_free": 0, "eta_paid": 0,
                  "lanes": {"normal": {"queued": 10, "wait": 600},
                            "slow": {"queued": 300, "wait": 13 * H}}}
        from cloud import alerts

        async def fake_send(subject, body):
            w.sent.append((subject, body))
        monkeypatch.setattr(alerts, "send_admin_alert", fake_send)
        qw.configure(snapshot=lambda: dict(w.snap), busy_elsewhere=lambda: 0,
                     state_dir=w.dir, is_active=lambda: w.active, now=0)
        return w

    def test_modes_run_without_an_alert_channel(self, world):
        assert asyncio.run(qw.tick(now=100_000.0)) is None
        assert qm.brake_active() and qm.recent_first(qm.SLOW) and not qm.recent_first(qm.NORMAL)
        saved = json.load(open(os.path.join(world.dir, qm.STATE_FILE)))
        assert saved["brake"]["active"]
        assert world.sent == []

    def test_other_instance_follows_and_hands_over_refusals(self, world):
        asyncio.run(qw.tick(now=100_000.0))
        qm.reset()                       # the old container: plain queue in memory
        world.active = False
        qm.note_rejected()
        qm.note_rejected()
        asyncio.run(qw.tick(now=100_100.0))
        assert qm.brake_active()
        saved = json.load(open(os.path.join(world.dir, qm.STATE_FILE)))
        assert saved["brake"]["rejected"] == 2
        world.active = True
        qm.note_rejected()
        asyncio.run(qw.tick(now=100_200.0))
        assert qm.current()["brake"]["rejected"] == 3

    def test_mode_switch_reorders_the_app_queue(self, world, out):
        _job(out, "a", SLOW, 1.0)
        _job(out, "b", SLOW, 2.0)
        asyncio.run(qw.tick(now=100_000.0))
        assert _peek_order() == ["b", "a"]
        world.snap["lanes"]["slow"]["wait"] = 0
        asyncio.run(qw.tick(now=100_200.0))
        assert _peek_order() == ["a", "b"] and not qm.brake_active()

    def test_alerts_name_the_modes_and_the_refusals(self, world, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
        kind = asyncio.run(qw.tick(now=100_000.0))
        assert "brake-on" in kind
        subject, body = next(s for s in world.sent if "Freno" in s[0])
        assert "Carril gratis lento" in body and "FRENO activo" in body
        assert "Carril gratis normal" in body and "por orden de llegada" in body
        for _ in range(4):
            qm.note_rejected()
        world.snap["lanes"]["slow"]["wait"] = 0
        kind = asyncio.run(qw.tick(now=100_300.0))
        assert "brake-off" in kind
        subject, body = world.sent[-1]
        assert "rechazado 4 envíos" in body
        assert "XA" not in body and "XB" not in body

    def test_queue_alert_carries_lane_lines(self, world, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
        world.snap["waiting"] = [{"id": f"f{i}", "paid": False, "lane": "slow"} for i in range(300)]
        qw.configure(snapshot=lambda: dict(world.snap), busy_elsewhere=lambda: 0,
                     wait_started=lambda jid: 100_000.0 - 5 * H,
                     state_dir=world.dir, is_active=lambda: True, now=0)
        asyncio.run(qw.tick(now=100_000.0))
        queue_msg = next(b for s, b in world.sent if "atascada" in s)
        assert "Carril gratis lento" in queue_msg and "rechazados" in queue_msg

    def test_braked_hours_never_become_the_usual_rate(self, world):
        lim = qw.limits()
        state = {}
        m = {"done_last_hour": 80, "slots": 5, "running": 5, "queued": 50,
             "running_elsewhere": 0, "reduced_clips": True}
        qw._full_since = 1.0
        _k, _sub, samples, _at = qw.rate_incident(state, m, 2 * H, lim, False)
        assert samples == []
        m["reduced_clips"] = False
        qw._full_since = 1.0
        _k, _sub, samples, _at = qw.rate_incident(state, m, 2 * H, lim, False)
        assert samples == [16.0]

    def test_reduced_clips_tail_after_release(self, sat):
        s = qm.current()
        s["brake"].update(active=False, off_at=1000.0)
        qm.adopt(s)
        assert qm.reduced_clips_recently(1000.0 + 1800)
        assert not qm.reduced_clips_recently(1000.0 + 2 * H)


# --- Volume: the loop stays responsive ---------------------------------------------------

def _worst_stall(coro_fn):
    async def run():
        worst, stop = 0.0, False

        async def ticker():
            nonlocal worst
            last = time.perf_counter()
            while not stop:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                worst = max(worst, now - last - 0.005)
                last = now
        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        result = await coro_fn()
        stop = True
        await t
        return worst, result
    return asyncio.run(run())


class TestVolume:
    N = 3000

    def test_resume_scan_reads_manifests_off_the_loop(self, out, sat, monkeypatch):
        for i in range(800):
            d = out / f"m{i}"
            d.mkdir()
            (d / app_module._RESUME_FILE).write_text(json.dumps(
                {"cmd": ["python"], "priority": FREE, "user_id": None,
                 "reservation_id": f"r{i}", "submitted_at": float(i)}))
        real = app_module._manifest_names

        def slow_names():
            names = real()
            time.sleep(0.5)          # a slow shared disk listing + reads
            return names
        monkeypatch.setattr(app_module, "_manifest_names", slow_names)
        worst, keep = _worst_stall(app_module._resume_interrupted_jobs_async)
        assert len(keep) == 800 and app_module.job_queue.qsize() == 800
        assert worst < 0.15, f"loop held {worst * 1000:.0f} ms"
        assert _order()[:3] == ["m0", "m1", "m2"]

    def test_thousands_of_jobs_mode_switch_status_and_expiry(self, out, sat, monkeypatch):
        import random
        rng = random.Random(3)
        for i in range(self.N):
            _job(out, f"j{i}", rng.choice([FREE, SLOW]), rng.uniform(0, 1e5), manifest=False)
        t0 = time.perf_counter()
        _set_recent(normal=True, slow=True)
        assert time.perf_counter() - t0 < 0.1
        _assert_heap()
        # 300 tabs polling: one ranking, then lookups.
        t0 = time.perf_counter()
        for i in range(300):
            assert app_module.queue_snapshot(f"j{i}") is not None
        assert time.perf_counter() - t0 < 0.25

        async def noop(*a, **k):
            return None
        monkeypatch.setattr(app_module, "_settle_reservation", noop)
        monkeypatch.setattr(app_module, "_notify_job_webhook", noop)
        monkeypatch.setattr(app_module, "_autopilot_job_finished", noop)
        now = 1e5 + app_module.ACTIVE_JOB_MAX_SECONDS - 5e4   # about half are past it
        worst, gone = _worst_stall(lambda: app_module._expire_starved_jobs(now))
        assert 0 < len(gone) < self.N
        assert worst < 0.15, f"loop held {worst * 1000:.0f} ms"
        assert app_module.job_queue.qsize() == self.N - len(gone)
        _assert_heap()


# --- A slow-lane video that never gets to run ---------------------------------------

class TestNeverRuns:
    """Through many deploys it is neither dropped nor swept, its reservation
    stays live, and past ACTIVE_JOB_MAX_SECONDS it ends with a reason that
    both instances report (no 404) and its minutes / first video back."""

    def _handover(self, out):
        """The next container: memory gone, the queue rebuilt from manifests."""
        app_module.jobs.clear()
        app_module.job_queue = asyncio.PriorityQueue()
        return app_module._resume_interrupted_jobs()

    def test_survives_many_handovers_untouched(self, out, sat, monkeypatch):
        sub = time.time() - 20 * H
        d = out / "slowold"
        d.mkdir()
        (d / ".owner").write_text("u")
        (d / app_module._RESUME_FILE).write_text(json.dumps(
            {"cmd": ["python"], "priority": SLOW, "lane": qm.SLOW, "user_id": "u",
             "reservation_id": "r-old", "submitted_at": sub, "attempts": 0}))
        upload = out.parent / "uploads"
        upload.mkdir()
        (upload / "slowold_src.mp4").write_bytes(b"x" * 10)
        monkeypatch.setattr(app_module, "UPLOAD_DIR", str(upload))
        monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 1e-9)     # disk "full"
        monkeypatch.setattr(app_module, "UPLOADS_MAX_GB", 1e-9)
        monkeypatch.setattr(app_module, "JOB_RETENTION_SECONDS", 1)
        for _ in range(12):
            keep = self._handover(out)
            assert keep == {"r-old"}
            app_module._sweep_expired_job_dirs(time.time() + 10)
            app_module._enforce_output_size_cap()
            app_module._enforce_uploads_size_cap()
            app_module._sweep_expired_uploads(time.time() + 10)
        m = json.loads((d / app_module._RESUME_FILE).read_text())
        assert m["attempts"] == 0 and m["submitted_at"] == sub
        assert (upload / "slowold_src.mp4").exists()
        assert app_module.jobs["slowold"]["status"] == "queued"
        live = asyncio.run(app_module._live_reservation_ids())
        assert live["r-old"] >= time.time() - 5        # queued here: never "hung"

    def test_expires_cleanly_and_both_instances_say_why(self, out, sat, monkeypatch):
        released = []

        async def settle(jid, job=None):
            released.append((jid, job["status"]))

        async def noop(*a, **k):
            return None
        monkeypatch.setattr(app_module, "_settle_reservation", settle)
        monkeypatch.setattr(app_module, "_notify_job_webhook", noop)
        monkeypatch.setattr(app_module, "_autopilot_job_finished", noop)
        now = time.time()
        _job(out, "slow_old", SLOW, now - app_module.ACTIVE_JOB_MAX_SECONDS - 1, user="u")
        (out / "slow_old" / ".owner").write_text("u")
        asyncio.run(app_module._expire_starved_jobs(now))
        assert released == [("slow_old", "failed")]      # minutes back, grant kept
        # The other instance (or this one after a restart) never held it.
        app_module.jobs.clear()
        view = app_module._job_view_from_disk("slow_old")
        assert view["status"] == "failed"
        assert view["logs"][-1] == app_module.QUEUE_EXPIRED_LOG
        assert self._handover(out) == set()               # never resumed again
        assert "slow_old" not in app_module.jobs
