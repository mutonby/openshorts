"""The per-plan job limit holds across the two containers of a deploy.

``jobs`` is one instance's memory. During a handover the new container has
not adopted the jobs the old one took after it booted, and the old one never
sees the new one's, so a free account (one job at a time) could queue a
second job on the other instance. The ledger is shared: every queued or
running clip job holds a ``reserved`` row from submission to its end, and the
limit check counts those too.
"""
import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cloud import config

app_module = pytest.importorskip("app")

from test_support_ops import _seed, clean, db, needs_db  # noqa: E402,F401


def _req():
    return SimpleNamespace(state=SimpleNamespace(), client=None, headers={})


@pytest.fixture()
def cloud_user(monkeypatch):
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
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "_submissions", {})
    return user


def _metering(reserved, balance_calls):
    class _M:
        @staticmethod
        async def reserved_process_jobs(uid):
            if isinstance(reserved, Exception):
                raise reserved
            return reserved

        @staticmethod
        async def get_balance(uid):
            balance_calls.append(uid)
            return {"remaining": 0, "plan": "free"}
    return _M


def _submit():
    return asyncio.run(app_module.reserve_process_minutes(
        _req(), "https://youtu.be/a", None, "job-new"))


def test_a_job_queued_on_the_other_instance_counts(cloud_user, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "_metering", _metering(1, calls))
    with pytest.raises(HTTPException) as e:
        _submit()
    assert e.value.status_code == 429
    assert calls == []  # refused before any reservation is attempted
    assert app_module._submissions_in_flight("u1") == 0


def test_under_the_limit_the_submission_goes_on(cloud_user, monkeypatch):
    calls = []
    cloud_user.plan = "starter"  # two at a time
    monkeypatch.setattr(app_module, "_metering", _metering(1, calls))
    with pytest.raises(HTTPException) as e:
        _submit()
    assert e.value.status_code == 402  # reached the balance check: not the job limit
    assert calls == ["u1"]


def test_the_count_fails_open(cloud_user, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, "_metering", _metering(RuntimeError("db down"), calls))
    with pytest.raises(HTTPException) as e:
        _submit()
    assert e.value.status_code == 402
    assert calls == ["u1"]


@needs_db
def test_only_open_clip_job_reservations_count(clean):
    from cloud import database, metering
    from cloud.models import UsageLedger

    async def run():
        uid = await _seed("free@example.com")
        other = await _seed("other@example.com")
        async with database.session() as s:
            async with s.begin():
                for job_type, status, owner in [
                    ("process", "reserved", uid),     # queued elsewhere: counts
                    ("process", "committed", uid),    # finished
                    ("process", "released", uid),     # failed / expired
                    ("thumbnail", "reserved", uid),   # not a clip job
                    ("process", "reserved", other),   # someone else's
                ]:
                    s.add(UsageLedger(user_id=owner, job_id=str(uuid.uuid4()),
                                      job_type=job_type, minutes=2, minutes_from_plan=2,
                                      minutes_from_topup=0, status=status))
        return await metering.reserved_process_jobs(uid)

    assert asyncio.run(run()) == 1
