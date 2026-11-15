"""Support-ops API v1 (cloud/support_ops.py): the support back-office looks up
and fixes one account.

The first half needs no database: the switches, the keys, the CORS guard and
every check that runs before a query. The second half runs against a real
Postgres (the models use citext/jsonb/uuid) and is skipped unless
``TEST_DATABASE_URL`` points at a throwaway database, which CI provides; it
drops and recreates every table there.
"""
import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from cloud import support_ops

READ = "r" * 40
WRITE = "w" * 40
BASE = support_ops.PREFIX


def _app():
    app = FastAPI()
    app.include_router(support_ops.router)
    app.add_middleware(CORSMiddleware, allow_origins=["https://dashboard.example"],
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(support_ops.CorsGuard)
    return app


def _h(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def keys():
    support_ops.load_keys({support_ops.READ_KEY_ENV: READ, support_ops.WRITE_KEY_ENV: WRITE})
    yield
    support_ops.load_keys({})


@pytest.fixture
def client(keys):
    return TestClient(_app(), raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Switches and keys
# --------------------------------------------------------------------------- #
class TestSwitches:
    def test_off_without_keys(self):
        assert support_ops.load_keys({}) is False

    def test_a_short_key_keeps_the_whole_feature_off(self, capsys):
        assert support_ops.load_keys({support_ops.READ_KEY_ENV: "short",
                                      support_ops.WRITE_KEY_ENV: WRITE}) is False
        assert "shorter than" in capsys.readouterr().out

    def test_either_key_turns_it_on(self):
        assert support_ops.load_keys({support_ops.READ_KEY_ENV: READ}) is True
        assert support_ops.load_keys({support_ops.WRITE_KEY_ENV: WRITE}) is True
        support_ops.load_keys({})

    def test_every_route_is_404_when_off(self):
        support_ops.load_keys({})
        c = TestClient(_app())
        uid = str(uuid.uuid4())
        for method, path in (("GET", "/users?email=a@b.com"), ("GET", f"/users/{uid}"),
                             ("GET", f"/users/{uid}/activity"), ("GET", "/actions"),
                             ("POST", f"/users/{uid}/actions/ban"), ("PUT", "/actions"),
                             ("GET", "/nothing-here")):
            r = c.request(method, BASE + path, headers=_h(WRITE))
            assert r.status_code == 404, (method, path)
            assert r.json() == {"detail": "Not Found"}


class TestAuth:
    def test_missing_or_wrong_key_is_401(self, client):
        assert client.get(f"{BASE}/actions").status_code == 401
        assert client.get(f"{BASE}/actions", headers=_h("x" * 40)).status_code == 401
        assert client.get(f"{BASE}/actions", headers={"Authorization": READ}).status_code == 401

    def test_read_key_reads(self, client):
        r = client.get(f"{BASE}/actions", headers=_h(READ))
        assert r.status_code == 200
        names = [a["name"] for a in r.json()["actions"]]
        assert names == ["set_plan", "add_credits", "reset_usage", "change_email", "ban",
                         "unban", "delete_account", "update_field", "restore"]
        for a in r.json()["actions"]:
            assert set(a) == {"name", "params", "description"}

    def test_read_key_cannot_write(self, client):
        r = client.post(f"{BASE}/users/{uuid.uuid4()}/actions/ban", headers=_h(READ),
                        json={"params": {"reason": "x"}})
        assert r.status_code == 403

    def test_write_key_reads_too(self, client):
        assert client.get(f"{BASE}/actions", headers=_h(WRITE)).status_code == 200

    def test_not_in_the_openapi_schema(self, client):
        assert not any(p.startswith(BASE) for p in _app().openapi()["paths"])


class TestNoCors:
    def test_no_cors_headers_even_for_an_allowed_origin(self, client):
        r = client.get(f"{BASE}/actions", headers={**_h(READ), "Origin": "https://dashboard.example"})
        assert r.status_code == 200
        assert not [k for k in r.headers if k.lower().startswith("access-control-")]

    def test_preflight_is_404(self, client):
        r = client.options(f"{BASE}/actions", headers={
            "Origin": "https://dashboard.example", "Access-Control-Request-Method": "GET"})
        assert r.status_code == 404
        assert not [k for k in r.headers if k.lower().startswith("access-control-")]


class TestRequestChecks:
    """Everything here is refused before the database is touched."""

    def _post(self, client, action, body, key=WRITE):
        return client.post(f"{BASE}/users/{uuid.uuid4()}/actions/{action}",
                           headers=_h(key), json=body)

    def test_unknown_action_is_404(self, client):
        assert self._post(client, "make_coffee", {}).status_code == 404

    def test_real_write_needs_a_ticket(self, client):
        r = self._post(client, "ban", {"params": {"reason": "spam"}, "dry_run": False})
        assert r.status_code == 400 and "ticket" in r.json()["detail"]

    @pytest.mark.parametrize("action,params", [
        ("set_plan", {"plan": "enterprise"}),
        ("set_plan", {"plan": "pro", "until": "2001-01-01"}),
        ("set_plan", {"plan": "pro", "until": "next tuesday"}),
        ("set_plan", {"plan": "pro", "stripe_subscription_id": "cus_123"}),
        ("set_plan", {"plan": "free", "until": "2999-01-01"}),
        ("add_credits", {"amount": "60"}),
        ("add_credits", {"amount": True}),
        ("add_credits", {"amount": 0}),
        ("add_credits", {}),
        ("reset_usage", {"anything": 1}),
        ("change_email", {"new_email": "not an email"}),
        ("ban", {"reason": "  "}),
        ("delete_account", {}),
        ("update_field", {"field": "email", "value": "x@y.com"}),
        ("update_field", {"field": "stripe_customer_id", "value": "cus_1"}),
        ("update_field", {"field": "marketing_opt_out", "value": "yes"}),
        ("update_field", {"field": "free_plan_denied", "value": "Not A Slug!"}),
        ("restore", {"before": {}}),
        ("restore", {"before": {"password": "x"}}),
        ("restore", {"before": {"ban_reason": "x"}}),
        ("restore", {"before": {"usage": 3}}),
    ])
    def test_invalid_params_are_400(self, client, action, params):
        r = self._post(client, action, {"params": params})
        assert r.status_code == 400, r.text
        assert r.json()["detail"]

    def test_malformed_bodies_are_400(self, client):
        uid = uuid.uuid4()
        url = f"{BASE}/users/{uid}/actions/unban"
        assert client.post(url, headers=_h(WRITE), content=b"{nope").status_code == 400
        assert client.post(url, headers=_h(WRITE), json=[1]).status_code == 400
        assert client.post(url, headers=_h(WRITE), json={"dry_run": "no"}).status_code == 400
        assert client.post(url, headers=_h(WRITE), json={"params": []}).status_code == 400
        assert client.post(url, headers=_h(WRITE), json={"surprise": 1}).status_code == 400

    def test_a_non_uuid_user_is_404(self, client):
        r = client.post(f"{BASE}/users/not-a-uuid/actions/unban", headers=_h(WRITE), json={})
        assert r.status_code == 404

    def test_lookup_needs_exactly_one_key(self, client):
        assert client.get(f"{BASE}/users", headers=_h(READ)).status_code == 400
        assert client.get(f"{BASE}/users?email=a@b.com&id=x", headers=_h(READ)).status_code == 400


def test_add_interval_clamps_the_day():
    jan31 = datetime(2026, 1, 31, 12, tzinfo=timezone.utc)
    assert support_ops._add_interval(jan31, "month") == datetime(2026, 2, 28, 12, tzinfo=timezone.utc)
    assert support_ops._add_interval(jan31, "year") == datetime(2027, 1, 31, 12, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Against Postgres
# --------------------------------------------------------------------------- #
DB_URL = os.environ.get("TEST_DATABASE_URL")
try:
    import asyncpg  # noqa: F401
    _HAVE_ASYNCPG = True
except ImportError:
    _HAVE_ASYNCPG = False

needs_db = pytest.mark.skipif(not (DB_URL and _HAVE_ASYNCPG),
                              reason="TEST_DATABASE_URL (Postgres) and asyncpg needed")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def db():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    from cloud import database, models  # noqa: F401

    # NullPool: TestClient runs the app on its own event loop, the fixtures
    # on asyncio.run's; a pooled connection must never cross between them.
    engine = create_async_engine(DB_URL, poolclass=NullPool)

    async def init():
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
            await conn.run_sync(database.Base.metadata.drop_all)
            await conn.run_sync(database.Base.metadata.create_all)
    _run(init())
    saved = database._engine, database._sessionmaker
    database._engine = engine
    database._sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield database
    database._engine, database._sessionmaker = saved
    _run(engine.dispose())


@pytest.fixture
def clean(db):
    from sqlalchemy import text

    async def wipe():
        names = ", ".join(t.name for t in db.Base.metadata.sorted_tables)
        async with db.session() as s:
            await s.execute(text(f"TRUNCATE {names} CASCADE"))
            await s.commit()
    _run(wipe())
    return db


@pytest.fixture
def plan_hooks(monkeypatch):
    calls = []

    async def _fake(user_id, plan, became_live):
        calls.append((str(user_id), plan, became_live))
    monkeypatch.setattr(support_ops, "_on_plan_live", _fake)
    return calls


NOW = datetime.now(timezone.utc)


async def _seed(email="alice@example.com", *, google=False, sub=None, topups=(), used=0,
                customer=None, **user_kw):
    from cloud import database
    from cloud.models import CreditTopup, Subscription, UsageLedger, User
    async with database.session() as s:
        async with s.begin():
            u = User(email=email, google_sub="g-" + email if google else None,
                     stripe_customer_id=customer, last_login_at=NOW, **user_kw)
            s.add(u)
            await s.flush()
            period_end = None
            if sub:
                row = Subscription(
                    user_id=u.id, stripe_subscription_id=sub.get("id", "sub_" + uuid.uuid4().hex[:10]),
                    plan=sub.get("plan", "creator"), interval=sub.get("interval", "month"),
                    status=sub.get("status", "active"),
                    minutes_per_period=sub.get("minutes", 300),
                    current_period_start=NOW - timedelta(days=3),
                    current_period_end=sub.get("end", NOW + timedelta(days=27)),
                    cancel_at_period_end=False)
                s.add(row)
                period_end = row.current_period_end
            else:
                from cloud import metering
                period_end = metering.free_period_end()
            for total, consumed in topups:
                s.add(CreditTopup(user_id=u.id, stripe_session_id="cs_" + uuid.uuid4().hex,
                                  minutes_total=total, minutes_consumed=consumed,
                                  created_at=NOW - timedelta(days=10)))
            if used:
                s.add(UsageLedger(user_id=u.id, job_id="job-used", job_type="process",
                                  minutes=used, minutes_from_plan=used, minutes_from_topup=0,
                                  status="committed", period_end=period_end))
            return u.id


async def _user(uid):
    from cloud import database
    from cloud.models import User
    async with database.session() as s:
        return await s.get(User, uid)


def _act(client, uid, action, params=None, dry_run=None, ticket="T-1", key=WRITE):
    body = {"params": params or {}, "ticket": ticket, "actor": "tests"}
    if dry_run is not None:
        body["dry_run"] = dry_run
    return client.post(f"{BASE}/users/{uid}/actions/{action}", headers=_h(key), json=body)


def _view(client, uid):
    r = client.get(f"{BASE}/users/{uid}", headers=_h(READ))
    assert r.status_code == 200, r.text
    return r.json()["user"]


USER_VIEW_KEYS = {"id", "email", "name", "created_at", "last_login_at", "auth_provider", "plan",
                  "plan_status", "billing_interval", "stripe_customer_id", "stripe_subscription_id",
                  "usage", "credits", "banned", "ban_reason", "deleted", "extra"}


@needs_db
class TestLookup:
    def test_email_lookup_is_exact_and_case_insensitive(self, client, clean):
        uid = _run(_seed("alice@example.com"))
        _run(_seed("alice2@example.com"))
        r = client.get(f"{BASE}/users", params={"email": "Alice@Example.COM"}, headers=_h(READ))
        assert r.status_code == 200
        assert [u["id"] for u in r.json()["users"]] == [str(uid)]

    def test_gmail_aliases_find_the_normalised_account(self, client, clean):
        uid = _run(_seed("foobar@gmail.com", google=True))
        r = client.get(f"{BASE}/users", params={"email": "Foo.Bar@gmail.com"}, headers=_h(READ))
        assert [u["id"] for u in r.json()["users"]] == [str(uid)]

    def test_id_and_stripe_customer_lookups(self, client, clean):
        uid = _run(_seed(customer="cus_ABC"))
        for q in ({"id": str(uid)}, {"stripe_customer": "cus_ABC"}):
            r = client.get(f"{BASE}/users", params=q, headers=_h(READ))
            assert [u["id"] for u in r.json()["users"]] == [str(uid)]

    def test_no_match_is_an_empty_list(self, client, clean):
        for q in ({"email": "nobody@example.com"}, {"id": "not-a-uuid"},
                  {"id": str(uuid.uuid4())}, {"stripe_customer": "cus_none"}):
            r = client.get(f"{BASE}/users", params=q, headers=_h(READ))
            assert r.status_code == 200 and r.json() == {"users": []}

    def test_get_user_404(self, client, clean):
        assert client.get(f"{BASE}/users/{uuid.uuid4()}", headers=_h(READ)).status_code == 404
        assert client.get(f"{BASE}/users/nope", headers=_h(READ)).status_code == 404

    def test_an_erased_account_shows_as_deleted(self, client, clean):
        from cloud import account, database
        from cloud.models import AccountDeletion
        former = str(uuid.uuid4())

        async def add():
            async with database.session() as s:
                async with s.begin():
                    s.add(AccountDeletion(former_user_id=former,
                                          email_sha256=account.email_fingerprint("gone@example.com"),
                                          stripe_customer_id="cus_gone", plan_at_deletion="pro"))
        _run(add())
        r = client.get(f"{BASE}/users", params={"email": "gone@example.com"}, headers=_h(READ))
        (v,) = r.json()["users"]
        assert v["deleted"] is True and v["id"] == former and set(v) == USER_VIEW_KEYS
        assert _view(client, former)["deleted"] is True
        assert client.get(f"{BASE}/users", params={"stripe_customer": "cus_gone"},
                          headers=_h(READ)).json()["users"][0]["id"] == former


@needs_db
class TestUserView:
    def test_shape_and_figures(self, client, clean):
        uid = _run(_seed(sub={"plan": "creator", "minutes": 300}, topups=[(60, 10)], used=50,
                         customer="cus_1"))
        v = _view(client, uid)
        assert set(v) == USER_VIEW_KEYS
        assert v["plan"] == "creator" and v["plan_status"] == "active"
        assert v["billing_interval"] == "month" and v["auth_provider"] == "magic_link"
        assert v["usage"]["used"] == 50 and v["usage"]["limit"] == 300
        assert v["usage"]["unit"] == "minutes" and v["usage"]["period_end"]
        assert v["credits"] == 50
        assert v["banned"] is False and v["deleted"] is False and v["name"] is None

    def test_free_account(self, client, clean):
        uid = _run(_seed("free@gmail.com", google=True, used=5))
        v = _view(client, uid)
        assert v["plan"] == "free" and v["plan_status"] is None
        assert v["usage"]["limit"] == 20 and v["usage"]["used"] == 5
        assert v["auth_provider"] == "google"

    def test_no_secret_ever_reaches_the_view(self, client, clean):
        from cloud import database
        from cloud.models import ApiKey, LoginCode, MagicLinkToken
        uid = _run(_seed("sec@example.com", google=True))

        async def add():
            async with database.session() as s:
                async with s.begin():
                    s.add(ApiKey(user_id=uid, name="ci", key_hash="KEYHASHSECRET",
                                 prefix="osk_PREFIXSECRET"))
                    s.add(MagicLinkToken(email="sec@example.com", token_hash="MAGICSECRET",
                                         expires_at=NOW + timedelta(hours=1)))
                    s.add(LoginCode(code_hash="LOGINSECRET", user_id=uid, challenge="CHALLENGESECRET",
                                    expires_at=NOW + timedelta(minutes=5)))
        _run(add())
        v = _view(client, uid)
        dump = json.dumps(v)
        for secret in ("KEYHASHSECRET", "PREFIXSECRET", "MAGICSECRET", "LOGINSECRET",
                       "CHALLENGESECRET", "g-sec@example.com"):
            assert secret not in dump
        assert v["extra"]["api_keys_active"] == 1
        activity = client.get(f"{BASE}/users/{uid}/activity", headers=_h(READ)).text
        assert "SECRET" not in activity


@needs_db
class TestActivity:
    def test_items_newest_first_with_jobs_logins_and_topups(self, client, clean, monkeypatch):
        from cloud import database
        from cloud.models import EvidenceEvent, Project, UsageLedger
        uid = _run(_seed(topups=[(60, 0)]))

        async def add():
            async with database.session() as s:
                async with s.begin():
                    s.add(UsageLedger(user_id=uid, job_id="job-ok", job_type="process", minutes=12,
                                      minutes_from_plan=12, status="committed",
                                      created_at=NOW - timedelta(hours=3)))
                    s.add(UsageLedger(user_id=uid, job_id="job-bad", job_type="process", minutes=8,
                                      minutes_from_plan=8, status="released",
                                      created_at=NOW - timedelta(hours=1)))
                    s.add(Project(user_id=uid, job_id="job-ok", title="My talk",
                                  metadata_r2_key="k"))
                    s.add(EvidenceEvent(user_id=uid, email="alice@example.com", kind="login",
                                        method="google", ip="203.0.113.7", user_agent="UA-SECRET",
                                        created_at=NOW - timedelta(hours=2)))
        _run(add())
        monkeypatch.setattr(support_ops, "_local_jobs", lambda user_id: {
            "job-bad": {"status": "failed", "error": "Gemini Error: empty response",
                        "source": "url", "created_at": None}})
        r = client.get(f"{BASE}/users/{uid}/activity", headers=_h(READ))
        assert r.status_code == 200
        items = r.json()["items"]
        assert [i["type"] for i in items] == ["job", "login", "job", "topup"]
        for i in items:
            assert set(i) == {"type", "id", "created_at", "status", "title", "error", "minutes",
                              "source", "detail"}
        bad, login, ok, topup = items
        assert bad["id"] == "job-bad" and bad["status"] == "failed" and bad["minutes"] == 0
        assert bad["error"].startswith("Gemini Error") and bad["source"] == "url"
        assert ok["status"] == "charged" and ok["minutes"] == 12 and ok["title"] == "My talk"
        assert login["source"] == "google" and "203.0.113.7" not in json.dumps(items)
        assert "UA-SECRET" not in json.dumps(items)
        assert topup["minutes"] == 60 and topup["source"] == "stripe"
        r = client.get(f"{BASE}/users/{uid}/activity?limit=2", headers=_h(READ))
        assert len(r.json()["items"]) == 2

    def test_limit_checks_and_unknown_user(self, client, clean):
        uid = _run(_seed())
        assert client.get(f"{BASE}/users/{uid}/activity?limit=0", headers=_h(READ)).status_code == 400
        assert client.get(f"{BASE}/users/{uid}/activity?limit=x", headers=_h(READ)).status_code == 400
        assert client.get(f"{BASE}/users/{uid}/activity?limit=9999", headers=_h(READ)).status_code == 200
        assert client.get(f"{BASE}/users/{uuid.uuid4()}/activity", headers=_h(READ)).status_code == 404


@needs_db
class TestActions:
    def _roundtrip(self, client, uid, action, params, field_of_view):
        """dry-run changes nothing; apply changes it; restore puts it back."""
        start = field_of_view(_view(client, uid))
        r = _act(client, uid, action, params)                       # dry_run omitted = dry run
        assert r.status_code == 200, r.text
        dry = r.json()
        assert dry["dry_run"] is True and dry["changed"] is True
        assert field_of_view(_view(client, uid)) == start
        r = _act(client, uid, action, params, dry_run=False)
        assert r.status_code == 200, r.text
        applied = r.json()
        assert applied["dry_run"] is False and applied["changed"] is True
        assert applied["before"] == dry["before"]
        assert field_of_view(_view(client, uid)) != start
        r = _act(client, uid, "restore", {"before": applied["before"]}, dry_run=False)
        assert r.status_code == 200, r.text
        assert r.json()["after"] == applied["before"]
        assert field_of_view(_view(client, uid)) == start
        return dry, applied

    def test_add_credits(self, client, clean, capsys):
        uid = _run(_seed(topups=[(60, 10)]))
        dry, applied = self._roundtrip(client, uid, "add_credits", {"amount": 200},
                                       lambda v: v["credits"])
        assert dry["before"] == {"credits": 50} and dry["after"] == {"credits": 250}
        out = capsys.readouterr().out
        audit = [json.loads(ln.split("support-ops audit ", 1)[1]) for ln in out.splitlines()
                 if ln.startswith("support-ops audit ")]
        assert [a["action"] for a in audit] == ["add_credits", "restore"]
        assert audit[0]["ticket"] == "T-1" and audit[0]["actor"] == "tests"
        assert audit[0]["user_id"] == str(uid) and audit[0]["after"] == {"credits": 250}

    def test_negative_credits_never_below_zero(self, client, clean):
        uid = _run(_seed(topups=[(60, 10), (200, 0)]))
        r = _act(client, uid, "add_credits", {"amount": -100.5}, dry_run=False)
        assert r.json()["after"] == {"credits": 149.5}
        assert _view(client, uid)["credits"] == 149.5
        r = _act(client, uid, "add_credits", {"amount": -150}, dry_run=False)
        assert r.status_code == 400
        assert _view(client, uid)["credits"] == 149.5

    def test_fractional_restore_is_exact(self, client, clean):
        uid = _run(_seed(topups=[(60, Decimal("12.25"))]))
        before = _act(client, uid, "add_credits", {"amount": -47.75}, dry_run=False).json()["before"]
        assert _view(client, uid)["credits"] == 0
        _act(client, uid, "restore", {"before": before}, dry_run=False)
        assert _view(client, uid)["credits"] == 47.75

    def test_reset_usage(self, client, clean):
        uid = _run(_seed(sub={"plan": "pro", "minutes": 750}, used=120))
        dry, _ = self._roundtrip(client, uid, "reset_usage", {}, lambda v: v["usage"]["used"])
        assert dry["before"]["usage"]["used"] == 120 and dry["after"]["usage"]["used"] == 0
        # Nothing to reset: changed false.
        _act(client, uid, "reset_usage", {}, dry_run=False)
        r = _act(client, uid, "reset_usage", {}, dry_run=False)
        assert r.json()["changed"] is False

    def test_usage_restore_refuses_another_period(self, client, clean):
        uid = _run(_seed(sub={"plan": "pro", "minutes": 750}, used=10))
        r = _act(client, uid, "restore", {"before": {"usage": {
            "used": 99, "period_end": "2001-01-01T00:00:00+00:00"}}}, dry_run=False)
        assert r.status_code == 400

    def test_change_email(self, client, clean):
        uid = _run(_seed("old@example.com"))
        _run(_seed("taken@example.com"))
        r = _act(client, uid, "change_email", {"new_email": "Taken@Example.com"})
        assert r.status_code == 409
        dry, applied = self._roundtrip(client, uid, "change_email", {"new_email": "New@Example.com"},
                                       lambda v: v["email"])
        assert applied["after"] == {"email": "new@example.com"}

    def test_ban_and_unban(self, client, clean):
        uid = _run(_seed())
        dry, applied = self._roundtrip(client, uid, "ban", {"reason": "chargeback fraud"},
                                       lambda v: (v["banned"], v["ban_reason"]))
        assert applied["after"]["ban_reason"] == "chargeback fraud"
        _act(client, uid, "ban", {"reason": "spam"}, dry_run=False)
        assert _view(client, uid)["banned"] is True
        r = _act(client, uid, "unban", {}, dry_run=False)
        assert r.json()["after"] == {"ban_reason": None, "banned_at": None}
        assert _act(client, uid, "unban", {}, dry_run=False).json()["changed"] is False

    def test_update_field(self, client, clean):
        uid = _run(_seed(free_plan_denied="disposable_mx"))
        dry, applied = self._roundtrip(client, uid, "update_field",
                                       {"field": "free_plan_denied", "value": None},
                                       lambda v: (v["plan"], v["extra"]["free_plan_denied"]))
        assert applied["before"] == {"free_plan_denied": "disposable_mx"}
        self._roundtrip(client, uid, "update_field", {"field": "marketing_opt_out", "value": True},
                        lambda v: v["extra"]["marketing_opt_out"])

    def test_set_plan_needs_a_subscription_id_without_a_row(self, client, clean):
        uid = _run(_seed())
        r = _act(client, uid, "set_plan", {"plan": "pro"})
        assert r.status_code == 400 and "stripe_subscription_id" in r.json()["detail"]

    def test_set_plan_creates_the_missing_row_and_restore_removes_it(self, client, clean, plan_hooks):
        uid = _run(_seed())
        until = (NOW + timedelta(days=20)).isoformat()
        dry, applied = self._roundtrip(
            client, uid, "set_plan", {"plan": "pro", "stripe_subscription_id": "sub_missed",
                                      "until": until},
            lambda v: (v["plan"], v["usage"]["limit"], v["stripe_subscription_id"]))
        assert dry["before"] == {"subscription": None}
        assert applied["after"]["subscription"]["plan"] == "pro"
        assert applied["after"]["subscription"]["minutes_per_period"] == 750
        assert plan_hooks == [(str(uid), "pro", True)]     # only on the real write

    def test_set_plan_changes_a_live_row(self, client, clean, plan_hooks):
        uid = _run(_seed(sub={"plan": "starter", "minutes": 100}))
        dry, applied = self._roundtrip(client, uid, "set_plan", {"plan": "creator"},
                                       lambda v: (v["plan"], v["usage"]["limit"]))
        assert applied["after"]["subscription"]["minutes_per_period"] == 300
        assert plan_hooks[0] == (str(uid), "creator", False)

    def test_set_plan_free_cancels_app_side(self, client, clean, plan_hooks):
        uid = _run(_seed(sub={"plan": "pro", "minutes": 750}))
        dry, applied = self._roundtrip(client, uid, "set_plan", {"plan": "free"},
                                       lambda v: (v["plan"], v["plan_status"]))
        assert applied["after"]["subscription"]["status"] == "canceled"

    def test_set_plan_refuses_another_accounts_subscription(self, client, clean):
        _run(_seed("other@example.com", sub={"id": "sub_theirs"}))
        uid = _run(_seed())
        r = _act(client, uid, "set_plan", {"plan": "pro", "stripe_subscription_id": "sub_theirs"})
        assert r.status_code == 409

    def test_unknown_user_is_404(self, client, clean):
        assert _act(client, uuid.uuid4(), "unban", {}).status_code == 404

    def test_never_touches_other_users(self, client, clean):
        uid = _run(_seed(topups=[(60, 0)]))
        other = _run(_seed("bystander@example.com", topups=[(60, 0)], used=3))
        before = _view(client, other)
        for action, params in (("add_credits", {"amount": 5}), ("reset_usage", {}),
                               ("ban", {"reason": "x"}),
                               ("update_field", {"field": "marketing_opt_out", "value": True})):
            assert _act(client, uid, action, params, dry_run=False).status_code == 200
        assert _view(client, other) == before


@needs_db
class TestBanBlocksAccess:
    def test_banned_accounts_authenticate_as_nobody(self, client, clean, monkeypatch):
        from fastapi import HTTPException
        from cloud import auth, database
        monkeypatch.setenv("JWT_SECRET", "test-secret-" + "x" * 32)
        uid = _run(_seed())
        token = auth.issue_jwt(uid, "alice@example.com")

        class _Req:
            headers = {"Authorization": f"Bearer {token}"}
            method, url = "GET", type("U", (), {"path": "/api/me"})()

        async def who():
            return await auth.get_current_user_optional(_Req())
        assert str(_run(who()).id) == str(uid)
        _act(client, uid, "ban", {"reason": "abuse"}, dry_run=False)
        with pytest.raises(HTTPException) as e:
            _run(who())
        assert e.value.status_code == 403

        async def load():
            async with database.session() as s:
                return await auth._load_current_user(s, uid)
        assert _run(load()) is None                  # Autopilot and friends see nobody
        _act(client, uid, "unban", {}, dry_run=False)
        assert str(_run(who()).id) == str(uid)

    def test_magic_link_sign_in_is_refused(self, clean, monkeypatch):
        import hashlib
        from cloud import auth, database
        from cloud.models import MagicLinkToken
        monkeypatch.setenv("JWT_SECRET", "test-secret-" + "x" * 32)
        _run(_seed("banned@example.com", ban_reason="abuse", banned_at=NOW))

        async def add():
            async with database.session() as s:
                async with s.begin():
                    s.add(MagicLinkToken(email="banned@example.com",
                                         token_hash=hashlib.sha256(b"raw-token").hexdigest(),
                                         expires_at=NOW + timedelta(hours=1)))
        _run(add())
        app = FastAPI()
        app.include_router(auth.router)
        r = TestClient(app).post("/api/auth/magic-link/verify", json={"token": "raw-token"})
        assert r.status_code == 403


@needs_db
class TestDeleteAccount:
    @pytest.fixture(autouse=True)
    def quiet(self, monkeypatch):
        from cloud import alerts, analytics, emails
        self.goodbyes = []

        async def _noop(*a, **k):
            return None

        async def _goodbye(email):
            self.goodbyes.append(email)
        monkeypatch.setattr(alerts, "send_telegram", _noop)
        monkeypatch.setattr(analytics, "track", lambda *a, **k: None)
        monkeypatch.setattr(emails, "send_account_deleted_email", _goodbye)

    def test_confirmation_must_match(self, client, clean):
        uid = _run(_seed("bye@example.com"))
        r = _act(client, uid, "delete_account", {"confirm_email": "other@example.com"})
        assert r.status_code == 400

    def test_a_live_subscription_row_blocks_it(self, client, clean):
        uid = _run(_seed("bye@example.com", sub={"plan": "pro"}))
        r = _act(client, uid, "delete_account", {"confirm_email": "bye@example.com"})
        assert r.status_code == 409

    def test_dry_run_then_erase(self, client, clean):
        from cloud import account, database
        from cloud.models import AccountDeletion
        from sqlalchemy import select
        uid = _run(_seed("bye@example.com", topups=[(60, 0)], customer="cus_bye",
                         sub={"plan": "pro", "status": "canceled"}))
        r = _act(client, uid, "delete_account", {"confirm_email": "Bye@Example.com"})
        assert r.status_code == 200 and r.json()["after"] == {"account": None}
        assert _run(_user(uid)) is not None
        r = _act(client, uid, "delete_account", {"confirm_email": "bye@example.com"}, dry_run=False)
        assert r.status_code == 200, r.text
        assert r.json()["changed"] is True
        assert _run(_user(uid)) is None
        assert self.goodbyes == []                     # never mails the customer

        async def record():
            async with database.session() as s:
                return (await s.execute(select(AccountDeletion))).scalars().all()
        (rec,) = _run(record())
        assert rec.former_user_id == str(uid)
        assert rec.email_sha256 == account.email_fingerprint("bye@example.com")
        assert rec.stripe_customer_id == "cus_bye"
        assert _view(client, str(uid))["deleted"] is True
        assert _act(client, uid, "unban", {}).status_code == 404
