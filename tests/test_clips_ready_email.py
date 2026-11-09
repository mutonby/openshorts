"""The clips-ready email goes to dashboard jobs only (``app._notify_clips_ready``).

A job sent with an osk_ key (REST, MCP, OAuth clients) is followed by polling
or the webhook, so mailing its owner on every run is noise."""
import asyncio
import os
import sys
from types import SimpleNamespace

import app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _req(headers):
    return SimpleNamespace(headers=headers)


def test_api_key_detection():
    assert app.request_uses_api_key(_req({"Authorization": "Bearer osk_abc"}))
    assert app.request_uses_api_key(_req({"X-API-Key": "osk_abc"}))
    assert not app.request_uses_api_key(_req({"Authorization": "Bearer eyJhbGciOi"}))
    assert not app.request_uses_api_key(_req({}))


def test_api_job_sends_no_email(monkeypatch):
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    job = {"user_id": "u1", "status": "completed", "via_api_key": True,
           "result": {"clips": [{"title": "t"}]}}
    monkeypatch.setitem(app.jobs, "job-api", job)
    asyncio.run(app._notify_clips_ready("job-api"))
    assert not job.get("email_sent")


def _fake_cloud(monkeypatch, sent):
    """Stand-ins for the cloud modules _notify_clips_ready imports lazily, so
    the test runs without the database stack."""
    from contextlib import asynccontextmanager

    class _Session:
        async def get(self, _model, _uid):
            return SimpleNamespace(email="owner@example.com")

    @asynccontextmanager
    async def session():
        yield _Session()

    async def send_clips_ready_email(email, title, count, url):
        sent.append({"email": email, "title": title, "count": count, "url": url})

    monkeypatch.setitem(sys.modules, "cloud.database", SimpleNamespace(session=session))
    monkeypatch.setitem(sys.modules, "cloud.models", SimpleNamespace(User=object))
    monkeypatch.setitem(sys.modules, "cloud.emails",
                        SimpleNamespace(send_clips_ready_email=send_clips_ready_email))


def test_email_button_opens_history(monkeypatch):
    """"View my clips" must land on History, which lists the clips from any
    device. Plain #app opened the Clip Generator, which only shows a finished
    job in the browser that ran it (support: "I don't see them")."""
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "_cloud_config", SimpleNamespace(
        settings=SimpleNamespace(frontend_url="https://app.example")))
    sent = []
    _fake_cloud(monkeypatch, sent)
    job = {"user_id": "u1", "status": "completed",
           "result": {"clips": [{"title": "t"}] * 6}}
    monkeypatch.setitem(app.jobs, "job-dash", job)
    asyncio.run(app._notify_clips_ready("job-dash"))
    assert len(sent) == 1 and sent[0]["count"] == 6
    assert sent[0]["url"] == "https://app.example/#app?tab=history"


def test_history_deep_link_survives_sign_in():
    """The dashboard side of the same link: signed out, History asks to sign in
    and the sign-in comes back to History, not to the generator/tutorial."""
    def read(path):
        with open(os.path.join(REPO, "dashboard/src", path)) as f:
            return f.read()

    app_jsx = read("App.jsx")
    auth = read("contexts/AuthContext.jsx")
    lib = read("lib/tabAfterLogin.js")
    assert "'history'" in app_jsx.split("const DEEP_LINK_TABS", 1)[1].split("\n", 1)[0]
    assert "rememberTabAfterLogin(activeTab)" in app_jsx
    assert "'history'" in lib.split("const TABS", 1)[1].split("\n", 1)[0]
    assert "takeTabAfterLogin()" in auth
    assert "#app?tab=${afterLoginTab}" in auth
    assert "!afterLoginTab" in auth  # no first-run tutorial on top of History
