"""API smoke tests via FastAPI TestClient."""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    import opendispatch.api.queue as q
    importlib.reload(q)
    import opendispatch.api.app as appmod
    importlib.reload(appmod)
    from fastapi.testclient import TestClient
    return TestClient(appmod.app)


def test_profile_env_restores_environment_after_nested_contexts():
    from opendispatch.profiles import Profile, profile_env
    import os
    original = os.environ.get("TELEGRAM_BOT_TOKEN")
    os.environ.pop("TELEGRAM_BOT_TOKEN", None)
    try:
        with profile_env(Profile(platforms={"telegram": {"bot_token": "token-a"}})):
            assert os.environ["TELEGRAM_BOT_TOKEN"] == "token-a"
        assert "TELEGRAM_BOT_TOKEN" not in os.environ
    finally:
        if original is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = original


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_platforms_configured_covers_new_adapters(client, monkeypatch):
    """Health dashboard + composer dots must recognize tiktok/facebook/discord creds."""
    from opendispatch.api import app as appmod
    for var in ("TIKTOK_ACCESS_TOKEN", "FACEBOOK_PAGE_ID", "FACEBOOK_ACCESS_TOKEN",
                "DISCORD_WEBHOOK_URL", "TWITTER_API_KEY", "TWITTER_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("FACEBOOK_PAGE_ID", "123")
    monkeypatch.setenv("FACEBOOK_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/x")
    configured = appmod._platforms_configured()
    assert {"tiktok", "facebook", "discord"} <= configured
    assert "twitter" not in configured


def test_ai_adapt_malformed_json_returns_400(client):
    r = client.post("/ai/adapt", content=b"{", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "invalid JSON body" in r.text


def test_dispatch_validation_error(client):
    r = client.post("/dispatch", json={"targets": [], "formats": {}})
    assert r.status_code == 400


def test_dispatch_malformed_json_returns_400(client):
    r = client.post("/dispatch", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "invalid JSON body" in r.text


def test_dispatch_non_object_json_returns_400(client):
    r = client.post("/dispatch", content=b'["a list"]',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "must be an object" in r.text


def test_dispatch_rejects_private_webhook_and_traversal(client):
    body = {"targets": ["telegram"], "formats": {"telegram_message": {"text": "x"}},
            "webhook_url": "http://127.0.0.1:8000/secret"}
    assert client.post("/dispatch", json=body).status_code == 400
    body["webhook_url"] = "https://hooks.example.test/callback"
    body["formats"]["telegram_message"]["photo_path"] = "../../etc/passwd"
    assert client.post("/dispatch", json=body).status_code == 400


def test_dispatch_enqueues(client):
    body = {
        "targets": ["telegram:main"],
        "formats": {"telegram_message": {"text": "hi"}},
    }
    r = client.post("/dispatch", json=body)
    assert r.status_code == 202
    j = r.json()
    assert "unit_id" in j
    assert len(j["enqueued"]) == 1
    assert j["enqueued"][0]["target"] == "telegram:main"


def test_queue_list(client):
    body = {"targets": ["telegram"], "formats": {"telegram_message": {"text": "x"}}}
    client.post("/dispatch", json=body)
    r = client.get("/queue")
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_campaign_status_and_cancel(client):
    body = {
        "id": "campaign-api-1",
        "targets": ["telegram:main", "bluesky:main"],
        "formats": {
            "telegram_message": {"text": "hello"},
            "bluesky_post": {"text": "hello"},
        },
    }
    dispatch = client.post("/dispatch", json=body)
    assert dispatch.status_code == 202
    unit_id = dispatch.json()["unit_id"]

    status = client.get(f"/campaign/{unit_id}")
    assert status.status_code == 200
    assert status.json()["unit_id"] == unit_id
    assert status.json()["count"] == 2
    assert {r["status"] for r in status.json()["rows"]} == {"queued"}

    canceled = client.post(f"/campaign/{unit_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["unit_id"] == unit_id
    assert canceled.json()["canceled"] == 2
    assert {r["status"] for r in canceled.json()["rows"]} == {"canceled"}
    assert {r["status"] for r in client.get(f"/campaign/{unit_id}").json()["rows"]} == {"canceled"}


def test_campaign_not_found(client):
    r = client.get("/campaign/no-such-campaign")
    assert r.status_code == 404


def test_retry_does_not_revive_canceled_row(client):
    body = {"id": "campaign-retry-1", "targets": ["telegram"], "formats": {"telegram_message": {"text": "x"}}}
    unit_id = client.post("/dispatch", json=body).json()["unit_id"]
    row_id = client.get(f"/campaign/{unit_id}").json()["rows"][0]["id"]
    client.post(f"/campaign/{unit_id}/cancel")
    retry = client.post(f"/queue/{row_id}/retry")
    assert retry.status_code == 409
    assert client.get(f"/queue/{row_id}/json").json()["status"] == "canceled"


def test_delete_row(client):
    body = {"targets": ["telegram"], "formats": {"telegram_message": {"text": "bye"}}}
    r = client.post("/dispatch", json=body)
    row_id = r.json()["enqueued"][0]["id"]
    # Row exists
    assert client.get(f"/queue/{row_id}/json").status_code == 200
    # Delete it
    dr = client.delete(f"/queue/{row_id}")
    assert dr.status_code == 200
    assert dr.json()["deleted"] == row_id
    # Gone
    assert client.get(f"/queue/{row_id}/json").status_code == 404


def test_delete_nonexistent(client):
    r = client.delete("/queue/no-such-id")
    assert r.status_code == 404


def test_purge_published(client):
    body = {"targets": ["telegram"], "formats": {"telegram_message": {"text": "x"}}}
    r = client.post("/dispatch", json=body)
    row_id = r.json()["enqueued"][0]["id"]
    # Mark it published via direct queue manipulation
    from opendispatch.api.queue import get_queue
    get_queue().mark_published(row_id, "post-123")
    # Purge published
    pr = client.post("/_purge?status=published")
    assert pr.status_code == 200
    assert pr.json()["purged"] == 1
    assert pr.json()["status"] == "published"
    # Verify gone
    assert client.get(f"/queue/{row_id}/json").status_code == 404


def test_purge_rejects_queued(client):
    """Can't purge active rows — only published or dead."""
    r = client.post("/_purge?status=queued")
    assert r.status_code == 400


def test_bulk_import_wraps_plain_text_correctly(client):
    """Plain-text CSV cells must land in the field each adapter reads."""
    csv_body = (
        "targets,format_key,text_or_json,scheduled_for\n"
        "twitter,twitter_thread,hello tweeps,\n"
        "discord,discord_message,hello discord,\n"
        "facebook,facebook_post,hello fb,\n"
        "instagram,instagram_post,hello ig,\n"
    )
    r = client.post("/dispatch/bulk", content=csv_body,
                    headers={"Content-Type": "text/csv"})
    assert r.status_code == 202
    j = r.json()
    assert j["imported"] == 4, j.get("error_details")
    assert j["errors"] == 0
    rows = client.get("/queue").json()["rows"]
    fmt_by_platform = {r2["platform"]: r2["unit"]["formats"] for r2 in rows}
    assert fmt_by_platform["twitter:default"]["twitter_thread"]["tweets"] == ["hello tweeps"]
    assert fmt_by_platform["discord:default"]["discord_message"]["content"] == "hello discord"
    assert fmt_by_platform["facebook:default"]["facebook_post"]["text"] == "hello fb"
    assert fmt_by_platform["instagram:default"]["instagram_post"]["caption"] == "hello ig"


def test_bulk_import_rejects_bad_rows_without_blocking_good_ones(client):
    csv_body = (
        "twitter,twitter_thread,ok row,\n"
        "no-such-platform,x_key,bad row,\n"
    )
    r = client.post("/dispatch/bulk", content=csv_body,
                    headers={"Content-Type": "text/csv"})
    assert r.status_code == 202
    j = r.json()
    assert j["imported"] == 1
    assert j["errors"] == 1
