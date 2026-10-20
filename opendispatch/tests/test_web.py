"""Web UI smoke tests — HTML routes + HTMX fragments."""

from __future__ import annotations

import importlib

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


def test_dashboard_html(client):
    r = client.get("/", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "open-dispatch" in r.text
    assert "Queue" in r.text


def test_dashboard_filter_status(client):
    r = client.get("/?status=queued", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert "queued" in r.text


def test_compose_page_renders(client):
    r = client.get("/compose", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert "Compose" in r.text
    # Every adapter platform shows up as a checkbox
    for p in ["telegram", "twitter", "instagram", "bluesky", "linkedin"]:
        assert f'value="{p}"' in r.text


def test_compose_submit_creates_row(client):
    r = client.post(
        "/_compose",
        data={"text": "hello world", "platforms": ["telegram"]},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    assert "Dispatched" in r.text or "dispatched" in r.text.lower()
    # And it should show up in the queue
    rows = client.get("/queue").json()["rows"]
    assert any(r2["platform"] == "telegram:default" for r2 in rows)


def test_compose_submit_discord_and_facebook(client):
    """The composer must produce usable default formats for the two newest adapters."""
    r = client.post(
        "/_compose",
        data={"text": "hello world", "platforms": ["discord", "facebook"]},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    assert "Dispatched" in r.text or "dispatched" in r.text.lower()
    rows = client.get("/queue").json()["rows"]
    by_platform = {r2["platform"] for r2 in rows}
    assert "discord:default" in by_platform
    assert "facebook:default" in by_platform
    # The enqueued units must carry the correct default format payloads
    for r2 in rows:
        if r2["platform"] == "discord:default":
            assert r2["unit"]["formats"]["discord_message"]["content"] == "hello world"
        if r2["platform"] == "facebook:default":
            assert r2["unit"]["formats"]["facebook_post"]["text"] == "hello world"


def test_compose_submit_validation_error(client):
    r = client.post(
        "/_compose",
        data={"text": "", "platforms": []},
        headers={"Accept": "text/html"},
    )
    assert r.status_code == 200
    assert "Failed" in r.text or "failed" in r.text.lower()


def test_queue_fragment_returns_html(client):
    r = client.get("/_queue-fragment", headers={"Accept": "text/html"})
    assert r.status_code == 200
    # Fragment is just the table (or empty state), no <html> wrapper
    assert "<html" not in r.text.lower()


def test_queue_id_content_negotiation(client):
    # Seed a row
    submitted = client.post(
        "/_compose",
        data={"text": "neg test", "platforms": ["telegram"]},
        headers={"Accept": "text/html"},
    )
    assert submitted.status_code == 200
    rows = client.get("/queue").json()["rows"]
    row_id = rows[0]["id"]

    # JSON consumer → JSON
    r_json = client.get(f"/queue/{row_id}", headers={"Accept": "application/json"})
    assert r_json.status_code == 200
    assert r_json.json()["id"] == row_id

    # Browser → HTML
    r_html = client.get(f"/queue/{row_id}", headers={"Accept": "text/html"})
    assert r_html.status_code == 200
    assert "text/html" in r_html.headers["content-type"]
    assert row_id[:8] in r_html.text


def test_retry_browser_returns_fragment(client):
    # Seed and fail one row, then retry as a browser
    submitted = client.post(
        "/_compose",
        data={"text": "retry test", "platforms": ["telegram"]},
        headers={"Accept": "text/html"},
    )
    row_id = client.get("/queue").json()["rows"][0]["id"]
    r = client.post(f"/queue/{row_id}/retry", headers={"Accept": "text/html"})
    assert r.status_code == 200
    # Fragment of the queue table
    assert "<html" not in r.text.lower()


def test_retry_api_returns_json(client):
    submitted = client.post(
        "/_compose",
        data={"text": "retry api", "platforms": ["telegram"]},
        headers={"Accept": "text/html"},
    )
    row_id = client.get("/queue").json()["rows"][0]["id"]
    r = client.post(f"/queue/{row_id}/retry", headers={"Accept": "application/json"})
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
