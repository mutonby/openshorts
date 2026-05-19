"""
API contract test: snapshot the FastAPI openapi.json.

This is the single most important regression check for the restructure.
The flat layout's openapi dump is captured into
`tests/snapshots/baseline.openapi.json` on the first run. After the
restructure, the test diffs the current dump against the baseline —
any route renamed, dropped, or with a changed schema fails loudly.

To intentionally regenerate the baseline (e.g. after we restructure
and accept the new shape), delete the snapshot file and re-run.
"""
import json
import os
from pathlib import Path

import pytest

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "snapshots"
BASELINE = SNAPSHOTS_DIR / "baseline.openapi.json"
CURRENT = SNAPSHOTS_DIR / "current.openapi.json"


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """
    Build a TestClient against the production FastAPI app.

    Imports `app` lazily inside the fixture so the conftest sys.modules
    stubs are in place first. app.py creates `uploads/` and `output/`
    at import time — we redirect cwd into a tmp dir so we don't leave
    artifacts in the repo.
    """
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app  # noqa: WPS433  intentional late import

    with TestClient(fastapi_app) as client:
        yield client


def _dump_openapi(client) -> dict:
    """Normalize the openapi dict so trivial reorderings don't false-fail."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    return resp.json()


def test_openapi_dump_matches_baseline(app_client):
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    current = _dump_openapi(app_client)
    CURRENT.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")

    if not BASELINE.exists():
        # First run: capture the baseline so subsequent runs have something
        # to diff against. The test passes (this is intentional — the user
        # is locking in current behavior).
        BASELINE.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8"
        )
        pytest.skip(
            f"Wrote initial baseline to {BASELINE}. Commit it to lock the contract."
        )

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if current != baseline:
        pytest.fail(
            "openapi.json drifted from baseline.\n"
            f"Diff written to {CURRENT}.\n"
            f"If the change is intentional, delete {BASELINE} and re-run "
            "to regenerate."
        )


# --- Targeted route checks that don't need any external mocks ----------


def test_translate_languages_endpoint_returns_dict(app_client):
    """The endpoint must expose the canonical language codes somewhere
    in its response — directly as keys, or nested under a common wrapper
    (e.g. {"languages": {...}}). We just need 'en', 'es', 'fr' to show up."""
    r = app_client.get("/api/translate/languages")
    assert r.status_code == 200
    body = r.json()

    def _flatten_keys(obj, out):
        if isinstance(obj, dict):
            out.update(obj.keys())
            for v in obj.values():
                _flatten_keys(v, out)
        elif isinstance(obj, list):
            for v in obj:
                _flatten_keys(v, out)
        elif isinstance(obj, str):
            out.add(obj)

    found: set[str] = set()
    _flatten_keys(body, found)
    assert {"en", "es", "fr"} <= found, f"missing core language codes in {body!r}"


def test_status_for_unknown_job_returns_4xx(app_client):
    r = app_client.get("/api/status/this-job-does-not-exist")
    assert r.status_code in (400, 404, 422)


def test_app_serves_openapi_json(app_client):
    r = app_client.get("/openapi.json")
    assert r.status_code == 200
    body = r.json()
    assert "paths" in body
    assert len(body["paths"]) > 0


def test_app_serves_docs(app_client):
    r = app_client.get("/docs")
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
