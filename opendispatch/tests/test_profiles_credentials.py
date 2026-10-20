"""Profile credentials must be clearable through the edit form.

The form renders every credential input with value="" (secrets are never sent
to the browser), and _profile_from_form only overwrites a field when the
submitted value is non-empty. So "empty" is overloaded: it means both
"leave unchanged" and "clear this credential" — and the second is
unreachable. A rotated-then-leaked token can never be removed through the
UI; the only way to drop it is editing profiles.json by hand.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from opendispatch.profiles import Profile, ProfileStore


class Form(dict):
    """Minimal stand-in for a submitted HTML form."""

    def get(self, key: str, default: Any = None) -> Any:
        return dict.get(self, key, default)


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileStore:
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    return ProfileStore()


def _from_form(form: dict, profile_id: str | None = None) -> Profile:
    from opendispatch.api.app import _profile_from_form

    return _profile_from_form(Form(form), profile_id=profile_id)


def test_omitted_field_keeps_the_existing_secret(store: ProfileStore) -> None:
    p = Profile(name="work", platforms={"twitter": {"api_key": "K", "api_secret": "S"}})
    store.save(p)

    updated = _from_form(
        {"id": p.id, "name": "work", "twitter__api_key": "K2"}, profile_id=p.id
    )
    # Not submitted at all -> preserved.
    assert updated.platforms["twitter"]["api_secret"] == "S"
    assert updated.platforms["twitter"]["api_key"] == "K2"


def test_explicit_clear_marker_removes_the_secret(store: ProfileStore) -> None:
    p = Profile(name="work", platforms={"twitter": {"api_key": "K", "api_secret": "S"}})
    store.save(p)

    # The form sends "<platform>__<field>__clear" when the user ticks
    # "remove this credential".
    updated = _from_form(
        {"id": p.id, "name": "work",
         "twitter__api_key": "K",
         "twitter__api_secret__clear": "1"},
        profile_id=p.id,
    )
    assert "api_secret" not in updated.platforms.get("twitter", {})


def test_clear_marker_does_not_erase_other_fields(store: ProfileStore) -> None:
    p = Profile(name="work", platforms={
        "twitter": {"api_key": "K", "api_secret": "S", "access_token": "AT"},
    })
    store.save(p)

    updated = _from_form(
        {"id": p.id, "name": "work", "twitter__api_secret__clear": "1"}, profile_id=p.id
    )
    tw = updated.platforms["twitter"]
    assert "api_secret" not in tw
    assert tw["api_key"] == "K"
    assert tw["access_token"] == "AT"


def test_clearing_every_field_drops_the_platform(store: ProfileStore) -> None:
    p = Profile(name="work", platforms={"twitter": {"api_key": "K", "api_secret": "S"}})
    store.save(p)

    updated = _from_form(
        {"id": p.id, "name": "work",
         "twitter__api_key__clear": "1", "twitter__api_secret__clear": "1"},
        profile_id=p.id,
    )
    assert "twitter" not in updated.platforms


def test_rendered_form_leaks_no_stored_secret(tmp_path, monkeypatch) -> None:
    """Render the real template and prove no stored value reaches the HTML.

    Static source inspection is not enough: a Jinja expression anywhere in the
    template could interpolate a secret into value="". So render it with a
    profile that has real-looking secrets and assert none of them appear.
    """
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    from starlette.requests import Request

    from opendispatch.api.app import _profile_ctx, templates

    p = Profile(name="work", platforms={
        "twitter": {"api_key": "AKIAEXAMPLE", "api_secret": "SUPERSECRETVALUE"},
        "tiktok": {"access_token": "TIKTOKSECRETVALUE"},
        "discord": {"webhook_url": "https://discord.com/api/webhooks/SECRETPATH"},
    })
    ProfileStore().save(p)

    resp = templates.TemplateResponse(
        Request({"type": "http", "method": "GET", "path": "/",
                 "headers": [], "query_string": b""}),
        "profile_form.html", {**_profile_ctx("0.4.0", "now"), "profile": p},
    )
    body = resp.body.decode()

    for secret in ("AKIAEXAMPLE", "SUPERSECRETVALUE", "TIKTOKSECRETVALUE", "SECRETPATH"):
        assert secret not in body, f"{secret!r} was rendered into the page"

    # The clear affordance must exist, otherwise the __clear marker is
    # unreachable from the UI.
    assert 'name="twitter__api_secret__clear"' in body
    # Every credential platform must be reachable in the form.
    for platform in ("twitter", "bluesky", "telegram", "instagram", "linkedin",
                     "threads", "youtube", "tiktok", "facebook", "discord"):
        assert f"{platform}__" in body, f"{platform} missing from the credential form"


def test_profiles_json_endpoint_returns_no_secrets(store: ProfileStore) -> None:
    store.save(Profile(name="work", platforms={
        "twitter": {"api_key": "SECRET_KEY", "api_secret": "SECRET_SECRET"},
    }))
    payload = [
        {"id": p.id, "name": p.name, "emoji": p.emoji,
         "color": p.color, "configured": p.configured_platforms()}
        for p in store.list()
    ]
    blob = str(payload)
    assert "SECRET_KEY" not in blob
    assert "SECRET_SECRET" not in blob


def test_stored_file_is_owner_only(store: ProfileStore) -> None:
    p = Profile(name="work", platforms={"twitter": {"api_secret": "S"}})
    store.save(p)
    mode = store._path.stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)


def test_env_injection_restores_previous_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from opendispatch.profiles import profile_env

    monkeypatch.setenv("TWITTER_API_SECRET", "ORIGINAL")
    p = Profile(name="x", platforms={"twitter": {"api_secret": "PROFILE"}})
    with profile_env(p):
        assert os.environ["TWITTER_API_SECRET"] == "PROFILE"
    assert os.environ["TWITTER_API_SECRET"] == "ORIGINAL"


def test_env_injection_removes_vars_that_were_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from opendispatch.profiles import profile_env

    monkeypatch.delenv("TWITTER_API_SECRET", raising=False)
    p = Profile(name="x", platforms={"twitter": {"api_secret": "PROFILE"}})
    with profile_env(p):
        assert os.environ["TWITTER_API_SECRET"] == "PROFILE"
    assert "TWITTER_API_SECRET" not in os.environ
