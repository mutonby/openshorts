"""OAuth 2.1 for the MCP endpoint: discovery documents, dynamic registration
input rules, PKCE and redirect building. The database-backed pieces
(register/authorize/token) are exercised against prod after deploy; what is
pinned here is every rule a client can trip over without a database."""
import base64
import hashlib
import secrets

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cloud import mcp_oauth


class TestHelpers:
    def test_redirect_uri_rules(self):
        ok = ["https://claude.ai/api/mcp/auth_callback", "https://chatgpt.com/connector_platform_oauth_redirect",
              "http://localhost:3334/callback", "http://127.0.0.1:8080/cb"]
        bad = ["http://evil.com/cb", "https://x.com/cb#frag", "ftp://x/cb", "not a url", ""]
        assert all(mcp_oauth.valid_redirect_uri(u) for u in ok)
        assert not any(mcp_oauth.valid_redirect_uri(u) for u in bad)

    def test_pkce_s256(self):
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        assert mcp_oauth.pkce_matches(verifier, challenge)
        assert not mcp_oauth.pkce_matches(verifier + "x", challenge)
        assert not mcp_oauth.pkce_matches("short", challenge)   # < 43 chars
        assert not mcp_oauth.pkce_matches("", challenge)

    def test_add_query_keeps_existing_params_and_drops_none(self):
        out = mcp_oauth.add_query("https://c.ai/cb?a=1", code="xyz", state=None)
        assert out == "https://c.ai/cb?a=1&code=xyz"

    def test_consent_url_targets_the_hash_router(self):
        url = mcp_oauth.consent_url("https://www.openshorts.app/", {"client_id": "abc", "state": "s"})
        assert url.startswith("https://www.openshorts.app/#/oauth/authorize?")
        assert "client_id=abc" in url and "state=s" in url

    def test_metadata_points_at_the_same_host(self):
        m = mcp_oauth.server_metadata("https://mcp.openshorts.app")
        assert m["issuer"] == "https://mcp.openshorts.app"
        assert m["token_endpoint"] == "https://mcp.openshorts.app/oauth/token"
        assert m["code_challenge_methods_supported"] == ["S256"]
        assert m["token_endpoint_auth_methods_supported"] == ["none"]
        r = mcp_oauth.resource_metadata("https://mcp.openshorts.app")
        assert r["resource"] == "https://mcp.openshorts.app/mcp"
        assert r["authorization_servers"] == ["https://mcp.openshorts.app"]


class TestDiscoveryEndpoints:
    def _client(self):
        # The router is mounted by cloud.setup_sync in cloud mode only; test it
        # on a bare app so the checks don't depend on BILLING_ENABLED.
        app = FastAPI()
        app.include_router(mcp_oauth.router)
        return TestClient(app)

    def test_well_known_documents_are_served(self):
        c = self._client()
        for path in ("/.well-known/oauth-protected-resource",
                     "/.well-known/oauth-protected-resource/mcp",
                     "/.well-known/oauth-authorization-server",
                     "/.well-known/oauth-authorization-server/mcp"):
            r = c.get(path)
            assert r.status_code == 200, path
            assert r.headers.get("access-control-allow-origin") == "*"
            body = r.json()
            assert ("resource" in body) or ("issuer" in body)

    def test_openai_apps_challenge(self, monkeypatch):
        c = self._client()
        monkeypatch.delenv("OPENAI_APPS_CHALLENGE", raising=False)
        assert c.get("/.well-known/openai-apps-challenge").status_code == 404
        monkeypatch.setenv("OPENAI_APPS_CHALLENGE", " tok123 ")
        r = c.get("/.well-known/openai-apps-challenge")
        assert r.status_code == 200 and r.text == "tok123"
        assert r.headers["content-type"].startswith("text/plain")

    def test_register_rejects_bad_redirects_and_secrets(self):
        c = self._client()
        r = c.post("/oauth/register", json={"client_name": "x", "redirect_uris": ["http://evil.com/cb"]})
        assert r.status_code == 400 and r.json()["error"] == "invalid_redirect_uri"
        r = c.post("/oauth/register", json={"redirect_uris": ["https://ok.com/cb"],
                                           "token_endpoint_auth_method": "client_secret_basic"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_client_metadata"
        r = c.post("/oauth/register", json={"redirect_uris": []})
        assert r.status_code == 400

    def test_token_rejects_other_grants(self):
        r = self._client().post("/oauth/token", data={"grant_type": "client_credentials"})
        assert r.status_code == 400 and r.json()["error"] == "unsupported_grant_type"


class TestConsentDecisions:
    """The consent page's own URL is attacker-controlled (anyone can link to
    it): Cancel must go where the API says, and only a session may consent."""

    def _client(self, monkeypatch, redirect_uris=("https://claude.ai/api/mcp/auth_callback",)):
        import json as _json
        from types import SimpleNamespace

        async def _load(client_id):
            if client_id != "c1":
                return None
            return SimpleNamespace(id="c1", client_name="Claude",
                                   redirect_uris=_json.dumps(list(redirect_uris)))
        monkeypatch.setattr(mcp_oauth, "_load_client", _load)
        app = FastAPI()
        app.include_router(mcp_oauth.router)
        return TestClient(app)

    def test_deny_redirects_only_to_a_registered_uri(self, monkeypatch):
        c = self._client(monkeypatch)
        r = c.post("/api/oauth/deny", json={"client_id": "c1", "state": "s",
                                            "redirect_uri": "https://claude.ai/api/mcp/auth_callback"})
        assert r.status_code == 200
        assert r.json()["redirect"] == "https://claude.ai/api/mcp/auth_callback?error=access_denied&state=s"
        for bad in ("javascript:alert(document.domain)//", "https://evil.example/cb"):
            r = c.post("/api/oauth/deny", json={"client_id": "c1", "redirect_uri": bad})
            assert r.status_code == 400
        r = c.post("/api/oauth/deny", json={"client_id": "nope",
                                            "redirect_uri": "https://claude.ai/api/mcp/auth_callback"})
        assert r.status_code == 400

    def test_client_info_shows_where_access_goes(self, monkeypatch):
        c = self._client(monkeypatch)
        body = c.get("/api/oauth/client/c1").json()
        assert body["redirect_hosts"] == ["claude.ai"]

    def test_an_api_key_cannot_consent(self, monkeypatch):
        c = self._client(monkeypatch)
        r = c.post("/api/oauth/authorize",
                   headers={"Authorization": "Bearer osk_" + "a" * 40},
                   json={"client_id": "c1", "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
                         "code_challenge": "x" * 43})
        assert r.status_code == 403


class _FakeSession:
    """Answers each ``execute`` with the next queued row, in order."""

    def __init__(self, rows):
        self.rows, self.added, self.committed = list(rows), [], False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, _stmt):
        from types import SimpleNamespace
        row = self.rows.pop(0) if self.rows else None
        return SimpleNamespace(scalar_one_or_none=lambda: row)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


class TestRefreshAndRevoke:
    """Gemini's connected apps want refresh tokens that never rotate
    ("persistent linking") and a revocation endpoint for unlinking."""

    def _client(self, monkeypatch, rows):
        sess = _FakeSession(rows)
        monkeypatch.setattr(mcp_oauth.database, "session", lambda: sess)
        app = FastAPI()
        app.include_router(mcp_oauth.router)
        return TestClient(app), sess

    def _key(self, **kw):
        from types import SimpleNamespace
        base = dict(id="k1", key_hash="old", prefix="osk_old", revoked_at=None,
                    scope="openshorts", name="Gemini · connected via OAuth")
        base.update(kw)
        return SimpleNamespace(**base)

    def test_metadata_advertises_refresh_and_revocation(self):
        m = mcp_oauth.server_metadata("https://mcp.openshorts.app")
        assert m["grant_types_supported"] == ["authorization_code", "refresh_token"]
        assert m["revocation_endpoint"] == "https://mcp.openshorts.app/oauth/revoke"
        assert "code" in m["response_types_supported"]
        assert m["registration_endpoint"].endswith("/oauth/register")

    def test_refresh_token_is_never_a_bearer_key(self):
        from cloud import api_keys
        assert not api_keys.looks_like_key(mcp_oauth.new_refresh_token())

    def test_unknown_refresh_token_is_invalid_grant(self, monkeypatch):
        c, _ = self._client(monkeypatch, [None])
        r = c.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": "osr_x"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"

    def test_refresh_from_another_client_is_refused(self, monkeypatch):
        from types import SimpleNamespace
        rt = SimpleNamespace(client_id="c1", api_key_id="k1", last_used_at=None)
        c, _ = self._client(monkeypatch, [rt])
        r = c.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": "osr_x",
                                         "client_id": "other"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"

    def test_refresh_on_a_revoked_key_is_refused(self, monkeypatch):
        from datetime import datetime, timezone
        from types import SimpleNamespace
        rt = SimpleNamespace(client_id="c1", api_key_id="k1", last_used_at=None)
        key = self._key(revoked_at=datetime.now(timezone.utc))
        c, _ = self._client(monkeypatch, [rt, key])
        r = c.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": "osr_x",
                                         "client_id": "c1"})
        assert r.status_code == 400 and r.json()["error"] == "invalid_grant"

    def test_refresh_swaps_the_secret_and_keeps_the_refresh_token(self, monkeypatch):
        from types import SimpleNamespace
        from cloud import api_keys
        rt = SimpleNamespace(client_id="c1", api_key_id="k1", last_used_at=None)
        key = self._key()
        c, sess = self._client(monkeypatch, [rt, key])
        r = c.post("/oauth/token", data={"grant_type": "refresh_token", "refresh_token": "osr_keep",
                                         "client_id": "c1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["refresh_token"] == "osr_keep"
        assert body["access_token"].startswith("osk_")
        assert key.key_hash == api_keys.hash_key(body["access_token"]) != "old"
        assert body["scope"] == "openshorts" and sess.committed
        assert r.headers["cache-control"] == "no-store"

    def test_revoke_is_always_200(self, monkeypatch):
        c, _ = self._client(monkeypatch, [None])
        assert c.post("/oauth/revoke", data={}).status_code == 200
        assert c.post("/oauth/revoke", data={"token": "osk_" + "z" * 40}).status_code == 200

    def test_revoke_disconnects_an_oauth_key(self, monkeypatch):
        key = self._key()
        c, sess = self._client(monkeypatch, [key])
        assert c.post("/oauth/revoke", data={"token": "osk_" + "z" * 40}).status_code == 200
        assert key.revoked_at is not None and sess.committed

    def test_revoke_by_refresh_token(self, monkeypatch):
        from types import SimpleNamespace
        key = self._key()
        rt = SimpleNamespace(client_id="c1", api_key_id="k1")
        c, _ = self._client(monkeypatch, [rt, key])
        assert c.post("/oauth/revoke", data={"token": "osr_x"}).status_code == 200
        assert key.revoked_at is not None

    def test_revoke_never_touches_a_hand_made_key(self, monkeypatch):
        key = self._key(scope=None, name="my laptop")
        c, _ = self._client(monkeypatch, [key])
        assert c.post("/oauth/revoke", data={"token": "osk_" + "z" * 40}).status_code == 200
        assert key.revoked_at is None
