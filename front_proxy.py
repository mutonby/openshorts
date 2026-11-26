"""Routing and proxying of a front process (front.py runs it).

Kept apart from front.py so it imports nothing of the API (tests use it with
stand-in apps). See front.py and docs/architecture/multiprocess-api.md.
"""
import os
import re

import httpx

ENGINE_SOCKET = os.environ.get("ENGINE_SOCKET") or "/tmp/openshorts-engine.sock"
# Requests one front may have open on the engine at once. Each is a socket in
# both processes: fdlimit.py lifts the open-file limit well above
# API_FRONT_WORKERS times this.
ENGINE_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# (method, exact path). Each was checked to need none of the engine's memory.
_LOCAL_EXACT = frozenset({
    ("GET", "/health"),
    ("GET", "/api/config"),
    ("POST", "/api/media-token"),
    ("GET", "/api/social/user"),
    # cloud/auth.py, cloud/oauth.py: the database and signed cookies only.
    ("GET", "/api/me"),
    ("POST", "/api/auth/magic-link"),
    ("POST", "/api/auth/magic-link/verify"),
    ("POST", "/api/auth/exchange"),
    ("POST", "/api/auth/login-method"),
    ("POST", "/api/auth/review-login"),
    ("POST", "/api/auth/attribution"),
    ("GET", "/api/auth/google"),
    ("GET", "/api/auth/google/callback"),
    ("POST", "/api/onboarding/survey"),
    # cloud/billing.py: Stripe and the database. Not the webhook nor
    # end-trial: a plan change moves the user's queued jobs (engine memory).
    ("GET", "/api/billing/plans"),
    ("GET", "/api/billing/invoices"),
    ("POST", "/api/billing/checkout"),
    ("POST", "/api/billing/portal"),
})
# The remote worker's calls after its claim read and write only the job
# directory (remote_worker.py, file locks), as either container of a deploy
# already could. Not /claim (the queue) nor /source (one bandwidth budget).
_LOCAL_PATTERNS = (
    ("GET", re.compile(rf"^/internal/worker/jobs/{_UUID}$")),
    ("GET", re.compile(rf"^/internal/worker/jobs/{_UUID}/transcript$")),
    ("POST", re.compile(rf"^/internal/worker/jobs/{_UUID}/log$")),
    ("PUT", re.compile(rf"^/internal/worker/jobs/{_UUID}/file$")),
    ("POST", re.compile(rf"^/internal/worker/jobs/{_UUID}/done$")),
)
_STATUS = re.compile(rf"^/api/status/({_UUID})$")


def _is_preflight(scope) -> bool:
    if scope.get("method") != "OPTIONS":
        return False
    return any(k == b"access-control-request-method" for k, _ in scope.get("headers") or ())


def serves_locally(method: str, path: str) -> bool:
    """Whether a front answers this request itself (preflights and status
    snapshots aside)."""
    if (method, path) in _LOCAL_EXACT:
        return True
    return any(m == method and rx.match(path) for m, rx in _LOCAL_PATTERNS)


# Hop-by-hop headers (RFC 9110 7.6.1) stay on their own hop.
_HOP = frozenset({b"connection", b"keep-alive", b"proxy-connection", b"te", b"trailer",
                  b"transfer-encoding", b"upgrade"})
# The front's own server adds these to every response: the engine's copies
# would go out twice.
_RESPONSE_DROP = _HOP | {b"date", b"server"}


class _ClientGone(Exception):
    pass


class Front:
    """The ASGI app of a front process."""

    def __init__(self, local, status_view=None, status_key=None,
                 socket_path=None, transport=None):
        self.local = local
        self.status_view = status_view   # job id -> view a front may answer from, or None
        self.status_key = status_key     # ASGI scope key the local app reads it from
        self.socket_path = socket_path or ENGINE_SOCKET
        self._transport = transport
        self._client = None

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # The limits go on the transport: httpx ignores a client's
            # ``limits`` once it is handed a transport, and the default pool is
            # 100 connections. A video the browser reads slowly holds one for as
            # long as it plays, so a single viewer seeking through a long source
            # filled a front's pool and every other request of that front
            # waited 10 s for a 503.
            transport = self._transport or httpx.AsyncHTTPTransport(
                uds=self.socket_path, retries=1, limits=ENGINE_LIMITS)
            # No read/write timeout: uploads, long polls and MCP streams last
            # as long as they last, as they did when the API was one process.
            self._client = httpx.AsyncClient(
                transport=transport, base_url="http://engine",
                timeout=httpx.Timeout(None, connect=5.0, pool=10.0))
        return self._client

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind == "lifespan":
            return await self._lifespan(scope, receive, send)
        if kind != "http":
            return await self.local(scope, receive, send)
        method, path = scope["method"], scope["path"]
        if _is_preflight(scope) or serves_locally(method, path):
            return await self.local(scope, receive, send)
        if method in ("GET", "HEAD"):
            m = _STATUS.match(path)
            if m:
                view = self.status_view(m.group(1)) if self.status_view else None
                if view is not None:
                    scope = dict(scope)
                    scope[self.status_key] = view
                    return await self.local(scope, receive, send)
        return await self.proxy(scope, receive, send)

    async def _lifespan(self, scope, receive, send):
        async def receive_wrapped():
            message = await receive()
            if message["type"] == "lifespan.shutdown" and self._client is not None:
                await self._client.aclose()
                self._client = None
            return message
        await self.local(scope, receive_wrapped, send)

    async def proxy(self, scope, receive, send):
        headers = [(k, v) for k, v in scope["headers"] if k.lower() not in _HOP]
        names = {k.lower() for k, _ in headers}
        # Through Traefik these are already set; a direct hit (the health
        # check, a local run) gets them from this hop so the engine still
        # knows the client.
        client = scope.get("client")
        if b"x-forwarded-for" not in names and client:
            headers.append((b"x-forwarded-for", client[0].encode("latin-1")))
        if b"x-forwarded-proto" not in names:
            headers.append((b"x-forwarded-proto", scope.get("scheme", "http").encode("latin-1")))
        target = scope.get("raw_path") or scope["path"].encode("utf-8")
        if scope.get("query_string"):
            target += b"?" + scope["query_string"]

        length = next((v for k, v in headers if k.lower() == b"content-length"), None)
        has_body = (length is not None and length.strip() != b"0") \
            or any(k.lower() == b"transfer-encoding" for k, _ in scope["headers"])
        gone = []

        async def body():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    gone.append(True)
                    raise _ClientGone()
                chunk = message.get("body", b"")
                if chunk:
                    yield chunk
                if not message.get("more_body", False):
                    return

        request = self.client().build_request(
            scope["method"], target.decode("latin-1"), headers=headers,
            content=body() if has_body else None)
        try:
            response = await self.client().send(request, stream=True)
        except Exception as e:
            if gone:
                return   # the client left while its body was being passed on
            print(f"⚠️ Front: engine unreachable ({type(e).__name__}) for "
                  f"{scope['method']} {scope['path']}")
            return await self._unavailable(send)
        try:
            await send({"type": "http.response.start", "status": response.status_code,
                        "headers": [(k, v) for k, v in response.headers.raw
                                    if k.lower() not in _RESPONSE_DROP]})
            async for chunk in response.aiter_raw():
                if chunk:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except (OSError, httpx.HTTPError, _ClientGone):
            # The browser left, or the engine cut the stream: nothing to tell.
            pass
        finally:
            await response.aclose()

    @staticmethod
    async def _unavailable(send):
        body = b'{"detail":"The server is restarting, retry in a few seconds."}'
        await send({"type": "http.response.start", "status": 503,
                    "headers": [(b"content-type", b"application/json"),
                                (b"retry-after", b"3"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


