"""What an OAuth-issued token may reach.

OAuth tokens are ordinary ``osk_`` API keys (cloud/mcp_oauth.py). The consent
screen asks for one thing, the MCP connection, so the key it mints carries
exactly that and not the rest of the REST API (projects, Autopilot, social
accounts...).

``openshorts`` is the only scope, and the one every MCP client asks for (it is
the single entry of ``scopes_supported``). It covers what the MCP tools do,
which call back into this API in-process with the caller's own header, plus
downloading the results. Keys a user creates by hand have no scope and keep
the whole API, as before.
"""
import re
from typing import Optional

MCP_SCOPE = "openshorts"
SUPPORTED_SCOPES = (MCP_SCOPE,)

# Keys minted by the OAuth flow before the scope column existed. Their name is
# set by the token endpoint and cannot be renamed afterwards; a hand-made key
# that copies the suffix only narrows its own reach.
OAUTH_KEY_NAME_SUFFIX = "· connected via OAuth"

_ID = r"[A-Za-z0-9_-]{1,64}"
_MCP_ROUTES = tuple((method, re.compile(pattern)) for method, pattern in (
    ("POST", r"/mcp"),
    ("GET", r"/mcp"),
    # The tools (mcp_server._TOOL_IMPLS) and the restore their retry uses.
    ("POST", r"/api/process"),
    ("POST", r"/api/uploads"),
    ("PUT", rf"/api/uploads/{_ID}"),
    ("DELETE", rf"/api/uploads/{_ID}"),
    ("GET", rf"/api/status/{_ID}"),
    ("POST", rf"/api/projects/{_ID}/restore"),
    ("GET", r"/api/me"),
    ("POST", r"/api/subtitle"),
    ("POST", r"/api/clip/rerender"),
    ("POST", r"/api/social/post"),
    ("GET", r"/api/translate/languages"),
    ("POST", r"/api/translate"),
    # Getting the results out.
    ("GET", r"/videos/.+"),
    ("GET", r"/thumbnails/.+"),
    ("GET", rf"/api/jobs/{_ID}/download-all"),
    ("GET", rf"/api/source/{_ID}"),
    ("GET", rf"/api/source-url/{_ID}"),
    ("POST", r"/api/media-token"),
))


def normalise(requested: Optional[str]) -> str:
    """The scope to grant for what a client asked: the supported ones it
    named, or the default when it named none of them (RFC 6749 3.3 lets the
    server grant less than asked, and says so in the token response)."""
    asked = [s for s in (requested or "").split() if s in SUPPORTED_SCOPES]
    return " ".join(dict.fromkeys(asked)) or MCP_SCOPE


def effective(scope: Optional[str], key_name: Optional[str]) -> Optional[str]:
    """The scope a stored key acts under; None means the whole API."""
    if scope:
        return scope
    if key_name and key_name.endswith(OAUTH_KEY_NAME_SUFFIX):
        return MCP_SCOPE
    return None


def allows(scope: Optional[str], method: str, path: str) -> bool:
    """Whether a key with ``scope`` may make this request."""
    if not scope:
        return True
    granted = set(scope.split())
    if MCP_SCOPE in granted:
        method = "GET" if method == "HEAD" else method
        return any(m == method and rx.fullmatch(path or "") for m, rx in _MCP_ROUTES)
    return False
