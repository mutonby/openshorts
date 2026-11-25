"""Direct YouTube publishing (self-host): upload a clip with the YouTube Data
API instead of through Upload-Post, optionally scheduled.

Connect once from the host:

    docker exec -it openshorts-backend /opt/venv/bin/python youtube_direct.py \
        connect /app/deploy/client_secret.json

``client_secret.json`` is a Google Cloud OAuth client of type "Desktop app"
with the YouTube Data API v3 enabled. The refresh token and client are kept in
``YOUTUBE_DIRECT_TOKEN`` (default deploy/youtube_token.json, git-ignored, mode 600).
While that file exists, /api/social/post sends YouTube here and only the other
platforms to Upload-Post.

Uploads from an unaudited Google Cloud project are locked to private by
YouTube, scheduled or not, until the project passes YouTube's API audit.
"""
import base64
import hashlib
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import httpx

TOKEN_PATH = os.environ.get("YOUTUBE_DIRECT_TOKEN") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "deploy", "youtube_token.json")
SCOPE = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
# Desktop OAuth clients accept any loopback port. Nothing listens there: the
# browser fails to load it and the user pastes the address back.
REDIRECT_URI = "http://127.0.0.1:8765/"
CATEGORY_PEOPLE_BLOGS = "22"

# Tests swap in an httpx.MockTransport.
_transport = None
# Access token cache: (token, monotonic expiry).
_access = {"token": None, "expires": 0.0}


class YouTubeDirectError(Exception):
    pass


def _client(timeout=60.0):
    return httpx.Client(timeout=timeout, transport=_transport)


def _load():
    with open(TOKEN_PATH) as f:
        return json.load(f)


def _save(data):
    os.makedirs(os.path.dirname(TOKEN_PATH) or ".", exist_ok=True)
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)


def is_connected() -> bool:
    try:
        return bool(_load().get("refresh_token"))
    except (OSError, ValueError):
        return False


def _access_token() -> str:
    if _access["token"] and time.monotonic() < _access["expires"] - 60:
        return _access["token"]
    cfg = _load()
    with _client() as c:
        r = c.post(TOKEN_URL, data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "refresh_token": cfg["refresh_token"],
            "grant_type": "refresh_token",
        })
    if r.status_code != 200:
        # invalid_grant: revoked, or the consent screen is still in "Testing"
        # (Google expires those refresh tokens after 7 days).
        raise YouTubeDirectError(
            f"YouTube login refresh failed ({r.status_code}): {r.text[:300]}. "
            "Reconnect with: youtube_direct.py connect")
    tok = r.json()
    _access["token"] = tok["access_token"]
    _access["expires"] = time.monotonic() + int(tok.get("expires_in", 3600))
    return _access["token"]


def publish_at(scheduled_date, tz_name=None):
    """RFC 3339 UTC time for YouTube's status.publishAt, or None.

    The dashboard sends either an absolute ISO time ("...Z" from the single
    post dialog) or a naive local time plus an IANA timezone (schedule week).
    """
    if not scheduled_date:
        return None
    dt = datetime.fromisoformat(str(scheduled_date).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz_name or "UTC"))
    dt = dt.astimezone(timezone.utc)
    if dt <= datetime.now(timezone.utc):
        raise YouTubeDirectError(f"Scheduled time {dt.isoformat()} is in the past")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def upload(file_path, title, description, scheduled_date=None, tz_name=None):
    """Upload ``file_path`` to the connected channel. Blocking: call it through
    asyncio.to_thread from a request handler. Returns the video id and URL."""
    when = publish_at(scheduled_date, tz_name)
    status = {"privacyStatus": "private" if when else "public",
              "selfDeclaredMadeForKids": False}
    if when:
        status["publishAt"] = when
    meta = {
        "snippet": {"title": title, "description": description or "",
                    "categoryId": CATEGORY_PEOPLE_BLOGS},
        "status": status,
    }
    size = os.path.getsize(file_path)
    headers = {"Authorization": f"Bearer {_access_token()}"}
    with _client(timeout=600.0) as c:
        r = c.post(f"{UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                   headers={**headers, "X-Upload-Content-Type": "video/mp4",
                            "X-Upload-Content-Length": str(size)},
                   json=meta)
        if r.status_code != 200 or "location" not in r.headers:
            raise YouTubeDirectError(f"YouTube refused the upload ({r.status_code}): {r.text[:500]}")
        with open(file_path, "rb") as f:
            r = c.put(r.headers["location"], content=f,
                      headers={**headers, "Content-Type": "video/mp4",
                               "Content-Length": str(size)})
    if r.status_code not in (200, 201):
        raise YouTubeDirectError(f"YouTube upload failed ({r.status_code}): {r.text[:500]}")
    video = r.json()
    vid = video.get("id")
    print(f"📺 YouTube direct upload: {vid} scheduled={when or 'now'} "
          f"privacy={video.get('status', {}).get('privacyStatus')}")
    return {"platform": "youtube", "success": True, "video_id": vid,
            "url": f"https://youtube.com/shorts/{vid}", "publish_at": when}


def _pkce():
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def connect(client_secret_path):
    with open(client_secret_path) as f:
        raw = json.load(f)
    client = raw.get("installed") or raw.get("web") or raw
    client_id, client_secret = client["client_id"], client["client_secret"]
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    url = AUTH_URL + "?" + urlencode({
        "client_id": client_id, "redirect_uri": REDIRECT_URI, "response_type": "code",
        "scope": SCOPE, "access_type": "offline", "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    })
    print("\n1. Open this link, sign in, and pick the channel to post to:\n")
    print(url)
    print("\n2. The browser then fails to load a 127.0.0.1:8765 page. That is expected.")
    pasted = input("   Copy that page's full address from the address bar and paste it here:\n> ").strip()
    q = parse_qs(urlparse(pasted).query)
    if q.get("state", [None])[0] != state:
        sys.exit("That address is not from this login attempt (state mismatch). Run connect again.")
    if "code" not in q:
        sys.exit(f"No authorization code in that address: {q.get('error', ['?'])[0]}")
    with _client() as c:
        r = c.post(TOKEN_URL, data={
            "client_id": client_id, "client_secret": client_secret, "code": q["code"][0],
            "code_verifier": verifier, "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        })
    if r.status_code != 200:
        sys.exit(f"Token exchange failed ({r.status_code}): {r.text[:300]}")
    tok = r.json()
    if not tok.get("refresh_token"):
        sys.exit("Google returned no refresh token. Remove the app's access at "
                 "myaccount.google.com/permissions and run connect again.")
    _save({"client_id": client_id, "client_secret": client_secret,
           "refresh_token": tok["refresh_token"]})
    _access["token"], _access["expires"] = None, 0.0
    print(f"\nConnected. Channel: {channel_title()}")
    print(f"Saved to {TOKEN_PATH}. YouTube posts from the dashboard now go direct.")


def channel_title():
    with _client() as c:
        r = c.get(CHANNELS_URL, params={"part": "snippet", "mine": "true"},
                  headers={"Authorization": f"Bearer {_access_token()}"})
    items = r.json().get("items") or [] if r.status_code == 200 else []
    return items[0]["snippet"]["title"] if items else f"(could not read: {r.status_code})"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "connect" and len(sys.argv) == 3:
        connect(sys.argv[2])
    elif cmd == "status":
        print(f"Connected: {channel_title()}" if is_connected() else "Not connected")
    elif cmd == "disconnect":
        if os.path.exists(TOKEN_PATH):
            os.remove(TOKEN_PATH)
        print("Disconnected: YouTube posts go through Upload-Post again.")
    else:
        sys.exit("usage: youtube_direct.py connect <client_secret.json> | status | disconnect")
