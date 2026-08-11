"""youtube_publish.py — Direct YouTube upload via Google OAuth.

Zero-budget publishing to the user's OWN YouTube channel, no third party:

  * /api/youtube/auth-url  → Google consent URL (scope: youtube.upload)
  * /api/youtube/callback  → OAuth callback; stores the refresh token
  * /api/youtube/post      → uploads a clip straight to the channel

Setup (one-time, free): create OAuth credentials in Google Cloud Console
(APIs & Services → Credentials → OAuth Client ID → Web application) and add
this backend's URL + /api/youtube/callback as an authorized redirect URI.
Enable the YouTube Data API v3 for the project. Then set:

    GOOGLE_YT_CLIENT_ID=...
    GOOGLE_YT_CLIENT_SECRET=...

The refresh token is stored server-side in server_settings.json (never in the
browser). Uploads count against YouTube's free API quota (~6 uploads/day at
the default 10,000 units/day — each upload costs 1,600 units).
"""

import json
import os
import time
from typing import Any, Dict, Optional

import httpx

# --- OAuth endpoints ---
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def is_configured() -> bool:
    return bool(os.environ.get("GOOGLE_YT_CLIENT_ID")
                and os.environ.get("GOOGLE_YT_CLIENT_SECRET"))


def client_id() -> str:
    return os.environ.get("GOOGLE_YT_CLIENT_ID", "").strip()


def client_secret() -> str:
    return os.environ.get("GOOGLE_YT_CLIENT_SECRET", "").strip()


def _settings_path() -> str:
    data_dir = os.environ.get("DATA_DIR", "").strip() or "output"
    return os.path.join(data_dir, "server_settings.json")


def _load_settings() -> dict:
    try:
        with open(_settings_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    return data


def _save_settings(settings: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_settings_path()), exist_ok=True)
        with open(_settings_path(), "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"⚠️ [youtube] Could not persist settings: {e}")


def build_auth_url(redirect_uri: str, state: str = "") -> str:
    import urllib.parse
    params = {
        "client_id": client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, redirect_uri: str) -> Optional[dict]:
    """Exchange the OAuth code for tokens; store the refresh token."""
    resp = httpx.post(TOKEN_URL, data={
        "client_id": client_id(),
        "client_secret": client_secret(),
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }, timeout=30.0)
    if resp.status_code != 200:
        print(f"⚠️ [youtube] Token exchange failed: {resp.text[:300]}")
        return None
    data = resp.json()
    refresh = data.get("refresh_token")
    if not refresh:
        return None
    settings = _load_settings()
    youtube = settings.setdefault("youtube", {})
    youtube["refresh_token"] = refresh
    youtube["access_token"] = data.get("access_token", "")
    youtube["expires_at"] = time.time() + int(data.get("expires_in", 3600)) - 60
    _save_settings(settings)
    return data


def _access_token() -> Optional[str]:
    """A valid access token, refreshing it when expired."""
    settings = _load_settings()
    youtube = settings.get("youtube") or {}
    refresh = youtube.get("refresh_token")
    if not refresh:
        return None
    access = youtube.get("access_token", "")
    expires_at = float(youtube.get("expires_at") or 0)
    if access and time.time() < expires_at:
        return access
    resp = httpx.post(TOKEN_URL, data={
        "client_id": client_id(),
        "client_secret": client_secret(),
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }, timeout=30.0)
    if resp.status_code != 200:
        print(f"⚠️ [youtube] Refresh failed: {resp.text[:300]}")
        return None
    data = resp.json()
    access = data.get("access_token", "")
    youtube["access_token"] = access
    youtube["expires_at"] = time.time() + int(data.get("expires_in", 3600)) - 60
    _save_settings(settings)
    return access


def connection_status() -> Dict[str, Any]:
    """Whether a YouTube account is connected + its channel name."""
    settings = _load_settings()
    youtube = settings.get("youtube") or {}
    if not youtube.get("refresh_token"):
        return {"connected": False}
    return {
        "connected": True,
        "channelTitle": youtube.get("channel_title", ""),
        "channelId": youtube.get("channel_id", ""),
    }


def fetch_channel_info() -> Optional[dict]:
    """Look up the connected channel's name/id and cache it."""
    access = _access_token()
    if not access:
        return None
    resp = httpx.get(CHANNELS_URL,
                     params={"part": "snippet", "mine": "true"},
                     headers={"Authorization": f"Bearer {access}"},
                     timeout=30.0)
    if resp.status_code != 200:
        print(f"⚠️ [youtube] Channel lookup failed: {resp.text[:300]}")
        return None
    items = resp.json().get("items") or []
    if not items:
        return None
    snippet = items[0].get("snippet") or {}
    info = {
        "channel_title": snippet.get("title", ""),
        "channel_id": items[0].get("id", ""),
    }
    settings = _load_settings()
    settings.setdefault("youtube", {}).update(info)
    _save_settings(settings)
    return info


def upload_video(file_path: str, title: str, description: str,
                 privacy: str = "public") -> Optional[dict]:
    """Upload a video to the connected channel (multipart, <256MB clips).

    Returns {"video_id", "url"} on success, None on failure.
    """
    access = _access_token()
    if not access:
        raise RuntimeError("YouTube not connected — connect it in Settings first.")
    if not os.path.exists(file_path):
        raise RuntimeError(f"Video file not found: {file_path}")

    metadata = {
        "snippet": {
            "title": (title or "OpenShorts+ clip")[:100],
            "description": (description or "")[:4900],
        },
        "status": {
            "privacyStatus": privacy if privacy in ("public", "unlisted", "private") else "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    url = f"{UPLOAD_URL}?uploadType=multipart&part=snippet,status"
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json; boundary=openshorts_boundary",
    }
    body = _multipart_body(metadata, file_path)
    resp = httpx.post(url, headers=headers, content=body, timeout=600.0)
    if resp.status_code not in (200, 201):
        print(f"⚠️ [youtube] Upload failed ({resp.status_code}): "
              f"{resp.text[:400]}")
        # Quota or permission errors read clearer when surfaced.
        detail = resp.text[:400]
        if "quota" in detail.lower():
            raise RuntimeError(
                "YouTube API quota exceeded for today (~6 free uploads/day). "
                "Try again tomorrow or use Upload-Post for more.")
        if "accessNotConfigured" in detail:
            raise RuntimeError(
                "YouTube Data API v3 is not enabled on your Google Cloud "
                "project — enable it in the API Library.")
        raise RuntimeError(f"YouTube upload failed: {detail}")
    data = resp.json()
    video_id = data.get("id")
    return {
        "video_id": video_id,
        "url": f"https://youtu.be/{video_id}" if video_id else "",
    }


def disconnect() -> None:
    settings = _load_settings()
    settings.pop("youtube", None)
    _save_settings(settings)


def _multipart_body(metadata: dict, file_path: str) -> bytes:
    """Build a multipart/related body: JSON metadata + raw video bytes."""
    boundary = b"openshorts_boundary"
    json_part = json.dumps(metadata).encode("utf-8")
    with open(file_path, "rb") as f:
        video = f.read()
    parts = [
        b"--" + boundary + b"\r\n",
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
        json_part + b"\r\n",
        b"--" + boundary + b"\r\n",
        b"Content-Type: video/mp4\r\n",
        b"Content-Transfer-Encoding: binary\r\n\r\n",
        video + b"\r\n",
        b"--" + boundary + b"--\r\n",
    ]
    return b"".join(parts)
