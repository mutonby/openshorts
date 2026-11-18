"""Posting a finished Short to every network through Upload-Post.

One upload request carries all networks; each gets its own caption from
``viral.platform_posts``. Spacing and a daily cap keep an account from looking
like a spam bot (a burst of near-identical uploads is the fastest way to get
reach throttled): when the last post is more recent than ``min_gap_minutes``
the next one is scheduled after the gap instead of going out at once.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

API_BASE = "https://api.upload-post.com/api"
UPLOAD_URL = f"{API_BASE}/upload"
NETWORKS = ("tiktok", "instagram", "youtube", "facebook", "x", "threads", "bluesky", "linkedin")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]{3,40}$")


class UploadPostError(RuntimeError):
    pass


def _http(session):
    if session is None:
        import requests as session
    return session


def _err(r) -> str:
    try:
        d = r.json()
        msg = d.get("message") or d.get("error") or d.get("detail") or ""
    except ValueError:
        msg = r.text[:200]
    return f"Upload-Post answered {r.status_code}" + (f": {msg}" if msg else "")


def list_profiles(api_key: str, session=None, timeout: float = 30) -> List[dict]:
    """Your Upload-Post profiles: [{"username", "connected": [networks]}].

    A profile is the container your social accounts are connected to; its
    ``username`` is what ``post.upload_post_user`` must be set to (it is not
    your login email).
    """
    r = _http(session).get(f"{API_BASE}/uploadposts/users",
                           headers={"Authorization": f"Apikey {api_key}"}, timeout=timeout)
    if r.status_code != 200:
        raise UploadPostError(_err(r))
    data = r.json() if r.content else {}
    out = []
    for p in (data.get("profiles") or []) if isinstance(data, dict) else []:
        name = p.get("username") if isinstance(p, dict) else None
        if not isinstance(name, str) or not name:
            continue
        socials = p.get("social_accounts") or {}
        connected = [n for n in NETWORKS if isinstance(socials.get(n), dict) and socials.get(n)]
        out.append({"username": name, "connected": connected})
    return out


def create_profile(api_key: str, username: str, session=None, timeout: float = 30) -> str:
    """Create a profile; one that already exists counts as created."""
    if not _PROFILE_NAME.match(username or ""):
        raise UploadPostError("profile names are 3-40 letters, digits, - or _")
    r = _http(session).post(f"{API_BASE}/uploadposts/users", json={"username": username},
                            headers={"Authorization": f"Apikey {api_key}"}, timeout=timeout)
    text = (r.text or "").lower()
    if r.status_code in (200, 201) or r.status_code == 409 or (
            r.status_code == 400 and ("exist" in text or "already" in text)):
        return username
    if "limit" in text:
        raise UploadPostError("your Upload-Post plan has no free profile left: use an existing one")
    raise UploadPostError(_err(r))


def connect_url(api_key: str, username: str, return_url: str = "", session=None,
                timeout: float = 30) -> str:
    """A one-time link (about an hour) to connect social accounts to ``username``."""
    if not _PROFILE_NAME.match(username or ""):
        raise UploadPostError("pick or create a profile first")
    payload = {"username": username, "connectTitle": "Connect your accounts for livewatch",
               "connectDescription": "Link the networks livewatch should post your Shorts to."}
    if return_url:
        payload.update(redirectUrl=return_url, redirectButtonText="Back to livewatch")
    r = _http(session).post(f"{API_BASE}/uploadposts/users/generate-jwt", json=payload,
                            headers={"Authorization": f"Apikey {api_key}"}, timeout=timeout)
    if r.status_code not in (200, 201):
        raise UploadPostError(_err(r))
    url = (r.json() or {}).get("access_url") or (r.json() or {}).get("url") or ""
    if not url.startswith("https://"):
        raise UploadPostError("Upload-Post did not return a connect link")
    return url


def upload_fields(posts: Dict[str, Dict[str, str]]) -> dict:
    """Upload-Post form fields for the per-network texts of ``viral.platform_posts``."""
    data: dict = {}
    for p, t in posts.items():
        title, desc = t.get("title", ""), t.get("description", "")
        if p == "tiktok":
            data["tiktok_title"] = desc or title
        elif p == "instagram":
            data["instagram_title"] = desc or title
            data["media_type"] = "REELS"
        elif p == "youtube":
            data["youtube_title"] = title
            data["youtube_description"] = desc
            data["privacyStatus"] = "public"
        elif p == "facebook":
            data["facebook_title"] = title
            data["facebook_description"] = desc or title
            data["facebook_media_type"] = "REELS"
        elif p == "x":
            data["x_title"] = title
        elif p == "threads":
            data["threads_title"] = title
        elif p == "bluesky":
            data["bluesky_title"] = title
    first = next(iter(posts.values()), {})
    data["title"] = first.get("title", "") or "clip"
    return data


class PostLedger:
    """When posts went out, kept on disk so a restart keeps the spacing and cap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.times: List[datetime] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self.times = [datetime.fromisoformat(t) for t in raw.get("posts", [])]
            except (ValueError, OSError):
                self.times = []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keep = [t for t in self.times if t > datetime.now(timezone.utc) - timedelta(days=3)]
        self.path.write_text(json.dumps({"posts": [t.isoformat() for t in keep]}), encoding="utf-8")

    def slot(self, now: datetime, min_gap: timedelta, daily_cap: int) -> Optional[datetime]:
        """When the next post may go out (``now`` or later), None if today's cap is used."""
        with self._lock:
            day_ago = now - timedelta(hours=24)
            recent = sorted(t for t in self.times if t > day_ago)
            if len(recent) >= daily_cap:
                return None
            if recent and recent[-1] + min_gap > now:
                return recent[-1] + min_gap
            return now

    def record(self, when: datetime):
        with self._lock:
            self.times.append(when)
            self._save()


def post_video(video: Path, posts: Dict[str, Dict[str, str]], *, api_key: str, user: str,
               when: Optional[datetime] = None, session=None, timeout: float = 300) -> dict:
    """Upload one video to Upload-Post. Returns its JSON answer; raises on failure."""
    if session is None:
        import requests as session
    http = session
    data = upload_fields(posts)
    data["user"] = user
    data["async_upload"] = "true"
    fields = list(data.items()) + [("platform[]", p) for p in posts]
    now = datetime.now(timezone.utc)
    if when and when > now + timedelta(minutes=1):
        fields.append(("scheduled_date", when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    with open(video, "rb") as fh:
        r = http.post(UPLOAD_URL, headers={"Authorization": f"Apikey {api_key}"},
                      data=fields, files={"video": (Path(video).name, fh, "video/mp4")},
                      timeout=timeout)
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"Upload-Post {r.status_code}: {r.text[:300]}")
    try:
        return r.json()
    except ValueError:
        return {"raw": r.text[:300]}
