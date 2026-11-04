"""What the stream's own viewers clipped, as a signal for the moment picker.

A Twitch VOD comes with evidence no transcript has: the clips viewers made
while watching (or afterwards, from the VOD). Each Helix clip records the VOD
it came from (``video_id``), where in it it starts (``vod_offset``), how long
it is and how many views it got. Gaming is where that matters most: the
picker reads what was said, and a chase, a crash or a heist finale is often a
scream and twelve seconds of nobody talking, which a transcript ranks low.

Everything here is best effort. No ``TWITCH_CLIENT_ID``/``TWITCH_CLIENT_SECRET``,
a VOD Twitch will not describe, a network error: the job carries on without
the signal, exactly as it did before. The calls use an app access token
(client credentials), which reads public VODs and clips; no user account is
involved.

The pure helpers (``parse_vod_id``, ``shift_clips``, ``window_heat``,
``promote_hot_windows``) are standard-library only and unit-tested.
"""
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

HELIX = "https://api.twitch.tv/helix"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
MAX_CLIP_PAGES = 5          # 100 clips a page; Helix returns them by views
CLIP_LOOKBACK = timedelta(days=60)  # clips cut from the VOD after the stream

_VOD_PATH = re.compile(r"^/(?:[^/]+/)?(?:videos?|v)/(\d+)/?$")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def parse_vod_id(url) -> Optional[str]:
    """The numeric VOD id of a ``twitch.tv/videos/<id>`` link, else None.

    Clip links (``clips.twitch.tv/...``, ``twitch.tv/<user>/clip/...``) and
    channel pages are not VODs and return None.
    """
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in ("twitch.tv", "www.twitch.tv", "m.twitch.tv"):
        return None
    m = _VOD_PATH.match(parsed.path or "")
    return m.group(1) if m else None


def parse_helix_duration(value) -> float:
    """Helix VOD durations look like ``3h2m1s``; seconds as a float."""
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", str(value or "")):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def shift_clips(clips: List[dict], start_seconds: float, duration: float) -> List[dict]:
    """Move VOD-time clips into the time of the file being clipped.

    When the user clipped only a stretch of the VOD (``SOURCE_START_SECONDS``)
    the local file starts there. Clips are kept when their middle falls inside
    the file, and clamped to it.
    """
    out = []
    start = float(start_seconds or 0)
    for c in clips or []:
        try:
            s = float(c["start"]) - start
            e = s + max(1.0, float(c.get("duration") or 0))
        except (KeyError, TypeError, ValueError):
            continue
        mid = (s + e) / 2
        if mid < 0 or mid > duration:
            continue
        out.append({"start": round(max(0.0, s), 3), "end": round(min(duration, e), 3),
                    "views": int(c.get("views") or 0)})
    return out


def window_heat(windows: List[dict], clips: List[dict]) -> Dict[str, dict]:
    """``{window_id: {"clips": n, "views": v, "spans": [[s, e], ...]}}``.

    A viewer clip belongs to the window holding its midpoint, so one clip is
    never counted twice across neighbouring windows.
    """
    heat: Dict[str, dict] = {}
    for c in clips or []:
        mid = (c["start"] + c["end"]) / 2
        for w in windows:
            if w["start"] <= mid < w["end"] or (mid == w["end"] and w is windows[-1]):
                h = heat.setdefault(w["id"], {"clips": 0, "views": 0, "spans": []})
                h["clips"] += 1
                h["views"] += c.get("views", 0)
                h["spans"].append([c["start"], c["end"]])
                break
    for h in heat.values():
        h["spans"].sort()
    return heat


def promote_hot_windows(shortlist: List[dict], scored_order: List[dict],
                        heat: Dict[str, dict], target: int) -> List[dict]:
    """Make sure the windows viewers clipped most reach the detail pass.

    The model's ranking stays in charge of most of the shortlist; up to a
    third of it (at least one slot) is reserved for the hottest windows by
    views, then by clip count. A promoted window replaces the lowest-scored
    window that viewers did not clip, so the shortlist keeps its size.
    """
    if not heat or target <= 0:
        return shortlist
    reserve = max(1, target // 3)
    hottest = sorted(heat.items(), key=lambda kv: (-kv[1]["views"], -kv[1]["clips"]))
    by_id = {w["id"]: w for w in scored_order}
    out = list(shortlist)
    have = {w["id"] for w in out}
    promoted = 0
    for wid, _ in hottest:
        if promoted >= reserve:
            break
        if wid in have or wid not in by_id:
            continue
        if len(out) < target:
            out.append(by_id[wid])
        else:
            # Drop the weakest window viewers did not touch (list is best first).
            for i in range(len(out) - 1, -1, -1):
                if out[i]["id"] not in heat:
                    del out[i]
                    out.append(by_id[wid])
                    break
            else:
                break
        have.add(wid)
        promoted += 1
    return out


def best_stretch(clips: List[dict], total_seconds: float, length_seconds: float,
                 lead_seconds: float = 60.0):
    """The ``length_seconds`` stretch of a VOD holding the most viewer-clip views.

    For Autopilot, which clips at most N minutes of each stream: the first N
    minutes of a stream are usually a "starting soon" screen and small talk,
    and the moments viewers clipped are hours in. ``clips`` are VOD-time
    ``{"start", "duration", "views"}`` (``fetch_viewer_clips``). Each candidate
    stretch opens ``lead_seconds`` before a clip, so the first moment keeps its
    setup. Returns ``(start, end)`` or None when there is no signal or the VOD
    already fits.
    """
    total = float(total_seconds or 0)
    length = float(length_seconds or 0)
    if not clips or length <= 0 or (total and total <= length):
        return None
    best, best_views = None, -1
    for anchor in clips:
        start = max(0.0, float(anchor["start"]) - lead_seconds)
        if total:
            start = min(start, max(0.0, total - length))
        end = start + length
        views = sum(int(c.get("views") or 0) + 1 for c in clips   # +1: a clip counts
                    if start <= float(c["start"]) and
                    float(c["start"]) + float(c.get("duration") or 0) <= end)
        if views > best_views:
            best, best_views = (round(start, 3), round(end, 3)), views
    return best


# --------------------------------------------------------------------------- #
# Helix
# --------------------------------------------------------------------------- #
def _credentials():
    cid = os.environ.get("TWITCH_CLIENT_ID", "").strip()
    secret = os.environ.get("TWITCH_CLIENT_SECRET", "").strip()
    return (cid, secret) if cid and secret else (None, None)


def _app_token(client, cid, secret) -> Optional[str]:
    resp = client.post(TOKEN_URL, data={
        "client_id": cid, "client_secret": secret, "grant_type": "client_credentials"})
    if resp.status_code != 200:
        return None
    return (resp.json() or {}).get("access_token")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_viewer_clips(vod_id: str, timeout: float = 15.0) -> List[dict]:
    """``[{"start", "duration", "views"}]`` in VOD seconds, most viewed first.

    Empty on any failure or when credentials are not configured.
    """
    cid, secret = _credentials()
    if not cid or not vod_id:
        return []
    import httpx
    try:
        with httpx.Client(timeout=timeout) as client:
            token = _app_token(client, cid, secret)
            if not token:
                print("   ℹ️ Twitch: app token refused; no viewer-clip signal.")
                return []
            headers = {"Client-Id": cid, "Authorization": f"Bearer {token}"}
            r = client.get(f"{HELIX}/videos", params={"id": vod_id}, headers=headers)
            data = (r.json() or {}).get("data") or [] if r.status_code == 200 else []
            if not data:
                return []
            video = data[0]
            created = datetime.fromisoformat(
                str(video.get("created_at", "")).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            params = {"broadcaster_id": video.get("user_id"), "first": 100,
                      "started_at": _iso(created),
                      "ended_at": _iso(min(now, created + CLIP_LOOKBACK))}
            out, cursor = [], None
            for _ in range(MAX_CLIP_PAGES):
                if cursor:
                    params["after"] = cursor
                r = client.get(f"{HELIX}/clips", params=params, headers=headers)
                if r.status_code != 200:
                    break
                body = r.json() or {}
                for c in body.get("data") or []:
                    if str(c.get("video_id") or "") != str(vod_id):
                        continue
                    if c.get("vod_offset") is None:
                        continue  # the VOD was not available when it was cut
                    out.append({"start": float(c["vod_offset"]),
                                "duration": float(c.get("duration") or 0),
                                "views": int(c.get("view_count") or 0)})
                cursor = (body.get("pagination") or {}).get("cursor")
                if not cursor:
                    break
    except Exception as e:  # best effort: never fail a job over a hint
        print(f"   ℹ️ Twitch viewer clips unavailable ({type(e).__name__}: {e}).")
        return []
    out.sort(key=lambda c: -c["views"])
    return out


def viewer_clips_for_job(url, duration: float) -> List[dict]:
    """Viewer clips for the file this job clips, in its own seconds.

    Reads ``SOURCE_START_SECONDS`` when the job clips a stretch of the VOD
    (``MAX_SOURCE_MINUTES``), like ``main.cap_source_duration`` does.
    """
    vod_id = parse_vod_id(url)
    if not vod_id:
        return []
    clips = fetch_viewer_clips(vod_id)
    if not clips:
        return []
    start = 0.0
    if os.environ.get("MAX_SOURCE_MINUTES", "").strip():
        try:
            start = float(os.environ.get("SOURCE_START_SECONDS", "0") or 0)
        except ValueError:
            start = 0.0
    shifted = shift_clips(clips, start, float(duration or 0) or math.inf)
    print(f"   👀 Twitch: {len(shifted)} viewer clip(s) inside this source "
          f"({len(clips)} on the VOD).")
    return shifted
