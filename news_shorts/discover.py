"""Find today's videos on a topic from licensed sources only."""
import datetime
import os

import httpx

from .licensing import is_reusable

YT = "https://www.googleapis.com/youtube/v3"


def discover_youtube_cc(query, api_key=None, limit=5, client=None, now=None):
    """Creative Commons YouTube videos published in the last 24h.

    The search filter is only a hint: each hit is re-read with videos.list and
    kept only if status.license is really ``creativeCommon``.
    """
    api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    after = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    http = client or httpx
    r = http.get(f"{YT}/search", params={
        "part": "id", "q": query, "type": "video", "videoLicense": "creativeCommon",
        "publishedAfter": after, "order": "date", "maxResults": limit, "key": api_key})
    r.raise_for_status()
    ids = [i["id"]["videoId"] for i in r.json().get("items", []) if i.get("id", {}).get("videoId")]
    if not ids:
        return []
    r = http.get(f"{YT}/videos", params={
        "part": "snippet,status", "id": ",".join(ids), "key": api_key})
    r.raise_for_status()
    out = []
    for item in r.json().get("items", []):
        cand = {
            "id": item["id"],
            "title": item["snippet"].get("title", ""),
            "source_url": f"https://www.youtube.com/watch?v={item['id']}",
            "attribution": item["snippet"].get("channelTitle", ""),
            "license": item.get("status", {}).get("license"),
        }
        if is_reusable(cand):
            out.append(cand)
    return out


def manual_candidate(url, license, attribution, title=""):
    """Operator-declared source (Assemblée nationale, own footage). Rejected if unlicensed."""
    cand = {"id": url.rstrip("/").rsplit("/", 1)[-1], "title": title, "source_url": url,
            "attribution": attribution, "license": license}
    if not is_reusable(cand):
        raise ValueError("source lacks a clear allowed licence/attribution; skipped")
    return cand
