"""Pexels video discovery and bounded imports into the existing B-roll store."""

import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

import broll
import ssrf_guard


API = "https://api.pexels.com/v1/videos"


class StockMediaError(ValueError):
    pass


def configured():
    return bool(os.getenv("PEXELS_API_KEY", "").strip())


def _metadata(path, params=None):
    key = os.getenv("PEXELS_API_KEY", "").strip()
    if not key:
        raise StockMediaError("Stock search needs a Pexels API key in the server settings.")
    with ssrf_guard.guarded(), httpx.Client(timeout=20, trust_env=False) as client:
        response = client.get(f"{API}/{path}", params=params, headers={"Authorization": key})
    if response.status_code == 429:
        raise StockMediaError("The stock provider's request limit has been reached. Try again later.")
    if response.status_code in (401, 403):
        raise StockMediaError("The stock provider rejected the configured API key.")
    if response.status_code != 200:
        raise StockMediaError("The stock provider is temporarily unavailable.")
    try:
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError
        return result
    except ValueError as exc:
        raise StockMediaError("The stock provider returned an unreadable response.") from exc


def _public_asset_url(url):
    if not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        return False
    return parsed.scheme == "https" and port in (None, 443) and not parsed.username and not parsed.password and (
        host == "pexels.com" or host.endswith(".pexels.com") or host == "player.vimeo.com")


def _view(item):
    try:
        duration = float(item["duration"])
        if not 0.1 <= duration <= broll.MAX_ASSET_SECONDS:
            return None
        files = [entry for entry in item["video_files"]
                 if entry.get("file_type") == "video/mp4" and _public_asset_url(entry.get("link"))
                 and 0 < max(entry.get("width") or 0, entry.get("height") or 0) <= 1920]
        selected = max(files, key=lambda entry: (entry.get("width") or 0) * (entry.get("height") or 0), default=None)
        if selected is None or not _public_asset_url(item.get("url")) or not _public_asset_url(item.get("image")):
            return None
        return {"id": int(item["id"]), "duration": duration, "preview_url": selected["link"],
                "image": item["image"], "source_url": item["url"],
                "creator": str((item.get("user") or {}).get("name") or "Pexels contributor")[:120]}
    except (ValueError, TypeError, KeyError):
        return None


def search(query, page=1):
    try:
        data = _metadata("search", {"query": query, "page": page, "per_page": 12})
        videos = [result for item in data.get("videos", []) if (result := _view(item))]
        return {"videos": videos, "page": page, "has_more": page * 12 < int(data.get("total_results") or 0),
                "provider": "Pexels", "provider_url": "https://www.pexels.com/"}
    except httpx.HTTPError as exc:
        raise StockMediaError("Could not reach the stock provider.") from exc


def download(video_id, target):
    """Resolve an ID through the provider and download only its published file."""
    try:
        data = _metadata(f"videos/{video_id}")
        video = _view(data)
        if video is None or video["id"] != video_id:
            raise StockMediaError("This stock video is unavailable or exceeds the supported duration.")
        with ssrf_guard.guarded(), httpx.Client(timeout=30, trust_env=False) as client:
            url = video["preview_url"]
            for attempt in range(6):
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        redirect = urljoin(url, response.headers.get("location", ""))
                        if attempt == 5 or not response.headers.get("location") or not _public_asset_url(redirect):
                            raise StockMediaError("The stock provider returned an unsupported download link.")
                        url = redirect
                        continue
                    _download_response(response, target)
                    return video
        raise StockMediaError("The stock video could not be downloaded.")
    except httpx.HTTPError as exc:
        raise StockMediaError("The stock video could not be downloaded.") from exc


def _download_response(response, target):
    if response.status_code != 200:
        raise StockMediaError("The stock video could not be downloaded.")
    try:
        length = int(response.headers.get("content-length") or 0)
    except ValueError as exc:
        raise StockMediaError("The stock provider returned an invalid file size.") from exc
    if length > broll.MAX_UPLOAD_BYTES:
        raise StockMediaError("This stock video exceeds the 40 MB asset limit.")
    size = 0
    created = False
    try:
        with Path(target).open("xb") as handle:
            created = True
            for block in response.iter_bytes(1024 * 1024):
                size += len(block)
                if size > broll.MAX_UPLOAD_BYTES:
                    raise StockMediaError("This stock video exceeds the 40 MB asset limit.")
                handle.write(block)
    except BaseException:
        if created:
            Path(target).unlink(missing_ok=True)
        raise
