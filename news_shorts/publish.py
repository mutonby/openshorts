"""Publishers. Credentials come from env vars only; dry-run is the default."""
import os

import httpx

from .licensing import attribution_text


def _env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError("missing env vars: " + ", ".join(missing))
    return [os.environ[n] for n in names]


def description(clip):
    return f"{clip.get('title', '')}\n\n{attribution_text(clip)}\n#shorts"


def publish_youtube(clip, dry_run=True, client=None):
    if dry_run:
        return {"dry_run": True, "platform": "youtube"}
    cid, secret, refresh = _env("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")
    http = client or httpx
    tok = http.post("https://oauth2.googleapis.com/token", data={
        "client_id": cid, "client_secret": secret, "refresh_token": refresh,
        "grant_type": "refresh_token"})
    tok.raise_for_status()
    auth = {"Authorization": f"Bearer {tok.json()['access_token']}"}
    meta = {"snippet": {"title": (clip.get("title") or "Short")[:95] + " #shorts",
                        "description": description(clip), "categoryId": "25"},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}}
    init = http.post("https://www.googleapis.com/upload/youtube/v3/videos"
                     "?uploadType=resumable&part=snippet,status", headers=auth, json=meta)
    init.raise_for_status()
    with open(clip["clip_path"], "rb") as f:
        up = http.put(init.headers["Location"], headers=auth, content=f.read())
    up.raise_for_status()
    return {"platform": "youtube", "id": up.json().get("id")}


def publish_tiktok(clip, dry_run=True, client=None):
    if dry_run:
        return {"dry_run": True, "platform": "tiktok"}
    (token,) = _env("TIKTOK_ACCESS_TOKEN")
    http = client or httpx
    data = open(clip["clip_path"], "rb").read()
    auth = {"Authorization": f"Bearer {token}"}
    init = http.post("https://open.tiktokapis.com/v2/post/publish/video/init/", headers=auth, json={
        "post_info": {"title": description(clip)[:2200], "privacy_level": "SELF_ONLY"},
        "source_info": {"source": "FILE_UPLOAD", "video_size": len(data),
                        "chunk_size": len(data), "total_chunk_count": 1}})
    init.raise_for_status()
    d = init.json()["data"]
    up = http.put(d["upload_url"], content=data, headers={
        "Content-Type": "video/mp4",
        "Content-Range": f"bytes 0-{len(data) - 1}/{len(data)}"})
    up.raise_for_status()
    return {"platform": "tiktok", "id": d.get("publish_id")}


PUBLISHERS = {"youtube": publish_youtube, "tiktok": publish_tiktok}


def publish_approved(queue, platforms=("youtube", "tiktok"), dry_run=True):
    items = queue.load()
    results = []
    for i in items:
        if i["status"] != "approved":
            continue
        for p in platforms:
            if p in i["published"]:
                continue
            res = PUBLISHERS[p](i, dry_run=dry_run)
            results.append((i["clip_path"], res))
            if not dry_run:
                i["published"][p] = res
        if not dry_run and all(p in i["published"] for p in platforms):
            i["status"] = "published"
    if not dry_run:
        queue.save(items)
    return results
