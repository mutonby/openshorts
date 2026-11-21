"""OpenShorts CLI: clip long videos into vertical shorts from the terminal.

Zero dependencies by design so `uvx openshorts` and `pipx run openshorts`
start instantly. Talks to the same REST API the dashboard and the MCP server
use; nothing here can drift from what the app actually does.

Auth and target come from the environment:
  OPENSHORTS_API_KEY  osk_... key from the account page (cloud only)
  OPENSHORTS_API_URL  defaults to https://api.openshorts.app; set to
                      http://localhost:8000 for a self-hosted instance,
                      where no key is needed.
"""

import argparse
from contextlib import closing, nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_API = "https://api.openshorts.app"
POLL_SECONDS = 10
FRAMING_METHODS = ("auto", "motion", "detail", "center", "fit", "gameplay", "legacy")
TREND_PROFILES = {
    "explainer": {"clip_min_seconds": 20, "clip_max_seconds": 45,
                  "auto_hook": True, "auto_hook_style": "pill"},
    "process": {"clip_min_seconds": 15, "clip_max_seconds": 40, "auto_hook": False},
    "story": {"clip_min_seconds": 35, "clip_max_seconds": 65,
              "auto_hook": True, "auto_hook_style": "outline"},
}


def _base():
    return os.environ.get("OPENSHORTS_API_URL", DEFAULT_API).rstrip("/")


def _request(method, path, body=None):
    headers = {"Accept": "application/json"}
    key = os.environ.get("OPENSHORTS_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(_base() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except Exception:
            payload = {"detail": str(e.reason)}
        return e.code, payload
    except urllib.error.URLError as e:
        print(f"error: cannot reach {_base()} ({e.reason})", file=sys.stderr)
        sys.exit(2)


def _die(status, payload):
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    if not isinstance(detail, str):
        detail = json.dumps(detail)
    print(f"error ({status}): {detail}", file=sys.stderr)
    sys.exit(1)


def _quality_hold(payload):
    if isinstance(payload, dict) and payload.get("needs_confirmation"):
        height = (payload.get("quality_check") or {}).get("max_height")
        resolution = f" ({height}p)" if height else ""
        return f"Source resolution{resolution} requires confirmation."
    return None


def _absolutize(url):
    if url and url.startswith("/"):
        return _base() + url
    return url


def _print_clips(result):
    clips = (result or {}).get("clips") or []
    if not clips:
        print("no clips in result")
        return
    for i, clip in enumerate(clips):
        title = clip.get("title") or clip.get("video_title_for_youtube_short") or f"clip {i}"
        print(f"[{i}] {title}")
        # Cloud: download_url is signed and opens without the API key.
        print(f"    {clip.get('download_url') or _absolutize(clip.get('video_url') or '')}")


def cmd_process(args):
    body = {"url": args.url, "acknowledged": True}
    if args.layouts:
        body["layouts"] = args.layouts
    if args.framing:
        body["framing_method"] = args.framing
    if args.format:
        body["output_format"] = args.format
    if args.webhook:
        body["webhook_url"] = args.webhook
    if args.webhook_secret:
        body["webhook_secret"] = args.webhook_secret
    if args.force_low_quality:
        body["force_low_quality"] = True
    status, payload = _request("POST", "/api/process", body)
    if status >= 400:
        _die(status, payload)
    hold = _quality_hold(payload)
    if hold:
        _die(409, f"{hold} Retry with --force-low-quality to continue.")
    job_id = payload.get("job_id")
    if args.json and not args.wait:
        print(json.dumps(payload))
        return
    print(f"job queued: {job_id}")
    if args.wait:
        _watch(job_id, as_json=args.json)
    else:
        print(f"follow it with: openshorts status {job_id} --watch")


def _watch(job_id, as_json=False):
    seen_logs = 0
    while True:
        status, payload = _request("GET", f"/api/status/{job_id}")
        if status >= 400:
            _die(status, payload)
        logs = payload.get("logs") or []
        for line in logs[seen_logs:]:
            print(f"  {line}")
        seen_logs = len(logs)
        state = payload.get("status")
        if state == "completed":
            if as_json:
                print(json.dumps(payload.get("result") or {}))
            else:
                print("completed:")
                _print_clips(payload.get("result"))
            return
        if state == "failed":
            print("job failed; last log lines above", file=sys.stderr)
            sys.exit(1)
        time.sleep(POLL_SECONDS)


def cmd_status(args):
    if args.watch:
        _watch(args.job_id, as_json=args.json)
        return
    status, payload = _request("GET", f"/api/status/{args.job_id}")
    if status >= 400:
        _die(status, payload)
    if args.json:
        print(json.dumps(payload))
        return
    print(f"status: {payload.get('status')}")
    logs = payload.get("logs") or []
    if logs:
        print(f"last log: {logs[-1]}")


def cmd_clips(args):
    status, payload = _request("GET", f"/api/status/{args.job_id}")
    if status >= 400:
        _die(status, payload)
    if payload.get("status") != "completed":
        print(f"job is {payload.get('status')}, no clips yet", file=sys.stderr)
        sys.exit(1)
    if args.json:
        print(json.dumps((payload.get("result") or {}).get("clips") or []))
        return
    _print_clips(payload.get("result"))


def cmd_quota(args):
    status, payload = _request("GET", "/api/me")
    # 401 anonymous, 404 self-host (the cloud router is not mounted): neither
    # is an error, there is simply no minute quota to report.
    if status in (401, 404):
        print("no cloud account in play: self-hosted or anonymous, no minute quota")
        return
    if status >= 400:
        _die(status, payload)
    if args.json:
        print(json.dumps(payload))
        return
    print(f"plan: {payload.get('plan')}")
    print(f"minutes: {payload.get('minutes')}")
    print(f"entitled: {payload.get('entitled')}")


def cmd_publish(args):
    body = {
        "job_id": args.job_id,
        "clip_index": args.clip_index,
        "platforms": [p.strip() for p in args.platforms.split(",") if p.strip()],
    }
    if args.title:
        body["title"] = args.title
    if args.schedule:
        body["scheduled_date"] = args.schedule
    if args.timezone:
        body["timezone"] = args.timezone
    status, payload = _request("POST", "/api/social/post", body)
    if status >= 400:
        _die(status, payload)
    print(json.dumps(payload) if args.json else f"publishing: {payload}")


def _regional_code(value):
    region = value.upper()
    if not re.fullmatch(r"[A-Z]{2}", region):
        raise argparse.ArgumentTypeError("use a two-letter country code")
    return region


def _discover(url, label, headers=None):
    request = urllib.request.Request(url)
    for name, value in (headers or {}).items():
        request.add_unredirected_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(2 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"{label} returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(f"could not reach {label}") from exc
    if len(data) > 2 * 1024 * 1024:
        raise ValueError(f"{label} returned an oversized response")
    return data


def _search_trends(region):
    data = _discover(f"https://trends.google.com/trending/rss?geo={region}", "Google Trends")
    try:
        channel = ET.fromstring(data).find("channel")
    except ET.ParseError:
        raise ValueError("Google Trends returned invalid RSS") from None
    if channel is None:
        raise ValueError("Google Trends RSS channel is missing")
    return [
        {"topic": (item.findtext("title") or "").strip(),
         "approx_searches": item.findtext("{*}approx_traffic"),
         "published_at": item.findtext("pubDate")}
        for item in channel.findall("item") if (item.findtext("title") or "").strip()
    ]


def _youtube_data(resource, params, key):
    url = f"https://www.googleapis.com/youtube/v3/{resource}?{urllib.parse.urlencode(params)}"
    # Keep the API key out of request URLs and any error output.
    data = _discover(url, "YouTube Data API", headers={"X-Goog-Api-Key": key})
    try:
        items = json.loads(data)["items"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("YouTube Data API returned an invalid response") from None
    if not isinstance(items, list):
        raise ValueError("YouTube Data API returned an invalid item list")
    return items


def _short_video_references(topic, region, days, limit, key):
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    search = _youtube_data("search", {
        "part": "snippet", "type": "video", "videoDuration": "short", "order": "viewCount",
        "publishedAfter": since, "regionCode": region, "q": topic, "maxResults": 50,
        "fields": "items(id/videoId)",
    }, key)
    ids = list(dict.fromkeys(
        row["id"]["videoId"] for row in search
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", row.get("id", {}).get("videoId", ""))
    ))
    if not ids:
        return []
    videos = _youtube_data("videos", {
        "part": "snippet,contentDetails,statistics", "id": ",".join(ids),
        "fields": "items(id,snippet(title,channelTitle,publishedAt),contentDetails/duration,statistics/viewCount)",
    }, key)
    references = []
    for video in videos:
        if video.get("id") not in ids:
            continue
        metadata = video.get("contentDetails", {})
        duration = re.fullmatch(r"PT(?:(\d+)M)?(?:(\d+)S)?", metadata.get("duration", ""))
        if not duration:
            continue
        minutes, seconds = (int(value or 0) for value in duration.groups())
        total_seconds = minutes * 60 + seconds
        if 0 < total_seconds <= 180:
            snippet = video.get("snippet", {})
            references.append({
                "url": f"https://www.youtube.com/watch?v={video['id']}",
                "title": snippet.get("title", video["id"]), "channel": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"), "seconds": total_seconds,
                "views": int(video.get("statistics", {}).get("viewCount", 0)),
            })
    return sorted(references, key=lambda row: -row["views"])[:limit]


def _safe_line(value):
    return "".join(char if char.isprintable() else " " for char in str(value))


def cmd_trends(args):
    if args.topic:
        key = os.environ.get("YOUTUBE_DATA_API_KEY")
        if not key:
            raise ValueError("set YOUTUBE_DATA_API_KEY for YouTube short-video references")
        rows = _short_video_references(args.topic, args.region, args.days, args.limit, key)
        source = "youtube_short_video_references"
        heading = f"YouTube short-video references for {_safe_line(args.topic)} | {args.region} | last {args.days} days"
    else:
        rows = _search_trends(args.region)[:args.limit]
        source = "google_trending_searches"
        heading = f"Google trending searches | {args.region}"
    if args.json:
        print(json.dumps({"source": source, "region": args.region, "items": rows}))
        return
    print(heading)
    for row in rows:
        if args.topic:
            print(f"{_safe_line(row['title'])} | {row['views']:,} views | {row['seconds']}s\n  {row['url']}")
        else:
            searches = row["approx_searches"] or "unknown"
            print(f"{_safe_line(row['topic'])} | {_safe_line(searches)} searches")


def _topic_words(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def _load_trend_sources(path):
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list):
        raise ValueError("source manifest must contain a JSON array")
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict) or not isinstance(source.get("url"), str):
            raise ValueError(f"source {index} needs a URL")
        url = urllib.parse.urlsplit(source["url"])
        if url.scheme not in {"http", "https"} or not url.hostname or url.username is not None:
            raise ValueError(f"source {index} needs an HTTP(S) URL without credentials")
        topics = source.get("topics")
        if not isinstance(topics, list) or not topics or not all(
            isinstance(topic, str) and _topic_words(topic) for topic in topics
        ):
            raise ValueError(f"source {index} needs a nonempty topics array")
        profile = source.get("profile", "explainer")
        if not isinstance(profile, str) or profile not in TREND_PROFILES:
            raise ValueError(f"source {index} profile must be explainer, process or story")
        start, end = source.get("source_start"), source.get("source_end")
        if start is not None or end is not None:
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end)):
                raise ValueError(f"source {index} needs numeric source_start and source_end")
            if not 0 <= start < end:
                raise ValueError(f"source {index} needs 0 <= source_start < source_end")
    return sources


def _trend_matches(sources, topics):
    seen = set()
    source_topics = []
    for source in sources:
        start, end = source.get("source_start"), source.get("source_end")
        identity = (source["url"], None if start is None else float(start),
                    None if end is None else float(end))
        tags = [f" {_topic_words(topic)} " for topic in source["topics"]]
        source_topics.append((source, identity, tags))
    for topic in topics:
        title = f" {_topic_words(topic)} "
        for source, identity, tags in source_topics:
            if identity not in seen and any(tag in title for tag in tags):
                seen.add(identity)
                yield source, topic, identity


def _submit_trend_job(db, identity, body):
    identity = hashlib.sha256(json.dumps([_base(), *identity]).encode()).hexdigest()
    with db:
        # Flush the claim before sending. If the API accepted a request whose
        # response was lost, the next scheduled run will skip it.
        claimed = db.execute("INSERT OR IGNORE INTO jobs(source) VALUES (?)", (identity,)).rowcount
    if not claimed:
        if db.execute("SELECT job_id FROM jobs WHERE source = ?", (identity,)).fetchone() == (None,):
            print(f"unconfirmed trend job {identity}; check OpenShorts before resubmitting", file=sys.stderr)
        return None
    status, payload = _request("POST", "/api/process", body)
    hold = _quality_hold(payload)
    if status >= 400 or hold:
        if hold or (400 <= status < 500 and status != 408):
            with db:
                db.execute("DELETE FROM jobs WHERE source = ?", (identity,))
        if hold:
            _die(409, hold)
        _die(status, payload)
    job_id = payload.get("job_id") if isinstance(payload, dict) else None
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("could not confirm job submission; check OpenShorts before resubmitting")
    with db:
        db.execute("UPDATE jobs SET job_id = ? WHERE source = ?", (job_id, identity))
    return job_id


def cmd_trend_generate(args):
    if args.submit and not args.acknowledge:
        raise ValueError("--submit requires --acknowledge for the rights to use the listed sources")
    sources = _load_trend_sources(args.manifest)
    topics = args.topic or [item["topic"] for item in _search_trends(args.region)]
    matches = _trend_matches(sources, topics)
    results = []
    failure = None
    state = args.state or args.manifest.with_suffix(".trends.sqlite3")
    if state.resolve() == args.manifest.resolve():
        raise ValueError("state file and source manifest must use different paths")
    db = sqlite3.connect(state) if args.submit else None
    with (closing(db) if db is not None else nullcontext()):
        if db is not None:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (source TEXT PRIMARY KEY, job_id TEXT)")
        for source, topic, identity in matches:
            profile = source.get("profile", "explainer")
            source_range = {name: source[name] for name in ("source_start", "source_end")
                            if source.get(name) is not None}
            body = {"url": source["url"], "acknowledged": True, "output_format": "vertical",
                    "framing_method": "auto", "target_clips": args.clips,
                    **TREND_PROFILES[profile], **source_range}
            if args.webhook:
                body["webhook_url"] = args.webhook
                secret = os.environ.get("OPENSHORTS_WEBHOOK_SECRET")
                if secret:
                    body["webhook_secret"] = secret
            result = {"topic": topic, "url": source["url"], "profile": profile, **source_range}
            if db is not None:
                try:
                    job_id = _submit_trend_job(db, identity, body)
                except (SystemExit, ValueError) as exc:
                    failure = exc
                    break
                if job_id is None:
                    continue
                result["job_id"] = job_id
            results.append(result)
            if len(results) >= args.limit:
                break
    if args.json:
        print(json.dumps({"dry_run": not args.submit, "items": results}))
    elif not results and failure is None:
        print("No matching sources to queue." if args.submit else "No matching sources.")
    else:
        for result in results:
            status = f"Queued {_safe_line(result['job_id'])}" if args.submit else "Would queue"
            span = (f" ({result['source_start']:g}–{result['source_end']:g}s)"
                    if "source_start" in result else "")
            print(f"{status} | {_safe_line(result['topic'])} | {result['profile']} | {_safe_line(result['url'])}{span}")
    if failure is not None:
        raise failure


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openshorts",
        description="Clip long videos into vertical shorts via the OpenShorts API.",
    )
    parser.add_argument("--json", action="store_true", help="raw JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="submit a video URL for clipping")
    p.add_argument("url", help="YouTube or direct video URL")
    p.add_argument("--layouts", help="comma list: auto,split,screencast,speaker_cut,punch_in")
    p.add_argument("--framing", choices=FRAMING_METHODS, help="vertical framing method")
    p.add_argument("--format", help="output shape: vertical, square or horizontal")
    p.add_argument("--webhook", help="webhook URL fired once when the job ends")
    p.add_argument("--webhook-secret", help="HMAC secret for X-OpenShorts-Signature")
    p.add_argument("--force-low-quality", action="store_true", help="accept a low-resolution source")
    p.add_argument("--wait", action="store_true", help="stream logs until the job ends")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("status", help="job status and last log line")
    p.add_argument("job_id")
    p.add_argument("--watch", action="store_true", help="stream logs until the job ends")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("clips", help="list finished clips with links")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_clips)

    p = sub.add_parser("quota", help="plan and remaining minutes")
    p.set_defaults(func=cmd_quota)

    p = sub.add_parser("publish", help="post or schedule one clip to social platforms")
    p.add_argument("job_id")
    p.add_argument("clip_index", type=int)
    p.add_argument("--platforms", required=True, help="comma list: tiktok,instagram,youtube,facebook,linkedin,threads,bluesky")
    p.add_argument("--title")
    p.add_argument("--schedule", help="ISO datetime for scheduled posting")
    p.add_argument("--timezone", help="IANA timezone for --schedule")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("trends", help="discover regional searches or recent YouTube short-video references")
    p.add_argument("--region", type=_regional_code, default="US", help="country code (default: US)")
    p.add_argument("--topic", help="search YouTube for short-video references for this topic")
    p.add_argument("--days", type=int, choices=range(1, 31), default=7, metavar="1-30",
                   help="YouTube reference lookback (default: 7 days)")
    p.add_argument("--limit", type=int, choices=range(1, 51), default=10, metavar="1-50")
    p.set_defaults(func=cmd_trends)

    p = sub.add_parser("trend-generate", help="match trends to your video library and queue clips")
    p.add_argument("manifest", type=Path, help="JSON array of authorized URLs and their topics")
    p.add_argument("--region", type=_regional_code, default="US", help="country code (default: US)")
    p.add_argument("--topic", action="append", help="use a reviewed topic instead of Google Trends (repeatable)")
    p.add_argument("--submit", action="store_true", help="queue jobs (default: preview matches)")
    p.add_argument("--acknowledge", action="store_true", help="confirm your rights to use the listed sources")
    p.add_argument("--state", type=Path, help="SQLite job history (default: manifest.trends.sqlite3)")
    p.add_argument("--limit", type=int, choices=range(1, 11), default=1, metavar="1-10",
                   help="maximum jobs per run (default: 1)")
    p.add_argument("--clips", type=int, choices=range(1, 16), default=3, metavar="1-15",
                   help="target clips per matching source (default: 3)")
    p.add_argument("--webhook", help="job completion URL (uses OPENSHORTS_WEBHOOK_SECRET if set)")
    p.set_defaults(func=cmd_trend_generate)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
