"""Watch streamers' channels and queue every new VOD for highlight review.

Run next to the API (it lists VODs with yt-dlp) and leave it running; a review
of each finished stream is waiting when you look:

    python -m channel_watch --acknowledge twitch:somestreamer youtube:@somechannel

A VOD is submitted once it is finished: not live, and its duration unchanged
across checks at least SETTLE_SECONDS apart (a Twitch VOD grows while the
stream is live). It goes to POST /api/process as an audio-first highlight
review (hype engine, review before rendering) with the chosen framing and the
viral edit on. Chat recorded live by chat_recorder.py is attached by the job.

The first pass records the VODs that already exist without submitting them
(``--backfill N`` submits the N newest of each channel instead). State is a
SQLite file (``--state``, default output/.channel-watch.sqlite3). The API is
OPENSHORTS_API_URL (default http://127.0.0.1:8000); GEMINI_API_KEY, when set,
is sent with each job so the clip judge runs; OPENSHORTS_API_KEY is sent as a
bearer token for an API that needs one. ``--acknowledge`` confirms you have
the rights to process these channels' videos.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

SETTLE_SECONDS = 600
RECENT = 5


def listing(channel):
    """(kind, URL of the channel's past streams) for ``twitch:<login>``,
    ``youtube:@handle``, or a channel URL; None for anything else."""
    value = str(channel or "").strip()
    kind, _, rest = value.partition(":")
    if kind.lower() == "twitch" and re.fullmatch(r"[A-Za-z0-9_]{3,25}", rest):
        return "twitch", f"https://www.twitch.tv/{rest.lower()}/videos?filter=archives&sort=time"
    if kind.lower() == "youtube" and re.fullmatch(r"@[A-Za-z0-9._-]{3,100}", rest):
        return "youtube", f"https://www.youtube.com/{rest}/streams"
    match = re.match(r"^https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]{3,25})/?$", value)
    if match:
        return listing(f"twitch:{match.group(1)}")
    match = re.match(r"^https?://(?:www\.)?youtube\.com/(@[A-Za-z0-9._-]{3,100}|channel/UC[A-Za-z0-9_-]{22})", value)
    if match:
        return "youtube", f"https://www.youtube.com/{match.group(1)}/streams"
    return None


class _QuietLogger:
    """A pass every few minutes: only yt-dlp's warnings and errors reach the log."""

    def debug(self, message):
        pass

    info = debug

    def warning(self, message):
        print(f"[watch] yt-dlp: {str(message)[:200]}", flush=True)

    error = warning


def _ydl(opts):
    import yt_dlp
    from youtube_io import runtime_options
    return yt_dlp.YoutubeDL({**runtime_options(), "quiet": True, "noprogress": True, "cachedir": False,
                             "socket_timeout": 20, **opts, "logger": _QuietLogger()})


def recent(channel, limit=RECENT):
    """The newest VODs of ``channel``: [{"id", "url", "live"}], newest first."""
    import livestream
    import ssrf_guard

    found = listing(channel)
    if not found:
        raise ValueError(f"not a channel: {channel!r} (use twitch:<login> or youtube:@handle)")
    _kind, url = found
    with ssrf_guard.guarded(), _ydl({"extract_flat": True, "playlistend": limit}) as ydl:
        info = ydl.extract_info(url, download=False)
    out = []
    for entry in (info or {}).get("entries") or []:
        vod = livestream.staged_url(entry.get("url") or entry.get("webpage_url") or "")
        if not vod and entry.get("id") and re.fullmatch(r"[A-Za-z0-9_-]{11}", str(entry["id"])):
            vod = livestream.staged_url(f"https://www.youtube.com/watch?v={entry['id']}")
        if vod:
            out.append({"id": vod, "url": vod,
                        "live": entry.get("live_status") in ("is_live", "is_upcoming", "post_live")})
    return out[:limit]


def finished_duration(url):
    """The VOD's duration if it is a finished recording, else None."""
    import ssrf_guard

    with ssrf_guard.guarded(), _ydl({}) as ydl:
        info = ydl.extract_info(url, download=False, process=False)
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
        return None
    duration = float(info.get("duration") or 0)
    return duration if duration > 0 else None


class State:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS vods (
            id TEXT PRIMARY KEY, channel TEXT, duration REAL, checked REAL, stable_since REAL,
            job_id TEXT, submitted REAL)""")
        # A channel seen with no VODs yet must not be "first sight" again later.
        self.db.execute("CREATE TABLE IF NOT EXISTS channels (channel TEXT PRIMARY KEY, first_seen REAL)")

    def get(self, vod):
        row = self.db.execute("SELECT duration, stable_since, job_id FROM vods WHERE id=?", (vod,)).fetchone()
        return None if row is None else {"duration": row[0], "stable_since": row[1], "job_id": row[2]}

    def seen(self, vod, channel, duration, now):
        known = self.get(vod)
        stable = now if (not known or known["duration"] is None or duration is None
                         or abs(known["duration"] - duration) > 2) else known["stable_since"]
        with self.db:
            self.db.execute("""INSERT INTO vods(id, channel, duration, checked, stable_since) VALUES (?,?,?,?,?)
                               ON CONFLICT(id) DO UPDATE SET duration=excluded.duration,
                               checked=excluded.checked, stable_since=excluded.stable_since""",
                            (vod, channel, duration, now, stable))
        return stable

    def baseline(self, vod, channel, now):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO vods(id, channel, checked, job_id) VALUES (?,?,?,'baseline')",
                            (vod, channel, now))

    def submitted(self, vod, job_id, now):
        with self.db:
            self.db.execute("UPDATE vods SET job_id=?, submitted=? WHERE id=?", (job_id, now, vod))

    def known_channel(self, channel):
        return self.db.execute("SELECT 1 FROM channels WHERE channel=?", (channel,)).fetchone() is not None

    def add_channel(self, channel, now):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO channels(channel, first_seen) VALUES (?,?)", (channel, now))


def job_body(url, *, framing="streamer", clips=None, viral_edit=True):
    body = {"url": url, "acknowledged": True, "output_format": "vertical", "framing_method": framing,
            "review_candidates": True, "highlight_discovery": True, "audio_first": True,
            "viral_edit": "1" if viral_edit else "0"}
    if clips:
        body["target_clips"] = int(clips)
    return body


def submit(api, body):
    """POST /api/process; returns (status, payload)."""
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if os.environ.get("GEMINI_API_KEY"):
        headers["X-Gemini-Key"] = os.environ["GEMINI_API_KEY"]
    if os.environ.get("OPENSHORTS_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENSHORTS_API_KEY']}"
    request = urllib.request.Request(api.rstrip("/") + "/api/process", data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode() or "{}")
        except ValueError:
            return exc.code, {"detail": str(exc.reason)}


def check(state, channels, *, api, settle=SETTLE_SECONDS, backfill=0, now=None,
          lister=recent, duration_of=finished_duration, sender=submit, body=job_body):
    """One pass over ``channels``. Returns the job ids submitted."""
    now = time.time() if now is None else now
    jobs = []
    for channel in channels:
        backfilled = set()
        try:
            vods = lister(channel)
        except Exception as exc:
            print(f"[watch] {channel}: could not list VODs ({type(exc).__name__})", flush=True)
            continue
        if not state.known_channel(channel):
            # First sight of this channel: what already exists is the baseline.
            for vod in vods[backfill:]:
                state.baseline(vod["id"], channel, now)
            state.add_channel(channel, now)
            vods = vods[:backfill]
            # Older VODs asked for by --backfill are finished: no settling.
            backfilled = {vod["id"] for vod in vods}
        for vod in vods:
            known = state.get(vod["id"])
            if known and known["job_id"]:
                continue
            if vod["live"]:
                state.seen(vod["id"], channel, None, now)
                continue
            try:
                duration = duration_of(vod["url"])
            except Exception as exc:
                print(f"[watch] {vod['url']}: metadata failed ({type(exc).__name__})", flush=True)
                continue
            stable_since = state.seen(vod["id"], channel, duration, now)
            if duration is None or (vod["id"] not in backfilled and now - stable_since < settle):
                continue
            status, payload = sender(api, body(vod["url"]))
            job_id = payload.get("job_id") if isinstance(payload, dict) else None
            if status < 300 and job_id:
                state.submitted(vod["id"], job_id, now)
                jobs.append(job_id)
                print(f"[watch] queued {vod['url']} as job {job_id}", flush=True)
            else:
                detail = payload.get("detail") if isinstance(payload, dict) else payload
                print(f"[watch] {vod['url']}: the API refused it ({status}: {str(detail)[:200]})", flush=True)
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Queue each new stream VOD for highlight review.")
    parser.add_argument("channels", nargs="+", help="twitch:<login>, youtube:@handle or a channel URL")
    parser.add_argument("--acknowledge", action="store_true",
                        help="confirm you have the rights to process these channels' videos")
    parser.add_argument("--api", default=os.environ.get("OPENSHORTS_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--interval", type=int, default=600, help="seconds between passes (default 600)")
    parser.add_argument("--settle", type=int, default=SETTLE_SECONDS,
                        help="seconds a VOD's length must stay unchanged before it is queued")
    parser.add_argument("--backfill", type=int, default=0, help="on first sight, queue the N newest VODs")
    parser.add_argument("--framing", default="streamer", choices=("streamer", "gameplay", "auto"))
    parser.add_argument("--clips", type=int, choices=range(1, 16), metavar="1-15")
    parser.add_argument("--no-viral-edit", action="store_true")
    parser.add_argument("--state", default=os.path.join(os.environ.get("OUTPUT_DIR", "output"),
                                                        ".channel-watch.sqlite3"))
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args(argv)
    if not args.acknowledge:
        parser.error("--acknowledge is required: confirm you have the rights to these channels' videos")
    bad = [c for c in args.channels if not listing(c)]
    if bad:
        parser.error(f"not a channel: {', '.join(bad)} (use twitch:<login> or youtube:@handle)")
    os.makedirs(os.path.dirname(os.path.abspath(args.state)), exist_ok=True)
    state = State(args.state)
    body = lambda url: job_body(url, framing=args.framing, clips=args.clips,
                                viral_edit=not args.no_viral_edit)
    print(f"[watch] {len(args.channels)} channel(s), every {args.interval}s -> {args.api}", flush=True)
    while True:
        check(state, args.channels, api=args.api, settle=args.settle, backfill=args.backfill, body=body)
        if args.once:
            return 0
        time.sleep(max(60, args.interval))


if __name__ == "__main__":
    sys.exit(main())
