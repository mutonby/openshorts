"""Record Twitch chat while a channel is live, for the hype engine.

A VOD's chat is hard to get afterwards, but reading chat while the stream is
live is allowed and free: Twitch's IRC accepts anonymous read-only
connections (nick ``justinfanNNNNN``, no token). This keeps one connection,
joins the configured channels and appends every chat message as one JSON line
to ``<dir>/<channel>/<YYYY-MM-DD>.jsonl`` (UTC day of the message):

    {"ts": 1795301680.123, "author_id": "123456", "text": "KEKW"}

``ts`` is Twitch's own send time (``tmi-sent-ts``), so lines line up with the
VOD's start time without trusting this machine's clock. Known bots are
skipped; author ids are kept only so one person spamming counts once.

When a Twitch VOD is reviewed later, ``attach`` converts the stretch the VOD
covers into a VOD-relative CSV the hype engine reads like an uploaded replay
(livestream.imported_chat). Recordings never leave this machine.

    python -m chat_recorder somechannel otherchannel
    CHAT_RECORD_CHANNELS=a,b python -m chat_recorder

Directory: CHAT_RECORDINGS_DIR (default output/.chat-recordings).
"""
import csv
import json
import os
import re
import socket
import ssl
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOST, PORT = "irc.chat.twitch.tv", 6697
BOTS = frozenset({"nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "wizebot",
                  "soundalerts", "sery_bot", "botrixoficial", "pokemoncommunitygame", "kofistreambot"})
_CHANNEL = re.compile(r"^[a-z0-9_]{3,25}$")
_LINE = re.compile(r"^(?:@(?P<tags>\S+) )?(?::(?P<prefix>\S+) )?(?P<command>\S+)(?: (?P<params>.*))?$")


def recordings_dir():
    configured = os.environ.get("CHAT_RECORDINGS_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(os.environ.get("OUTPUT_DIR", "output")) / ".chat-recordings"


def channel_name(value):
    """A Twitch login from a name or channel URL, or None."""
    value = str(value or "").strip().lower()
    match = re.match(r"^(?:https?://)?(?:www\.)?twitch\.tv/([a-z0-9_]+)", value)
    name = match.group(1) if match else value.lstrip("#")
    return name if _CHANNEL.match(name) else None


def _tags(raw):
    out = {}
    for part in (raw or "").split(";"):
        key, _, value = part.partition("=")
        out[key] = value.replace("\\s", " ").replace("\\:", ";").replace("\\\\", "\\")
    return out


def parse(line, now=None):
    """(channel, record) for a chat message line, ("PING", payload) for a
    ping, ("RECONNECT", None) when Twitch asks for one, else None."""
    match = _LINE.match(line.rstrip("\r\n"))
    if not match:
        return None
    command = match.group("command")
    params = match.group("params") or ""
    if command == "PING":
        return "PING", params.lstrip(":")
    if command == "RECONNECT":
        return "RECONNECT", None
    if command != "PRIVMSG":
        return None
    target, _, text = params.partition(" :")
    channel = target.lstrip("#").lower()
    tags = _tags(match.group("tags"))
    login = (match.group("prefix") or "").split("!", 1)[0].lower()
    if login in BOTS or not text.strip():
        return None
    try:
        ts = int(tags.get("tmi-sent-ts", "")) / 1000.0
    except ValueError:
        ts = now if now is not None else time.time()
    return channel, {"ts": round(ts, 3), "author_id": tags.get("user-id") or login,
                     "text": text.strip()[:500]}


class _Writer:
    """Append records to <dir>/<channel>/<UTC day>.jsonl, one open file per
    channel, rolled over at midnight UTC."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.open = {}

    def write(self, channel, record):
        day = datetime.fromtimestamp(record["ts"], timezone.utc).strftime("%Y-%m-%d")
        key, handle = self.open.get(channel, (None, None))
        if key != day:
            if handle:
                handle.close()
            folder = self.directory / channel
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = open(folder / f"{day}.jsonl", "a", encoding="utf-8", buffering=1)
            self.open[channel] = (day, handle)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self):
        for _, handle in self.open.values():
            handle.close()
        self.open = {}


def _connect(channels):
    raw = socket.create_connection((HOST, PORT), timeout=30)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=HOST)
    nick = f"justinfan{int(time.time()) % 90000 + 10000}"
    for line in ("CAP REQ :twitch.tv/tags twitch.tv/commands", "PASS SCHMOOPIIE", f"NICK {nick}",
                 *(f"JOIN #{c}" for c in channels)):
        sock.sendall((line + "\r\n").encode())
    sock.settimeout(360)          # Twitch pings about every 5 minutes
    return sock


def run(channels, directory=None, *, stop=None):
    """Record until ``stop()`` is true (forever by default), reconnecting with
    backoff. ``channels``: Twitch logins."""
    channels = sorted({c for c in (channel_name(x) for x in channels) if c})
    if not channels:
        raise ValueError("Name at least one Twitch channel to record.")
    writer = _Writer(directory or recordings_dir())
    backoff = 1
    print(f"Recording chat for: {', '.join(channels)} -> {writer.directory}", flush=True)
    try:
        while not (stop and stop()):
            try:
                sock = _connect(channels)
                backoff = 1
                buffer = b""
                while not (stop and stop()):
                    data = sock.recv(65536)
                    if not data:
                        raise ConnectionError("chat connection closed")
                    buffer += data
                    *lines, buffer = buffer.split(b"\r\n")
                    for raw in lines:
                        parsed = parse(raw.decode("utf-8", errors="replace"))
                        if not parsed:
                            continue
                        kind, payload = parsed
                        if kind == "PING":
                            sock.sendall(f"PONG :{payload}\r\n".encode())
                        elif kind == "RECONNECT":
                            raise ConnectionError("Twitch asked for a reconnect")
                        else:
                            writer.write(kind, payload)
            except (OSError, ConnectionError) as exc:
                print(f"Chat connection lost ({type(exc).__name__}); retrying in {backoff}s.", flush=True)
                time.sleep(backoff)
                backoff = min(60, backoff * 2)
    finally:
        writer.close()


def records(channel, start, end, directory=None):
    """Recorded messages of ``channel`` with start <= ts < end (epoch s)."""
    folder = Path(directory or recordings_dir()) / channel
    day = datetime.fromtimestamp(start, timezone.utc).date()
    last = datetime.fromtimestamp(end, timezone.utc).date()
    while day <= last:
        path = folder / f"{day.isoformat()}.jsonl"
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                        ts = float(item["ts"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    if start <= ts < end:
                        yield item
        day += timedelta(days=1)


def attach(channel, vod_start, duration, dest, directory=None, *, source_start=0.0):
    """Write the recorded chat a VOD covers as a VOD-relative CSV at ``dest``
    (time_seconds,author_id,text). Returns the number of messages, 0 when
    nothing was recorded (no file is written then)."""
    channel = channel_name(channel)
    if not channel or not vod_start or not duration:
        return 0
    start = vod_start + source_start
    rows = [(round(float(r["ts"]) - vod_start, 3), str(r.get("author_id") or ""), str(r.get("text") or ""))
            for r in records(channel, start, start + duration + 60, directory)]
    if not rows:
        return 0
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        out = csv.writer(handle)
        out.writerow(["time_seconds", "author_id", "text"])
        out.writerows(sorted(rows))
    tmp.replace(dest)
    return len(rows)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Record Twitch chat for highlight discovery.")
    parser.add_argument("channels", nargs="*", help="Twitch channel names or URLs")
    parser.add_argument("--out", help="recordings directory (default CHAT_RECORDINGS_DIR)")
    args = parser.parse_args(argv)
    channels = args.channels or [c for c in os.environ.get("CHAT_RECORD_CHANNELS", "").split(",") if c.strip()]
    try:
        run(channels, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
