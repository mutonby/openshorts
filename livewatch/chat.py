"""Live chat readers. Each one calls ``on_message(ChatMessage)`` from its own thread.

- Twitch: the public IRC gateway, read-only and anonymous (``justinfan``
  nick), over TLS. No account or key is needed to read a public chat.
- YouTube: ``yt-dlp --write-subs --sub-langs live_chat`` appends the live
  chat as JSON lines while the stream runs; the reader tails that file.
- Kick: no reader. Kick's supported chat access is its official webhook API,
  which needs a public URL a home PC does not have, so Kick moments come from
  the audio signal (``detector``) alone.

The parsers are pure functions and unit-tested; the readers reconnect with
backoff and never raise into the caller.
"""
from __future__ import annotations

import json
import random
import socket
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import procs


@dataclass(frozen=True)
class ChatMessage:
    ts: float      # wall-clock seconds (epoch) the message was sent
    user: str
    text: str


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def _unescape_tag(v: str) -> str:
    return (v.replace(r"\s", " ").replace(r"\:", ";").replace(r"\\", "\\")
            .replace(r"\r", "").replace(r"\n", ""))


def parse_twitch_irc(line: str, now: Optional[float] = None) -> Optional[ChatMessage]:
    """A ``PRIVMSG`` line (with IRCv3 tags) as a ChatMessage, else None."""
    if not line:
        return None
    tags = {}
    rest = line
    if rest.startswith("@"):
        tag_part, _, rest = rest.partition(" ")
        for kv in tag_part[1:].split(";"):
            k, _, v = kv.partition("=")
            tags[k] = _unescape_tag(v)
    if not rest.startswith(":"):
        return None
    prefix, _, rest = rest[1:].partition(" ")
    cmd, _, rest = rest.partition(" ")
    if cmd != "PRIVMSG":
        return None
    _chan, _, text = rest.partition(" :")
    if not text and not rest.endswith(" :"):
        return None
    user = tags.get("display-name") or prefix.split("!", 1)[0]
    try:
        ts = int(tags["tmi-sent-ts"]) / 1000.0
    except (KeyError, ValueError):
        ts = time.time() if now is None else now
    if text.startswith("\x01ACTION ") and text.endswith("\x01"):
        text = text[8:-1]
    return ChatMessage(ts=ts, user=user.lower(), text=text.strip())


def _runs_text(message: dict) -> str:
    out = []
    for run in (message or {}).get("runs", []) or []:
        if "text" in run:
            out.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"] or {}
            if emoji.get("isCustomEmoji"):
                sc = (emoji.get("shortcuts") or [""])[0]
                out.append(f" {sc.strip(':')} " if sc else " ")
            else:
                out.append(emoji.get("emojiId", ""))
    return "".join(out).strip()


def parse_youtube_chat_line(line: str):
    """Every text message in one line of yt-dlp's ``live_chat.json``."""
    try:
        data = json.loads(line)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    actions = (data.get("replayChatItemAction") or {}).get("actions") or []
    out = []
    for a in actions:
        item = (a.get("addChatItemAction") or {}).get("item") or {}
        r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if not r:
            continue
        try:
            ts = int(r.get("timestampUsec")) / 1e6
        except (TypeError, ValueError):
            continue
        user = (r.get("authorExternalChannelId")
                or (r.get("authorName") or {}).get("simpleText") or "?")
        text = _runs_text(r.get("message") or {})
        if text:
            out.append(ChatMessage(ts=ts, user=str(user).lower(), text=text))
    return out


def parse_youtube_replay_line(line: str):
    """[(seconds into the video, ChatMessage)] from one line of an ended stream's chat replay.

    A replay line carries ``videoOffsetTimeMsec``: where in the recording the
    message appeared, which is what clips are cut by.
    """
    try:
        data = json.loads(line)
        offset = int((data.get("replayChatItemAction") or {}).get("videoOffsetTimeMsec")) / 1000.0
    except (ValueError, TypeError, AttributeError):
        return []
    return [(offset, m) for m in parse_youtube_chat_line(line)]


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
class _Reader(threading.Thread):
    def __init__(self, name: str, on_message: Callable[[ChatMessage], None], log):
        super().__init__(name=name, daemon=True)
        self.on_message = on_message
        self.log = log
        self.stop_event = threading.Event()
        self.count = 0

    def stop(self):
        self.stop_event.set()

    def _emit(self, msg: ChatMessage):
        self.count += 1
        try:
            self.on_message(msg)
        except Exception as e:  # a detector bug must not kill the reader
            self.log(f"chat handler error: {e}")


class TwitchChatReader(_Reader):
    HOST = "irc.chat.twitch.tv"
    PORT = 6697

    def __init__(self, channel: str, on_message, log=print):
        super().__init__(f"twitch-chat-{channel}", on_message, log)
        self.channel = channel.lower().lstrip("#")

    def run(self):
        backoff = 2.0
        while not self.stop_event.is_set():
            try:
                self._session()
                backoff = 2.0
            except Exception as e:
                if self.stop_event.is_set():
                    break
                self.log(f"twitch chat {self.channel}: {e}; reconnecting in {backoff:.0f}s")
            self.stop_event.wait(backoff)
            backoff = min(backoff * 2, 60)

    def _session(self):
        raw = socket.create_connection((self.HOST, self.PORT), timeout=10)
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(raw, server_hostname=self.HOST)
        sock.settimeout(1.0)
        nick = f"justinfan{random.randint(10000, 99999)}"
        sock.sendall(f"CAP REQ :twitch.tv/tags\r\nNICK {nick}\r\nJOIN #{self.channel}\r\n".encode())
        buf = b""
        last_data = time.time()
        try:
            while not self.stop_event.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    if time.time() - last_data > 360:  # server PINGs every ~5 min
                        raise ConnectionError("chat went silent")
                    continue
                if not chunk:
                    raise ConnectionError("connection closed")
                last_data = time.time()
                buf += chunk
                while b"\r\n" in buf:
                    line, buf = buf.split(b"\r\n", 1)
                    text = line.decode("utf-8", "replace")
                    if text.startswith("PING"):
                        sock.sendall(("PONG" + text[4:] + "\r\n").encode())
                        continue
                    msg = parse_twitch_irc(text)
                    if msg:
                        self._emit(msg)
        finally:
            try:
                sock.close()
            except OSError:
                pass


class YouTubeChatReader(_Reader):
    def __init__(self, url: str, work_dir: Path, on_message, log=print):
        super().__init__("youtube-chat", on_message, log)
        self.url = url
        self.work_dir = Path(work_dir)
        self.proc: Optional[subprocess.Popen] = None

    def stop(self):
        super().stop()
        procs.terminate(self.proc)

    def run(self):
        backoff = 5.0
        while not self.stop_event.is_set():
            try:
                self._session()
            except Exception as e:
                self.log(f"youtube chat: {e}")
            if self.stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, 120)

    def _session(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        out_tmpl = str(self.work_dir / f"chat{stamp}.%(ext)s")
        cmd = [procs.ytdlp(), "--no-warnings", "--quiet", "--skip-download",
               "--no-playlist", "--write-subs", "--sub-langs", "live_chat",
               "-o", out_tmpl, self.url]
        self.proc = procs.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = time.time()
        part = self.work_dir / f"chat{stamp}.live_chat.json.part"
        final = self.work_dir / f"chat{stamp}.live_chat.json"
        fh = None
        pending = b""
        try:
            while not self.stop_event.is_set():
                if fh is None:
                    for p in (part, final):
                        if p.exists():
                            fh = open(p, "rb")
                            break
                if fh is not None:
                    chunk = fh.read()
                    if chunk:
                        pending += chunk
                        *lines, pending = pending.split(b"\n")
                        for ln in lines:
                            for msg in parse_youtube_chat_line(ln.decode("utf-8", "replace")):
                                if msg.ts >= started - 30:  # skip the history dump
                                    self._emit(msg)
                if self.proc.poll() is not None and (fh is None or not chunk):
                    break
                self.stop_event.wait(1.0)
        finally:
            if fh:
                fh.close()
            procs.terminate(self.proc)
            for p in self.work_dir.glob(f"chat{stamp}*"):
                try:
                    p.unlink()
                except OSError:
                    pass
