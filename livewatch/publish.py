"""Free publishing: YouTube and Instagram through their official APIs, TikTok to your phone.

No paid service in between. Each network is posted with your own free
developer credentials:

- **YouTube Shorts** (YouTube Data API v3, ``videos.insert`` resumable upload).
  A free Google Cloud project with an OAuth client of type "Desktop app";
  "Connect YouTube" in the dashboard opens Google's consent page and keeps a
  refresh token. If Google keeps uploads from an unaudited project private,
  the post is reported as "uploaded private" instead of silently succeeding.
- **Instagram Reels** (Instagram API with Instagram Login, graph.instagram.com).
  A free Meta developer app; generate an access token for your own
  professional (Creator/Business) account in the app dashboard. Instagram
  fetches the video from a public URL, so for each post a one-file server
  with a random path is exposed through a temporary Cloudflare quick tunnel
  (``cloudflared``) and closed right after: the dashboard is never exposed.
- **Phone handoff** (Telegram bot): TikTok (and anything else without a free
  public-posting API) is sent to your phone with its caption ready, posted in
  a few taps. TikTok's API keeps posts from unaudited apps private.

Every function takes a ``session`` so it can be tested without the network.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlencode

from . import procs

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
YT_UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
YT_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
IG_API = "https://graph.instagram.com"
IG_VERSION = "v27.0"
TG_API = "https://api.telegram.org"
TG_MAX_BYTES = 49 * 1024 * 1024


class PublishError(RuntimeError):
    pass


def _http(session):
    if session is None:
        import requests as session
    return session


def _msg(r) -> str:
    try:
        d = r.json()
        err = d.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        return str(d.get("error_description") or d.get("description") or err or d)[:300]
    except ValueError:
        return (r.text or "")[:300]


# --------------------------------------------------------------------------- #
# YouTube
# --------------------------------------------------------------------------- #
def youtube_auth_url(client_id: str, redirect_uri: str) -> tuple:
    """(url to open, state, code_verifier) for Google's consent page (PKCE)."""
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(16)
    q = {"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code",
         "scope": YT_SCOPE, "access_type": "offline", "prompt": "consent",
         "code_challenge": challenge, "code_challenge_method": "S256", "state": state}
    return f"{GOOGLE_AUTH}?{urlencode(q)}", state, verifier


def youtube_exchange_code(client_id: str, client_secret: str, code: str, redirect_uri: str,
                          verifier: str, session=None) -> str:
    """The refresh token for an authorisation code."""
    r = _http(session).post(GOOGLE_TOKEN, data={
        "client_id": client_id, "client_secret": client_secret, "code": code,
        "redirect_uri": redirect_uri, "grant_type": "authorization_code",
        "code_verifier": verifier}, timeout=30)
    if r.status_code != 200:
        raise PublishError(f"Google refused the sign-in: {_msg(r)}")
    tok = (r.json() or {}).get("refresh_token")
    if not tok:
        raise PublishError("Google returned no refresh token: remove livewatch's access at "
                           "myaccount.google.com/permissions and connect again")
    return tok


def youtube_access_token(client_id: str, client_secret: str, refresh_token: str, session=None) -> str:
    r = _http(session).post(GOOGLE_TOKEN, data={
        "client_id": client_id, "client_secret": client_secret,
        "refresh_token": refresh_token, "grant_type": "refresh_token"}, timeout=30)
    if r.status_code != 200:
        raise PublishError("YouTube sign-in expired or was revoked: press Connect YouTube again "
                           f"({_msg(r)})")
    return r.json()["access_token"]


def youtube_upload(video: Path, title: str, description: str, tags: List[str], *, client_id: str,
                   client_secret: str, refresh_token: str, session=None) -> dict:
    """Upload a Short. Returns {"id", "url", "privacy"}; privacy "private" means Google
    kept it private (an unaudited project), which the caller reports."""
    http = _http(session)
    token = youtube_access_token(client_id, client_secret, refresh_token, session=session)
    size = Path(video).stat().st_size
    meta = {"snippet": {"title": title[:100], "description": description[:4900],
                        "tags": [t for t in tags if t][:15], "categoryId": "20"},
            "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}}
    r = http.post(f"{YT_UPLOAD}?uploadType=resumable&part=snippet,status", json=meta, timeout=60,
                  headers={"Authorization": f"Bearer {token}", "X-Upload-Content-Length": str(size),
                           "X-Upload-Content-Type": "video/mp4"})
    if r.status_code != 200 or not r.headers.get("Location"):
        raise PublishError(f"YouTube refused the upload: {_msg(r)}")
    with open(video, "rb") as fh:
        r = http.put(r.headers["Location"], data=fh, timeout=900,
                     headers={"Authorization": f"Bearer {token}", "Content-Type": "video/mp4"})
    if r.status_code not in (200, 201):
        raise PublishError(f"YouTube upload failed: {_msg(r)}")
    body = r.json() or {}
    vid = body.get("id", "")
    return {"id": vid, "url": f"https://youtube.com/shorts/{vid}" if vid else "",
            "privacy": ((body.get("status") or {}).get("privacyStatus") or "")}


# --------------------------------------------------------------------------- #
# Instagram
# --------------------------------------------------------------------------- #
def instagram_me(token: str, session=None) -> dict:
    r = _http(session).get(f"{IG_API}/{IG_VERSION}/me",
                           params={"fields": "user_id,username", "access_token": token}, timeout=30)
    if r.status_code != 200:
        raise PublishError(f"Instagram refused the token: {_msg(r)}")
    d = r.json() or {}
    uid = str(d.get("user_id") or d.get("id") or "")
    if not uid:
        raise PublishError("Instagram did not return the account id")
    return {"id": uid, "username": d.get("username", "")}


def instagram_refresh(token: str, session=None) -> Optional[str]:
    """A fresh 60-day token (Instagram accepts it once the old one is a day old)."""
    r = _http(session).get(f"{IG_API}/refresh_access_token",
                           params={"grant_type": "ig_refresh_token", "access_token": token}, timeout=30)
    if r.status_code != 200:
        return None
    return (r.json() or {}).get("access_token")


class OneFileTunnel:
    """Serve one file at a random path, reachable from the internet through a
    Cloudflare quick tunnel, for as long as the ``with`` block lasts."""

    def __init__(self, path: Path, timeout: float = 90):
        self.path = Path(path)
        self.secret = secrets.token_urlsafe(24)
        self.timeout = timeout
        self.url = ""
        self._srv = None
        self._proc: Optional[subprocess.Popen] = None

    def _handler(self):
        path, secret = self.path, self.secret

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _head(self) -> bool:
                if self.path != f"/{secret}/{path.name}":
                    self.send_error(404)
                    return False
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                return True

            def do_HEAD(self):
                self._head()

            def do_GET(self):
                if not self._head():
                    return
                with open(path, "rb") as fh:
                    while chunk := fh.read(256 * 1024):
                        try:
                            self.wfile.write(chunk)
                        except OSError:
                            return
        return H

    def __enter__(self):
        exe = procs.cloudflared()
        if not exe:
            raise PublishError("Instagram needs cloudflared (winget install Cloudflare.cloudflared)")
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        port = self._srv.server_address[1]
        self._proc = procs.popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        found = []

        def read():
            for raw in self._proc.stdout:
                m = re.search(rb"https://[a-z0-9-]+\.trycloudflare\.com", raw)
                if m and not found:
                    found.append(m.group(0).decode())
        threading.Thread(target=read, daemon=True).start()
        deadline = time.time() + self.timeout
        while not found and time.time() < deadline and self._proc.poll() is None:
            time.sleep(0.3)
        if not found:
            self.__exit__(None, None, None)
            raise PublishError("could not open the temporary public link for Instagram (cloudflared)")
        self.url = f"{found[0]}/{self.secret}/{self.path.name}"
        return self

    def __exit__(self, *exc):
        procs.terminate(self._proc)
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()


def instagram_publish(video: Path, caption: str, *, token: str, session=None,
                      tunnel_factory: Callable = OneFileTunnel, poll_seconds: float = 10,
                      max_wait: float = 600) -> dict:
    """Publish a Reel. Returns {"id", "username"}."""
    http = _http(session)
    me = instagram_me(token, session=session)
    base = f"{IG_API}/{IG_VERSION}/{me['id']}"
    with tunnel_factory(video) as tun:
        # A fresh quick-tunnel name takes a few seconds to exist in DNS. Asking this
        # PC's resolver too early makes it cache "not found" for a minute, so ask
        # Cloudflare's public DNS instead, as Instagram's servers will.
        host = re.sub(r"^https://([^/]+)/.*$", r"\1", tun.url)
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                d = http.get("https://cloudflare-dns.com/dns-query", timeout=10,
                             params={"name": host, "type": "A"},
                             headers={"accept": "application/dns-json"}).json()
                if d.get("Status") == 0 and d.get("Answer"):
                    break
            except Exception:
                pass
            time.sleep(3)
        time.sleep(3)
        r = http.post(f"{base}/media", timeout=60, data={
            "media_type": "REELS", "video_url": tun.url, "caption": caption[:2200],
            "share_to_feed": "true", "access_token": token})
        if r.status_code != 200 or not (r.json() or {}).get("id"):
            raise PublishError(f"Instagram refused the Reel: {_msg(r)}")
        container = r.json()["id"]
        waited, status = 0.0, ""
        while waited < max_wait:
            time.sleep(poll_seconds)
            waited += poll_seconds
            s = http.get(f"{IG_API}/{IG_VERSION}/{container}", timeout=30,
                         params={"fields": "status_code,status", "access_token": token})
            status = ((s.json() or {}).get("status_code") or "") if s.status_code == 200 else ""
            if status in ("FINISHED", "ERROR", "EXPIRED"):
                break
        if status != "FINISHED":
            raise PublishError(f"Instagram could not process the video ({status or 'timed out'})")
    r = http.post(f"{base}/media_publish", timeout=60,
                  data={"creation_id": container, "access_token": token})
    if r.status_code != 200 or not (r.json() or {}).get("id"):
        raise PublishError(f"Instagram did not publish the Reel: {_msg(r)}")
    return {"id": r.json()["id"], "username": me["username"]}


# --------------------------------------------------------------------------- #
# Telegram (phone handoff)
# --------------------------------------------------------------------------- #
def telegram_find_chat(token: str, session=None) -> dict:
    """The chat of whoever last messaged the bot: {"chat_id", "name"}."""
    r = _http(session).get(f"{TG_API}/bot{token}/getUpdates", timeout=30)
    if r.status_code != 200:
        raise PublishError(f"Telegram refused the bot token: {_msg(r)}")
    for upd in reversed((r.json() or {}).get("result") or []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            name = chat.get("username") or chat.get("first_name") or chat.get("title") or ""
            return {"chat_id": str(chat["id"]), "name": name}
    raise PublishError("no message found: open your bot in Telegram, press Start (or send it "
                       "any message), then try again")


def telegram_send_text(token: str, chat_id: str, text: str, session=None) -> None:
    r = _http(session).post(f"{TG_API}/bot{token}/sendMessage", timeout=30,
                            data={"chat_id": chat_id, "text": text[:4096],
                                  "disable_web_page_preview": "true"})
    if r.status_code != 200:
        raise PublishError(f"Telegram: {_msg(r)}")


def telegram_handoff(video: Path, captions: Dict[str, str], *, token: str, chat_id: str,
                     title: str = "", session=None) -> dict:
    """Send the Short and one ready-to-paste caption per network to your phone."""
    http = _http(session)
    if Path(video).stat().st_size > TG_MAX_BYTES:
        raise PublishError("the Short is over Telegram's 50 MB limit for bots")
    nets = ", ".join(n.capitalize() for n in captions) or "your networks"
    with open(video, "rb") as fh:
        r = http.post(f"{TG_API}/bot{token}/sendVideo", timeout=300,
                      data={"chat_id": chat_id, "supports_streaming": "true",
                            "caption": f"New Short for {nets}: {title}"[:1024]},
                      files={"video": (Path(video).name, fh, "video/mp4")})
    if r.status_code != 200:
        raise PublishError(f"Telegram refused the video: {_msg(r)}")
    for net, text in captions.items():
        telegram_send_text(token, chat_id, f"{net.upper()} caption (tap to copy):\n\n{text}",
                           session=session)
    return {"message_id": ((r.json() or {}).get("result") or {}).get("message_id")}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
API_NETWORKS = ("youtube", "instagram")


def route(platforms: List[str], youtube_via: str, instagram_via: str, others_via: str) -> Dict[str, str]:
    """{network: "api" | "phone" | "upload-post" | "off"} for the networks to post to."""
    out = {}
    for p in platforms:
        via = youtube_via if p == "youtube" else instagram_via if p == "instagram" else others_via
        out[p] = via if via in ("api", "phone", "upload-post", "off") else "phone"
        if out[p] == "api" and p not in API_NETWORKS:
            out[p] = "phone"
    return {p: v for p, v in out.items() if v != "off"}


def credentials() -> dict:
    g = os.environ.get
    return {"yt_id": g("YOUTUBE_CLIENT_ID", ""), "yt_secret": g("YOUTUBE_CLIENT_SECRET", ""),
            "yt_refresh": g("YOUTUBE_REFRESH_TOKEN", ""), "ig_token": g("INSTAGRAM_ACCESS_TOKEN", ""),
            "tg_token": g("TELEGRAM_BOT_TOKEN", ""), "tg_chat": g("TELEGRAM_CHAT_ID", ""),
            "up_key": g("UPLOADPOST_API_KEY", "") or g("UPLOAD_POST_API_KEY", "")}


def missing(routes: Dict[str, str], creds: dict, upload_post_user: str = "") -> List[str]:
    """What is not set up yet for these routes, as messages for the user."""
    out = []
    if routes.get("youtube") == "api" and not (creds["yt_id"] and creds["yt_secret"] and creds["yt_refresh"]):
        out.append("YouTube is not connected (Settings > Posting accounts > Connect YouTube)")
    if routes.get("instagram") == "api" and not creds["ig_token"]:
        out.append("no Instagram access token (Settings > Posting accounts)")
    if "phone" in routes.values() and not (creds["tg_token"] and creds["tg_chat"]):
        out.append("the Telegram bot for phone handoff is not set up (Settings > Posting accounts)")
    if "upload-post" in routes.values() and not (creds["up_key"] and upload_post_user):
        out.append("Upload-Post key or profile missing")
    return out


def publish(video: Path, posts: Dict[str, Dict[str, str]], routes: Dict[str, str], *,
            tags: List[str], upload_post_user: str = "", log=print, session=None,
            tunnel_factory: Callable = OneFileTunnel) -> dict:
    """Post ``video`` to every routed network; one network failing never stops the others.

    Returns {"status": posted|partial|failed|private, "networks": {net: {...}}}.
    """
    creds = credentials()
    results: Dict[str, dict] = {}
    phone: Dict[str, str] = {}
    up: Dict[str, Dict[str, str]] = {}
    for net, via in routes.items():
        text = posts.get(net) or {}
        if via == "phone":
            phone[net] = (text.get("description") or text.get("title") or "").strip()
            continue
        if via == "upload-post":
            up[net] = text
            continue
        try:
            if net == "youtube":
                res = youtube_upload(video, text.get("title", ""), text.get("description", ""), tags,
                                     client_id=creds["yt_id"], client_secret=creds["yt_secret"],
                                     refresh_token=creds["yt_refresh"], session=session)
                if res["privacy"] == "private":
                    results[net] = {"status": "private", "url": res["url"],
                                    "why": "Google keeps API uploads private until the Cloud project "
                                           "passes its free audit; make it public in YouTube Studio"}
                    log(f"   youtube: uploaded but PRIVATE ({res['url']}): Google's unaudited-project rule")
                else:
                    results[net] = {"status": "posted", "url": res["url"]}
                    log(f"   youtube: posted {res['url']}")
            elif net == "instagram":
                caption = text.get("description") or text.get("title") or ""
                res = instagram_publish(video, caption, token=creds["ig_token"], session=session,
                                        tunnel_factory=tunnel_factory)
                results[net] = {"status": "posted", "id": res["id"]}
                log(f"   instagram: posted a Reel on @{res['username']}")
        except Exception as e:  # noqa: BLE001 - reported per network
            results[net] = {"status": "failed", "error": str(e)[:300]}
            log(f"   {net}: failed: {e}")
    if phone:
        try:
            telegram_handoff(video, phone, token=creds["tg_token"], chat_id=creds["tg_chat"],
                             title=next(iter(posts.values()), {}).get("title", ""), session=session)
            for net in phone:
                results[net] = {"status": "sent to phone"}
            log(f"   sent to your phone for {', '.join(phone)}")
        except Exception as e:  # noqa: BLE001
            for net in phone:
                results[net] = {"status": "failed", "error": str(e)[:300]}
            log(f"   phone handoff failed: {e}")
    if up:
        from . import poster
        try:
            poster.post_video(video, up, api_key=creds["up_key"], user=upload_post_user, session=session)
            for net in up:
                results[net] = {"status": "posted", "via": "upload-post"}
            log(f"   posted through Upload-Post: {', '.join(up)}")
        except Exception as e:  # noqa: BLE001
            for net in up:
                results[net] = {"status": "failed", "error": str(e)[:300]}
            log(f"   Upload-Post failed: {e}")
    states = {r["status"] for r in results.values()}
    ok = {"posted", "sent to phone"}
    if states and states <= ok:
        status = "posted"
    elif states & (ok | {"private"}):
        status = "partial"
    else:
        status = "failed"
    return {"status": status, "networks": results}
