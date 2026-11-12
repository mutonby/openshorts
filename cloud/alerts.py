"""Operational email alerts for the admin (proxy out of credits, high failure rate).

Tracks recent managed-job outcomes in memory and emails ADMIN_EMAIL (via Resend)
when something looks broken, with a per-alert cooldown so it never spams.
"""
import html as _html
import asyncio
import os
import time
from collections import deque

from .config import settings
from .emails import send_email

_recent = deque(maxlen=12)          # rolling window of recent job outcomes (ok bool)
_last_alert = {}                    # alert kind -> last-sent epoch
_ALERT_COOLDOWN = 3600              # 1 hour between repeats of the same alert
_FAIL_WINDOW_MIN = 6               # need at least this many recent jobs to judge a rate
_FAIL_THRESHOLD = 5                # ...and this many failures among them

# Specific signatures of a genuine proxy / download-stage failure. Kept precise
# on purpose: a bare "proxy"/"credit"/"balance" match fired on any job whose logs
# merely echoed the proxy URL (yt-dlp debug) or whose video title contained one of
# those words, producing false "out of credits" alerts for jobs that actually
# failed later in processing. These phrases only appear in real proxy failures.
_PROXY_HINTS = (
    "proxyerror",
    "cannot connect to proxy",
    "failed to connect to proxy",
    "unable to connect to proxy",
    "proxy authentication required",
    "407 proxy",
    "http error 407",
    "tunnel connection failed",
    "402 payment required",
    "out of credits",
    "insufficient balance",
)


def _looks_like_proxy_error(err: str) -> bool:
    e = (err or "").lower()
    return any(k in e for k in _PROXY_HINTS)


def _classify_failure(err: str) -> str:
    """One-word category for the last error, so the alert points the right way."""
    e = (err or "").lower()
    if _looks_like_proxy_error(e):
        return "proxy"
    if "no_audio" in e or "no audio" in e:
        return "no audio"
    if "sign in to confirm" in e or "not a bot" in e or "http error 403" in e \
            or "http error 429" in e or "video unavailable" in e or "read timed out" in e:
        return "youtube download"
    if "whisper" in e or "faster_whisper" in e or "transcrib" in e or "av/container" in e:
        return "transcription"
    # User content rejected by the AI provider's policy filter — deterministic,
    # not actionable on our side. Named so the alert doesn't read as an outage.
    if "prohibited_content" in e or "blocked this video" in e or "blocked its answer" in e:
        return "blocked content (user video)"
    # The source held nothing clip-shaped (typically an already-short video) —
    # content-shaped like the policy block above, not an outage.
    if "did not return usable clips" in e or "clip detection failed" in e:
        return "no clips found (user video)"
    if "gemini" in e or "google.genai" in e:
        return "gemini"
    if "ffmpeg" in e or "reframe" in e:
        return "ffmpeg/render"
    return "mixed"


def _cooldown_ok(kind: str) -> bool:
    now = time.time()
    if now - _last_alert.get(kind, 0) < _ALERT_COOLDOWN:
        return False
    _last_alert[kind] = now
    return True


# Prefix on every OpenShorts Telegram message. The chat is shared with other
# products (Upload-Post, …), so this tags which one each alert is from.
TELEGRAM_PREFIX = "OPENSHORTS ✂️ - "


def user_ref(user_id) -> str:
    """How a user is named in an operational alert: never by email address.

    These alerts go to Telegram, whose Bot API is operated from outside the EEA
    with no adequacy decision and no DPA available to us — so the messages must
    not carry personal data. They used to read "alice@example.com bought +60
    minutes". The first 8 hex of the account uuid is enough for the operator to
    find the row (``select * from users where id::text like '3f9a1c2b%'``) and
    is not, on its own, an identifier of a person to anyone reading the chat.
    """
    ref = str(user_id or "").replace("-", "")
    return f"user {ref[:8]}" if ref else "a user"


class TelegramSendError(RuntimeError):
    """A failed Bot API call, its text already free of the bot token and the
    chat id (httpx puts the request URL, token included, in its messages)."""


def redact_telegram(text) -> str:
    """``text`` without the bot token or the chat id, for logs and errors."""
    out = str(text)
    for secret, label in ((settings.telegram_bot_token, "<bot-token>"),
                          (settings.telegram_chat_id, "<chat-id>")):
        secret = str(secret or "").strip()
        if len(secret) >= 4:
            out = out.replace(secret, label)
    # Any other bot URL that slipped in (a token rotated mid-flight).
    import re
    return re.sub(r"/bot[0-9]+:[A-Za-z0-9_-]+", "/bot<bot-token>", out)


async def send_telegram(text: str, *, raise_errors: bool = False):
    """Push a plain-text message to the admin's Telegram chat. No-op if unset.

    Best-effort by default: never raises — an alert failing must not break a
    webhook or job. ``raise_errors=True`` is for callers with their own retry
    (the daily digest), where swallowing the failure means losing the message.
    """
    if not settings.telegram_configured:
        return
    failure = None
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        payload = {"chat_id": settings.telegram_chat_id,
                   "text": TELEGRAM_PREFIX + text,
                   "disable_web_page_preview": True}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except Exception as e:
        failure = redact_telegram(f"{type(e).__name__}: {e}")
    if failure is None:
        return
    if raise_errors:
        # Raised outside the handler, so the original (whose message, args and
        # request carry the token-bearing URL) is not even its context.
        raise TelegramSendError(failure)
    print(f"⚠️  Telegram alert failed: {failure}")


async def send_admin_alert(subject: str, body: str):
    """Notify the admin via every configured channel (email + Telegram)."""
    # Telegram first — instant, and configured independently of email.
    await send_telegram(f"{subject}\n\n{body}")

    to = settings.admin_email
    if not to or not settings.smtp_configured:
        if not settings.telegram_configured:
            print(f"⚠️  [ADMIN ALERT] {subject}\n{body[:500]}"
                  + ("" if to else "  (set ADMIN_EMAIL + SMTP_* or TELEGRAM_* to receive these)"))
        return
    # The body quotes job logs and error text: text, never markup.
    html = ("<pre style='font:13px/1.5 monospace;white-space:pre-wrap'>"
            f"{_html.escape(str(body))}</pre>")
    await send_email(to, f"[OpenShorts] {subject}", html)


async def record_job_outcome(ok: bool, error_text: str = ""):
    """Record a managed job's result and fire an alert if the picture looks bad."""
    _recent.append(bool(ok))
    if ok:
        return

    # 1) Proxy / credits problem — most urgent, alert immediately AND open the
    # incident so the proxy watcher keeps nagging until the balance is back
    # (one alert at 3am is easy to miss; an exhausted proxy takes all
    # YouTube-URL ingest down until someone tops it up).
    if _looks_like_proxy_error(error_text):
        if not _watch_down.get(_PAID_TARGET):
            _watch_down[_PAID_TARGET] = time.time()
            _watch_nag[_PAID_TARGET] = time.time()
        if _cooldown_ok("proxy"):
            await send_admin_alert(
                "🔴 Proxy error — may be out of credits",
                "A managed job failed with a proxy-related error. Check your proxy "
                "balance — downloads will keep failing until it's topped up.\n"
                "This repeats every 2 h until the proxy answers again.\n\n"
                f"Error:\n{error_text[:1200]}",
            )
        return

    # 2) High failure rate — report it honestly and classify the last error
    # instead of always blaming the download path (it's often transcription of
    # a silent upload, a bad video, etc.).
    recent = list(_recent)
    fails = recent.count(False)
    if len(recent) >= _FAIL_WINDOW_MIN and fails >= _FAIL_THRESHOLD and _cooldown_ok("failrate"):
        await send_admin_alert(
            f"⚠️ High job failure rate ({_classify_failure(error_text)})",
            f"{fails} of the last {len(recent)} managed jobs failed.\n\n"
            f"Last error:\n{error_text[:1200]}",
        )


# --- Proxy watcher --------------------------------------------------------------
# A dead download route takes YouTube-URL ingest down (or silently moves it to
# per-GB billing), and a single alert at the moment it dies is easy to miss.
# The watcher probes every configured route on a schedule, keeps nagging on
# Telegram until it's fixed, and confirms recovery (incident of 19-aug-2026:
# DataImpulse 407 TRAFFIC_EXHAUSTED, found by accident hours later).
#
# Watched targets:
# - "static proxy pool" (STATIC_PROXY_URLS): flat-rate ISP IPs, the primary
#   route. Ingest still works when they die (paid fallback) but every GB
#   starts costing money — that's worth a nag of its own.
# - "paid proxy" (PROXY_URL): the per-GB last resort; dead = ingest at risk.
_PROXY_PROBE_INTERVAL = 1800        # seconds between probes (30 min)
_PROXY_RENOTIFY = 7200              # keep nagging every 2 h while it stays down
# 204-No-Content endpoint: the probe costs a handful of bytes of paid traffic.
# Plain HTTP on purpose: HTTPS through DataImpulse fails in httpx with an SSL
# record-layer error even when the proxy is healthy (yt-dlp's stack is fine
# with it), which would make an HTTPS probe cry wolf. An exhausted balance
# rejects the request before forwarding, so HTTP still detects the 407.
_PROXY_PROBE_URL = "http://www.google.com/generate_204"
# The static pool gets probed against YouTube itself, not google.com: on
# 28-aug-2026 YouTube refused the static IPs for hours ("Video unavailable" /
# bot-check) while google.com kept answering 204, so the watcher said UP and
# every job silently moved to the per-GB proxy ($14 that day). A healthy
# static must return the watch page with a playable player response; bytes
# through the statics are flat-rate, so the ~600 KB page is free.
_STATIC_PROBE_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
_STATIC_OK_MARKERS = ('"playabilityStatus"', '"status":"OK"')
# The probe has to ask the way the DOWNLOAD asks, or it measures a question
# nobody in production is asking. Anonymously, a datacenter IP making ~400
# YouTube hits a day is rate-limited into "Sign in to confirm you're not a
# bot" — playabilityStatus comes back LOGIN_REQUIRED — while the very same IP
# serves every real download fine, because those carry these cookies.
# Measured 8-sep-2026 on all three statics: anonymous LOGIN_REQUIRED, with
# cookies OK, while proxy_usage showed 119 YouTube jobs on HD-static1 in 24 h
# and zero paid bytes. Without this the watcher cries wolf every 2 h and the
# alert tells you money is burning when none is.
_static_cookie_jar = None
_static_cookie_src = None

# The cookies themselves are a route: once Google rotates the session they
# came from, every download is anonymous whatever the file says, the statics
# get rate-limited into the bot-check one by one and jobs fall to per-GB
# billing. That is invisible to the pool probe (one IP still answering = UP)
# and to the ledger alert (it names the IPs, not the session). 30-sep-2026:
# ~900 MB paid in 70 min, cap hit, and the watcher had said nothing; yt-dlp
# had been warning "cookies are no longer valid" in every job log. The watch
# page carries the same signal yt-dlp reads: ytcfg "LOGGED_IN".
_COOKIE_TARGET = "YouTube session cookies"
_session_seen = {"logged_in": None}


def _note_session(body):
    """Record what the watch page says about the cookie session, if anything."""
    import re as _re
    m = _re.search(r'"LOGGED_IN":(true|false)', body or "")
    if m:
        _session_seen["logged_in"] = m.group(1) == "true"


def _probe_cookies():
    """The download's YOUTUBE_COOKIES as a jar httpx can send, or None.

    Parsed once and cached (the probe runs every 30 minutes). Never logged:
    the blob is a live YouTube session.
    """
    global _static_cookie_jar, _static_cookie_src
    raw = os.environ.get("YOUTUBE_COOKIES", "")
    if not raw:
        return None
    if raw == _static_cookie_src:
        return _static_cookie_jar
    jar = None
    try:
        import tempfile
        from http.cookiejar import MozillaCookieJar
        # Pre-validate: handed a blob that is not a cookie file, the stdlib
        # prints "http.cookiejar bug!" with a traceback before raising. That
        # would land in the container log every 30 minutes, so recognise the
        # shape ourselves (a Netscape record is 7 tab-separated fields).
        if not any(len(ln.split("\t")) == 7
                   for ln in raw.splitlines() if ln and not ln.startswith("#")):
            raise ValueError("not a Netscape cookie file")
        body = raw if raw.lstrip().startswith("# Netscape") else \
            "# Netscape HTTP Cookie File\n" + raw
        path = os.path.join(tempfile.mkdtemp(), "yt_cookies.txt")
        with open(path, "w") as fh:
            fh.write(body if body.endswith("\n") else body + "\n")
        jar = MozillaCookieJar(path)
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        print(f"\u26a0\ufe0f  Static probe could not read YOUTUBE_COOKIES ({type(e).__name__}); "
              f"probing anonymously, which YouTube bot-checks.")
        jar = None
    _static_cookie_src, _static_cookie_jar = raw, jar
    return jar


def _playability(body):
    """The playabilityStatus value in a watch page, for the alert detail."""
    import re as _re
    m = _re.search(r'"playabilityStatus":\{"status":"([A-Z_]+)"', body or "")
    return m.group(1) if m else "no player response"
_PROXY_STRIKES = 2                  # consecutive failed probes before alerting
_watch_down = {}                    # target name -> down-since epoch
_watch_nag = {}                     # target name -> last-nag epoch
_watch_strikes = {}                 # target name -> consecutive failed probes

_PAID_TARGET = "paid proxy"
_STATIC_TARGET = "static proxy pool"


def _watch_targets():
    """[(name, [proxy urls])] — a target is up if ANY of its urls answers."""
    targets = []
    statics = [p.strip() for p in
               os.environ.get("STATIC_PROXY_URLS", "").split(",") if p.strip()]
    if statics:
        targets.append((_STATIC_TARGET, statics))
    paid = os.environ.get("PROXY_URL", "").strip()
    if paid:
        targets.append((_PAID_TARGET, [paid]))
    return targets


async def _probe_one(proxy):
    """(ok, detail) for one request through one proxy.

    The paid proxy keeps the plain-HTTP 204 probe (HTTPS through DataImpulse
    breaks in httpx even when healthy). A static IP is probed against a real
    YouTube watch page: answering is not enough, the page must carry a
    playable player response, or the IP is flagged and every download is
    about to fall through to per-GB billing.
    """
    is_paid = proxy == os.environ.get("PROXY_URL", "").strip()
    try:
        import httpx
        kwargs = {"proxy": proxy, "timeout": 20}
        if not is_paid:
            jar = _probe_cookies()
            if jar is not None:
                kwargs["cookies"] = jar
        async with httpx.AsyncClient(**kwargs) as client:
            resp = await client.get(_PROXY_PROBE_URL if is_paid else _STATIC_PROBE_URL,
                                    follow_redirects=not is_paid)
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}"
        if is_paid:
            return True, ""
        body = resp.text or ""
        if "cookies" in kwargs:
            _note_session(body)
        if all(m in body for m in _STATIC_OK_MARKERS):
            return True, ""
        # Name what YouTube actually said. "LOGIN_REQUIRED" with cookies
        # configured means the session expired (rotate them); the old wording
        # blamed the IP for every case and sent us hunting the wrong thing.
        return False, (f"YouTube answered but the video is not playable "
                       f"(playabilityStatus: {_playability(body)})")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _watch_severity(name):
    """(emoji, headline, impact) for a route being down, judged by what is
    actually affected. A dead fallback while the primary answers is a lost
    safety net, not an outage: the first version of this alert said "jobs
    will keep failing" for that case and read as the platform being down
    (27-aug-2026, a single ReadTimeout on the paid proxy)."""
    if name == _COOKIE_TARGET:
        return "🟠", "YouTube session cookies expired", (
            "YouTube no longer recognises the session in YOUTUBE_COOKIES, so every "
            "download runs anonymously: the static IPs get bot-checked and jobs fall "
            "to the PER-GB proxy (and fail once its daily budget is hit). Fix: log in "
            "to YouTube in a private window, export the cookies, close the window "
            "without logging out, update YOUTUBE_COOKIES and redeploy.")
    static_down = bool(_watch_down.get(_STATIC_TARGET))
    paid_down = bool(_watch_down.get(_PAID_TARGET))
    other_configured = len(_watch_targets()) > 1
    if not other_configured:
        return "🔴", f"{name} DOWN", "It is the only YouTube route: YouTube-URL jobs are failing until it is fixed."
    if static_down and paid_down:
        return "🔴", "ALL YouTube proxies DOWN", "Static pool and paid proxy both fail: YouTube-URL jobs are failing. Direct file URLs are unaffected."
    if name == _STATIC_TARGET:
        return "🟠", "static proxy pool down, running on the paid proxy", ("YouTube jobs still work through the PER-GB paid proxy, so every download costs money until the statics are back.")
    return "ℹ️", "paid proxy (fallback) not answering", ("No impact right now: the static pool is UP and carries all YouTube jobs. You only lose the safety net if the statics fail too.")


async def _watch_update(name, ok, detail):
    """Shared incident state machine: alert on down, nag, confirm recovery.

    A route has to fail ``_PROXY_STRIKES`` probes in a row (30 min apart)
    before anyone is paged: one ReadTimeout on a per-GB proxy is routine."""
    now = time.time()
    if ok:
        _watch_strikes[name] = 0
        if _watch_down.get(name):
            mins = int((now - _watch_down[name]) / 60)
            await send_admin_alert(
                f"✅ {name} recovered",
                f"The {name} answers again after ~{mins} min down.",
            )
            _watch_down[name] = None
            _watch_nag[name] = 0.0
        return
    _watch_strikes[name] = _watch_strikes.get(name, 0) + 1
    # A rotated session is deterministic (the same answer on every IP), so
    # it does not get the benefit of the doubt a flaky proxy gets.
    strikes = 1 if name == _COOKIE_TARGET else _PROXY_STRIKES
    if not _watch_down.get(name):
        if _watch_strikes[name] < strikes:
            return  # one miss: wait for the next probe before saying anything
        _watch_down[name] = now
        _watch_nag[name] = now
        icon, headline, impact = _watch_severity(name)
        await send_admin_alert(
            f"{icon} {headline}",
            f"{impact}\n\nThe {name} failed {strikes} probe(s) in a row "
            f"({_PROXY_PROBE_INTERVAL // 60} min apart). This repeats every 2 h "
            f"until it answers again.\n\nProbe error: {detail[:400]}",
        )
    elif now - _watch_nag[name] >= _PROXY_RENOTIFY:
        _watch_nag[name] = now
        hours = (now - _watch_down[name]) / 3600
        icon, headline, impact = _watch_severity(name)
        await send_admin_alert(
            f"{icon} {headline} ({hours:.1f} h)",
            f"{impact}\n\nProbe error: {detail[:400]}",
        )


async def proxy_watch_tick():
    """One probe cycle over every configured route."""
    # Paid-proxy events held back by the ledger's cooldown go out with the
    # next tick at the latest (cloud/proxy_ledger.flush_alerts).
    try:
        from . import proxy_ledger
        await proxy_ledger.flush_alerts()
    except Exception:
        pass
    _session_seen["logged_in"] = None
    for name, urls in _watch_targets():
        ok, detail = False, "no urls"
        for u in urls:
            ok, detail = await _probe_one(u)
            if ok:
                break
        await _watch_update(name, ok, detail)
    # Judged from the watch pages the static probes just fetched with the
    # cookies attached; nothing to say when no static answered at all.
    if _probe_cookies() is not None and _session_seen["logged_in"] is not None:
        await _watch_update(_COOKIE_TARGET, _session_seen["logged_in"],
                            'watch page ytcfg says "LOGGED_IN":false with the '
                            'configured cookies attached')


async def proxy_watch_loop():
    """Background task started from the app lifespan (managed mode only)."""
    while True:
        try:
            await proxy_watch_tick()
        except Exception as e:
            print(f"⚠️  Proxy watch error: {e}")
        await asyncio.sleep(_PROXY_PROBE_INTERVAL)
