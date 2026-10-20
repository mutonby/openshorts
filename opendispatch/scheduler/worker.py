"""Scheduler worker — loops, picks up due rows, calls adapters, retries on failure.

Run alongside the API:
    python -m scheduler.worker
or via docker-compose (worker service).
"""

from __future__ import annotations

import logging
import os
import random
import socket
from urllib.parse import urlparse
import ipaddress
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from opendispatch.adapters import ADAPTERS
from opendispatch.adapters.errors import AUTH, PERMANENT, PUBLISHED, clip
from opendispatch.api.queue import get_queue
from opendispatch.api.schema import ContentUnit
from opendispatch.profiles import ProfileStore, profile_env

# Health heartbeat — written every poll loop so /healthz can show worker status.
_REPO_ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_PATH = Path(os.getenv("OPEN_DISPATCH_DATA", str(_REPO_ROOT / "data"))) / ".worker_heartbeat"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# httpx logs every request at INFO with the FULL URL. Telegram puts the bot
# token in the URL path and Instagram put the access token in the query
# string, so those lines wrote credentials into the log file. Verified:
#   INFO httpx: HTTP Request: POST https://api.telegram.org/bot<TOKEN>/sendMessage
for _noisy in ("httpx", "httpcore", "tweepy", "atproto"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("open-dispatch.worker")

POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))
MAX_ATTEMPTS = int(os.getenv("WORKER_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = int(os.getenv("WORKER_BACKOFF_BASE", "30"))  # seconds


def _write_heartbeat() -> None:
    """Write the current UTC timestamp so /healthz can show worker liveness."""
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_PATH.write_text(datetime.now(tz=timezone.utc).isoformat(), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass  # never crash the worker loop over a heartbeat write


def _backoff_seconds(attempts: int) -> int:
    return BACKOFF_BASE * (2 ** max(0, attempts - 1)) + random.randint(0, 10)


# Set by run_once for the duration of one pass. Kept off the _publish
# signature so existing callers that monkeypatch `_publish(row)` still match.
_RESOLVE = None
_QUEUE = None


def _publish(row: dict) -> tuple[bool, str, str]:
    platform_key = row.get("platform", "")
    platform, _, account = platform_key.partition(":")
    if account == "default":
        account = None
    adapter = ADAPTERS.get(platform)
    if not adapter:
        return False, "", f"no adapter for platform '{platform}'"
    unit = ContentUnit.from_dict(row["unit"])

    creds = None
    if _RESOLVE is not None and unit.creds_ref:
        creds = _RESOLVE(unit.creds_ref, platform)

    # Inject profile credentials if the unit references a profile.
    # Explicit creds win inside the adapter; the env overlay stays for the
    # standalone worker, which has no creds object.
    profile_id = unit.profile_id or (row.get("unit") or {}).get("profile_id")
    profile = ProfileStore().get(profile_id) if profile_id else None

    with profile_env(profile):
        if platform in ("instagram", "youtube"):
            from opendispatch.shorts import dispatch_unit
            return dispatch_unit(unit, platform, creds)
        if creds is not None:
            return adapter.publish(unit, account, creds=getattr(creds, platform, None))
        return adapter.publish(unit, account)


def _safe_webhook_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    try:
        addresses = {ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except (OSError, ValueError):
        return False
    return bool(addresses) and all(not (a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_unspecified or a.is_multicast) for a in addresses)


def _fire_webhook(url: str, payload: dict) -> None:
    if not _safe_webhook_url(url):
        log.warning("webhook rejected by egress policy")
        return
    try:
        httpx.post(url, json=payload, timeout=10, follow_redirects=False)
    except Exception as e:  # noqa: BLE001
        log.warning("webhook failed: %s", e)


def run_once(resolve=None, queue=None) -> int:
    global _RESOLVE, _QUEUE
    previous_resolve, previous_queue = _RESOLVE, _QUEUE
    _RESOLVE = resolve
    _QUEUE = queue
    try:
        return _run_once_body()
    finally:
        _RESOLVE, _QUEUE = previous_resolve, previous_queue


def _run_once_body() -> int:
    q = _QUEUE if _QUEUE is not None else get_queue()
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    due = q.list_due(now_iso)
    if not due:
        return 0
    log.info("publishing %d due row(s)", len(due))
    for row in due:
        rid = row["id"]
        if not q.mark_publishing(rid):
            log.info("skip %s: claim lost", rid)
            continue
        try:
            ok, post_id, err = _publish(row)
        except Exception as e:  # noqa: BLE001
            # An adapter that raises used to leave the row stuck in "publishing"
            # forever: list_due() only returns "queued" rows and nothing reaps
            # a stuck claim, so the post was silently lost. Any adapter
            # exception must become a recorded failure.
            log.exception("adapter raised for %s", rid)
            ok, post_id, err = False, "", f"retryable: adapter raised {type(e).__name__}: {clip(e)}"
        webhook = (row.get("unit") or {}).get("webhook_url")
        if ok:
            log.info("✓ %s → %s", rid, post_id)
            committed = q.mark_published(rid, post_id)
            if committed and webhook:
                _fire_webhook(webhook, {"event": "published", "id": rid, "post_id": post_id,
                                        "platform": row["platform"]})
        else:
            err = clip(err)
            cls, _, detail = err.partition(": ")
            # A post that went out but whose id we could not read must never be
            # retried — that is how you get duplicate posts.
            if cls == PUBLISHED:
                log.error("✘ %s published but unconfirmed: %s", rid, detail)
                committed = q.mark_published(rid, post_id or "")
                if committed and webhook:
                    _fire_webhook(webhook, {"event": "published", "id": rid,
                                            "post_id": post_id, "unconfirmed": True,
                                            "platform": row["platform"]})
                continue
            # Auth and permanent failures cannot succeed on retry; dead-letter
            # immediately instead of burning two more attempts and backoffs.
            non_retryable = cls in (AUTH, PERMANENT)
            attempts = int(row.get("attempts", 0)) + 1
            dead = non_retryable or attempts >= MAX_ATTEMPTS
            log.error("✘ %s (attempt %d): %s", rid, attempts, err)
            # Compute the backoff up front and hand it to mark_failed, so the
            # status flip and the reschedule land in one guarded write. Doing it
            # as a separate _update() afterwards left the row briefly "queued"
            # with a past scheduled_for, which another worker could claim — the
            # backoff was silently skipped and the stale reschedule landed on an
            # already-publishing row.
            retry_at = None
            if not dead:
                backoff = _backoff_seconds(attempts)
                new_sf = datetime.now(tz=timezone.utc).timestamp() + backoff
                retry_at = datetime.fromtimestamp(new_sf, tz=timezone.utc).isoformat()
            committed = q.mark_failed(rid, err, dead=dead, retry_at=retry_at)
            if non_retryable:
                log.error("  %s — not retryable, dead-lettered", cls)
            if committed and not dead:
                log.info("  retry in %ds", backoff)
            if committed and webhook:
                _fire_webhook(webhook, {"event": "failed", "id": rid, "error": err,
                                        "platform": row["platform"], "dead": dead})
    return len(due)


def main() -> int:
    log.info("scheduler worker starting (poll=%ds, max_attempts=%d, backoff_base=%d)",
             POLL_INTERVAL, MAX_ATTEMPTS, BACKOFF_BASE)
    while True:
        try:
            _write_heartbeat()
            run_once()
        except KeyboardInterrupt:
            log.info("interrupted; exiting")
            return 0
        except Exception as e:  # noqa: BLE001
            log.exception("worker loop error: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
