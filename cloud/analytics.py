"""Server-side product events to OpenPanel.

The browser already reports the funnel (Signup, CheckoutStarted, Subscribed,
QuotaWall*, UpsellModal*). This module reports what the browser cannot be
trusted for:

* **Job outcomes.** Ad-blockers eat a chunk of client-side analytics, and the
  interesting moment — a render finishing minutes later — often happens after
  the tab is gone. Server-side is the only place these are complete.
* **Repeat usage.** The one number that decides whether the product retains:
  on 26-jul-2026, 491 of 564 users who ever processed a video did it exactly
  once. Every clip-quality fix shipped that day (1080p, captions, loudness) was
  aimed at that 87%, and nothing in the stack could measure whether it moved.
  ``job_index`` on each event answers it directly: count distinct users at
  index 1 versus index >= 2, before and after.

Fire-and-forget by design. Analytics must never slow down or break a job, so
every failure is swallowed and every send runs in the background.
"""
import asyncio
import os

import httpx

_TIMEOUT = 5.0


def _config():
    """(api_url, client_id, client_secret) or None when not configured.

    Absent config is the normal state for self-hosted installs, so this stays
    silent rather than warning.
    """
    client_id = os.environ.get("OPENPANEL_CLIENT_ID", "").strip()
    if not client_id:
        return None
    api = (os.environ.get("OPENPANEL_API_URL", "").strip()
           or "https://api.openpanel.fotoexamen.com").rstrip("/")
    return api, client_id, os.environ.get("OPENPANEL_CLIENT_SECRET", "").strip()


# One client for every event, kept open. A client per event built a new TLS
# context and handshake each time: with an event on most requests that was a
# good share of the API's one core. Bound to the loop it was made on (tests
# and the MCP stdio entry point run their own loops).
_client = None
_client_loop = None


def _shared_client():
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client_loop is not loop or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
        _client_loop = loop
    return _client


async def _post(payload):
    cfg = _config()
    if not cfg:
        return
    api, client_id, secret = cfg
    headers = {"openpanel-client-id": client_id, "Content-Type": "application/json"}
    if secret:
        headers["openpanel-client-secret"] = secret
    try:
        await _shared_client().post(f"{api}/track", json=payload, headers=headers)
    except Exception:
        pass  # analytics must never surface into a job


def track(name: str, user_id=None, **props):
    """Queue an event. Returns immediately; never raises.

    ``user_id`` becomes OpenPanel's profileId, which is what makes retention
    and repeat-usage questions answerable per user.
    """
    if not _config():
        return
    payload = {
        "type": "track",
        "payload": {
            "name": name,
            "properties": {k: v for k, v in props.items() if v is not None},
        },
    }
    if user_id is not None:
        payload["payload"]["profileId"] = str(user_id)
    try:
        asyncio.get_running_loop().create_task(_post(payload))
    except RuntimeError:
        pass  # no loop (CLI/self-host path) — nothing to report to anyway


def track_revenue(user_id, amount_cents, currency="usd", **props):
    """Queue a ``revenue`` event in OpenPanel's native shape. Never raises.

    ``amount_cents`` is Stripe's integer minor-unit amount, negative for a
    refund. OpenPanel stores ``__revenue`` in minor units and divides by 100 in
    its own views (Overview card, revenue trend), so a $59 sale is sent as 5900;
    ``amount`` (major units) rides along because custom reports read raw
    properties without that formatting. Same contract as every other product
    on this OpenPanel instance, so revenue is comparable across them.
    """
    try:
        cents = int(amount_cents or 0)
    except (TypeError, ValueError):
        return
    if cents == 0:
        return
    props["__revenue"] = cents
    props["amount"] = round(cents / 100, 2)
    props["currency"] = (currency or "usd").upper()
    track("revenue", user_id=user_id, **props)


def acquisition_properties(attribution):
    """Flatten a ``SignupAttribution`` row (or any object with the same
    attributes, or None) into the props a revenue event carries.

    A Stripe webhook has no browser session, so the only way to know where a
    paying user came from is the first-touch snapshot recorded at sign-up.
    ``channel`` collapses it to one label: utm_source > referring host > direct.
    """
    if attribution is None:
        return {"channel": "direct"}
    get = lambda k: (getattr(attribution, k, None) or "").strip() or None  # noqa: E731
    created = getattr(attribution, "created_at", None)
    props = {
        "channel": get("utm_source") or get("referrer_host") or "direct",
        "utm_source": get("utm_source"),
        "utm_medium": get("utm_medium"),
        "utm_campaign": get("utm_campaign"),
        "referrer": get("referrer_host"),
        "landing_path": get("landing_path"),
        "signup_date": created.isoformat() if hasattr(created, "isoformat") else None,
    }
    return {k: v for k, v in props.items() if v}
