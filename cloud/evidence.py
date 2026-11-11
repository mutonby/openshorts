"""Evidence for payment disputes: a timestamped log of sign-ins and checkouts.

When a cardholder disputes a charge, the strongest answer is that the person
who paid is the person who used the account: the same address signing in from
the same network and browser before and after the payment. This module keeps
exactly that, one row per successful login and per checkout started, with the
client IP and user agent the request arrived with.

Best effort by design. ``record_event`` returns at once and the row is written
in the background in its own short transaction, so a slow or broken database
never delays or fails a sign-in or a checkout; a lost row costs one line of
evidence, nothing more. Failures are logged without the address or the IP.

One other reader: the free queue asks for an account's last address
(``last_ips``) to place a job it holds from before its lane was decided
(``app._move_backlog_to_slow_lane``). The address is looked up in the local
country database in memory and goes nowhere else.

Rows go with the account on erasure (``account.USER_OWNED_TABLES``) and age
out after ``RETENTION_DAYS`` (``purge_stale_events``, run by the retention
sweeper in cloud/videos.py).
"""
import asyncio
from datetime import datetime, timedelta, timezone

from . import database

KINDS = ("login", "checkout")
USER_AGENT_MAX = 300
# Card networks allow disputes up to ~120 days after the charge (longer for
# some reasons), and the evidence that matters is the activity around it.
RETENTION_DAYS = 400

_pending = set()   # strong refs: the loop only keeps weak references to tasks


def _client(request):
    """(ip, user agent) of a request, or (None, None). Never raises.

    The IP is ``request.client.host``, which uvicorn resolves from
    X-Forwarded-For behind the trusted proxy (see cloud/auth.py)."""
    if request is None:
        return None, None
    try:
        ip = request.client.host if request.client else None
    except Exception:
        ip = None
    try:
        ua = request.headers.get("user-agent") or None
    except Exception:
        ua = None
    if ua:
        ua = ua[:USER_AGENT_MAX]
    return ip, ua


async def _write(row: dict, session_factory):
    from .models import EvidenceEvent
    try:
        factory = session_factory or database.session
        async with factory() as session:
            async with session.begin():
                session.add(EvidenceEvent(**row))
    except Exception as e:
        # Type only: the message can carry the row's values (address, IP).
        print(f"⚠️  evidence event not recorded ({row.get('kind')}/{row.get('method')}): "
              f"{type(e).__name__}", flush=True)


def record_event(*, user_id, email, kind, method, request=None, meta=None,
                 session_factory=None):
    """Queue one evidence row. Returns the background task (or None); never raises.

    ``request`` is read here, synchronously, so the row carries the IP and
    user agent even though the write happens after the response is sent.
    """
    try:
        ip, ua = _client(request)
        row = dict(user_id=user_id, email=email, kind=kind, method=method,
                   ip=ip, user_agent=ua, meta=dict(meta or {}))
        task = asyncio.get_running_loop().create_task(_write(row, session_factory))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
        return task
    except Exception as e:
        print(f"⚠️  evidence event not queued ({kind}/{method}): {type(e).__name__}",
              flush=True)
        return None


async def purge_stale_events():
    """Drop evidence rows older than ``RETENTION_DAYS``. Called from the
    retention sweeper."""
    from sqlalchemy import delete
    from .models import EvidenceEvent

    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    async with database.session() as session:
        async with session.begin():
            await session.execute(
                delete(EvidenceEvent).where(EvidenceEvent.created_at < cutoff))



LAST_IPS_CHUNK = 500


async def last_ips(user_ids) -> dict:
    """``{str(user_id): ip}`` of the latest row with an address for each of
    ``user_ids`` that has one. One query per LAST_IPS_CHUNK accounts; ids that
    are not uuids (self-host, test doubles) are skipped."""
    import uuid as _uuid
    from sqlalchemy import select
    from .models import EvidenceEvent

    ids = []
    for u in {str(x) for x in (user_ids or ()) if x is not None}:
        try:
            ids.append(_uuid.UUID(u))
        except ValueError:
            continue
    out = {}
    if not ids:
        return out
    async with database.session() as session:
        for i in range(0, len(ids), LAST_IPS_CHUNK):
            rows = (await session.execute(
                select(EvidenceEvent.user_id, EvidenceEvent.ip)
                .where(EvidenceEvent.user_id.in_(ids[i:i + LAST_IPS_CHUNK]),
                       EvidenceEvent.ip.is_not(None))
                .order_by(EvidenceEvent.created_at)
            )).all()
            out.update({str(uid): str(ip) for uid, ip in rows})   # the latest wins
    return out
