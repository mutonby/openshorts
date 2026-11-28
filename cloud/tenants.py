"""Tenant resolution and tenant-minute reservations.

Phase 1 keeps the existing user/JWT metering intact. When
``TENANT_CREDITS_ENABLED`` is on and a request resolves to a tenant (by explicit
header, custom domain, or Firebase membership), the job's minutes are also
deducted from that tenant atomically in the user's reservation transaction.
Phase 2 replaces the header trust with verified Firebase ID tokens; the
functions here are the boundary the control plane and worker both call.
"""
import uuid
from decimal import Decimal

from sqlalchemy import select, text

from . import database
from .config import DEFAULT_TENANT_BRAND, DEFAULT_TENANT_ID, settings
from .models import Tenant, TenantCreditLedger, TenantJob, TenantMember


class TenantNotFound(Exception):
    pass


class TenantQuotaExceeded(Exception):
    def __init__(self, tenant_id, remaining, required):
        self.tenant_id = str(tenant_id)
        self.remaining = float(remaining)
        self.required = float(required)
        super().__init__(
            f"Tenant quota exceeded: need {required} min, {remaining} remaining")


def enabled() -> bool:
    return settings.tenant_credits_enabled


def _D(value) -> Decimal:
    return Decimal(str(value))


def _uuid():
    return uuid.uuid4()


def _clean_domain(value) -> str:
    """Turn a Host / X-Tenant-Domain header into a bare lowercase domain."""
    text = str(value or "").strip().lower()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.split("/", 1)[0]
    if text.startswith("[") and "]" in text:  # IPv6 literal
        return text[1:text.index("]")]
    return text.split(":", 1)[0]  # strip :port


async def ensure_default_tenant(conn):
    """Create the fallback tenant row on a fresh ``create_all`` database."""
    await conn.execute(text(
        "INSERT INTO tenants (id, brand_name, credit_balance) "
        "VALUES (:id, :brand, 0) ON CONFLICT (id) DO NOTHING"),
        {"id": DEFAULT_TENANT_ID, "brand": DEFAULT_TENANT_BRAND})


async def resolve_tenant_id_from_request(request, firebase_uid=None):
    """Resolve a tenant without trusting a caller to name someone else's data.

    The explicit id/domain headers are for a trusted control plane (Cloudflare
    Tunnel or an internal caller). Firebase membership is the user-facing path
    Phase 2 will use. An unresolved request returns None — legacy clouds keep
    their existing user metering and rows land in the default tenant.
    """
    tid = str(request.headers.get(settings.tenant_id_header) or "").strip()
    if tid:
        try:
            tid = uuid.UUID(tid)
        except (TypeError, ValueError, AttributeError):
            return None
        async with database.session() as session:
            found = await session.scalar(select(Tenant.id).where(Tenant.id == tid))
        return tid if found is not None else None

    domain = _clean_domain(
        request.headers.get(settings.tenant_domain_header)
        or request.headers.get("host"))
    if domain and domain not in ("localhost", "127.0.0.1", "::1"):
        async with database.session() as session:
            found = await session.scalar(
                select(Tenant.id).where(Tenant.custom_domain == domain))
        if found is not None:
            return found

    uid = str(firebase_uid or request.headers.get("x-firebase-uid") or "").strip()
    if uid:
        async with database.session() as session:
            return await session.scalar(
                select(TenantMember.tenant_id).where(TenantMember.firebase_uid == uid))
    return None


async def get_balance(tenant_id) -> dict:
    async with database.session() as session:
        tenant = await session.get(Tenant, tenant_id)
        if tenant is None:
            raise TenantNotFound(str(tenant_id))
        return {"remaining": float(_D(tenant.credit_balance))}


async def reserve_tenant_minutes(session, tenant_id, minutes, job_id) -> str:
    """Deduct tenant minutes under a row lock; return the ledger row id.

    The caller passes the same session as the user-minute reservation, so a
    failed user charge rolls back the tenant charge and vice versa.
    """
    required = _D(minutes)
    tenant = (await session.execute(
        select(Tenant).where(Tenant.id == tenant_id).with_for_update()
    )).scalar_one_or_none()
    if tenant is None:
        raise TenantNotFound(str(tenant_id))
    balance = _D(tenant.credit_balance)
    if required > balance:
        raise TenantQuotaExceeded(tenant_id, balance, required)
    balance_after = balance - required
    tenant.credit_balance = balance_after
    row = TenantCreditLedger(
        id=_uuid(), tenant_id=tenant_id, job_id=str(job_id), minutes=required,
        status="reserved", balance_after=balance_after)
    session.add(row)
    await session.flush()
    return str(row.id)


async def settle_tenant_minutes(session, reservation_id, status: str) -> None:
    """Commit or refund a tenant reservation alongside its user-minute row."""
    row = await session.get(TenantCreditLedger, reservation_id)
    if row is None or row.status != "reserved":
        return
    if status == "released":
        tenant = (await session.execute(
            select(Tenant).where(Tenant.id == row.tenant_id).with_for_update()
        )).scalar_one_or_none()
        if tenant is not None:
            balance = _D(tenant.credit_balance) + _D(row.minutes)
            tenant.credit_balance = balance
            row.balance_after = balance
    row.status = status


async def trim_tenant_minutes(session, reservation_id, refund) -> float:
    """Give back the unused part of a tenant reservation when the source shrank."""
    refund_d = _D(refund)
    if refund_d <= 0:
        return 0.0
    row = await session.get(TenantCreditLedger, reservation_id)
    if row is None or row.status != "reserved":
        return 0.0
    give = min(refund_d, _D(row.minutes))
    tenant = (await session.execute(
        select(Tenant).where(Tenant.id == row.tenant_id).with_for_update()
    )).scalar_one_or_none()
    if tenant is None:
        return 0.0
    balance = _D(tenant.credit_balance) + give
    tenant.credit_balance = balance
    row.minutes = _D(row.minutes) - give
    row.balance_after = balance
    return float(give)


async def record_job(job_id, tenant_id, user_id=None, minutes=None, source_key=None):
    """Create the durable queue row the local render node claims in Phase 3."""
    if tenant_id is None:
        return
    async with database.session() as session:
        async with session.begin():
            row = await session.get(TenantJob, str(job_id))
            if row is None:
                row = TenantJob(job_id=str(job_id), tenant_id=tenant_id)
                session.add(row)
            row.user_id = None if user_id is None else str(user_id)
            if minutes is not None:
                row.requested_minutes = _D(minutes)
            if source_key is not None:
                row.source_key = str(source_key)


async def mark_job(job_id, status: str, error=None):
    async with database.session() as session:
        async with session.begin():
            row = await session.get(TenantJob, str(job_id))
            if row is None:
                return
            row.status = status
            row.error = str(error)[:1000] if error else None
