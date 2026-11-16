"""User API keys (``osk_...``) for programmatic access: MCP clients, scripts, CI.

Two halves:
  - Management endpoints (create / list / revoke). These require a SESSION JWT,
    never an API key — a leaked key must not be able to mint replacement keys
    for the same account.
  - ``user_id_for_key``: the resolver cloud/auth.py calls so a Bearer
    ``osk_...`` (or ``X-API-Key``) authenticates like a session everywhere else.
    Quota, entitlement and job ownership then apply unchanged.
"""
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, func, and_

from . import database
from .models import ApiKey

router = APIRouter()

KEY_PREFIX = "osk_"
MAX_ACTIVE_KEYS = 10
# last_used_at is informational; refreshing it on every /api/status poll would
# write the row hundreds of times per job for no extra signal.
LAST_USED_REFRESH = timedelta(minutes=5)


def _now():
    return datetime.now(timezone.utc)


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Return (raw, sha256, display_prefix) for a fresh key."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_key(raw), raw[: len(KEY_PREFIX) + 6]


def looks_like_key(token: str) -> bool:
    return isinstance(token, str) and token.startswith(KEY_PREFIX)


async def user_id_for_key(raw: str) -> Optional[object]:
    """Resolve a raw ``osk_`` token to its owner's user id, or None."""
    principal = await key_principal(raw)
    return principal[0] if principal else None


async def key_principal(raw: str) -> Optional[tuple]:
    """``(user_id, scope)`` for a live ``osk_`` token, or None. ``scope`` is
    what the key may reach (cloud/scopes.py); None is the whole API.

    Refreshes ``last_used_at`` at most every LAST_USED_REFRESH so status polls
    don't turn every request into a DB write.
    """
    from . import scopes
    if not looks_like_key(raw):
        return None
    token_hash = hash_key(raw)
    async with database.session() as session:
        async with session.begin():
            row = (await session.execute(
                select(ApiKey).where(ApiKey.key_hash == token_hash)
            )).scalar_one_or_none()
            if row is None or row.revoked_at is not None:
                return None
            if row.last_used_at is None or _now() - row.last_used_at > LAST_USED_REFRESH:
                row.last_used_at = _now()
            return row.user_id, scopes.effective(row.scope, row.name)


# --------------------------------------------------------------------------- #
# Management endpoints (session JWT only)
# --------------------------------------------------------------------------- #
async def _session_user(request: Request):
    """The signed-in user, refusing API-key auth for key management."""
    from .auth import get_current_user_required

    auth = request.headers.get("Authorization", "")
    if looks_like_key(auth[7:].strip()) or looks_like_key(request.headers.get("X-API-Key", "")):
        raise HTTPException(status_code=403,
                            detail="API keys cannot manage API keys. Sign in to the dashboard.")
    return await get_current_user_required(request)


class CreateKeyRequest(BaseModel):
    name: Optional[str] = None


@router.post("/api/keys")
async def create_key(payload: CreateKeyRequest, request: Request):
    user = await _session_user(request)
    raw, token_hash, prefix = generate_key()
    name = (payload.name or "").strip()[:60] or "API key"
    async with database.session() as session:
        async with session.begin():
            active = (await session.execute(
                select(func.count(ApiKey.id)).where(and_(
                    ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None),
                ))
            )).scalar_one()
            if active >= MAX_ACTIVE_KEYS:
                raise HTTPException(status_code=400,
                                    detail=f"Limit of {MAX_ACTIVE_KEYS} active keys reached. Revoke one first.")
            row = ApiKey(user_id=user.id, name=name, key_hash=token_hash, prefix=prefix)
            session.add(row)
            await session.flush()
            key_id = row.id
    # The raw key appears here once and is never retrievable again.
    return {"id": str(key_id), "name": name, "prefix": prefix, "key": raw}


@router.get("/api/keys")
async def list_keys(request: Request):
    user = await _session_user(request)
    async with database.session() as session:
        rows = list((await session.execute(
            select(ApiKey).where(ApiKey.user_id == user.id)
            .order_by(ApiKey.created_at.desc()).limit(100)
        )).scalars())
    return {"keys": [{
        "id": str(k.id),
        "name": k.name,
        "prefix": k.prefix,
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
        "revoked": k.revoked_at is not None,
    } for k in rows]}


@router.delete("/api/keys/{key_id}")
async def revoke_key(key_id: str, request: Request):
    user = await _session_user(request)
    import uuid
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Key not found")
    async with database.session() as session:
        async with session.begin():
            row = (await session.execute(
                select(ApiKey).where(ApiKey.id == key_uuid)
            )).scalar_one_or_none()
            if row is None or str(row.user_id) != str(user.id):
                raise HTTPException(status_code=404, detail="Key not found")
            if row.revoked_at is None:
                row.revoked_at = _now()
    from . import owner_cache
    owner_cache.forget_user(user.id)
    return {"ok": True}
