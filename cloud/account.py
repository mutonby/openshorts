"""Account erasure: the GDPR Art. 17 "right to be forgotten", self-service.

The privacy policy already promises that "deleting your account from the
dashboard removes your content on the schedule above" — this is the code that
makes that sentence true. It is one endpoint, ``DELETE /api/account``, and it
is immediate and irreversible: erasure has to happen "without undue delay", and
once the account is gone there is no way left to authenticate whoever asks for
it back, so a recovery window would be a window nobody could safely use.

Two rules shape the order of operations below, and both come from asking what
happens when a step fails halfway:

1. **Stripe first, and abort if it refuses.** Deleting the user row while a
   live subscription keeps charging a card we can no longer map to anyone turns
   a privacy fix into a billing incident nobody can unwind. Cancelling first
   means the worst case is a cancelled subscription on an account that still
   exists — annoying, retryable, harmless.
2. **R2 before the database.** Those DB rows are the only index of which
   objects belong to this user. Drop them first and a failed R2 delete leaves
   objects that nothing in the system can ever find again; the bucket already
   carries orphans from an earlier version of this reasoning.

What deliberately survives: the Stripe customer and its invoices (six-year
retention under Spanish commercial law, privacy policy §5) and one
[AccountDeletion] row proving the erasure happened — a sha256 of the email
rather than the address, and no free text anywhere in it. The confirmation
email tells the user both, because a record they were never told about is not
accountability, it is retention.
"""
import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select, delete, func

from .config import settings, DELETION_LOG_RETENTION_DAYS
from . import database, email_policy, storage
from .models import (
    User, Subscription, CreditTopup, UsageLedger, MagicLinkToken,
    SignupAttribution, ApiKey, UploadPostProfile, UserVideo, ClipExpiryWarning,
    Project, AccountDeletion, OAuthCode, AutopilotRun, AutopilotSettings, CancellationFeedback,
)

# Every table that holds rows belonging to a user, child-first (clip_expiry_
# warnings points at user_videos as well as at users).
#
# These are all declared ON DELETE CASCADE, so in theory deleting the user row
# is enough. We delete them by hand anyway: the schema is bootstrapped with
# ``create_all``, which never ALTERs an existing table, so a constraint added
# after a table first appeared in production exists in the models and not in
# the database. Trusting the cascade would mean the difference shows up as a
# foreign-key violation the first time a real user tries to leave.
USER_OWNED_TABLES = (
    ClipExpiryWarning, UserVideo, Project, UsageLedger, CreditTopup,
    Subscription, ApiKey, SignupAttribution, UploadPostProfile, OAuthCode,
    AutopilotRun, AutopilotSettings, CancellationFeedback,
)

# The optional "why are you leaving" answer, as a closed list. It was a free
# text box first, and that was wrong: whatever the user types lands in a record
# that deliberately outlives their account, so a single "my name is X, please
# confirm" turns the one row meant to hold no readable identity into retained
# personal data. A fixed vocabulary cannot do that, and twenty deletions a month
# are far more legible counted than read.
DELETION_REASONS = (
    "too_expensive", "not_using_it", "clip_quality", "missing_feature",
    "found_alternative", "privacy", "other",
)

router = APIRouter()

# Job dirs on the API's own disk belong to app.py, which cannot be imported from
# here (the dependency runs the other way). It registers a purge callback at
# startup instead; when it hasn't, local files simply age out on their usual
# one-hour cleanup.
_local_purge = None


def register_local_purge(fn):
    """Let app.py hand us a ``(user_id) -> int`` local-file purge."""
    global _local_purge
    _local_purge = fn


def _now():
    return datetime.now(timezone.utc)


def email_fingerprint(email: str) -> str:
    """The only trace of an address that outlives the account."""
    return hashlib.sha256(email_policy.normalize_email(email).encode()).hexdigest()


async def free_plan_denial_for_signup(session, email: str) -> Optional[str]:
    """``User.free_plan_denied`` value for a brand-new account, or None.

    An address whose account was erased in the last FREE_REDO_BLOCK_DAYS
    comes back without the free allowance: the ledger is keyed by user id, so
    a new row would otherwise start the month with a fresh 20 minutes. Paid
    plans and top-ups work as for anyone. Matched on the same fingerprint the
    erasure stored, so it needs no new personal data.
    """
    from .config import FREE_REDO_BLOCK_DAYS
    if not email or FREE_REDO_BLOCK_DAYS <= 0:
        return None
    since = _now() - timedelta(days=FREE_REDO_BLOCK_DAYS)
    hit = (await session.execute(
        select(AccountDeletion.id).where(
            AccountDeletion.email_sha256 == email_fingerprint(email),
            AccountDeletion.deleted_at >= since,
        ).limit(1)
    )).scalar_one_or_none()
    return "recreated_after_deletion" if hit is not None else None


async def _session_user(request: Request):
    """The signed-in user, refusing API-key auth.

    Same reasoning as key management in cloud/api_keys.py: a leaked ``osk_``
    token must not be able to destroy the account it was issued from.
    """
    from . import api_keys
    from .auth import get_current_user_required

    auth = request.headers.get("Authorization", "")
    if api_keys.looks_like_key(auth[7:].strip()) or api_keys.looks_like_key(
            request.headers.get("X-API-Key", "")):
        raise HTTPException(status_code=403, detail=(
            "API keys cannot delete an account. Sign in to the dashboard."))
    return await get_current_user_required(request)


# --------------------------------------------------------------------------- #
# The steps
# --------------------------------------------------------------------------- #
async def _cancel_subscriptions(user_id) -> Optional[str]:
    """Cancel every Stripe subscription on the account, immediately.

    Returns the plan that was live (for the deletion record), or None. Raises
    on a Stripe error so the caller can abort before anything is erased — see
    rule 1 in the module docstring.
    """
    import stripe

    async with database.session() as session:
        rows = list((await session.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )).scalars())

    live = [r for r in rows if r.status not in ("canceled", "incomplete_expired")]
    plan = live[0].plan if live else (rows[0].plan if rows else None)
    for row in live:
        try:
            # cancel(), not cancel_at_period_end: the user is leaving now, and
            # billing them again for a service whose data we just erased would
            # be indefensible. Unused time is refunded as it always is.
            await asyncio.to_thread(lambda sid=row.stripe_subscription_id:
                                    stripe.Subscription.cancel(sid))
        except stripe.InvalidRequestError as e:
            # Stripe says this subscription is gone or already cancelled. Our
            # row can be stale — a cancellation made in the portal whose webhook
            # never landed leaves it reading "active" forever. Treating that as
            # a hard error would trap the user: the one account that cannot be
            # deleted is the one Stripe already stopped billing.
            print(f"ℹ️  Subscription {row.stripe_subscription_id} already gone at Stripe: {e}")
    return plan


async def _delete_upload_post_profile(user_id):
    """Remove the user's white-label profile (and its connected socials) at
    Upload-Post. Best effort: a stale empty profile there holds no content of
    ours, and failing the whole erasure over a third-party 500 would be worse.
    """
    import httpx
    from .social_profiles import API_BASE, _auth_headers

    if not settings.managed_upload_post_key:
        return
    async with database.session() as session:
        prof = await session.get(UploadPostProfile, user_id)
        if prof is None:
            return
        username = prof.profile_username

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.request(
                "DELETE", f"{API_BASE}/uploadposts/users",
                headers={**_auth_headers(), "Content-Type": "application/json"},
                json={"username": username},
            )
        if resp.status_code not in (200, 204, 404):
            print(f"⚠️  Upload-Post profile delete returned {resp.status_code} for {username}")
    except Exception as e:
        print(f"⚠️  Upload-Post profile delete failed for {username}: {e}")


async def _erase_rows(user_id, email, stripe_customer_id, plan, r2_deleted, reason) -> bool:
    """Swap the account for its erasure record, in one transaction.

    All-or-nothing on purpose: a half-erased account is worse than an un-erased
    one, because the user is told they are gone while some of their rows are
    not. Magic-link tokens are keyed by email rather than user id, so they need
    their own statement on top of [USER_OWNED_TABLES].

    Returns whether this call is the one that removed the account. Two requests
    can race here — a double-click, or two tabs — and everything before this
    point is idempotent, but the erasure record is not: writing it from the
    loser would leave two rows for one deletion, possibly disagreeing about the
    reason. Postgres serialises the two DELETEs, so the loser sees rowcount 0
    and writes nothing.
    """
    async with database.session() as session:
        async with session.begin():
            for model in USER_OWNED_TABLES:
                await session.execute(
                    delete(model).where(model.user_id == user_id))
            await session.execute(
                delete(MagicLinkToken).where(MagicLinkToken.email == email))
            result = await session.execute(delete(User).where(User.id == user_id))
            if result.rowcount == 0:
                return False
            session.add(AccountDeletion(
                former_user_id=str(user_id),
                email_sha256=email_fingerprint(email),
                stripe_customer_id=stripe_customer_id,
                plan_at_deletion=plan,
                r2_objects_deleted=r2_deleted,
                reason=reason,
            ))
    return True


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
class DeleteAccountRequest(BaseModel):
    # Typing the address is the confirmation step. Sessions last 30 days and
    # there is no password to re-enter, so this is the strongest thing we can
    # ask for without mailing a second token to a user who is leaving anyway.
    confirm_email: str
    reason: Optional[str] = None   # one of DELETION_REASONS, or ignored


@router.delete("/api/account")
async def delete_account(payload: DeleteAccountRequest, request: Request):
    user = await _session_user(request)

    if email_policy.normalize_email(payload.confirm_email) != \
            email_policy.normalize_email(user.email):
        raise HTTPException(status_code=400, detail=(
            "That doesn't match the email on this account."))

    async with database.session() as session:
        row = await session.get(User, user.id)
        if row is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        email, stripe_customer_id = row.email, row.stripe_customer_id
        # A reserved ledger row means metered work is in flight. Erasing now
        # would pull the output from under a job still writing it, and the
        # reservation would never settle. Minutes are refunded on failure, so
        # waiting costs the user nothing.
        #
        # This covers everything that costs minutes, which is everything long
        # enough to matter. A free action (burning captions on an untranslated
        # clip) writes no ledger row and so is not covered: worst case it fails
        # on a missing file, in a tab whose session is about to end anyway.
        in_flight = (await session.execute(
            select(func.count(UsageLedger.id)).where(
                UsageLedger.user_id == user.id, UsageLedger.status == "reserved")
        )).scalar_one()
    if in_flight:
        raise HTTPException(status_code=409, detail=(
            "A video is still processing. Wait for it to finish, then delete "
            "your account."))

    try:
        plan = await _cancel_subscriptions(user.id)
    except Exception as e:
        print(f"⚠️  Account deletion aborted, Stripe cancel failed for {user.id}: {e}")
        raise HTTPException(status_code=502, detail=(
            "We couldn't cancel your subscription, so nothing was deleted. "
            "Please try again, or email info@openshorts.app."))

    # Storage before the third party: if the bucket refuses, the only thing that
    # has changed so far is the cancelled subscription, and the error below can
    # say so truthfully.
    r2_deleted = None
    if settings.r2_configured:
        try:
            r2_deleted = await asyncio.to_thread(
                storage.delete_prefix, storage.user_prefix(user.id))
        except Exception as e:
            print(f"⚠️  Account deletion aborted, R2 purge failed for {user.id}: {e}")
            raise HTTPException(status_code=502, detail=(
                "We couldn't reach the storage that holds your clips, so your "
                "account was not deleted. Your subscription has already been "
                "cancelled. Please try again, or email info@openshorts.app."))

    if _local_purge is not None:
        try:
            # rmtree over gigabytes of video: off the event loop, or every other
            # request on the process waits for this user's disk to clear.
            await asyncio.to_thread(_local_purge, user.id)
        except Exception as e:
            print(f"⚠️  Local job purge failed for {user.id}: {e}")

    await _delete_upload_post_profile(user.id)

    # Anything not in the closed list is dropped rather than rejected: a
    # mismatched client must not be able to fail a deletion over a label.
    reason = payload.reason if payload.reason in DELETION_REASONS else None
    try:
        erased = await _erase_rows(
            user.id, email, stripe_customer_id, plan, r2_deleted, reason)
    except Exception as e:
        # The content is already gone; only the account row survived. Say that
        # rather than a bare 500 — the user needs to know a retry is safe and
        # that the account they can still sign into is now empty.
        print(f"⚠️  Account row delete failed for {user.id} after purge: {e}")
        raise HTTPException(status_code=500, detail=(
            "Your clips were deleted but we couldn't close the account itself. "
            "Please try again, or email info@openshorts.app."))
    # ``email`` was read before the delete on purpose: from here on the address
    # exists nowhere in our systems, and the goodbye email still has to go out.

    if not erased:
        # A concurrent request got there first and has already sent the email.
        return {"deleted": True, "r2_objects_deleted": r2_deleted}

    print(f"🗑️  Account erased: {user.id} (plan={plan}, r2_objects={r2_deleted})")

    # Everything below is after-the-fact and must never fail the deletion.
    try:
        # OpenPanel only ever received the account uuid, never the address (see
        # cloud/analytics.py), and it expires those identifiers on its own
        # 13-month schedule — so this event adds no identity, it just makes the
        # one churn signal we cannot otherwise see countable.
        from . import analytics
        analytics.track("AccountDeleted", user_id=user.id, plan=plan)
    except Exception:
        pass
    try:
        from .emails import send_account_deleted_email
        await send_account_deleted_email(email)
    except Exception as e:
        print(f"⚠️  Goodbye email failed: {e}")
    try:
        from .alerts import send_telegram
        await send_telegram(f"🗑️ Account deleted (plan: {plan or 'free'})")
    except Exception:
        pass

    return {"deleted": True, "r2_objects_deleted": r2_deleted}


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
async def purge_stale_deletion_records():
    """Drop erasure records past their own retention window.

    The log exists to answer a complaint; once claims are time-barred it is
    just a hashed address we have no reason to hold. Called from the video
    retention sweeper.
    """
    cutoff = _now() - timedelta(days=DELETION_LOG_RETENTION_DAYS)
    async with database.session() as session:
        async with session.begin():
            await session.execute(
                delete(AccountDeletion).where(AccountDeletion.deleted_at < cutoff))
