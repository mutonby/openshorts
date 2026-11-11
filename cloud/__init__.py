"""Optional cloud (paid / managed-keys) mode for OpenShorts.

This whole package is dormant unless the ``BILLING_ENABLED`` env flag is set.
``app.py`` imports it only in that case, so self-hosters never need the extra
dependencies in ``requirements-billing.txt``.

Wiring is split in two because Starlette forbids adding middleware after the app
has started:
  - ``setup_sync(app)``   -> runs at import time (middleware + routers)
  - ``setup_async(app)``  -> runs in the lifespan (DB engine, sweeper, orphans)

Both are built up phase by phase; each phase appends its own wiring.
"""
from .config import is_enabled, settings, validate_required  # noqa: F401


def setup_sync(app):
    """Synchronous wiring done at import time (before the app serves).

    Middleware and routers must be attached here. Fails fast on missing config
    so a misconfigured deploy never boots half-enabled.
    """
    validate_required()

    # SessionMiddleware backs the OAuth state handshake.
    from starlette.middleware.sessions import SessionMiddleware
    app.add_middleware(SessionMiddleware, secret_key=settings.jwt_secret)

    from . import (auth, oauth, billing, social_profiles, videos, api_keys,
                   account, mcp_oauth, marketing, autopilot, lifecycle, cancellation, onboarding)  # noqa: F401
    oauth.register()
    billing._init_stripe()
    app.include_router(auth.router)
    app.include_router(oauth.router)
    app.include_router(billing.router)
    # Cancel flow with reason + review (the portal can still cancel too).
    app.include_router(cancellation.router)
    # Sign-up survey: source, what they came to make, who they are.
    app.include_router(onboarding.router)
    app.include_router(social_profiles.router)
    app.include_router(videos.router)
    app.include_router(api_keys.router)
    app.include_router(mcp_oauth.router)
    app.include_router(account.router)
    # Autopilot: clip new videos from the user's connected YouTube channel.
    app.include_router(autopilot.router)
    # Unsubscribe from commercial email (LSSI art. 21.2): the out-of-minutes
    # upsell and the lifecycle emails (cloud/lifecycle.py).
    app.include_router(marketing.router)


async def setup_async(app, keep_reservation_ids=None, live_reservation_ids=None,
                      sweep_active=None, booted_at=None):
    """Async initialization done inside the FastAPI lifespan.

    DB engine, orphaned-reservation cleanup and the metering sweeper live here.
    ``keep_reservation_ids`` belong to jobs being resumed after a restart and
    must not be refunded (see app._resume_interrupted_jobs).
    ``live_reservation_ids`` is a callable the sweeper asks, every pass, which
    reservations belong to jobs still alive (app._live_reservation_ids);
    ``sweep_active`` is False while the instance drains. ``booted_at``: only
    reservations created before it can be orphans of this boot.
    """
    from . import database, digest, metering, videos
    await database.init_engine()
    await metering.release_orphaned_reservations(keep_ids=keep_reservation_ids,
                                                 created_before=booted_at)
    metering.start_sweeper(live_ids=live_reservation_ids, is_active=sweep_active)
    videos.start_sweeper()
    digest.start_digest()
    print("☁️  Cloud billing mode ENABLED (DB ready, metering active).")
