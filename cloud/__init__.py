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

    from . import auth, oauth, billing, social_profiles
    oauth.register()
    billing._init_stripe()
    app.include_router(auth.router)
    app.include_router(oauth.router)
    app.include_router(billing.router)
    app.include_router(social_profiles.router)


async def setup_async(app):
    """Async initialization done inside the FastAPI lifespan.

    DB engine, orphaned-reservation cleanup and the metering sweeper live here.
    """
    from . import database, metering
    await database.init_engine()
    await metering.release_orphaned_reservations()
    metering.start_sweeper()
    print("☁️  Cloud billing mode ENABLED (DB ready, metering active).")
