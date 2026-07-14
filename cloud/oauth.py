"""Google OAuth (redirect flow) for cloud mode.

GET /api/auth/google           -> redirect to Google consent
GET /api/auth/google/callback  -> exchange code, upsert user, redirect to the
                                  frontend with the JWT in the URL hash (never
                                  reaches server logs).

Registration happens in ``register()`` (called from setup_sync) only when Google
credentials are configured; otherwise the endpoints 404.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import RedirectResponse
from sqlalchemy import select

from .config import settings
from . import database
from .models import User
from .auth import issue_jwt

router = APIRouter()
oauth = None


def register():
    """Register the Google OAuth client if credentials are present."""
    global oauth
    if not settings.google_auth_enabled:
        return
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def _now():
    return datetime.now(timezone.utc)


@router.get("/api/auth/google")
async def google_login(request: Request):
    if oauth is None:
        raise HTTPException(status_code=404, detail="Google login not configured")
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/api/auth/google/callback", name="google_callback")
async def google_callback(request: Request):
    if oauth is None:
        raise HTTPException(status_code=404, detail="Google login not configured")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?error=oauth")

    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = await oauth.google.userinfo(token=token)
    email = (userinfo.get("email") or "").lower()
    google_sub = userinfo.get("sub")
    if not email:
        return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?error=noemail")

    async with database.session() as session:
        async with session.begin():
            user = (await session.execute(
                select(User).where(User.google_sub == google_sub)
            )).scalar_one_or_none()
            if user is None:
                # Merge with an existing magic-link account of the same email.
                user = (await session.execute(
                    select(User).where(User.email == email)
                )).scalar_one_or_none()
                if user is None:
                    user = User(email=email, google_sub=google_sub, last_login_at=_now())
                    session.add(user)
                    await session.flush()
                else:
                    user.google_sub = google_sub
                    user.last_login_at = _now()
            else:
                user.last_login_at = _now()
            user_id, user_email = user.id, user.email

    jwt_token = issue_jwt(user_id, user_email)
    return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?token={jwt_token}")
