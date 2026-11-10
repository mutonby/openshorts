"""Google OAuth (redirect flow) for cloud mode.

GET /api/auth/google           -> redirect to Google consent
GET /api/auth/google/callback  -> exchange code, upsert user, redirect to the
                                  frontend with a single-use login code in the
                                  URL hash; the dashboard trades it for the JWT
                                  at POST /api/auth/exchange with the verifier
                                  it kept when the sign-in started (cc=).

Registration happens in ``register()`` (called from setup_sync) only when Google
credentials are configured; otherwise the endpoints 404.
"""
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import RedirectResponse
from sqlalchemy import select

from .config import settings
from . import database, email_policy, evidence
from .models import User
from .auth import issue_jwt, valid_challenge, create_login_code

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


# Dashboards loaded before the code exchange existed start the sign-in
# without cc= and only understand ?token=. They get it until this date (an
# open tab, a sign-in started across the deploy); after it they are asked to
# sign in again from a fresh page.
LEGACY_TOKEN_REDIRECT_UNTIL = os.environ.get(
    "LEGACY_TOKEN_REDIRECT_UNTIL", "2026-11-12T00:00:00+00:00")


def _legacy_token_redirect_open() -> bool:
    try:
        until = datetime.fromisoformat(LEGACY_TOKEN_REDIRECT_UNTIL.replace("Z", "+00:00"))
    except ValueError:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    return _now() < until


@router.get("/api/auth/google")
async def google_login(request: Request):
    if oauth is None:
        raise HTTPException(status_code=404, detail="Google login not configured")
    # The dashboard's PKCE-style challenge, kept in the signed session cookie
    # until the callback binds the login code to it.
    challenge = request.query_params.get("cc")
    if valid_challenge(challenge):
        request.session["login_cc"] = challenge
    else:
        request.session.pop("login_cc", None)
    # Explicit override for setups where the Host header doesn't match a
    # Google-registered URI (e.g. local dev through the Vite proxy, where the
    # backend sees the internal Docker host "backend:8000").
    redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", "").strip()
    if not redirect_uri:
        redirect_uri = str(request.url_for("google_callback"))
        # Behind a TLS-terminating reverse proxy the app sees the request as http, so
        # url_for builds an http:// redirect_uri — but Google requires the exact
        # registered https URI for public hosts (→ redirect_uri_mismatch otherwise).
        # Force https except on localhost (where there's no proxy and http is used).
        if redirect_uri.startswith("http://") and not any(
            h in redirect_uri for h in ("localhost", "127.0.0.1")
        ):
            redirect_uri = "https://" + redirect_uri[len("http://"):]
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
    # Normalize (Gmail dots / +tags) so a Google login and a magic-link account
    # for the same real inbox key to one account.
    email = email_policy.normalize_email(userinfo.get("email") or "")
    google_sub = userinfo.get("sub")
    if not email:
        return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?error=noemail")
    # An unverified address would sign in to (and merge with) whatever
    # account owns that email here.
    if userinfo.get("email_verified") is False:
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
                    from .account import free_plan_denial_for_signup
                    user = User(email=email, google_sub=google_sub, last_login_at=_now(),
                                free_plan_denied=await free_plan_denial_for_signup(session, email))
                    session.add(user)
                    await session.flush()
                else:
                    user.google_sub = google_sub
                    user.last_login_at = _now()
            else:
                user.last_login_at = _now()
            user_id, user_email = user.id, user.email
    evidence.record_event(user_id=user_id, email=user_email, kind="login",
                          method="google", request=request)

    challenge = request.session.pop("login_cc", None)
    if valid_challenge(challenge):
        code = await create_login_code(user_id, challenge)
        return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?code={code}")
    if _legacy_token_redirect_open():
        jwt_token = issue_jwt(user_id, user_email)
        return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?token={jwt_token}")
    return RedirectResponse(f"{settings.frontend_url}/#/auth/callback?error=retry")
