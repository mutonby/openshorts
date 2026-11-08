"""Upload-Post white-label profiles for managed users.

Each OpenShorts user maps to one Upload-Post profile (a container for their
connected socials). We create it lazily and generate branded connection links so
the user connects TikTok/IG/YouTube on a page that looks like OpenShorts.

Docs: https://docs.upload-post.com/guides/user-profile-integration
"""
import httpx
from fastapi import APIRouter, Request, HTTPException

from .config import settings
from . import database
from .models import UploadPostProfile

router = APIRouter()

API_BASE = "https://api.upload-post.com/api"
CONNECT_PLATFORMS = ["tiktok", "instagram", "youtube"]


def connect_platforms():
    """Twitch joins the connect page only once Upload-Post supports it
    (``AUTOPILOT_TWITCH=1``): it is read-only there, an Autopilot source."""
    from .autopilot import twitch_enabled
    return CONNECT_PLATFORMS + (["twitch"] if twitch_enabled() else [])


def _auth_headers():
    return {"Authorization": f"Apikey {settings.managed_upload_post_key}"}


def profile_username_for(user_id) -> str:
    return f"os_{user_id.hex[:12]}"


async def _create_remote_profile(username: str):
    """Create the profile in Upload-Post. Treat 'already exists' as success."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{API_BASE}/uploadposts/users",
            headers=_auth_headers(),
            json={"username": username},
        )
        if resp.status_code in (200, 201):
            return
        text = resp.text.lower()
        # Idempotency: if it already exists, that's fine.
        # Upload-Post answers 409 "Username already in use"; older replies were a
        # 400 with "exists"/"already". Both mean the profile is there.
        if resp.status_code == 409 or (
            resp.status_code == 400 and ("exist" in text or "already" in text)
        ):
            return
        # Upload-Post plan cap reached — surface a clean, actionable error, not a 500.
        if "profile_limit_reached" in text or "limit of" in text:
            raise HTTPException(
                status_code=503,
                detail="Social posting is temporarily unavailable (provider profile limit). "
                       "Please try again later.",
            )
        resp.raise_for_status()


async def ensure_profile(user) -> str:
    """Return the user's Upload-Post profile username, creating it if needed."""
    async with database.session() as s:
        prof = await s.get(UploadPostProfile, user.id)
        if prof:
            return prof.profile_username

    username = profile_username_for(user.id)
    await _create_remote_profile(username)

    async with database.session() as s:
        async with s.begin():
            prof = await s.get(UploadPostProfile, user.id)  # re-check under txn
            if prof:
                return prof.profile_username
            s.add(UploadPostProfile(user_id=user.id, profile_username=username))
    return username


# Where the connect page sends the user back to. A closed list: the value
# comes from the client and ends up in a redirect.
RETURN_TARGETS = {
    "account": "/#/account?connected=1",
    "autopilot": "/#app?tab=autopilot&connected=1",
}


async def get_connect_url(username: str, return_to: str = "account") -> str:
    """Generate a branded, single-use connection URL (~1h) for the user's socials."""
    back = RETURN_TARGETS.get(return_to, RETURN_TARGETS["account"])
    payload = {
        "username": username,
        "logoImage": settings.openshorts_logo_url,
        "connectTitle": "Connect your social accounts",
        "connectDescription": "Link TikTok, Instagram and YouTube to post your shorts directly from OpenShorts.",
        "redirectUrl": f"{settings.frontend_url}{back}",
        "redirectButtonText": "Back to OpenShorts",
        "platforms": connect_platforms(),
        "showCalendar": True,  # keep the scheduling calendar available in the page
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{API_BASE}/uploadposts/users/generate-jwt",
            headers=_auth_headers(),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("access_url") or data.get("url")


async def profile_token(username: str):
    """A short-lived token scoped to one profile, or None.

    Upload-Post's history endpoint filters by profile only when called with a
    profile JWT; with the managed key it returns every OpenShorts user's posts.
    The token rides in generate-jwt's access_url.
    """
    from urllib.parse import parse_qs, urlparse
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(
            f"{API_BASE}/uploadposts/users/generate-jwt",
            headers=_auth_headers(),
            # generate-jwt stores the profile's connect-page platforms; a call
            # without them resets the list and Twitch (opt-in there) would
            # vanish from a connect page the user has open.
            json={"username": username, "platforms": connect_platforms()},
        )
        resp.raise_for_status()
        url = resp.json().get("access_url") or ""
    return (parse_qs(urlparse(url).query).get("token") or [None])[0]


@router.post("/api/social/connect")
async def social_connect(request: Request):
    """Return a branded Upload-Post connection URL for the signed-in managed user."""
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    username = await ensure_profile(user)
    access_url = await get_connect_url(username, request.query_params.get("return_to") or "account")
    return {"access_url": access_url}
