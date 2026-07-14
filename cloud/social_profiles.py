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
        if resp.status_code == 400 and ("exist" in text or "already" in text):
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


async def get_connect_url(username: str) -> str:
    """Generate a branded, single-use connection URL (~1h) for the user's socials."""
    payload = {
        "username": username,
        "logoImage": settings.openshorts_logo_url,
        "connectTitle": "Connect your social accounts",
        "connectDescription": "Link TikTok, Instagram and YouTube to post your shorts directly from OpenShorts.",
        "redirectUrl": f"{settings.frontend_url}/#/account?connected=1",
        "redirectButtonText": "Back to OpenShorts",
        "platforms": CONNECT_PLATFORMS,
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


@router.post("/api/social/connect")
async def social_connect(request: Request):
    """Return a branded Upload-Post connection URL for the signed-in managed user."""
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    username = await ensure_profile(user)
    access_url = await get_connect_url(username)
    return {"access_url": access_url}
