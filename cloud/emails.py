"""Transactional email via Resend (magic-link login).

If RESEND_API_KEY is unset (e.g. local dev), the link is printed to the server
log instead of sent, so the login flow is still testable without a real inbox.
"""
import httpx

from .config import settings


async def send_magic_link_email(email: str, link: str):
    if not settings.resend_api_key:
        print(f"✉️  [DEV magic link] {email} -> {link}")
        return

    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Sign in to OpenShorts</h2>
        <p>Click the button below to sign in. This link expires in 15 minutes.</p>
        <p><a href="{link}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">Sign in</a></p>
        <p style="color:#666;font-size:13px">If you didn't request this, ignore this email.</p>
      </div>
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {settings.resend_api_key}"},
            json={
                "from": settings.email_from,
                "to": [email],
                "subject": "Your OpenShorts sign-in link",
                "html": html,
            },
        )
        resp.raise_for_status()
