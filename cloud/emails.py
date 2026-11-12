"""Transactional + operational email via SMTP (e.g. Namecheap Private Email).

If SMTP isn't configured (local dev), messages are printed to the server log so
the magic-link flow and alerts still work without a real mailbox.
"""
import asyncio
import html as _html
import smtplib
import ssl
from email.message import EmailMessage

from .config import settings


def _send_sync(to: str, subject: str, html: str, headers: dict | None = None):
    msg = EmailMessage()
    msg["From"] = settings.email_from
    msg["To"] = to
    msg["Subject"] = subject
    for name, value in (headers or {}).items():
        msg[name] = value
    msg.set_content("This message requires an HTML-capable email client.")
    msg.add_alternative(html, subtype="html")

    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port,
                              context=ssl.create_default_context(), timeout=20) as s:
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)


async def send_email(to: str, subject: str, html: str, headers: dict | None = None):
    """Send an email (async wrapper). Logs instead of sending if SMTP is unset."""
    if not settings.smtp_configured:
        print(f"✉️  [DEV email → {to}] {subject}")
        return
    try:
        await asyncio.to_thread(_send_sync, to, subject, html, headers)
    except Exception as e:
        print(f"⚠️  Failed to send email to {to}: {e}")


async def send_commercial_email(user_id, to: str, subject: str, html: str) -> bool:
    """Send a commercial communication (LSSI art. 21): never to an account that
    unsubscribed, and always with a one-click way out (footer link plus the
    ``List-Unsubscribe`` header, see cloud/marketing.py). Returns False when
    the account had opted out and nothing was sent.
    """
    from . import marketing
    if user_id is not None and await marketing.is_opted_out(user_id):
        return False
    unsub = marketing.unsubscribe_url(user_id) if user_id is not None else ""
    headers = {}
    if unsub:
        footer = (f'<a href="{unsub}" style="color:#666">Unsubscribe</a> from '
                  "upgrade and tips emails.")
        headers = {"List-Unsubscribe": f"<{unsub}>",
                   "List-Unsubscribe-Post": "List-Unsubscribe=One-Click"}
    else:
        footer = "Reply to this email if you'd rather not get upgrade and tips emails."
    html = (f'{html}<p style="font-family:system-ui,sans-serif;max-width:480px;'
            f'margin:24px auto 0;color:#666;font-size:12px">{footer}</p>')
    await send_email(to, subject, html, headers)
    return True


async def send_magic_link_email(email: str, link: str):
    if not settings.smtp_configured:
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
    await send_email(email, "Your OpenShorts sign-in link", html)


GITHUB_REPO_URL = "https://github.com/mutonby/openshorts"


async def send_clips_ready_email(email: str, job_title: str, clip_count: int,
                                 dashboard_url: str):
    """Job-completion notice: lets the user close the tab during processing."""
    # The title comes from the video (and its model-written metadata): text,
    # never markup.
    plain = " ".join((job_title or "Your video").split())[:200] or "Your video"
    title = _html.escape(plain)
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your clips are ready 🎬</h2>
        <p><strong>{title}</strong> produced {clip_count} viral-ready
           clip{'s' if clip_count != 1 else ''}. They're waiting in the History tab
           of your dashboard: sign in with this email address to see them.</p>
        <p><a href="{dashboard_url}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">View my clips</a></p>
        <p style="color:#666;font-size:13px">Enjoying OpenShorts? A
           <a href="{GITHUB_REPO_URL}" style="color:#666">star on GitHub</a> helps a lot ⭐</p>
      </div>
    """
    await send_email(email, f"Your clips are ready — {plain}", html)


async def send_queue_expired_email(email: str, dashboard_url: str):
    """Job notice: a free video waited out a jammed queue and was not clipped."""
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>We could not get to your video in time</h2>
        <p>The free queue was far busier than usual and your video waited too
           long, so we stopped it instead of keeping you waiting. Sorry about
           that. <strong>No minutes were used.</strong></p>
        <p>Send it again whenever you like: while the queue is busy, the
           newest free videos go first.</p>
        <p><a href="{dashboard_url}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">Send it again</a></p>
        <p style="color:#666;font-size:13px">Paid plans skip the free queue.</p>
      </div>
    """
    await send_email(email, "Your video was not clipped: send it again", html)


async def send_autopilot_clips_email(email: str, video_title: str, clip_count: int,
                                     scheduled_count: int, dashboard_url: str):
    """Autopilot finished clipping a new channel video on its own.

    This is the email that makes Autopilot visible: the user did nothing, and
    this is where they find out the work got done anyway.
    """
    import html as _html
    title = _html.escape((video_title or "your new video").strip())
    plural = "s" if clip_count != 1 else ""
    if scheduled_count:
        posting = (f"<p>The best {scheduled_count} {'is' if scheduled_count == 1 else 'are'} "
                   f"scheduled to publish, one a day, on the accounts you picked.</p>")
    else:
        posting = "<p>Review them, tweak anything you like, and post the ones you want.</p>"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Autopilot clipped your new video 🎬</h2>
        <p>You published <strong>{title}</strong> and OpenShorts already turned it
           into {clip_count} short{plural}.</p>
        {posting}
        <p><a href="{dashboard_url}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">See my clips</a></p>
        <p style="color:#666;font-size:13px">You can pause Autopilot anytime from the
           Autopilot page in your dashboard.</p>
      </div>
    """
    await send_email(email, f"Autopilot: {clip_count} new short{plural} from your video", html)


async def send_clips_expiring_email(email: str, clip_count: int):
    """Free clips enter their last day before deletion — honest loss aversion.

    The deadline is real (FREE_CLIP_RETENTION_DAYS); this simply makes it
    visible instead of deleting silently. Doubles as re-engagement for users
    who clipped once and went quiet.
    """
    # The app has no /dashboard route and never has: it is a single-page app
    # routed entirely through the hash, and #app is what opens it (see
    # dashboard/src/main.jsx). /dashboard used to land on the SPA fallback, so
    # the button quietly showed the marketing page; now that unknown paths
    # return a real 404 it fails visibly.
    dash = f"{settings.frontend_url}/#app"
    pricing = f"{settings.frontend_url}/#/pricing?src=email_expiring"
    n = clip_count
    clips = f"{n} clip" + ("s" if n != 1 else "")
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your {clips} will be deleted tomorrow ⏳</h2>
        <p>Free clips are stored for 7 days, and {('these' if n != 1 else 'this one')}
           {'are' if n != 1 else 'is'} about to expire. Two ways to keep them:</p>
        <ul style="line-height:1.9;padding-left:20px">
          <li><strong>Download them now</strong> from your dashboard, or</li>
          <li><a href="{pricing}"><strong>Upgrade to Starter ($12/mo)</strong></a> &mdash; clips stored forever,
              no watermark, and 100 minutes every month.</li>
        </ul>
        <p><a href="{dash}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">Save my clips</a></p>
        <p style="color:#666;font-size:13px">After tomorrow they're gone for good
           &mdash; we can't recover deleted clips.</p>
      </div>
    """
    print(f"⏳ Clips-expiring email → {email} ({clips})")
    await send_email(email, f"Your {clips} will be deleted tomorrow", html)


async def send_out_of_minutes_email(email: str, upgrade_url: str, user_id=None):
    """Free user hit their monthly quota — the natural upgrade moment.

    Mirrors the in-app upgrade modal: lead with what they lose by staying free
    (watermark, clips deleted after 7 days) and one concrete plan, not a
    generic pricing link.
    """
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your video is still waiting 🎬</h2>
        <p>You've used this month's 20 free minutes — which usually means the
           clips are working for you. Here's what <strong>Starter ($12/mo)</strong>
           changes today:</p>
        <ul style="line-height:1.9;padding-left:20px">
          <li><strong>100 minutes</strong> every month (5&times; your free quota)</li>
          <li><strong>No watermark</strong> on your clips</li>
          <li>Clips stored <strong>forever</strong> &mdash; free clips are deleted after 7 days</li>
        </ul>
        <p><a href="{upgrade_url}" style="display:inline-block;background:#111;color:#fff;
           padding:12px 20px;border-radius:8px;text-decoration:none">Upgrade and finish your video</a></p>
        <p style="color:#666;font-size:13px">Cancel anytime. Your free minutes
           reset on the 1st of every month.</p>
      </div>
    """
    if await send_commercial_email(user_id, email,
                                   "Your video is waiting — you're out of free minutes", html):
        print(f"✉️  Out-of-minutes upsell email → {email}")


async def send_account_deleted_email(email: str):
    """Confirmation that an account was erased. Sent last, to an address we no
    longer hold: this is the only notice the user will ever get, and it is also
    the only signal they would have if the deletion had not been theirs.
    """
    html = """
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your OpenShorts account has been deleted</h2>
        <p>Everything is gone: your account, your projects, your clips and their
           transcripts, your API keys, and the connection to any social accounts
           you had linked. Any active subscription was cancelled.</p>
        <p>Two things we keep, and why:</p>
        <ul style="line-height:1.7;padding-left:20px">
          <li><strong>Your invoices</strong>, for six years &mdash; Spanish
              commercial law requires it, and the Stripe customer reference
              that finds them goes with it.</li>
          <li><strong>A record that this deletion happened</strong>, for five
              years, holding a one-way hash of your email address instead of
              the address itself.</li>
        </ul>
        <p>You can sign up again any time with the same address; it will be a
           brand-new, empty account.</p>
        <p style="color:#666;font-size:13px">If this wasn't you, reply to this
           email straight away &mdash; info@openshorts.app.</p>
      </div>
    """
    print(f"✉️  Account-deleted confirmation → {email}")
    await send_email(email, "Your OpenShorts account has been deleted", html)


# --------------------------------------------------------------------------- #
# Lifecycle emails (cloud/lifecycle.py decides who gets which, and when).
# All of them are commercial communications: send_commercial_email skips
# accounts that unsubscribed and adds the unsubscribe footer and header.
# --------------------------------------------------------------------------- #
def _cta(url: str, label: str) -> str:
    return (f'<p><a href="{url}" style="display:inline-block;background:#111;color:#fff;'
            f'padding:12px 20px;border-radius:8px;text-decoration:none">{label}</a></p>')


def _first_video_line(first_video_minutes: int) -> str:
    if first_video_minutes > 0:
        return (f"<p><strong>Your first video is on us:</strong> anything up to "
                f"{first_video_minutes} minutes is clipped whole, free.</p>")
    return "<p>You get 20 free minutes every month.</p>"


async def send_welcome_email(user_id, email: str, first_video_minutes: int) -> bool:
    """Right after sign-up: one concrete next step, not a feature tour."""
    app_url = f"{settings.frontend_url}/#app"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Welcome to OpenShorts 👋</h2>
        <p>Paste a YouTube link (or upload a video) and you get vertical clips
           with captions, ready for TikTok, Reels and Shorts, in a few minutes.</p>
        {_first_video_line(first_video_minutes)}
        {_cta(app_url, "Clip my first video")}
        <p style="color:#666;font-size:13px">Works best with people talking:
           podcasts, interviews, streams, tutorials.</p>
      </div>
    """
    return await send_commercial_email(user_id, email, "Your first video is on us", html)


async def send_first_clip_email(user_id, email: str, first_video_minutes: int) -> bool:
    """A day after sign-up with no video processed yet."""
    app_url = f"{settings.frontend_url}/#app"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your clips are one link away</h2>
        <p>You signed up yesterday but haven't clipped anything yet. It takes
           one step: paste the link of a video you already published.</p>
        {_first_video_line(first_video_minutes)}
        {_cta(app_url, "Paste a link")}
        <p style="color:#666;font-size:13px">Stuck on something? Just reply to
           this email.</p>
      </div>
    """
    return await send_commercial_email(user_id, email, "Your first video is still free", html)


async def send_winback_email(user_id, email: str, promo_code: str = "",
                             promo_label: str = "") -> bool:
    """Two days after the first free video, still no plan."""
    pricing = f"{settings.frontend_url}/#/pricing?src=email_winback"
    if promo_code:
        offer = (f"<p>Here's <strong>{promo_label or 'a discount'}</strong>: use code "
                 f"<strong style=\"font-family:monospace;font-size:16px\">{promo_code}</strong> "
                 f"at checkout.</p>")
        subject = f"{promo_label or 'A discount'} on OpenShorts, for you"
    else:
        offer = ""
        subject = "Keep your clips, lose the watermark"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Liked your clips?</h2>
        <p>Free clips carry a watermark and are deleted after 7 days. Starter
           ($12/mo) gives you 100 minutes a month, no watermark, and clips
           stored forever. The clips you already made lose the mark the
           moment you upgrade.</p>
        {offer}
        {_cta(pricing, "See plans")}
        <p style="color:#666;font-size:13px">Cancel anytime.</p>
      </div>
    """
    return await send_commercial_email(user_id, email, subject, html)


async def send_paid_idle_email(user_id, email: str, minutes: int, period_end,
                               ending: bool = False) -> bool:
    """A paying plan with nothing processed this period, days into it."""
    app = f"{settings.frontend_url}/#app"
    autopilot = f"{settings.frontend_url}/#app?tab=autopilot"
    until = f"{period_end:%B} {period_end.day}"
    tail = (f"Your plan ends on {until}, so they are yours until then."
            if ending else f"They reset on {until} and don't carry over.")
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>Your {minutes} minutes are waiting</h2>
        <p>You haven't clipped anything this month yet. {tail}</p>
        <p>Paste a YouTube link (a podcast, a stream, a talk) and you get
           ready-to-post vertical clips, no watermark.</p>
        {_cta(app, "Clip a video")}
        <p><strong>No time to do it by hand?</strong> Turn on Autopilot: pick
           your channel once and every new video you publish is clipped for
           you automatically.</p>
        <p><a href="{autopilot}">Set up Autopilot</a></p>
      </div>
    """
    return await send_commercial_email(user_id, email, f"Your {minutes} minutes are waiting", html)


async def send_comeback_email(user_id, email: str, free_minutes: int) -> bool:
    """A few days after a subscription ended: the free plan is still there."""
    app = f"{settings.frontend_url}/#app"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>You still have {free_minutes} free minutes every month</h2>
        <p>Your plan has ended, but your account hasn't: {free_minutes} minutes
           of video a month stay free, on the same GPU, no card needed.</p>
        <p>Got a new video? Paste the link and get your clips.</p>
        {_cta(app, "Clip a video")}
        <p style="color:#666;font-size:13px">Free clips carry a watermark and
           last 7 days. When you need more, plans are on the same page.</p>
      </div>
    """
    return await send_commercial_email(
        user_id, email, f"Your {free_minutes} free minutes are back", html)


async def send_checkout_recovery_email(user_id, email: str, recovery_url: str,
                                       cart_label: str = "") -> bool:
    """A Stripe Checkout expired unpaid. ``recovery_url`` reopens the same cart
    (valid 30 days, Stripe's ``after_expiration.recovery``). ``cart_label`` is
    ``lifecycle.cart_label``: "the Starter plan ($12.00)", "60 extra minutes"."""
    what = cart_label or "your OpenShorts plan"
    html = f"""
      <div style="font-family:system-ui,sans-serif;max-width:480px;margin:0 auto">
        <h2>You didn't finish checking out</h2>
        <p>You started checking out {what} but the payment wasn't completed.
           Your cart is saved: one click takes you back to it.</p>
        {_cta(recovery_url, "Finish checkout")}
        <p style="color:#666;font-size:13px">Card declined? PayPal and other
           methods are on the same page. Reply to this email if something
           didn't work.</p>
      </div>
    """
    return await send_commercial_email(user_id, email, "Your OpenShorts checkout is saved", html)
