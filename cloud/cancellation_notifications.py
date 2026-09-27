"""Best-effort forwarding of cancellation feedback to the existing admin chat."""
import re

from .alerts import TELEGRAM_PREFIX, send_telegram, user_ref


_EMAIL = re.compile(r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+")
_NUMBER = re.compile(r"(?<!\w)\+?\d(?:[\d ()\t.-]*\d)?(?!\w)")


def _redact(value) -> str:
    """Minimize obvious emails/phone/card-like numbers, NOT anonymization.

    Names, addresses and obfuscated identifiers can remain. No raw text is logged.
    """
    text = _EMAIL.sub("[email redacted]", str(value or ""))
    return _NUMBER.sub(
        lambda match: "[number redacted]"
        if sum(char.isdigit() for char in match.group()) >= 7 else match.group(),
        text,
    )


def _trim(text: str, limit: int) -> str:
    # Count UTF-16 units too, so emoji cannot put the Telegram payload over limit.
    encoded = text.encode("utf-16-le", errors="replace")
    if len(encoded) <= limit * 2:
        return text
    return encoded[: (limit - 1) * 2].decode("utf-16-le", errors="ignore") + "…"


def _field(value, limit: int) -> str:
    # Redact before trimming: never forward a sliced-off fragment of a secret.
    return _trim(" ".join(_redact(value).split()), limit)


async def notify_cancellation_feedback(
    user_id,
    reason: str,
    comment: str,
    rating: int | None = None,
    plan: str | None = None,
    feedback_id: str | None = None,
) -> None:
    """Forward received feedback, not a confirmation of Stripe cancellation.

    Caller should schedule as a background task if request latency matters.
    Uses existing Telegram configuration; unset configuration is a no-op.
    Redaction is intentionally heuristic, not a guarantee of anonymization.
    """
    try:
        lines = [
            "cancellation requested/feedback received, not confirmed",
            user_ref(user_id),
            f"Reason: {_field(reason, 200)}",
        ]
        if rating is not None:
            lines.append(f"Rating: {_field(rating, 16)}")
        if plan is not None:
            lines.append(f"Plan: {_field(plan, 80)}")
        if feedback_id is not None:
            lines.append(f"Feedback ID: {_field(feedback_id, 80)}")
        # Quote every user-content line to distinguish it from the stage/metadata.
        quoted = "\n".join(
            "> " + line for line in _redact(comment).splitlines()
        ) or "> (no comment)"
        lines.append("Comment (user-supplied text):\n" + _trim(quoted, 2500))
        budget = 4095 - len(TELEGRAM_PREFIX.encode("utf-16-le")) // 2
        message = _trim("\n".join(lines), budget)
        # Bypass the sender's exception printing: exception strings can contain
        # request bodies. Swallow here instead without logging feedback or errors.
        await send_telegram(message, raise_errors=True)
    except Exception:
        return
