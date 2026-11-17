"""Only content with a clear reuse licence is ever accepted."""

ALLOWED_LICENSES = {
    "creativeCommon": "YouTube Creative Commons (CC BY)",
    "assemblee-nationale": "Assemblée nationale open videos (attribution required)",
    "owned": "Owned by the channel operator",
}


def is_reusable(candidate: dict) -> bool:
    """A candidate needs an allowed licence, a source URL and an attribution."""
    return (
        candidate.get("license") in ALLOWED_LICENSES
        and bool(candidate.get("source_url"))
        and bool(candidate.get("attribution"))
    )


def attribution_text(candidate: dict) -> str:
    lic = ALLOWED_LICENSES.get(candidate.get("license"), "")
    return f"Source: {candidate.get('attribution')} - {candidate.get('source_url')} ({lic})"
