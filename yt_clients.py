"""yt-dlp client selection for YouTube, shared by the download (``main.py``)
and the duration probe (``cloud/metering.py``) so the two cannot drift.

Why the client list is explicit. With account cookies yt-dlp's defaults are
``tv_downgraded`` + ``web``, and for a share of videos both come back
UNPLAYABLE / SABR-only, which surfaces as "Video unavailable" with no
formats. That looked like an IP ban (it also happened on every static ISP IP)
and for a week sent ~26 downloads a week to the per-GB proxy, which then
downloaded 360p through the same dead client list. Measured on 6-sep-2026
inside the prod container, same static proxy, same video:

    cookies + default clients            -> Video unavailable
    cookies + mweb (+ PO token)          -> 1080p
    no cookies + default clients         -> 1080p
    cookies + default,mweb (+ PO token)  -> 1080p (also on a video that worked)

``mweb`` needs a GVS PO token for its https formats, which the bgutil
provider mints as long as the webpage is NOT skipped (``player_skip:
webpage`` drops the account's Data Sync ID and the token request fails).
The old fallback list (``tv_embed``, ``android``) is gone from yt-dlp: the
first is "unsupported", the second is skipped whenever cookies are present.
"""

HD_CLIENTS = ["default", "mweb"]


class NotASingleVideo(ValueError):
    """The URL is a YouTube page that is not one video (search, playlist,
    channel). Raised before yt-dlp ever sees it, by the probe and the
    download alike."""


class LiveStreamOnAir(NotASingleVideo):
    """The URL is a live stream still on air (or upcoming, or one YouTube is
    still turning into a video). Its metadata reports the time elapsed so far
    as the duration, but a download only gets the last seconds at the live
    edge: billing the reported length would charge for a video that is not
    there yet."""

    MESSAGE = ("This video is a live stream that is still on air. Paste the link "
               "again once the stream has ended and YouTube has finished processing it.")

    def __init__(self, message=None):
        super().__init__(message or self.MESSAGE)


# yt-dlp's ``live_status`` values whose download is not the whole video.
LIVE_NOT_READY = ("is_live", "is_upcoming", "post_live")


def live_not_ready(info) -> bool:
    """True when yt-dlp's metadata describes a stream that can't be fetched
    whole yet (see LiveStreamOnAir)."""
    if not isinstance(info, dict):
        return False
    return bool(info.get("is_live")) or info.get("live_status") in LIVE_NOT_READY


# Paths that carry one video id. Anything on youtube.com that is not one of
# these is refused: the list of pages that fan out is open-ended (channels,
# hashtags, the legacy /<vanity> channel URL) and each one costs a paginated
# browse before it fails, while a single-video shape we forgot costs the user
# one clear error message.
_SINGLE_VIDEO_PREFIXES = ("shorts", "live", "embed", "v", "e", "clip")

_NOT_ONE_VIDEO = {
    "results": "a search results page, not a video",
    "playlist": "a playlist, not a video; open one video in it and paste that link",
    "hashtag": "a hashtag page, not a video; open one video in it and paste that link",
    "channel": "a channel page, not a video",
    "c": "a channel page, not a video",
    "user": "a channel page, not a video",
    "feed": "a YouTube page with no video in it",
    "gaming": "a YouTube page with no video in it",
    "music": "a YouTube page with no video in it",
    "podcasts": "a YouTube page with no video in it",
    "playables": "a YouTube page with no video in it",
}


def youtube_non_video_reason(url):
    """Why this YouTube URL is not a single video, or None if it is one.

    A search page, a bare playlist, a channel or a hashtag is walked entry by
    entry by yt-dlp even with ``noplaylist`` (that option only strips the
    ``list=`` off a ``watch`` link). Measured in the prod container on
    17-sep-2026: a ``results?search_query=`` URL held the probe's executor
    thread for 2220 s and still failed on an entry that "premieres in 4 days",
    after the same URL had paid the per-GB proxy three times in a day. So the
    check is by path, up front and free: nothing that is not one video is
    worth a single request.

    It is an allowlist, not a list of bad paths. The first version named the
    pages it knew (search, playlist, channel, @handle) and let everything else
    through, and on 20-sep-2026 a URL whose browse yt-dlp titled "viralshorts"
    walked to page 23 on the static pool, escalated on the "Unable to download
    API page" errors that produced, and walked the same pages again through
    the per-GB proxy until YouTube rate-limited it.
    """
    from urllib.parse import urlparse, parse_qs
    try:
        parts = urlparse(url or "")
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host == "youtu.be":
        return None if parts.path.strip("/") else "youtu.be link without a video id"
    if not (host == "youtube.com" or host.endswith(".youtube.com")):
        return None
    path = parts.path.rstrip("/") or "/"
    segments = [seg for seg in path.split("/") if seg]
    if not segments:
        return "a YouTube page with no video in it"
    first = segments[0]
    if first in ("watch", "watch_popup"):
        return None if parse_qs(parts.query).get("v") else "a watch page without a video id"
    if first in _SINGLE_VIDEO_PREFIXES:
        return None if len(segments) > 1 else f"a /{first} link without a video id"
    if first.startswith("@") or first in _NOT_ONE_VIDEO:
        return _NOT_ONE_VIDEO.get(first, "a channel page, not a video")
    return ("a YouTube page that is not one video; open the video and paste "
            "the link of the video itself")


def pot_provider_args(bgutil_http, bgutil_script):
    """Extractor args selecting the PO token provider (bgutil http or script)."""
    if bgutil_http:
        return {"youtubepot-bgutilhttp": {"base_url": [bgutil_http]}}
    if bgutil_script:
        return {"youtubepot-bgutilscript": {"script_path": [bgutil_script]}}
    return {}


def hd_extractor_args(bgutil_http, bgutil_script):
    """Extractor args for the HD attempt, or None when no PO token provider
    is configured (the HD path needs one; the plan then skips it)."""
    provider = pot_provider_args(bgutil_http, bgutil_script)
    if not provider:
        return None
    return {**provider, "youtube": {"player_client": list(HD_CLIENTS)}}


def fallback_extractor_args(bgutil_http, bgutil_script):
    """Extractor args for the conservative attempt: same live client list,
    the provider when there is one, and never ``player_skip`` (see module
    docstring). Without a provider mweb still serves the progressive 360p
    format, which is enough for a probe and better than no download."""
    return {**pot_provider_args(bgutil_http, bgutil_script),
            "youtube": {"player_client": list(HD_CLIENTS)}}
