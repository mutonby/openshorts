"""The social networks OpenShorts posts clips to, through Upload-Post.

One place for the list, the display names and the per-network form fields of
an upload, so the clip post, the AI Shorts post and Autopilot send the same
thing. Field names follow the Upload-Post upload API: ``<platform>_title`` and
``<platform>_description`` override the generic ``title`` per network.

What each network shows of a clip:
- TikTok, Instagram: the caption, sent as ``*_title``.
- YouTube: a title (100 characters at most) and a description.
- Facebook (Reels), LinkedIn: a title and the caption as the post text. With
  several Facebook Pages, Upload-Post uses the profile's default Page.
- Threads: the caption only (Upload-Post posts ``threads_title`` when set and
  splits text over 500 characters into a thread).
- Bluesky: the caption only; Upload-Post cuts it to the 300-character limit.
"""
from typing import Iterable, List

PLATFORMS = ("tiktok", "instagram", "youtube", "facebook", "linkedin", "threads", "bluesky")

LABELS = {
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "youtube": "YouTube",
    "facebook": "Facebook",
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "bluesky": "Bluesky",
}

# YouTube refuses a title over 100 characters; the Gemini title is not bounded.
YOUTUBE_TITLE_MAX = 100


def youtube_title(title) -> str:
    """``title`` cut to YouTube's limit, on a word boundary when there is one."""
    t = " ".join(str(title or "").split())
    if len(t) <= YOUTUBE_TITLE_MAX:
        return t
    cut = t[:YOUTUBE_TITLE_MAX]
    space = cut.rfind(" ")
    return (cut[:space] if space >= YOUTUBE_TITLE_MAX // 2 else cut).rstrip(" ,.;:-")


def label(platform: str) -> str:
    return LABELS.get(platform, str(platform).capitalize())


def clean(values: Iterable[str]) -> List[str]:
    """The known platforms among ``values``, lower-cased, in a stable order."""
    wanted = {str(v).strip().lower() for v in (values or [])}
    return [p for p in PLATFORMS if p in wanted]


def unknown(values: Iterable[str]) -> List[str]:
    """The requested names that are not a platform OpenShorts posts to."""
    return [v for v in (values or []) if str(v).strip().lower() not in PLATFORMS]


def upload_fields(platforms: Iterable[str], *, title: str, description: str,
                  yt_title: str = None, tiktok_post_mode: str = "MEDIA_UPLOAD") -> dict:
    """The per-network form fields of one video upload to ``platforms``."""
    ps = set(platforms or [])
    title = title or ""
    description = description or title
    data = {}
    if "tiktok" in ps:
        data["tiktok_title"] = description
        data["post_mode"] = tiktok_post_mode
    if "instagram" in ps:
        data["instagram_title"] = description
        data["media_type"] = "REELS"
    if "youtube" in ps:
        data["youtube_title"] = youtube_title(yt_title or title)
        data["youtube_description"] = description
        data["privacyStatus"] = "public"
    if "facebook" in ps:
        data["facebook_title"] = title
        data["facebook_description"] = description
        data["facebook_media_type"] = "REELS"
    if "linkedin" in ps:
        data["linkedin_title"] = title
        data["linkedin_description"] = description
    if "threads" in ps:
        data["threads_title"] = description
    if "bluesky" in ps:
        data["bluesky_title"] = title
        data["bluesky_description"] = description
    return data


# A connection to Upload-Post that does not open in a few seconds is retried
# instead of waited on: from a busy host some connects to the API hung until
# the 20 s timeout and the user got a 500 (16-nov), while the next attempt
# connected in milliseconds. httpx retries only failures to connect, so a
# request that reached Upload-Post is never sent twice.
UPLOAD_POST_CONNECT_TIMEOUT = 5.0
UPLOAD_POST_CONNECT_RETRIES = 2


def upload_post_client(timeout: float):
    import httpx
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=UPLOAD_POST_CONNECT_TIMEOUT),
        transport=httpx.AsyncHTTPTransport(retries=UPLOAD_POST_CONNECT_RETRIES))
