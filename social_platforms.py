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
- X: the caption as the post text (``x_title``), with platform limits
  handled by Upload-Post.
- Pinterest: a title, pin description, and the selected board ID.
"""
import re
from typing import Iterable, List

import httpx

PLATFORMS = ("tiktok", "instagram", "youtube", "facebook", "linkedin",
             "threads", "bluesky", "x", "pinterest")
PINTEREST_BOARD_ID = r"^[0-9]{1,32}$"
PINTEREST_BOARDS_URL = "https://api.upload-post.com/api/uploadposts/pinterest/boards"

LABELS = {
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "youtube": "YouTube",
    "facebook": "Facebook",
    "linkedin": "LinkedIn",
    "threads": "Threads",
    "bluesky": "Bluesky",
    "x": "X",
    "pinterest": "Pinterest",
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
                  yt_title: str = None, tiktok_post_mode: str = "MEDIA_UPLOAD",
                  pinterest_board_id: str = None, source_credit_url: str = None) -> dict:
    """The per-network form fields of one video upload to ``platforms``."""
    ps = set(platforms or [])
    title = title or ""
    description = description or title
    x_text = description
    if source_credit_url:
        description = description.rstrip()
        # X's provider removes links for accounts without its X Links add-on.
        # Keep the URL on its own line so no empty attribution label is left.
        x_text = f"{description}\n\n{source_credit_url}"
        description = f"{description}\n\nSource: {source_credit_url}"
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
    if "x" in ps:
        data["x_title"] = x_text
    if "pinterest" in ps:
        if not isinstance(pinterest_board_id, str) or not re.fullmatch(PINTEREST_BOARD_ID, pinterest_board_id):
            raise ValueError("Choose a Pinterest board.")
        data["pinterest_board_id"] = pinterest_board_id
        data["pinterest_title"] = title[:100]
        data["pinterest_description"] = description
    return data


class PinterestBoardError(ValueError):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


async def pinterest_boards(api_key, profile):
    """List eligible boards from one explicit Upload-Post profile."""
    if not api_key or not profile:
        raise PinterestBoardError("Connect Pinterest for this publishing profile.", 409)
    try:
        async with upload_post_client(20.0) as client:
            response = await client.get(PINTEREST_BOARDS_URL, headers={"Authorization": f"Apikey {api_key}"},
                                        params={"profile": profile})
        if response.status_code in (401, 403, 404):
            raise PinterestBoardError("Connect or reconnect Pinterest for this publishing profile.", 409)
        if response.status_code == 429:
            raise PinterestBoardError("Pinterest board lookup is rate limited. Try again shortly.", 429)
        response.raise_for_status()
        boards = response.json()["boards"]
        if not isinstance(boards, list):
            raise ValueError
    except PinterestBoardError:
        raise
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise PinterestBoardError("Could not load Pinterest boards. Try again shortly.", 502) from exc

    eligible = {}
    for board in boards:
        if not isinstance(board, dict) or str(board.get("privacy") or "").upper() == "SECRET":
            continue
        board_id = str(board.get("id") or "")
        name = board.get("name")
        if re.fullmatch(PINTEREST_BOARD_ID, board_id) and isinstance(name, str) and name.strip():
            eligible[board_id] = {"id": board_id, "name": name.strip()}
    return list(eligible.values())


async def validate_pinterest_board(api_key, profile, board_id):
    if not board_id:
        raise PinterestBoardError("Choose a Pinterest board.", 400)
    boards = await pinterest_boards(api_key, profile)
    if not any(board["id"] == board_id for board in boards):
        raise PinterestBoardError("Choose an available Pinterest board for this publishing profile.", 400)


# A connection to Upload-Post that does not open in a few seconds is retried
# instead of waited on: from a busy host some connects to the API hung until
# the 20 s timeout and the user got a 500 (16-nov), while the next attempt
# connected in milliseconds. httpx retries only failures to connect, so a
# request that reached Upload-Post is never sent twice.
UPLOAD_POST_CONNECT_TIMEOUT = 5.0
UPLOAD_POST_CONNECT_RETRIES = 2


def upload_post_client(timeout: float):
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=UPLOAD_POST_CONNECT_TIMEOUT),
        transport=httpx.AsyncHTTPTransport(retries=UPLOAD_POST_CONNECT_RETRIES))
