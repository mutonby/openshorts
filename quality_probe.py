"""
Fast pre-flight probe: what maximum resolution does YouTube offer for a URL?

Run as a short subprocess by app.py BEFORE a job starts, so the user can decide
to abort (and e.g. refresh cookies) instead of burning 20+ minutes of
transcription and rendering on a 360p-only source.

Prints JSON to stdout: {"max_height": int, "mode": str, "cookies_invalid": bool}
Always exits 0 — the caller treats probe failures as "unknown" and starts the
job anyway (fail-open), where the in-job warning still applies.
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _find_cookies_path():
    # Mirrors main.py's cookie discovery.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(script_dir, 'www.youtube.com_cookies.txt'),
        os.path.join(script_dir, 'cookies.txt'),
        '/app/cookies.txt',
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe available YouTube quality for a URL.")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    result = {"max_height": 0, "mode": None, "cookies_invalid": False, "duration": 0}
    # Same gates as the metering probe and the download (app.py also checks
    # before spawning this): a URL on a private / tailnet address, or a
    # YouTube page that is not one video, never reaches yt-dlp. The empty
    # result is the fail-open "unknown", so the caller carries on and the
    # real probe / download refuse the URL with their own message.
    try:
        from security_utils import assert_public_url
        from yt_clients import youtube_non_video_reason
        assert_public_url(args.url)
        if youtube_non_video_reason(args.url):
            raise ValueError("not a single video")
    except Exception:
        print(json.dumps(result))
        return 0
    try:
        import yt_dlp
        import ssrf_guard

        warnings = []

        class _CollectLogger:
            def debug(self, msg):
                pass

            def info(self, msg):
                pass

            def warning(self, msg):
                warnings.append(str(msg))

            def error(self, msg):
                warnings.append(str(msg))

        import contextlib
        import yt_account
        # One of the configured accounts; a cookies file next to the script
        # (self-host) when the environment has none.
        session = yt_account.Session()
        cookies_path = session.path or _find_cookies_path()
        base_opts = {
            'quiet': True,
            'no_warnings': False,
            'logger': _CollectLogger(),
            'socket_timeout': 20,
            'retries': 2,
            'cachedir': False,
            # A watch?v=X&list=... link is the one video; without this
            # yt-dlp walks the whole list (see cloud/metering.py).
            'noplaylist': True,
        }
        # yt-dlp's default player clients (tv_downgraded/web_safari with
        # cookies, android_vr/web_safari anonymous) are the ones that still
        # serve HD without PO tokens — never override them here.
        attempts = []
        # Not while YouTube has the account limited (yt_account): the request
        # would fail and count against it again.
        if cookies_path and yt_account.cooling_down() <= 0:
            attempts.append(("cookie-auth", {**base_opts, 'cookiefile': cookies_path}))
        attempts.append(("anonymous", {**base_opts, 'cookiefile': None}))

        for mode, opts in attempts:
            authed = mode == "cookie-auth"
            try:
                with (session.request() if authed else contextlib.nullcontext()):
                    with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(args.url, download=False)
                if authed:
                    session.ok()
                formats = info.get('formats') or []
                heights = [
                    f.get('height') or 0
                    for f in formats
                    if f.get('vcodec', 'none') != 'none'
                    and f.get('protocol') != 'mhtml'
                    and f.get('ext') != 'mhtml'
                ]
                max_height = max(heights, default=0)
                if max_height > result["max_height"]:
                    result["max_height"] = max_height
                    result["mode"] = mode
                if not result["duration"]:
                    result["duration"] = int(info.get('duration') or 0)
                if result["max_height"] >= 1080:
                    break
            except Exception as e:
                if authed and yt_account.is_rate_limit(str(e)):
                    session.limited()
                continue

        result["cookies_invalid"] = any("no longer valid" in w for w in warnings)
    except Exception:
        pass

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
