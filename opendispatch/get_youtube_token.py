#!/usr/bin/env python3
"""
One-time script to get a YouTube OAuth refresh token.

Usage:
    pip install google-auth-oauthlib
    python get_youtube_token.py

It will open your browser. Sign in with the YouTube channel account.
After approval, the refresh token is printed here — paste it into .env as
YOUTUBE_REFRESH_TOKEN=<value>

The credentials file (youtube_credentials.json) must be in the same directory.
It is already in .gitignore so it won't be committed.
"""

from __future__ import annotations

import json
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CREDS_FILE = Path(__file__).parent / "youtube_credentials.json"


def main() -> None:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise SystemExit(
            "Missing dependency. Run:\n"
            "  pip install google-auth-oauthlib\n"
            "then re-run this script."
        )

    if not CREDS_FILE.exists():
        raise SystemExit(
            f"Credentials file not found: {CREDS_FILE}\n"
            "Download it from Google Cloud Console:\n"
            "  APIs & Services → Credentials → your OAuth client → Download JSON\n"
            "Save it as youtube_credentials.json in the opendispatch folder."
        )

    print("Opening browser for YouTube authorization...")
    print("Sign in with the Google account that OWNS your YouTube channel.\n")

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), scopes=SCOPES)

    # run_local_server opens browser + spins up localhost callback automatically
    credentials = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    print("\n" + "=" * 60)
    print("SUCCESS — Retrieved OAuth Credentials:")
    print("=" * 60)
    print(f"\nYOUTUBE_CLIENT_ID={credentials.client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={credentials.client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={credentials.refresh_token}")
    print("\n" + "=" * 60)

    # Attempt to automatically update .env
    env_file = Path(__file__).parent / ".env"
    if env_file.exists() and credentials.refresh_token:
        try:
            content = env_file.read_text(encoding="utf-8")
            import re
            content = re.sub(
                r"^YOUTUBE_REFRESH_TOKEN=.*$",
                f'YOUTUBE_REFRESH_TOKEN="{credentials.refresh_token}"',
                content,
                flags=re.MULTILINE,
            )
            # Ensure client id and secret are also aligned
            content = re.sub(
                r"^YOUTUBE_CLIENT_ID=.*$",
                f'YOUTUBE_CLIENT_ID="{credentials.client_id}"',
                content,
                flags=re.MULTILINE,
            )
            content = re.sub(
                r"^YOUTUBE_CLIENT_SECRET=.*$",
                f'YOUTUBE_CLIENT_SECRET="{credentials.client_secret}"',
                content,
                flags=re.MULTILINE,
            )
            env_file.write_text(content, encoding="utf-8")
            print("Auto-updated .env with YOUTUBE_REFRESH_TOKEN, YOUTUBE_CLIENT_ID, and YOUTUBE_CLIENT_SECRET!")
        except Exception as e:
            print(f"Notice: Could not auto-write to .env: {e}. Please copy/paste manually.")

    print("\nNote: Because the app is in 'Testing' mode in Google Cloud Console,")
    print("this token is valid for 7 days unless published to Production.")
    print("=" * 60)


if __name__ == "__main__":
    main()
