#!/usr/bin/env python3
"""
YouTube OAuth2 authentication setup.

This script guides you through getting YouTube Data API credentials and
saving them for use by youtube_utils.py.

Requirements:
1. Google Cloud Console project with YouTube Data API v3 enabled
2. OAuth2 client ID (Desktop application)
3. This script will open your browser to authorize

Usage:
    python youtube_auth.py --output secrets/youtube_token.json
"""

import argparse
import json
import os
from pathlib import Path

from google.auth.oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials


# YouTube Data API v3 scopes (upload + video management)
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_credentials(client_secrets_file: str, token_output: str) -> Credentials:
    """
    Authenticate with YouTube using OAuth2 and save credentials.

    Args:
        client_secrets_file: Path to OAuth client secrets JSON from Google Cloud Console
        token_output: Where to save the resulting token

    Returns:
        Credentials object (also saved to token_output)
    """
    print("\n🔐 YouTube Data API Authentication\n")
    print(f"Using client secrets: {client_secrets_file}")

    if not os.path.exists(client_secrets_file):
        print(f"\n❌ Client secrets file not found: {client_secrets_file}")
        print("\nTo set this up:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a new project or select existing one")
        print("  3. Enable YouTube Data API v3")
        print("  4. Create OAuth2 credentials (Desktop application)")
        print("  5. Download credentials as JSON")
        print("  6. Save it to the path specified above\n")
        exit(1)

    # Create authorization flow
    flow = InstalledAppFlow.from_client_secrets_file(
        client_secrets_file,
        scopes=SCOPES,
        redirect_uri="http://localhost:8888/callback"
    )

    print("\n👉 A browser window will open. Please authorize access to YouTube.")
    print("   (This allows uploading videos to your channel)\n")

    credentials = flow.run_local_server(port=8888, open_browser=True)

    # Save credentials for future use
    output_path = Path(token_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as token_file:
        token_file.write(credentials.to_json())

    print(f"\n✅ Credentials saved to: {output_path}")
    print(f"   Keep this file secure! It allows uploading to your YouTube channel.\n")

    return credentials


def refresh_token(token_path: str) -> Credentials:
    """Refresh an existing token."""
    print(f"\n🔄 Refreshing YouTube token from {token_path}...\n")

    with open(token_path) as f:
        token_data = json.load(f)

    credentials = Credentials.from_authorized_user_info(token_data, scopes=SCOPES)

    if credentials.expired:
        print("Token is expired, refreshing...")
        refresh_request = Request()
        credentials.refresh(refresh_request)

        with open(token_path, 'w') as f:
            json.dump(json.loads(credentials.to_json()), f, indent=2)

        print(f"✅ Token refreshed and saved to {token_path}\n")
    else:
        print("✅ Token is still valid\n")

    return credentials


def main():
    parser = argparse.ArgumentParser(
        description="YouTube OAuth2 authentication for openshorts"
    )
    parser.add_argument(
        "--client-secrets",
        default="secrets/youtube_client_secrets.json",
        help="Path to OAuth client secrets JSON from Google Cloud Console"
    )
    parser.add_argument(
        "--output",
        default="secrets/youtube_token.json",
        help="Path to save OAuth2 token"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh an existing token instead of creating new"
    )

    args = parser.parse_args()

    if args.refresh:
        if not os.path.exists(args.output):
            print(f"❌ Token file not found: {args.output}")
            print("Run without --refresh to create new credentials\n")
            exit(1)
        refresh_token(args.output)
    else:
        get_credentials(args.client_secrets, args.output)

    print("You can now use this token with publish_to_youtube.py")
    print(f"  --youtube-token {args.output}\n")


if __name__ == "__main__":
    main()
