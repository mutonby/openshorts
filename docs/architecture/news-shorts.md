# Licensed news shorts pipeline

`news_shorts/` discovers daily political-news videos, cuts vertical clips with the
existing clipper (`main.py`), queues them for human review, and publishes approved
clips to YouTube Shorts and TikTok.

## Licensing rules
Only three sources are accepted (`news_shorts/licensing.py`): YouTube
Creative Commons (re-verified via `videos.list` `status.license`, not only the
search filter), Assemblée nationale open videos (pass `--url --license
assemblee-nationale --attribution ...`), or `owned`. Anything without a clear
licence, source URL and attribution is skipped. Attribution is appended to every
description. Nothing here tries to evade copyright detection.

## Usage
```bash
export YOUTUBE_API_KEY=...                       # discovery
python -m news_shorts.cli discover "Mélenchon"   # CC videos of the last 24h -> clips -> queue
python -m news_shorts.cli list
python -m news_shorts.cli approve output/news/<id>/clip.mp4   # manual approval
python -m news_shorts.cli publish                # DRY RUN (default)
python -m news_shorts.cli publish --no-dry-run   # real publish, approved clips only
```
Publishing env vars (never committed): `YOUTUBE_CLIENT_ID`,
`YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` (scope `youtube.upload`),
`TIKTOK_ACCESS_TOKEN` (Content Posting API, `video.publish`). TikTok posts default
to `SELF_ONLY` until your app is audited.

## Local credentials file

The CLI also reads a local `.env` (git-ignored) with `KEY=VALUE` lines for the variables above, without overriding variables already set in the shell.

## 9:16 hook template

`python3 -m news_shorts.cli template CLIP --hook "Punchy line" --credit "Source : Channel (CC BY)"`
adds a high-contrast hook card for the first 3 seconds (fade in/out) and a permanent source credit on a 1080x1920 canvas.
Use the clip's `viral_hook_text` from the metadata as the hook, and keep the credit: it is the attribution the license requires.
