# Download routing and paid proxy accounting (`cloud/proxy_ledger.py`)

## Route order

Downloads go **direct → static ISP proxies (`STATIC_PROXY_URLS`, flat rate) →
per-GB proxy (`PROXY_URL`)**. The duration probe
(`cloud/metering.probe_url_minutes`) follows the same order, with one extra free
step before any per-GB attempt: the fallback clients through a static
(`fallback-static`).

## yt-dlp clients

**The client list is explicit and shared** (`yt_clients.py`: `default,mweb` +
the bgutil PO token provider). With account cookies yt-dlp's own defaults are
`tv_downgraded` + `web`, and on a share of videos both come back UNPLAYABLE /
SABR-only, which yt-dlp reports as "Video unavailable" and looks like an IP ban.
`mweb` needs the PO token and the token needs the webpage: **never put
`player_skip: webpage` back.** Every attempt asks for the 1080p format spec (the
old `best[ext=mp4]` spec was itself the 360p progressive file).

## Cookies: first attempt with, second without

The probe carries `YOUTUBE_COOKIES`, like the download does: an anonymous probe
from datacenter IPs gets "Sign in to confirm you're not a bot" in bursts, while
the authenticated request goes through the same IPs. But the **first** attempt
on a route carries the cookies and the second drops them, on the probe as on the
download: with cookies YouTube answers UNPLAYABLE for every client on a share of
videos, while the same video anonymously on the same IP returns 1080p. Without
the anonymous second attempt the probe reads a cookie problem as an IP problem
and escalates to the per-GB proxy, which carries the same cookies and fails
identically.

## When the per-GB proxy is allowed

- The probe reaches it **only** when a static route failed for a reason another
  IP can fix (`static_failure_warrants_paid`: bot-check, 403/429, proxy/network
  errors). Never for a private/removed/members-only video, an uploader's country
  block, or a live stream with no duration (those fail the same on every IP).
- **Never for a non-YouTube URL** (Twitch, Kick, Rumble, product pages...).
- Both probe and download pass `noplaylist`: a `watch?v=X&list=...` or mix link
  is the one video the user was watching; otherwise yt-dlp walks the whole list
  and dies on an entry nobody pasted.
- The probe keeps **every** attempt's error per static route, not the last one:
  the anonymous retry ends in a bot-check by design, and an age-gate from the
  cookie attempt is the real verdict, so it must not be overwritten into an
  escalation.
- A URL that is not one video is refused by path before any request
  (`yt_clients.youtube_non_video_reason`): `noplaylist` does nothing for those.
  That check is an **allowlist** of the paths that carry a video id, not a
  denylist of known bad pages (hashtag pages and legacy `/<vanity>` channel URLs
  slipped through a denylist).
- YouTube can bot-check anonymous requests from every non-residential IP
  **per video** (the same video fails on all of them whatever the client; only
  an authenticated session or a residential IP gets through). So when the probe
  saw every static bot-checked and the paid proxy answer, the job carries
  `DOWNLOAD_SKIP_STATICS=1` (`metering.pop_statics_bot_checked`, also in the
  resume manifest) and `plan_download_attempts` goes straight to the paid
  attempts instead of repeating anonymous hits for the same verdict.
- `PAID_PROXY_DAILY_MB` (default 500) is the hard ceiling: past it the paid
  proxy is dropped from the probe and from every new job's env until UTC
  midnight.
- Do not keep `PROXY_URL` in a local `.env`: every local `main.py` run would then
  bill the per-GB proxy.

## Ledger and alerts

`main.py` prints `PROXY_ROUTE=<json>` after every download (winner, paid bytes
across all attempts including failed paid ones, each free attempt's error);
`app.py` persists it as a `proxy_usage` row at job end and alerts when the paid
proxy carried bytes, folding a burst into one message per 5 min. The table, not
the in-memory monthly counter or the rotating container log, is what answers
"what did this day cost".

The watcher probes the static pool against a real YouTube watch page (playable
markers), not a generic connectivity endpoint: YouTube can refuse the static IPs
while other hosts keep answering. When `YOUTUBE_COOKIES` is set it also checks
the ytcfg `LOGGED_IN` marker and alerts on the first probe that says the session
is gone (`cloud/alerts.py`).
