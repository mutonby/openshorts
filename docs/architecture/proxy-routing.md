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
  Such a link goes from the server's own address (a connection that never
  came up is tried once more) and then through the statics, stopping at the
  first one that reaches the site. A static whose provider does not carry the
  site answers the CONNECT itself with a 403 (`main.exit_refused`): the next
  static is asked, and that refusal is never read as a proxy without credits
  (`alerts._looks_like_proxy_error`).
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
  midnight (`reserve_process_minutes` sets `request.state.paid_allowed`;
  `run_job` checks again when a queued job starts). Thumbnail Studio downloads
  run in the API process and take `allow_paid` / `route_out` from
  `download_youtube_video` instead of the env.
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

## The account's request budget (`yt_account.py`)

Every probe and every download that sends `YOUTUBE_COOKIES` spends the same
account's budget, and YouTube limits a busy account ("Your account has been
rate-limited by YouTube for up to an hour"). While it is limited every request
with the cookies fails on every route and counts against it again, and the
anonymous attempts are bot-checked on non-residential IPs, so downloads failed
outright.

- **Paced.** `yt_account.request()` wraps each authenticated extraction (the
  metering probe, the quality probe, the download): at most `YT_ACCOUNT_SLOTS`
  (2) at a time on the host, started `YT_ACCOUNT_MIN_INTERVAL` (2 s) apart.
  State is files in `output/`, shared by job processes and both containers of
  a deploy. Fail-open: a slot that never frees is skipped after
  `YT_ACCOUNT_MAX_WAIT`.
- **Per video.** After two "rate-limited" answers
  (`YT_ACCOUNT_ANSWERS_PER_VIDEO`) a probe or a download stops sending the
  cookies for that video: the remaining routes would get the same answer, and
  each one counts against the account. Not after one: the limit is often on a
  single exit IP and the next static route gets the video. The anonymous
  attempts still run.
- **Cooldown.** YouTube limits by degrees: stray "rate-limited" answers among
  requests that work are routine (hours of them on 12-nov-2026 with every
  download succeeding). `YT_ACCOUNT_STRIKES` (6) in a row within
  `YT_ACCOUNT_STRIKE_WINDOW` (180 s), with no authenticated success in
  between, start `YT_ACCOUNT_COOLDOWN_SECONDS` during which nobody sends the
  cookies at all. **Off by default (0)**: it only helps where a download can
  succeed without the account, and in production the anonymous attempts are
  bot-checked on every IP, so a cooldown stopped every download (12-nov-2026:
  a half-limited account went to one success in a hundred). Turn it on only
  with a route that works anonymously.
- **One account, one exit.** A probe and a download start on their
  account's own static route (`yt_account.home_exit`: the account's index
  over the routes as configured), not on a random one. A signed-in session
  that shows up from every address of the pool within seconds does not look
  like a person, and the accounts were being limited far below the request
  rate YouTube lets one through. With more accounts than routes they share,
  as people behind one address do. The other routes are still tried when the
  first fails. Keep `STATIC_PROXY_URLS` in the same order across deploys.
- **Every account resting.** A job's download waits for the first one to
  wake, up to `YT_ACCOUNT_REST_WAIT` (600 s, 0 for none), instead of asking
  the one closest to waking: that one answered "rate-limited" again, and a
  queue of downloads doing it kept every account limited for as long as
  there were downloads to try. Never a download that runs inside the API
  (`may_wait`), and the retry of a limited video is scheduled for when an
  account wakes (`yt_account.all_resting`).
- **Pace for a busy host.** The defaults (2 slots, 2 s apart per account)
  suit one account and a few jobs. With a queue behind several accounts,
  `YT_ACCOUNT_SLOTS=1` and `YT_ACCOUNT_MIN_INTERVAL` of 6 to 8 keep each
  account at a steady request every few seconds instead of bursts when many
  jobs start together; it still leaves room for hundreds of extractions an
  hour per account.
- **Several accounts.** `YOUTUBE_COOKIES_2` .. `YOUTUBE_COOKIES_9` add
  accounts next to `YOUTUBE_COOKIES`. Each video takes the next one in turn
  (`yt_account.Session`, one private cookie file per probe or download), and
  every account has its own slots and pacing. When YouTube answers
  "rate-limited" the video moves to another account that is not resting
  before it counts the answer against `YT_ACCOUNT_ANSWERS_PER_VIDEO`. An
  account with `YT_ACCOUNT_REST_STRIKES` (3) such answers in a row rests
  `YT_ACCOUNT_REST_SECONDS` (900) while the others carry the load; with every
  account resting, the one closest to waking is used, never none. The session
  watcher (`cloud/alerts.py`) only checks the first account.
- **Age-restricted videos.** YouTube opens them only to an age-verified
  account ("Sorry, this content is age-restricted ... provide more info so we
  can be sure you're an adult"; anonymous: "Sign in to confirm your age").
  `yt_account.is_age_gate` + `Session.age_gated()` move the video to an
  account it has not tried (awake first), with no strike against the one
  that refused: it is fine for every other video. The probe and the download
  both do it. With every account tried, the download stops there, before the
  per-GB proxy, and prints `main.AGE_RESTRICTED_MESSAGE` (download it and
  upload the file); unless an account was only rate-limited, in which case
  the job waits for the cooldown like any limited download.
- **An account served protected formats only.** YouTube answers some
  signed-in accounts "This video is DRM protected" for every video, for hours.
  Every job that drew that account then failed on each route, and the
  anonymous fallback only met the bot check. `yt_account.is_drm` +
  `Session.drm_refused()` move the video to an account it has not tried
  (awake first) and count the answer against the one that gave it, like a
  limit: enough of them in a row rest it, a request that works clears them.
  A video protected for everybody ends once every account said so. The probe
  and the download both do it.
- **One probe per submission.** The quality gate reuses what the metering
  probe just learned (`metering.probe_facts`) instead of asking again.
- **The download starts from the probe's answer.** The probe and the
  download asked YouTube the same thing minutes apart (the formats and their
  media links), each with the account's cookies. With the download cache on,
  the probe keeps yt-dlp's sanitized result per video id
  (`source_cache.info_put`, `<id>.info.json`) until 15 minutes before the
  first of its media links expires (they carry their own `expire`, about six
  hours), and `main._stored_extraction` hands it to the download, which then
  asks nothing of the account: for a whole download and for a ranged one
  alike, and for another user's job on the same video too. Media links are
  bound to the address that asked for them, so it is used only through the
  same exit (`source_cache.route_key`: a digest, the proxy URL with its
  credentials is never written) with the same client list, and never from
  or on the per-GB proxy, whose exit changes. A kept extraction that does
  not download is dropped and the usual attempts run as if it had never been
  there; that failure is a media request, not one of the account's.
  `SOURCE_CACHE_INFO=0` turns it off.
- **No probe for a submission that cannot be queued.** The cost line
  (`/api/process/estimate`) runs when a link is pasted, before any submit. An
  account the slow lane's brake refuses on submit (`app._slow_lane_of` and
  `queue_mode.brake_active`) gets no estimate and no probe: those probes were
  a large share of the accounts' requests while the brake was on, for jobs
  that were then refused. A length the account already has cached is still
  shown.
- **Wait, not fail.** A download that failed only because of the limit prints
  `YT_ACCOUNT_LIMITED`; `app._schedule_auto_retry` puts the job back in line
  for when the cooldown ends, up to `YT_LIMIT_RETRIES` (3) times, on a counter
  of its own. The user sees "YouTube is limiting downloads right now".


## Ranged downloads cut short (`main.ranged_cut_short`)

A ranged fetch (source far longer than the paid minutes) is ffmpeg reading
each stream in one long HTTP request. A connection dropped mid-stream ends
the input ("Error during demuxing: Input/output error") while ffmpeg still
exits 0, so the attempt used to "succeed" with the first minutes only.
- With the source's duration known, the file is checked against the range
  (`ranged_cut_short`, more than `RANGED_SHORTFALL_SECONDS` short):
  `RangedDownloadCutShort` retries once on the same route; cut short twice (a
  reused extraction counts as the first), the part that came in is kept and
  no other route, nor the per-GB proxy, is tried. The job is then billed for
  that part (`app._trim_to_downloaded_source`), and the attempt is recorded
  with `short` in the route line.
- ffmpeg is not told to reconnect (`-reconnect`..). A stream the site keeps
  ending is resumed a few MB at a time, far below playback speed: the fetch
  then lasts the best part of an hour, and every such job holds its
  preparation here and the slot of the worker waiting for its source. A
  dropped stream ends the fetch and the check above deals with it.
- A ranged fetch has a time and a speed limit (`main.RangedFetchGuard`):
  past `RANGED_FETCH_MAX_SECONDS` (300), or once it brings in less than
  `RANGED_FETCH_MIN_KBPS` (100) over a minute, ffmpeg is interrupted (SIGINT,
  so it closes the file it was writing) and the part that came in is kept at
  once, as `short` and `slow` in the route line. It is not fetched again: a
  slow route is slow again. Some exits pace those long reads far below
  playback speed for stretches of the day; without the limit the slow fetches
  pile up, every slot waits on one, and paid jobs queue behind them. `0`
  turns a limit off. An ffmpeg that ignores the interrupt is killed and the
  attempt fails like any other.


## Download cache (`source_cache.py`)

Many users clip the same popular YouTube videos within hours of each other,
and every repeat used to spend a request of a YouTube account at submit (the
metering probe) and again at download. With `SOURCE_CACHE_DIR` set (off when
empty; Cloud points it at `output/.source_cache`) a video is asked for once:

- **Keyed by the YouTube id** (`source_cache.video_id`: watch, youtu.be,
  shorts, live, embed; never `/clip/`, never another host). The rest of the
  URL (`t=`, `si=`, `list=`) does not matter.
- **The source.** `main.download_youtube_video` takes a fresh entry before
  any network call: hard-linked into the job dir (one inode, a copy only
  across filesystems), and the job prints `PROXY_ROUTE` with winner `cache`,
  so `proxy_usage` counts the hits. A miss holds the id's lock
  (`source_cache.claim`) while it downloads, so jobs on the same video wait
  for that one download (`SOURCE_CACHE_WAIT_SECONDS`, 900, then they download
  anyway). Only a whole source over a free route is kept
  (`is_whole_free_download`): never a ranged fetch (just the paid minutes)
  nor one won on the per-GB proxy (720p). Only a job's download
  (`use_cache=True`) uses it: the API's in-process downloads (Thumbnail
  Studio) run on a shared executor that must not wait on another job's lock. The job's own cut (`cap_source_duration`)
  replaces its file with `os.replace`, so the cached copy stays whole.
- **The transcript.** Stored next to the source after a job transcribes that
  very file (same inode or same size, so never a cut one) and reused for the
  same length (±0.5 s) and the same `TRANSCRIBE_BACKEND`.
- **The probe.** `metering.probe_url_minutes` answers from `<id>.probe.json`
  (length and best height, which also feeds `probe_facts` for the quality
  gate) and records every YouTube probe there. A video's length does not
  change; a video made private meanwhile fails at the download as before.
- **Bounded.** An entry lives `SOURCE_CACHE_HOURS` (48) from when it was
  stored; past `SOURCE_CACHE_MAX_GB` (150) the least recently used go first
  (`prune`, on every store). A source over `SOURCE_CACHE_MAX_ENTRY_GB` (4; 0
  for no limit) is not kept: a handful of many-hour streams of tens of GB
  each filled the cache and pushed the ordinary sources out within the hour. It is a dot entry of `output/`: never served by
  `/videos` (dot segments are refused), never swept, and not counted by the
  output size cap (`app._output_jobs_size`), which could only purge finished
  jobs for room it is not allowed to free.
- Public video files keyed by id, no user or job attached. What is read back
  is data: the title becomes a file name only through `source_cache.sanitize`
  and a basename. Fail-open: any error is a miss and the job downloads.
