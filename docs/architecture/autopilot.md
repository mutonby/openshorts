# Autopilot (`cloud/autopilot.py`, dashboard tab "Autopilot")

Cloud-only: every new video on the user's connected YouTube channel becomes clips
on its own, and optionally the best ones are scheduled on their socials, one a
day. Paid plans only (it spends minutes unattended).

- **Channel listing** comes from Upload-Post, not scraping:
  `GET /api/uploadposts/media?platform=youtube&user=os_<id>` reads the channel's
  uploads playlist with the user's own OAuth token, so it includes videos
  uploaded straight to YouTube. There is no push, so a loop polls (tick 5 min,
  each user at most once per 55 min).
- **Jobs go through the normal pipeline**: an in-process `POST /api/process`
  (`httpx.ASGITransport`, like the MCP server) authenticated with a freshly
  issued session JWT for the user. Metering, the probe, the per-plan job limit,
  the quality gate and the download proxies apply unchanged. The per-job content
  attestation is recorded once as `rights_ack_at`.
- **Guard rails**: only videos published after `enabled_at` (switching on never
  clips the back catalogue) and at most 3 days old; 1 automatic job per user per
  24 h; `max_minutes` per video (default 30, sent as the partial-clip
  `max_minutes`); YouTube Shorts are skipped (`HEAD /shorts/<id>` answers 200 for
  a Short and 303 to `/watch` for a regular video; the SOCS consent cookie is
  required, or EU requests get a 302 to consent.youtube.com).
- **Dedupe across the deploy handover**: two containers poll at once during a
  rolling deploy, so a video is claimed by INSERTing its `autopilot_runs` row
  (unique `user_id, video_id`) before anything is submitted. A draining instance
  stops polling. A 429 (job limit) or 5xx drops the claim to retry.
- **Completion**: `run_job_wrapper` calls `on_job_finished` after the R2 archive.
  A guarded UPDATE (`status='processing'` → final) makes it once-only; it sends
  the Autopilot email (replacing the generic clips-ready one) and, with
  autopublish on, uploads the top `clips_to_publish` clips by `predicted_score`
  to Upload-Post in a background task, scheduled one a day at `publish_hour` in
  the user's IANA `timezone`. Runs stuck in processing for 6 h are marked
  `timeout`.
- `AUTOPILOT_DISABLED=1` turns the poller off without touching the API.
- Both tables are in `account.USER_OWNED_TABLES`.

## Twitch as a source (`AUTOPILOT_TWITCH=1`)

Off by default: it needs Upload-Post to serve `platform=twitch` on the media
endpoint and to offer Twitch on the connect page. With the flag on:

- `fetch_all_videos` reads every connected source (`channel_sources`: YouTube,
  then Twitch). One broken connection does not stop the other; the error is
  raised only when no source could be read.
- Twitch ids are stored as `twitch:<id>` (`normalise_media`), so they never
  collide with YouTube ids in `autopilot_runs`; the URL is the VOD link.
- A broadcast still live (`is_live`) is not claimed; a later tick takes it.
- No private/Short checks. Instead of the first `max_minutes` (usually the
  "starting soon" screen), the job clips the stretch the stream's viewers
  clipped most (`twitch_signals.best_stretch`, one minute of lead), sent as
  `source_start`/`source_end` so only that stretch is billed. Without Twitch
  credentials or viewer clips it falls back to `max_minutes` from the start.
- The connect page lists Twitch (`social_profiles.connect_platforms`). It is
  read-only: Autopilot never publishes to Twitch.
- `/auto-clip`, `/platforms/twitch`, `/gta-6-clips` and `/wow-forever-clips`
  state these rules and the viewer-clips signal below (`TWITCH_AUTOPILOT_*` in
  `dashboard/seo/pages.js`); if they change, change the copy too.

## Viewer clips as a selection signal (`twitch_signals.py`)

Any job whose URL is a Twitch VOD (Autopilot or pasted) asks Helix, with an app
token from `TWITCH_CLIENT_ID` / `TWITCH_CLIENT_SECRET`, for the clips made from
that VOD (`video_id` + `vod_offset` + `view_count`). They are shifted into the
file's own time (`SOURCE_START_SECONDS`), shown to the scoring and detail
prompts on the windows that hold them (`viewers_clipped`, `viewer_moments`),
and up to a third of the shortlist is reserved for the hottest windows
(`promote_hot_windows`). The silent-footage vision pass gets them as a hint
too. Everything is best effort: no credentials or any error means no signal.
