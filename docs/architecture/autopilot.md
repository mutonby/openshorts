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
