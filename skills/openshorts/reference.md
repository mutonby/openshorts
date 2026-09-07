# OpenShorts HTTP reference

Base URL: `https://api.openshorts.app` (self-hosted: `http://localhost:8000`).
Every call carries `Authorization: Bearer osk_...` or `X-API-Key: osk_...`.
Self-hosted instances need no key, but do need a Gemini key, either in the
server's environment or as an `X-Gemini-Key` header.

The MCP server at `https://mcp.openshorts.app/mcp` wraps exactly these
endpoints, forwarding your auth headers, so the two paths cannot drift.

## Quota

`GET /api/me`

```json
{
  "plan": "free",
  "entitled": false,
  "minutes": { "plan_allowance": 20, "plan_used": 4, "plan_remaining": 16,
               "topup_remaining": 0, "remaining": 16 },
  "upload_post_profile": null
}
```

`upload_post_profile` is null when no social account is connected, which means
publishing will fail. On a self-hosted instance this endpoint is not mounted at
all and returns 404, which simply means there is no quota to report.

## Start a job

`POST /api/process`, JSON body for a URL source (use multipart form fields with
a `file` part to upload a local video instead):

```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "acknowledged": true,
  "layouts": ["auto"],
  "output_format": "auto",
  "target_clips": 4,
  "clip_min_seconds": 15,
  "clip_max_seconds": 60,
  "webhook_url": "https://example.com/hooks/openshorts",
  "webhook_secret": "change-me",
  "force_low_quality": false
}
```

`acknowledged: true` is the rights attestation and is required. `layouts`
accepts a JSON list or a comma separated string of `auto`, `split`,
`screencast`, `speaker_cut`, `punch_in`. `output_format` is one of `auto`,
`vertical`, `horizontal`, `square`. `target_clips` is 1 to 15;
`clip_min_seconds` 5 to 175; `clip_max_seconds` 10 to 180 and at least 5 above
the minimum.

Success:

```json
{ "job_id": "0f0a...", "status": "queued" }
```

Quality gate, also HTTP 200:

```json
{ "needs_confirmation": true,
  "quality_check": { "max_height": 480, "min_height": 720, "cookies_invalid": false } }
```

Ask the user, then resubmit the identical body with `force_low_quality: true`.

Errors: **402** `{"detail": {"error": "quota_exceeded", "minutes_required": 12,
"minutes_remaining": 3}}`, **429** too many jobs already running, **400** the
source duration could not be determined.

## Poll a job

`GET /api/status/{job_id}` returns `{"status": ..., "logs": [...], "result": ...}`
where status is `queued`, `processing`, `completed` or `failed`. `result.clips`
appears once completed, each entry carrying `title`,
`video_title_for_youtube_short`, `video_description_for_tiktok`,
`video_description_for_instagram`, `start`, `end` and `video_url`. A relative
`video_url` is relative to the API base.

Poll every 30 to 60 seconds. Prefer a webhook.

## Webhook

Fired once per job, on success and on failure, so a flow never hangs:

```json
{
  "event": "job.completed",
  "job_id": "0f0a...",
  "status": "completed",
  "clips": [ { "index": 0, "title": "...", "video_url": "...", "download_url": "..." } ]
}
```

A failed job sends `"event": "job.failed"` with an `error` string and no clips.
`download_url` is a presigned link valid for 24 hours (hosted service only).

The URL must be public HTTPS; it is validated at submit time and re-resolved at
delivery time, so a URL that points at a private address is refused. With a
`webhook_secret`, the **raw request body** is signed HMAC-SHA256 and sent as
`X-OpenShorts-Signature: sha256=<hex>`. Verify against the raw bytes, not
against a re-serialized object:

```javascript
const crypto = require('crypto');
const expected = 'sha256=' + crypto
  .createHmac('sha256', secret)
  .update(rawBody)
  .digest('hex');
```

## Restyle captions

`POST /api/subtitle`

```json
{ "job_id": "...", "clip_index": 0, "preset": "hormozi" }
```

`preset` is a ready-made look (font, colours, outline, effect, case): `hormozi`,
`mrbeast`, `bounce`, `karaoke`, `tiktok`, `reels`, `shorts`, `gold`, `neon`,
`cyber`, `minimal`, `boxed` or `classic` (`GET /api/config` lists them under
`captionPresets`). Any explicit field overrides the preset:

```json
{ "job_id": "...", "clip_index": 0, "style": "karaoke", "position": "bottom",
  "font_size": 48, "font_name": "...", "font_color": "#FFFFFF",
  "highlight_color": "#FFE600", "uppercase": true }
```

`style` is `classic` or `karaoke`; `position` is `top`, `middle` or `bottom`.
Only `job_id` and `clip_index` are required. To burn a preset on every clip of
a job instead, pass `caption_preset` to `POST /api/process`.

## Re-cut a clip

`POST /api/clip/rerender`

```json
{ "job_id": "...", "clip_index": 0,
  "segments": [ { "start": 812.4, "end": 831.0 }, { "start": 840.2, "end": 858.9 } ],
  "snap_to_words": true, "reapply_captions": true, "framing": "auto" }
```

Times are seconds in the **original source video**. Segments concatenate in the
order given. `framing` is `auto`, `full` (no side cropping) or `track` (force
subject tracking); omit it to keep the clip's current framing. Anything outside
the clip's original range, or any non-auto `framing`, re-runs the reframe engine
and needs the retained source video.

## Publish

`POST /api/social/post`

```json
{ "job_id": "...", "clip_index": 0, "platforms": ["tiktok", "instagram", "youtube"],
  "title": "...", "description": "...",
  "scheduled_date": "2026-08-25T18:00:00Z", "timezone": "Europe/Madrid" }
```

Omit `scheduled_date` to post now. Uploads are asynchronous. TikTok arrives as a
draft in the app; Instagram and YouTube publish directly.

Queue: `GET /api/social/scheduled`, and
`DELETE /api/social/scheduled/{job_id}` to cancel one before it goes out.

Analytics of what was published: `GET /api/social/analytics` (profile totals),
`GET /api/social/analytics/posts` (per post), and
`GET /api/social/analytics/impressions` (windowed, e.g. `?period=last_week`).

Full generated API docs: <https://api.openshorts.app/docs>.
