# Free-plan watermark: a served copy that paying removes (`watermarked.py`)

Every file the pipeline and the clip editor write is clean. On the free plan the
file that is *served* is `wm_<final>`, a copy of the final file with the mark
burned in (`main.mark_delivery`, the last step after captions; in the job the copy
is one more output of the ffmpeg that burns the captions, `auto_hook_clip(mark=)` /
`auto_caption_clip(mark=)`, and `mark_delivery` only finds it made; the editor endpoints get it from `app._deliver`, decided from the
caller's plan right now, not from the job's flag, which a restart loses). Both
twins live on disk and in R2 (`cloud.videos._twins_to_archive`). The mark must
never be burned into the canonical reframe: every derivative would inherit it
and "upgrade to remove the watermark" would be false for clips already made.

- **Upgrade**: `cloud/billing._upsert_subscription` fires
  `videos.unmark_user_library` on the free→live transition (`became_live`), and
  so does a paid `invoice.paid` after a lapse (past_due/unpaid → active). Per
  clip it checks the clean twin exists in R2, moves the history row and the
  project state to it, deletes the marked object and calls back into
  `app._unmark_local_job` so an open dashboard polls the clean URL. No render,
  no source video. The dashboard chases the job result after the plan flips
  (`App.jsx`, the `wm_` regex) and refreshes the durable map.
- **Paid while it waited**: a job submitted on the free plan that finishes
  after its owner paid (queued for hours, or running when the webhook came)
  still renders the `wm_` copy, but at delivery `app._unmark_if_owner_paid`
  asks for the owner's live plan and runs the same re-pointing for that job
  only (`unmark_user_library(user_id, job_id=)`), after the archive and before
  the caller's webhook and the clips-ready email. A failed lookup or a twin
  missing from R2 leaves the job marked.
- **Walk-backs**: `wm_` is the outermost layer; `_strip_burned_captions` /
  `_strip_burned_hook` drop it first, `_canonical_clip_file` globs
  `wm_*{clean}`.
- **`/videos` guard** (`app._media_guard`): a job rendered for the free plan
  carries a `.marked` dotfile (`watermarked.MARKER_FILE`, written by `main.py`,
  restored with the project, removed on unmark; `_deliver` marks only jobs that
  carry it, so a legacy job whose canonical already has the mark never gets a
  second one). While it exists the clean deliverables of that job are refused,
  so the twin cannot be fetched by stripping the prefix off a URL. The editor's
  `temp_*` scratch files and the caption sidecars stay servable.
- **Notice**: `WatermarkModal` shows once per job when the clips land on a free
  account (`source=results`, 2.5 s after the grid) and, if skipped, before the
  first download; tracked as `WatermarkNoticeSeen` / `WatermarkNoticeUpgrade`
  with `source`. Its Upgrade button carries the Starter price ("from $12/mo")
  and opens the upsell `TopUpModal` with starter put forward; checkouts from it
  carry `CheckoutStarted` `source=watermark`. From a download it downloads the
  clip first. Going straight to the Starter checkout instead (5-7 Oct 2026)
  left far more of those checkouts unpaid.
