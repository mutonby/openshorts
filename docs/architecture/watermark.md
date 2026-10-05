# Free-plan watermark: a served copy that paying removes (`watermarked.py`)

Every file the pipeline and the clip editor write is clean. On the free plan the
file that is *served* is `wm_<final>`, a copy of the final file with the mark
burned in (`main.mark_delivery`, the last step after captions, one NVENC pass
per free clip; the editor endpoints get it from `app._deliver`, decided from the
caller's plan right now, not from the job's flag, which a restart loses). Both
twins live on disk and in R2 (`cloud.videos._twins_to_archive`). The mark must
never be burned into the canonical reframe: every derivative would inherit it
and "upgrade to remove the watermark" would be false for clips already made.

- **Upgrade**: `cloud/billing._upsert_subscription` fires
  `videos.unmark_user_library` on the free→live transition (`became_live`). Per
  clip it checks the clean twin exists in R2, moves the history row and the
  project state to it, deletes the marked object and calls back into
  `app._unmark_local_job` so an open dashboard polls the clean URL. No render,
  no source video. The dashboard chases the job result after the plan flips
  (`App.jsx`, the `wm_` regex) and refreshes the durable map.
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
  with `source`. Its Upgrade opens the upsell `TopUpModal`.
