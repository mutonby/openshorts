# Job concurrency, failure handling and deploy handover

## Concurrency model

Async job queue with semaphore-based concurrency control, sized by
`MAX_CONCURRENT_JOBS` (default 5). Jobs auto-clean after 1 hour.

The real limit on a GPU host is VRAM, not CPU: each `main.py` job holds Parakeet
(onnxruntime CUDA), TransNetV2 (torch, ~2 GB at peak) and an NVENC session. When
the card is full a cut fails as "Generic error in an external library" / exit
187 with 0 bytes, TransNetV2 as CUDA OOM, and the reframe writer as a broken
pipe. The retry in `ffmpeg_utils.cut_clip` waits for a **busy** GPU; it cannot
help with a **full** one. Size `MAX_CONCURRENT_JOBS` against free VRAM first,
then check the CPU.

Guards that keep the card from filling:

- **Release the ASR model after transcribing.** `main.transcribe_video` calls
  `transcribe_backends.release_models()`; so do the in-process transcriptions in
  the API (thumbnail studio, `/api/subtitle` on a dubbed clip), otherwise the
  ASR singletons live in uvicorn for good. TransNetV2 `empty_cache()`s after each
  pass.
- **Parakeet runs lean** (`transcribe_backends.parakeet_providers`,
  `PARAKEET_VAD_BATCH`=4): 4 VAD segments per encoder batch, CUDA arena
  `kSameAsRequested`, no max cuDNN workspace. Same words, lower peak. Batch 2
  dropped a sentence and int8 was much slower with many words different: both
  rejected.
- **Parakeet must load the GPU build of onnxruntime.** `onnxruntime` and
  `onnxruntime-gpu` share one import package and the last wheel unpacked wins;
  with the CPU one on top, the CUDA request is only a UserWarning and every
  transcription runs on the CPU, holding its ASR slot many times longer. The
  GPU Dockerfile writes `onnxruntime-gpu` again after the lock and fails the
  build without `CUDAExecutionProvider` (`tests/test_dockerfile_onnxruntime.py`);
  `[job-cpu]` reports `asr_device` (`cuda` / `cpu`) per job.
- **Host-wide transcription slots** (`transcribe_backends.host_asr_slot`,
  `ASR_HOST_SLOTS`, default 2): flock files `output/.asr-gpu-N.lock`, shared by
  every job process and by both containers of a deploy.
- **Queue admission** (`app._wait_for_shared_gpu`): a job starts only when
  running-here + running-on-the-draining-instance < `MAX_CONCURRENT_JOBS` and
  `nvidia-smi` reports at least `GPU_MIN_FREE_MB` (4500) free; an idle card
  always starts, and the wait is bounded by the drain timeout.

CPU savings that must not be undone (output verified identical by decoded-frame
MD5 + audio):

- `parakeet_session_options()` turns onnxruntime thread spinning off for the ASR
  session only; the VAD must keep `load_vad`'s defaults (see its docstring).
- `frame_sampler.read_at` reads forward once instead of seeking per sampled
  frame in `analyze_scenes_strategy` and `split_layout` (each seek re-decoded the
  GOP).

Speed at equal quality:

- **Cut on the card** (`ffmpeg_utils.cut_clip`, NVDEC → NVENC with
  `-hwaccel_output_format cuda`) for 8-bit 4:2:0 sources; byte-identical to the
  CPU cut. Other sources and a failed GPU cut decode on the CPU.
- **Silero VAD on one CPU thread** (`vad_load_kwargs`): on CUDA it ran one 32 ms
  chunk per launch.
- **Blur at quarter size** (`ffmpeg_utils.blurred_backdrop`).
- **Watermark as a served copy** (`main.mark_delivery`, see `watermark.md`):
  after captions. It must not ride the reframe encode, which would make it
  permanent. The `wm_` copy is one more output of the ffmpeg that burns the
  captions (`add_hook_to_video(mark=)` with a hook, `burn_subtitles(mark=)`
  without); only a clip with no captions gets its own `mark_delivery` pass.
  Those multi-output passes can segfault on exit after every file is written
  (`ffmpeg_utils.finished_before_crash` keeps the files then).
- **hooked_ + subtitled_ from one ffmpeg** (`hooks.add_hook_to_video(also=)`):
  the editor still needs both files (re-caption walks back to hooked_, hook
  replace to the canonical), so nothing is skipped, only one decode.
- **Cheap cut** (`ffmpeg_utils.INTERMEDIATE`): the cut is decoded again by the
  reframe at once, so on NVENC it uses p1 at a lower `-cq` (`CUT_NVENC_PRESET`,
  `CUT_NVENC_CQ`) instead of p4 + hq + AQ. libx264 keeps its old settings.
- **Thread cap** (`bin/ffmpeg`, first on the image's `PATH`): every ffmpeg gets
  `-threads FFMPEG_THREADS` per input and the same for its filter graphs.
  Uncapped, each one sized its pools to the whole machine, so a few jobs kept
  more runnable threads than cores. Decoding and filtering are deterministic:
  the clips are byte-identical. Unset or 0 = uncapped.
- **Jobs run niced** (`JOB_NICE`, default 10, `app._lower_priority`): under CPU
  pressure the API and the editor endpoints, which run in the API process, go
  first. Containers compete by their CPU weight, not by nice, so this only
  orders work inside the instance.
- **NVENC encode slots** (`JOB_NVENC_SLOTS`, default 4, `app._job_encode_env`
  → `bin/ffmpeg`): at most that many job ffmpegs encode with NVENC at once,
  host-wide (flock files `output/.nvenc-N.lock`, shared by both containers of
  a deploy). A few encodes already keep the card's NVENC engines busy; with
  5 jobs × `CLIP_WORKERS` uncapped, dozens of sessions split the same engines
  and starved any other encoder on the card without finishing jobs sooner.
  Exempt: the nvenc probe (`NVENC_SLOTS=0` in its env), pipe-fed encoders,
  and the API process (editor renders). A queued encode runs anyway after
  `NVENC_SLOT_WAIT` (600 s). 0 turns it off.
- **NVENC on p3** (QUALITY, QUALITY_FAST, DELIVERY; was p5, then p4, oct-2026):
  same hq tune and AQs. p3 against p4 on real clips: ~25% less engine time,
  files ~13% larger, SSIM to the source within 0.001; p2 bought little more
  speed for ~21% larger files and p1 is no faster than p2.
- **`CLIP_WORKERS=6`** by default.
- **Decode on the card, keep the CPU's pixels** (`gpu_decode`): the scene
  pass, the sampled frames of the classifiers (`frame_sampler.read_at_path`),
  the trajectory, every reframe segment and the hook/caption/watermark burns
  decode with NVDEC. Frames come down as nv12 and are repacked to yuv420p
  before any other filter (`gpu_decode.DOWNLOAD`): nv12 converted straight to
  BGR differs from the CPU decode, the repacked frame does not. Only 8-bit
  4:2:0 H.264/HEVC without rotation, and only with `GPU_DECODE_MIN_FREE_MB`
  free on the card; any ffmpeg that fails on the card runs again on the CPU.
  Kill switches: `GPU_DECODE=0` (all), `GPU_DECODE_RENDER=0` (encodes only).
- **One-pass GPU render** (`gpu_render` + `osrender/`, GPU image only): the
  reframe render (`reframe_v2.encode_scenes`) runs as a single process that
  keeps the frames on the card (NVDEC -> CUDA kernels -> NVENC) instead of one
  ffmpeg per scene downloading every frame for crop/scale/blur on the CPU and
  a concat. The layers come from the scenes' own filtergraphs
  (`gpu_render.layers_from_graph`, which mirrors ffmpeg's even-coordinate
  overlay/pad/crop and sendcmd's frame timing); output matches the ffmpeg path
  at SSIM >= 0.97 per layout (`tests/test_gpu_render_equivalence.py`). It takes
  an NVENC slot like `bin/ffmpeg`. Unreadable graph, unsupported source, no
  binary or any failure: the scene-by-scene ffmpeg render, unchanged.
  `GPU_RENDER=0` turns it off. Punch-in only renders whole here: ffmpeg's crop
  ends the stream at the first size change a sendcmd makes.
- **Hooked-only picture on demand** (`hooks.materialize`, `HOOK_LAZY`): the
  job's hook + captions pass encodes only what is served (captioned, plus the
  free plan's `wm_` copy). The hooked-only `hooked_<ts>_<clean>` is needed
  only when the editor re-captions or removes the captions, so the pass keeps
  a recipe (`.hookrecipe_<name>.json` + `.png`: overlay image, position,
  window) and `app._strip_burned_captions` encodes it the first time it is
  asked for, off the event loop; `/api/hook` strips past it without encoding.
  Same filter and encoder as before, so the file is the same. No recipe or a
  failed encode: the walk-back stops as it always did. `HOOK_LAZY=0` encodes
  it in the pass again.
- **Sparse trajectory** (`reframe_v2._sparse_trajectory`, `SPARSE_TRAJECTORY`):
  the detectors look at one frame in `DETECT_STRIDE` plus scene starts, and only
  in TRACK scenes, so only those are scaled and piped (ffmpeg `select`). It
  needs the clip's frame count from the scene pass, kept only when that
  rawvideo pipe repeated or dropped no frame (`dup=`/`drop=` in its report):
  the trajectory counts the frames of a constant-frame-rate pipe, `select`
  counts decoded frames, and the two agree only then. Any mismatch restores the
  camera and runs the full pass.
- **Per-job CPU** (`JOB_CPU` marker at main.py exit, logged by the API as
  `[job-cpu]`): CPU-seconds of the job process plus every ffmpeg it waited for,
  wall time and the GPU decode counters, to compare pipeline versions per job.

Tried and dropped: `scale_cuda`/`scale_npp` for the analysis frames (their
pixels differ from swscale's, so the detectors would see other frames); yt-dlp chunking/concurrent fragments (no
gain). Method for such changes: record Gemini decisions once and replay them,
run old and new code side by side, compare clips by decoded-frame MD5 / SSIM.

## Parakeet writes some English in Cyrillic

`transcribe_backends._repair_script_drift`. Parakeet v3 picks the language per
VAD segment, so on short segments it can spell English phonetically in
Ukrainian/Russian ("Тудей Марк із гана такі сторі") while the file still reads
`en` and the whisper fallback never fires. Segments whose script cannot belong to the file's language are
re-transcribed with whisper forced to that language; any failure keeps the
parakeet text.

## Failures the user should never see (`app.run_job_wrapper`)

- **Auto-retry**: a failed job whose error text is transient (CUDA/OOM, NVENC
  "Generic error in an external library", cublas, Gemini 5xx, "No clips could be
  rendered") is re-queued once after `AUTO_RETRY_DELAY_SECONDS` (30), keeping its
  reservation and transcript checkpoint (`AUTO_RETRY_LIMIT`=1). Content failures
  (no audio, private video, no clips found, policy block) are final. Inside a
  job, `main.py` retries each failed clip once, alone, after a pause, and renders
  clips best-score first.
- **Shutdown is not failure**: when the drain timeout cancels a running job, the
  child is killed and the manifest + reservation are kept, so the next instance
  resumes it. A manifest next to a metadata file means "stopped mid-render" and
  is resumed, not recovered as completed.
- `/api/status` returns `queue: {position, ahead, eta_seconds}` while queued; the
  dashboard shows it with a "paid plans skip the line" upsell (paid plans do
  dispatch first: `PLAN_PRIORITY`).
- **Paying moves a waiting video up** (`app.reprioritize_user_jobs`, called by
  the subscription webhook through `billing.register_local_reprioritize` on
  every live upsert). The user's queued heap entries take the plan's priority
  and keep their seq (arrival order inside the new priority); their resume
  manifests are rewritten too, except the ones running here or heartbeated
  elsewhere. A webhook that lands on the other instance of a deploy can only
  rewrite the manifests: the resume scan (`_follow_manifest_priorities`) raises
  queued entries whose manifest says better, reading them only when the shared
  stamp `output/.priority_gen` (bumped by every rewrite) changed. Priorities
  only move up. Running jobs, and the cut of a partial job, are left as they
  are. Top-ups do not move anything: a pack does not change a new job's
  priority either. A paid `invoice.paid` after a lapse (past_due/unpaid ->
  active) moves jobs up and unmarks the library too.
- **Manifest scans run off the event loop.** With thousands of job dirs,
  reading every manifest is seconds of open + json.load, and the API, its
  health check and the queue share the loop: the rewrites, the follow and the
  sweeper's live set read the disk in `asyncio.to_thread` (with a copy of
  `_running_jobs`); only the heap is touched on the loop thread. Scans list
  job dirs with a manifest (`_manifest_names`), never the loose files of
  `output/`. `tests/test_queue_loop_blocking.py` holds the loop to a bound.
- **Re-enqueued from a manifest = the owner's plan now.** A job resumed at
  startup or by the resume scan, or re-queued by auto-retry, takes the better
  of its manifest's priority and its owner's current plan
  (`app._raise_to_owner_plans`, one `metering.live_plans` lookup for all the
  owners, in chunks). That covers a job that was running when its owner paid
  and was cut by a drain, and makes every deploy move up the waiting videos of
  owners who paid before the webhook path existed. A failed lookup keeps the
  manifest's priority. At startup the worker starts only after that lookup,
  so the first job it takes is already ranked by the current plan.

## Queue watch (`cloud/queue_watch.py`, Cloud only)

A saturated queue fails nothing, it only starts jobs hours late, so the
failure-rate alert stays silent. Every `QUEUE_ALERT_INTERVAL_SECONDS` (180) the
API measures its queue (`app._queue_watch_snapshot`: waiting jobs paid/free,
oldest wait of each, running vs `MAX_CONCURRENT_JOBS`, jobs finished in the last
hour, the `queue_estimate` wait for a new job) and alerts through
`alerts.send_admin_alert` (the admin Telegram chat, plus `ADMIN_EMAIL` with SMTP)
when:

- a paid job waits more than `QUEUE_ALERT_PAID_WAIT_MINUTES` (10), or
- the oldest free job waits more than `QUEUE_ALERT_FREE_WAIT_MINUTES` (30), or
- more than `QUEUE_ALERT_MAX_QUEUED` (200) jobs wait.

One message on entering, a reminder with the trend at most every
`QUEUE_ALERT_REMIND_MINUTES` (60), one "recovered" once every figure is under
half its threshold for two measurements in a row. `QUEUE_ALERTS_DISABLED=1` is
the kill switch; without `BILLING_ENABLED` or an alert channel it never starts.

With deploys:

- Only the instance that owns `output/.instance` and is not draining measures.
  The incident lives in `output/.queue_watch.json` (non-blocking flock on
  `.queue_watch.lock`), so the new container neither repeats the alert nor
  loses the recovery.
- A wait is read from disk (`app._queue_wait_started`): the `.owner` mtime
  (submission) or the manifest heartbeat, whichever is later, so a job resumed
  after a deploy or auto-retried waits from its last run, not its submission.
  A job raised to a paid priority (its owner paid) is paid from then on, told
  apart by its queue priority, and its paid wait starts at the raise
  (`raised_at` in the manifest and the record; the wait cache is keyed on
  id, attempt and paid-ness).
- While another instance still runs jobs, and for one paid threshold after
  that or after boot, nothing opens or closes: reminders only.
- If the state file cannot be written or read (disk full, read-only), the tick
  goes on from the last state kept in memory, so the incident does not re-open
  and re-alert on every tick.

## Deploys and running jobs (handover + drain)

Every push to `main` redeploys the API container as a rolling update: the NEW
container starts before the old one stops and both share `output/`, so `app.py`
coordinates them:

- Each instance writes its id to `output/.instance` at startup. An instance that
  sees another id there is the old one and **drains**: it finishes the jobs it is
  running, starts none, and leaves queued manifests on disk.
- A running job heartbeats its `.resume.json` every 10 s. The resume scan
  (startup + every 30 s) re-enqueues only manifests nobody heartbeated for 60 s,
  so no job runs twice and none is lost. Max 2 resume attempts.
- SIGTERM drains too, up to `DRAIN_TIMEOUT_SECONDS` (840), then hands the signal
  to uvicorn. **Keep it below the deployment platform's stop grace period.**
  `--timeout-graceful-shutdown 15` (Dockerfile) caps the wait for in-flight
  connections; uvicorn's default is unbounded and one open range download can
  keep a drained container alive for the whole grace period.
- `/health/ready` + the Dockerfile `HEALTHCHECK` keep the reverse proxy off a
  dying container (it only routes to `healthy` containers). An instance answers
  503 from the moment it gets SIGTERM, and a booting one gets no traffic until it
  answers. Only SIGTERM flips it, not the marker drain (at that point the new
  container is still booting). The platform health check replaces the
  Dockerfile one with a curl/wget command, so **the image must ship `curl`** or
  every deploy rolls back as unhealthy. The drain keeps serving for
  `PROXY_DRAIN_SECONDS` (20) after the jobs are done so the proxy notices the 503
  before the socket closes, and `HARD_EXIT_SECONDS` (30) after that the process
  is ended outright (an executor thread hung in a network probe would otherwise
  keep the interpreter alive). `/health` stays a plain liveness probe.
- `/api/status` answers from disk for a job this instance never held, so a poll
  landing on either container during the handover is fine.
- **Reservations of live jobs are never swept.** The metering sweeper
  (`metering._sweep_once`, every 15 min) releases `reserved` rows older than
  `STUCK_RESERVATION_HOURS` (3) only when `app._live_reservation_ids` does not
  name them: queued/running here, or any resume manifest on the shared disk
  (the other instance's running jobs, jobs left for the next instance). A
  drain-cancelled job's manifest keeps its reservation whatever the old
  instance's record says. Each live reservation carries when its job last
  moved: now for a job queued here (a long queue never counts), the start of
  the run (`run_started`, stamped at dispatch; not the heartbeat, which a hung
  child keeps beating) for a running one, the last write or heartbeat for a
  manifest nobody holds. One that has not moved for
  `LIVE_RESERVATION_MAX_HOURS` (24) is released as hung. A draining instance
  does not sweep (from there the new one's queue looks like manifests nobody
  took). If the live set cannot be read the pass is skipped. Every manifest
  write is atomic, since a half-written manifest reads as no manifest.
- `main.py` leaves `.transcript_checkpoint.json` in the job dir so a re-run job
  skips the paid transcription (download and Gemini repeat).

Every deploy is a full build plus a handover, so batch small commits (tests,
docs) with the next real change.
