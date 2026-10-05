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
  one NVENC pass per free clip, after captions. It must not ride the reframe
  encode, which would make it permanent.
- **hooked_ + subtitled_ from one ffmpeg** (`hooks.add_hook_to_video(also=)`):
  the editor still needs both files (re-caption walks back to hooked_, hook
  replace to the canonical), so nothing is skipped, only one decode.
- **`CLIP_WORKERS=6`** by default.

Tried and dropped: NVDEC for the analysis decodes and `scale_cuda` (slower, and
scale_cuda does not match swscale); yt-dlp chunking/concurrent fragments (no
gain). Method for such changes: record Gemini decisions once and replay them,
run old and new code side by side, compare clips by decoded-frame MD5 / SSIM.

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
- `main.py` leaves `.transcript_checkpoint.json` in the job dir so a re-run job
  skips the paid transcription (download and Gemini repeat).

Every deploy is a full build plus a handover, so batch small commits (tests,
docs) with the next real change.
