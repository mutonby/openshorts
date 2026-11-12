# Remote CPU worker

A second machine without a GPU can render part of the free queue. It is off
unless the API has `REMOTE_WORKER_TOKEN` and `REMOTE_WORKER_SLOTS` > 0.

## Shape

`worker.py` runs on the other host, from the same image built for the CPU
(`GPU=0`, the default). It only makes outbound HTTPS calls to the API:

1. `POST /internal/worker/claim`: the API hands it the next eligible queued
   job (`app._claim_remote_job`) and a claim id.
2. The API fetches the source itself, `main.py <the job's command>
   --prepare-only`: download and the cut to the paid minutes, then
   `SOURCE_READY <path>`. The worker has no proxy and no cookie, and a
   datacenter address that never downloaded before is bot-checked at once.
3. The worker polls `GET /internal/worker/jobs/{id}` until `ready`, fetches the
   source (`.../source`), and runs `main.py -i <source> -o output/<id>` with
   the job's options, at the same relative paths the API host uses.
4. Its output lines go to `.../log` every few seconds (also its heartbeat),
   each output file to `.../file` in chunks (a clip and the metadata as soon
   as `CLIP_READY` is printed, the rest at the end), the exit code to
   `.../done`.
5. The API follows those files (`app._run_job_remote`) as it follows a local
   subprocess, then delivers the job like any other: settle, R2 archive,
   watermark copy, webhook, emails.

Everything the worker sends lands as files in the job directory
(`remote_worker.py`: `.remote.json`, `.remote.log`, `.remote.seen`,
`.remote.done`), never in the memory of the instance that answered. During a
deploy handover either container can answer the worker while the one that
dispatched the job keeps following it.

## Which jobs go

`app._remote_eligible`: a free job (normal or slow lane) that never failed
here, whose command the worker can repeat (no precomputed transcript, not a
Twitch VOD) and that runs on the server's own model key. And only while a job
is already waiting for a GPU slot: with room on the card the job runs here,
where it is faster. Paid jobs never leave the GPU host.

A remote job is in `_running_jobs` (a drain waits for it, its manifest is
heartbeated, expiry and sweeps see it as live) and in `_remote_jobs`, so it
holds none of the `MAX_CONCURRENT_JOBS` slots: `_local_running()` is what the
GPU admission counts, and `_jobs_busy_elsewhere` skips a job with a claim.

## When it goes wrong

- The worker goes quiet for `REMOTE_LEASE_SECONDS` (180), takes longer than
  `REMOTE_MAX_SECONDS`, reports a failure of its own, or its claim disappears: the job fails with a transient error, so
  the auto-retry runs it on this host. A retried job is never remote again
  (a job resumed after a deploy starts over and can be).
- The download fails: the job fails as a local one would, with the same logs.
- The instance that dispatched the job is stopped before the worker ends: the
  manifest stays, the next instance resumes the job locally and
  `_register_resumed` clears the claim. The worker's next call gets `410` and
  it drops the job.

## What the worker can and cannot do

The worker host is treated as less trusted than the API host: it parses
user-supplied video. A worker that is no longer ours must not be a way into
the API host.

- It holds its own `GEMINI_API_KEY` and the worker token. No database,
  storage, mail, billing, proxy or cookie credential ever reaches it. Job
  settings cross by name from a closed list (`remote_worker.allowed_option`:
  `WATERMARK`, `BLUR_SENSITIVE`, the source caps, `AUTO_*`, `CLIP_*`), checked on both sides, and
  the command is `--format <choice>` or nothing (`worker_args`, `safe_args`).
- It can read one thing: the source of a job it holds a live claim for.
- It can write only render outputs into that job's directory
  (`remote_worker.safe_relpath`): one plain file name with a known extension,
  no subdirectory, no link, **no dotfile**, and never a file that was there
  before it (the source). The dotfiles of a job directory are what the API
  trusts (owner, manifest, the free-plan mark, encode recipes), so they are
  written on the API host only: the API sets `.marked` for a free job before
  the worker is given it, and remote jobs run with `HOOK_LAZY=0` so no recipe
  is needed.
- One upload request per job at a time (`upload_lock`, a file lock both
  containers of a deploy share). Files land atomically, and the claim and the
  budget are checked again at that moment on what is on disk:
  `REMOTE_MAX_FILE_MB` per video file, 8 MB for anything the API parses
  (JSON, subtitles), `REMOTE_MAX_JOB_MB` in all, a few hundred files, 2 MB of
  log.
- A `*_metadata.json` is rewritten as data when it lands
  (`remote_worker.clean_metadata`): no `video_url` (the API decides where a
  clip is served from), bounded lists, text fields as text.
- A remote run that does not complete leaves none of its files and none of
  its log lines, and is always run again on the API host, whatever the worker
  printed: the API host decides what a video's failure is.
- The claim ends with the job, also when the job completes while the
  instance is being stopped for a deploy.
- `REMOTE_MAX_SECONDS` bounds a render, and `REMOTE_BREAKER_FAILURES` of the
  last ten remote jobs ending badly stop the claims for
  `REMOTE_BREAKER_SECONDS`.
- Every refusal before a valid token is a `404` (a wrong HTTP method answers
  `405` like on any route); a claim that is no longer the job's is a `410`. A
  token shorter than 32 characters leaves the feature off.

What stays true whatever the code does: the worker sees the source videos of
the jobs it renders, and it decides what those jobs' clips look like. So a
worker that is not ours any more can hand a user wrong clips (and that job's
minutes are spent on them), and can send a free job's `wm_` copy without the
mark in it: the API checks names, sizes and shapes, not pictures. Anything
the API reads back from a job directory (metadata, layout ranges, subtitles)
is data from another machine: parse it as numbers and plain text, never paste
it into a filter graph, a path or markup.

Run it locked down (`docker-compose.worker.yml`): no published port,
non-root, no capabilities, read-only root filesystem, CPU, memory and pid
limits. The compose file gives it its own bridge network; the rules that keep
that network away from private addresses (the host, other Docker networks, a
VPN) are host firewall rules and must be added on the host. A VM around it
adds a kernel boundary when other services share the host. The model cache
(`/app/.cache`) is a writable volume the models are loaded from: a job that
broke out of the pipeline could change it for the jobs after it, on the
worker only.

## Settings

API: `REMOTE_WORKER_TOKEN` (32+ random characters), `REMOTE_WORKER_SLOTS` (jobs
it may hold at once), `REMOTE_LEASE_SECONDS`, `REMOTE_MAX_SECONDS`,
`REMOTE_MAX_FILE_MB`, `REMOTE_MAX_JOB_MB`, `REMOTE_BREAKER_FAILURES`,
`REMOTE_BREAKER_SECONDS`.

Worker: `WORKER_API_URL` (https), `WORKER_TOKEN`, `WORKER_SLOTS`,
`GEMINI_API_KEY`, plus the pipeline defaults the API host sets for itself
(`AUTO_LAYOUT`, `TRANSCRIBE_BACKEND`, `FFMPEG_THREADS`, `CLIP_WORKERS`...): only
per-job options travel with a job.

On a CPU each job costs several times the CPU of a GPU job. Size
`WORKER_SLOTS` by cores (a job keeps about eight busy), and keep
`REMOTE_WORKER_SLOTS` equal to it.
