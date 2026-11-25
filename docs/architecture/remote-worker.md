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
   datacenter address that never downloaded before is bot-checked at once
   (unless its operator gave it accounts: "A worker that fetches the link
   itself" below).
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

## A worker with a GPU of its own

The same `worker.py` runs from the GPU image (`--build-arg GPU=1`) on a host
with a card, with the container given the GPU: it then transcribes, finds the
scenes and encodes on that card, like the API host does. Set
`WORKER_SOURCE_ONLY=1` on it. Its claim says so (`source_only`, kept in the
claim file), and the API host's `--prepare-only` run for that job downloads
and cuts the source and nothing more: no transcription and no scene pass
(`remote_worker.source_only_env`). Preparing them there made such a worker
wait for a host that is busy by definition (a job only goes out while another
waits for a slot), while its own card sat idle. A transcript the download
cache already holds for the source still travels: it costs the API host
nothing. The worker is no more trusted for it: it still holds no download
credential and gets the source from the API host.

## A worker that fetches the link itself

With every source downloaded on the API host, that host's cores, its link and
its video-site accounts carry every job, and a worker waits for the file
longer than it takes to render it. A worker whose operator gave it accounts
of its own (`YOUTUBE_COOKIES`, `YOUTUBE_COOKIES_2`.. in the worker's
environment, used through `yt_account` exactly as on the API host, from the
worker's own address) sets `WORKER_DOWNLOADS=1`:

- Its claim says so (`downloads`, kept in the claim file). For a link of a
  site on the closed list (`remote_worker.safe_link`, checked again by the
  worker) the API host runs no `--prepare-only`: the claim goes straight to
  `ready` with the link (`set_ready_link`, `app._worker_link`). A whole copy
  already in the API host's download cache is still sent the usual way: it
  costs nobody's account a request. A cut is never served from that cache,
  so the worker fetches it.
- The worker runs `main.py -u <link> -o output/<id> --keep-original`. Where
  the user's stretch starts (`SOURCE_START_SECONDS`) travels only in this
  mode: a worker that is sent the source gets it already cut.
- No credential crosses: the accounts are the operator's, set on that
  machine, and a name like `YOUTUBE_COOKIES` in a job's options is dropped
  like any other name off the list.
- The source it fetched goes back with the clips (an ordinary `.mp4` of the
  job directory, within the per-file limit), because the editor on the API
  host works from it. One over the limit is not sent: the job completes and
  its editor has no source.
- A link the worker cannot fetch (its accounts are limited, the video needs
  another route) ends as `done` with `no_source`: the job runs again on the
  API host with its own routes, and the breaker is not told. The worker
  knows the source arrived by main.py's own `PROXY_ROUTE=` line.
- It fetches at most `WORKER_DOWNLOAD_SLOTS` (2) links at the same moment.
  A fetch per slot from one address, on a few accounts, had the media
  refused for a large share of them, and each refused link is a job that
  reaches the API host minutes late. A claim made while that many of its
  jobs are still fetching does not say `downloads`, so that job is sent its
  source the usual way: the offer is made per claim (`worker.claim_body`),
  and the number can grow while the hand-backs stay rare. A job that
  offered and is not ready within seconds was not given as a link (the API
  host had a copy): it stops counting as a fetch while it waits.
- It stops offering for `WORKER_DOWNLOAD_PAUSE_SECONDS` (900) once
  `WORKER_DOWNLOAD_STRIKES` (4) of its last eight fetches could not get the
  source, and while every account of its own rests (`yt_account.all_resting`,
  the worker's own state files). Meanwhile it works as a worker that never
  downloads.
- A job whose video is already in its directory on the API host (a job
  resumed after a deploy) is not given as a link: the worker never replaces a
  file that was there before it, so its own copy could not come back.

One account lives on one host: its pacing and its rest are files of that
host, so an account configured on the worker is not also configured on the
API host.

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
- The worker is stopped or restarted (SIGTERM): it hands its running jobs back
  with `stopping: true` (also the ones whose source this host is still
  preparing, and the ones in the middle of the source transfer; the compose
  file gives it `stop_grace_period` to do so), they are retried here like any
  remote failure, and they do not count towards the breaker (`REMOTE_BREAKER_FAILURES` of the last
  10 jobs failing parks the worker for `REMOTE_BREAKER_SECONDS`, 30 min).
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
  `WATERMARK`, the source caps, `AUTO_*`, `CLIP_*`), checked on both sides, and
  the command is `--format <choice>` or nothing (`worker_args`, `safe_args`).
- It can read three things, all of a job it holds a live claim for: the
  source and, when this host prepared them, the source's transcript
  (`.../transcript`) and scene probabilities (`.../scenes`, see below). It can
  never write either back: they are dotfiles.
- It can write only render outputs into that job's directory
  (`remote_worker.safe_relpath`): one plain file name with a known extension,
  no subdirectory, no link, **no dotfile**, and never a file that was there
  before it (the source). The dotfiles of a job directory are what the API
  trusts (owner, manifest, the free-plan mark, encode recipes), so they are
  written on the API host only: the API sets `.marked` for a free job before
  the worker is given it. A worker that says it sends hook recipes
  (`hook_recipes` in its claim) runs with `HOOK_LAZY=1` like this host and
  sends each recipe under its plain name (`hookrecipe_<hooked>.json` + `.png`,
  `worker.Uploader.publish_recipes`); the API reads the JSON as numbers and a
  plain clip name, checks the image is a PNG and writes the dotfiles itself
  (`remote_worker.adopt_recipe`). An older worker runs with `HOOK_LAZY=0`.
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
`REMOTE_BREAKER_SECONDS`, `REMOTE_QUIET_MINUTES` (minutes of the hour, UTC,
like `55-5`, with no new claim: when the worker's host shares its link with
something that peaks at fixed times), `REMOTE_QUIET_WORKERS` (the worker names
those minutes are for, comma separated; unset: every worker), `REMOTE_STANDBY_WORKERS`
(worker names kept in reserve: one of them is handed a job only while a job it
could take has waited `REMOTE_STANDBY_AFTER_SECONDS`, 1800, so a worker on a
shared host idles until the queue backs up), `REMOTE_SOURCE_MBPS` (cap for all source
transfers together, 0 for none), `PREPARE_TRANSCRIPT` (`1`, `cache` or unset:
whether the source's transcript is prepared for the worker), `PREPARE_SCENES`
(`1` or unset: whether the source's scene probabilities are), `SCENE_HOST_SLOTS` (2),
`PREPARE_SCENES_WAIT` (30), `PREPARE_SCENES_MAX_MINUTES` (45).

The source download is the worker's one long call: the API touches the lease
while it streams the file (`_source_chunks`), so a slow transfer is not taken
for a dead worker. Before that, a ~1 GB source from a busy host outlived
`REMOTE_LEASE_SECONDS`, the job came back mid-transfer and four of those
tripped the breaker, leaving the worker idle exactly when the API host was
short of CPU.

Worker: `WORKER_API_URL` (https), `WORKER_TOKEN`, `WORKER_SLOTS`,
`GEMINI_API_KEY`, `WORKER_SOURCE_ONLY` (`1` on a worker with a GPU: the source
alone, see above), `WORKER_DOWNLOADS` (`1` with `YOUTUBE_COOKIES`.. of its
own: it fetches links itself, `WORKER_DOWNLOAD_SLOTS` at a time), plus the pipeline defaults the API host sets for itself
(`AUTO_LAYOUT`, `TRANSCRIBE_BACKEND`, `FFMPEG_THREADS`, `CLIP_WORKERS`...): only
per-job options travel with a job.

On a CPU each job costs several times the CPU of a GPU job. Size
`WORKER_SLOTS` by cores (a job keeps about eight busy), and keep
`REMOTE_WORKER_SLOTS` equal to it.

What limits a CPU worker is rarely its slots:

- **Transcription.** On a CPU the transcript is the largest single cost of a
  job and about half of its wall time. With `PREPARE_TRANSCRIPT` set on the
  API host, `main.py --prepare-only` leaves the transcript in the job
  directory as main.py's own checkpoint (`.transcript_checkpoint.json`):
  `cache` takes only what the download cache holds for that very source, `1`
  otherwise transcribes there, on the card, inside the same `ASR_HOST_SLOTS`
  as any job, and only if a slot frees up within `PREPARE_TRANSCRIPT_WAIT`
  seconds (45): when the card's own jobs keep them busy the worker
  transcribes for itself, so neither host waits for the other. It
  transcribes on the card or not at all (`transcribe_backends.require_gpu`):
  a CUDA failure leaves the transcript to the worker instead of falling back
  to CPU whisper on the API host. The claim then announces `transcript_size`, the worker fetches
  it (`GET .../transcript`, re-serialised as data and shape-checked on both
  sides, `remote_worker.valid_transcript`), puts it at the same place in its
  own job directory and its main.py reuses it exactly like a resumed run.
  Anything wrong with it and the worker transcribes as before. A remote run
  that fails leaves the checkpoint for the retry on the API host; a completed
  one removes it. Watch `asr_wait_s` in the API host's `[job-cpu]` lines: the
  prepared transcripts share its transcription turns.
- **Scene detection.** Once the transcript comes prepared, TransNetV2 is
  about half of what is left: it runs on every clip, on the CPU (profiled on a
  worker, 19-nov-2026). With `PREPARE_SCENES=1` on the API host, `main.py
  --prepare-only` runs it once over the whole source on the card, inside one
  of `SCENE_HOST_SLOTS` (2) host slots of its own, never the transcription
  ones (sharing them made the transcripts wait behind the scenes)
  (`PREPARE_SCENES_WAIT` seconds at most, and not for sources over
  `PREPARE_SCENES_MAX_MINUTES`), and leaves the per-frame
  probabilities in `.scene_probs.json` (float32, base64, tied to the source by
  name and duration). The claim announces `scenes_size`, the worker fetches
  them (`GET .../scenes`, shape-checked on both sides,
  `remote_worker.valid_scenes`) and main.py maps them onto each clip by time
  from the clip's cut start (`scene_detection.use_prepared`,
  `register_clip`): the clip is still decoded small for its frame count, the
  network does not run. Measured over 24 clips of 8 corpus sources: every cut
  found clip by clip is found within 1.5 frames. Anything wrong with them and
  each clip runs the network as before; they are removed with the transcript.
- **Transcription turns.** `ASR_HOST_SLOTS` (default 2) is sized for a GPU's
  memory. On a CPU host a transcription takes minutes, so with 2 turns most
  slots sit waiting for one (the job's `asr_wait_s`). Raise it on a CPU host
  with cores to spare, and cap the math libraries' threads per process
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) so parallel
  transcriptions do not each start one thread per core.
- **Process limit.** Each job runs about two hundred threads; a container
  `pids_limit` below slots x 250 makes ffmpeg fail to start threads and a job
  hang or die.
- **Encoder effort.** Most of a CPU job is libx264. `X264_PRESET` (for example
  `veryfast`) replaces every tier's preset at the same CRF: less CPU per clip
  for a somewhat larger file. Leave it unset on a GPU host.
