# ROADMAP

Two sections:

1. **Product roadmap** — user-facing feature backlog. Honest about what's shipped vs stubbed.
2. **Technical roadmap** — designs for the three larger backend features (motion graphics, soundtracks, layouts) and the deferred restructure work.

---

## Product roadmap

Tiering:

- **Shipped** — verified end-to-end in a browser smoke test on `chore/restructure-and-docs`.
- **Stubbed in v1** — UI is in place; the backend feature is a no-op, placeholder, or partial loop. Each item lists the backend TODO that unblocks it.
- **Later** — not started.

### Short-form wizard

**Shipped**
- 4-step wizard (Upload → Categorize → Processing → Review)
- Batch upload up to 5 clips, MP4 / MOV ≤ 2 GB
- Server-side MP4/MOV signature validation (extension + `ftyp` magic bytes at byte offset 4)
- Wizard auto-resets to Upload on rehydrate when File handle is lost (no more stranded Categorize/Processing state)
- Real-time per-clip progress polling with abort-on-unmount + terminal-status guard against stale responses
- Mini Snake game during processing
- Split-view review: clip list + phone preview + Before/After toggle
- Download / Publish / Schedule buttons (UI; backend gaps below)
- Processing history tab
- Info tooltips on every major control

**Stubbed in v1**
- "AI auto-categorization per clip" — defaults are pre-selected; real classification is `POST /api/categorize` (backend TODO #2)
- "Auto color grading" — toggle exists; LUT application is backend TODO #5
- "Silence and dead-air removal" — toggle exists; `silencedetect` integration is backend TODO #4
- "Face-focus layout" — toggle exists; per-category pipeline branches are backend TODO #3
- "Batch endpoint" — wizard fires one `POST /api/process` per file in parallel; real `POST /api/process/batch` is backend TODO #1
- "Send to CapCut export format" — UI-only placeholder
- "Publish directly to connected platform" — `/api/social/post` is synchronous today; bell notification can't advance past `submitted` until backend TODO #9 lands (`publish_jobs` queue + `GET /api/social/publish/status/{publish_id}`, mirroring the thumbnail flow at `backend/app/main.py:1565-1620`)
- "Schedule upload for a specific date/time" — same as Publish; UI fires the notification but no scheduler runs the eventual publish

**Later**
- Auto layout detection (screen-share → 16:9 + face-cam below; face-only → full 9:16)
- Face-cam position configuration (corners)
- True per-clip settings cards (today the auto-edit toggles apply to the whole batch)
- Re-upload an externally edited CapCut export
- AI learns talking-head style from past content
- Confidence score per AI edit decision
- Auto-detect / highlight viral hook moments within a clip
- B-roll auto-insertion suggestions
- Auto zoom / punch-in on face at key moments
- Background music auto-suggestion + fade
- Auto intro / outro from brand-kit templates
- Clip re-ordering within a batch before export
- Duplicate clip with different settings (A/B export)
- Platform-specific export presets (TikTok 9:16, Reels, Shorts, Snap)
- Per-platform subtitle style overrides
- Auto-generate social-media caption text (separate from burn-in subtitles)
- Thumbnail auto-generation per clip
- Viral score prediction per clip
- Compare two clips side by side in review
- Bulk download / bulk schedule for a batch
- Favorites / pins / tags / folders
- Central media library across all content types
- Filter library by platform / type / date / campaign

### Long-form wizard

**Shipped**
- 4-step wizard (Upload → Settings → Processing → Editor)
- Single MP4 / MOV upload ≤ 8 GB (4K supported)
- Simulated 5-stage processing progress (StrictMode-safe: timer survives the dev-mode double-mount)
- Editor: video preview + chapter timeline scrubber + Chapters / Subtitles / Export tabs
- Inline chapter title rename
- "Export segment as short" modal (UI wired; backend route still TODO)
- Mini Snake game during processing
- Wizard auto-resets to Upload on rehydrate when File handle is lost
- Processing history tab
- Info tooltips

**Stubbed in v1**
- "AI auto-chapter detection" — Editor seeds 3 placeholder chapters (Intro / Main / Outro); real PySceneDetect-driven chapters are backend TODO #6
- "Auto-generate YouTube description, tags, chapter timestamps" — toggle exists; no backend code wired (TODO #6+)
- "Subtitle panel: click any line to edit" — read-only panel today; edit → re-render pipeline isn't wired
- "Color grade" / "Intro / outro" toggles — backend TODOs #5 / #8
- "Export segment as short" backend route — `POST /api/long-form/export-segment` is backend TODO #7

**Later**
- Jump cut + filler-word removal (um, uh, like)
- Auto intro / outro from brand kit
- Separate LUT for long-form
- Color-grade matching with short-form
- AI show notes / blog post from transcript
- Multi-track audio editing (voice vs background)
- B-roll markers + suggestions on the timeline
- AI highlight-reel generator (best 2–3 min auto-extracted)
- Transcript-driven cuts (edit words → edits video)
- Drag-and-drop chapter reordering
- Auto thumbnail with face detection (best-frame picker)
- Engagement heat-map overlay (predicted drop-off)
- Direct YouTube Studio integration (push title, description, thumbnail, chapters in one click)
- Long-form → multi-shorts auto-pipeline (5 shorts from one source)
- Speaker labelling (multi-person)
- Subtitle export as `.srt` (separate file alongside burn-in)

### Clip Generator (original `/api/process` flow)

**Shipped**
- Upload a long-form video file OR paste a YouTube URL
- AI extracts top viral moments (Gemini 2.5 Flash, 3–15 clips per video)
- Preview + download extracted clips
- Subtitle, Hook, Translate, Render modals on each result card
- Subprocess command fixed (`python -u -m app.cli`; the pre-restructure `python -u main.py` was broken on this branch and surfaced by the smoke test)

**Later**
- Batch-extract from multiple YouTube URLs at once
- Filter extracted clips by length (15s / 30s / 60s)
- Inline trim / subtitle edit before download
- Auto-rank clips by predicted virality
- Push extracted clips directly into the short-form wizard

### Dashboard

**Shipped**
- 3 StatCards (clips processed / scheduled / published) — counters derive from history + notifications stores
- "Upcoming uploads" panel (filters the notification feed)
- "Recent activity" panel

**Stubbed in v1**
- Live backend feed for the StatCards — today everything derives from localStorage; the real source is `GET /api/clips/recent?limit=20` (backend TODO #10)

**Later**
- Per-platform analytics (views, watch time, follower growth, engagement rate)
- "Best-performing clip of the week" panel
- Posting-consistency / streak tracker
- Recommended posting times per platform
- Quick-upload shortcut from the dashboard
- Revenue / monetization tracker (YouTube, TikTok creator fund)

### Settings

**Shipped**
- Brand Kit editor — colors, font, per-aspect layout, 3×3 text-position grid, live preview cycling
- Brand-kit font upload (`assets/fonts/user/`, volume-mounted, persisted across restarts)
- API Keys page (Gemini / Upload-Post / ElevenLabs / fal.ai)
- VS-Code 180 px section nav (General / Platforms / System)

**Stubbed in v1** (placeholder section pages render but expose no editable controls yet)
- Subtitle style (separate from Brand Kit)
- Color presets
- Export defaults
- Per-platform settings (YouTube / TikTok / Instagram / Snapchat / Facebook)
- Processing history

**Later**
- Multiple brand-kit profiles (per channel / brand)
- Team / multi-user with role permissions
- Template system (save full settings as a named template, apply in one click)
- Light / dark mode toggle
- White-label custom domain
- Webhook + Zapier / Make integrations
- Storage / usage stats (storage used, clips processed this month, API calls)
- Email + push notifications for processing-complete / upload-confirmed events

### Notifications system

**Shipped**
- Bell icon in the header with unread dot
- `pushNotification(...)` from publish / schedule actions in `ResultCard.jsx` + `ScheduleWeekModal.jsx`
- Dropdown with mark-all-read + clear
- Codex audit cleared the path of DOM/XSS concerns: untrusted text renders as React text nodes, not HTML

**Stubbed in v1**
- The bell terminates at `submitted` / `scheduled` because publish is synchronous. The async upgrade is backend TODO #9 (see Short-form Publish above). Once it lands, the bell can advance items through `submitted → published` and `submitted → error`.

**Later**
- Processing-complete notification
- Scheduling reminder (upcoming scheduled post)
- Platform error alerts (failed upload)
- Browser / mobile push
- Email

---

## Follow-ups from the smoke-test pass

Captured during the post-Phase-4 manual browser smoke test on
`chore/restructure-and-docs` (commit `43c2d96`). Each item maps to an
exact file:line so the next agent doesn't have to re-find it.

### Backend security baseline for `POST /api/process` (STATE-MUTATING)

The smoke-test commit landed C3 (input validation) and re-confirmed C8
(concurrency lock via the job-queue semaphore) and C9 (attestation log).
Three controls remain opted-out per the `security_baseline:` block —
a `/gsd-secure-phase` pass should land them:

- **C2 — Rate limit.** Per-IP and per-key caps. `MAX_CONCURRENT_JOBS` is process-wide, not per-caller.
- **C4 — Timeout / breaker.** `run_job` spawns a Python subprocess with no timeout. A 15-min hard cap + breaker on repeated subprocess crashes (e.g. yt-dlp 403s) would prevent zombie jobs.
- **C7 — Idempotency.** Accept `Idempotency-Key` header; dedup window keyed on `(api_key_fingerprint, file_sha256 OR url)` for ~5 min.
- **C10 — Abuse / cost cap.** BYOK means cost lands on the user, so the host-side concern is volume — burst-rate kill switch + per-IP/day quota.

### Frontend polish

- **`Dashboard.jsx`** — the "CLIPS PROCESSED" StatCard shows the count but the sub-caption still reads "No batches yet" even when the count is non-zero. The caption is a separate field; threshold-derived sub-copy (`count > 0 ? '{n} batches' : 'No batches yet'`) is the fix.
- **`ShortForm/steps/Processing.jsx:189-203`** — when every job ends in `error`, both **Skip** and **Review** are disabled (Skip gates on `hasAnyComplete`; Review on `overallStatus === 'complete'`). The user has no forward path. Either let Review unlock when *all* jobs reach a terminal status (including all-error), or add a "Start over" link.
- **`LongForm/steps/Processing.jsx`** — the import of `useRef` is still in place (used by `savedRef`); harmless, but worth a tidy if/when the `savedRef` pattern itself gets replaced by a status check.

### Infra / tooling

- **Docker Compose anonymous volume gotcha.** `/app/node_modules` in `docker-compose.yml` masks freshly-installed npm deps after a `package.json` change — `react-router-dom` was missing in the container on first smoke-test run despite being in `package.json`. Fix is either (a) document `docker compose down -v && docker compose up --build` in the README, or (b) drop the anonymous volume from the `frontend` service and accept slower first builds.
- **OpenAPI snapshot drift.** `backend/tests/api/test_openapi_contract.py` fails 1/62 in the current docker image — Pydantic emits `contentMediaType: application/octet-stream` for file-upload fields where the baseline has `format: binary`. No route changes; pure schema-serialization drift. Regenerate per the procedure in `HANDOFF.md §12`, or pin Pydantic to the baseline-generation version.
- **`assets/fonts/user/` mount.** Already persistent across restarts, but there's no UI to *delete* an uploaded font yet.

### Adversarial review re-run

The H1 / H2 / M3 remediations in `43c2d96` are based on the read-only Codex
audit (task `task-mpdeyzjz-vpdetv`, completed 2026-05-20 02:01 UTC).
A follow-up `/codex:adversarial-review` before merge is worth running —
both to verify the fixes land cleanly and to surface anything the first
pass deferred.

---

## Technical roadmap

Designs and ordering for the three larger backend features the user asked
about during the restructure planning, plus the refactors deliberately
deferred out of the restructure phase so it could ship safely.

The headline rule: **everything below depends on the package structure that
already shipped in Phase 1, plus the single-FFmpeg-wrapper convention.**
Each feature is sized so that it can land in a small handful of atomic
commits with the `pytest -m "not e2e"` suite green between commits.

### Ordering (lowest blast radius first)

1. **Feature C — Motion Graphics Library.** Reuses the proven
   FFmpeg-overlay pattern from `openshorts/overlays/hooks.py`. No changes
   to the pipeline hot loop. **Ships first** because the compositor it
   introduces is the prerequisite for feature A's audio batching.
2. **Feature A — Background Soundtracks + SFX with Ducking.** Self-contained
   at the audio layer once C's compositor exists. Integrates at the
   single audio-mux step in `openshorts/video/pipeline.py` — small
   surface area, but it needs the FFmpeg wrapper migration (below) done.
3. **Feature B — Layout Templates.** Last because it touches the hottest
   loop in the codebase. Once C and A have landed, layouts is a clean
   polymorphism extraction with no need to also be inventing infra.

The three deferred refactors interleave naturally:

- Before **A**: finish migrating every `subprocess.run(['ffmpeg', ...])` call to
  `openshorts/video/ffmpeg.py` (Phase 1.10 leftover).
- Before or alongside **B**: split `app.py` into the eleven planned routers
  under `openshorts/routes/` and centralize job state in
  `openshorts/core/job_store.py` (Phase 1.9 leftover).
- Independently: split `openshorts/saas/pipeline.py` into the five planned
  modules (research / scripting / media / compositing / pipeline) (Phase 1.8
  leftover).

---

### Feature C — Motion Graphics Library

#### Why first

The hook-overlay code in `openshorts/overlays/hooks.py:add_hook_to_video()`
already proves out the pattern: render PNG via PIL, burn onto video via
FFmpeg `overlay` filter. Generalizing that to "a library of effects, each
rendered to a PNG sequence or alpha .mov, then composited in one ffmpeg
invocation" is a small extension. No changes to the per-frame loop.

#### Architecture

```
openshorts/motion_graphics/
├── base.py
│   class MotionGraphicEffect(ABC):
│       def render(self, duration_sec, fps, out_dir) -> Path  # returns PNG seq or .mov with alpha
│       def get_overlay_filter(self, start_sec, end_sec, w, h) -> str   # the FFmpeg filter chain
│
├── compositor.py
│   class MotionGraphicsCompositor:
│       def add(self, effect: MotionGraphicEffect, start_sec, end_sec): ...
│       def render(self, input_video, output_video):
│           # 1. ask each effect for its PNG/mov
│           # 2. build ONE filter_complex chain ([0:v][1:v]overlay=...[v1];[v1][2:v]overlay=...[v2];...)
│           # 3. invoke openshorts.video.ffmpeg.run(...) ONCE — single re-encode
│
└── library/
    ├── lower_thirds.py     class LowerThirdsEffect
    ├── callout.py          class CalloutEffect
    ├── progress_bar.py     class ProgressBarEffect
    └── animated_emoji.py   class AnimatedEmojiEffect
```

#### Files to add

- `openshorts/motion_graphics/base.py`
- `openshorts/motion_graphics/compositor.py`
- `openshorts/motion_graphics/library/{lower_thirds,callout,progress_bar,animated_emoji}.py`
- `openshorts/routes/motion_graphics.py` — `GET /api/motion-graphics/library` (lists effects + thumbnails) and `POST /api/motion-graphics/render` (apply a timeline)
- `openshorts/models/motion_graphics.py` — Pydantic schemas (`EffectInstance`, `RenderTimeline`, etc.)
- Frontend: a `MotionGraphicsModal.jsx` matching the existing `HookModal` / `SubtitleModal` pattern (defer until UI work is in scope)

#### Integration

The compositor sits *after* the vertical-reframing step and *before* the
audio mux in `openshorts/video/pipeline.py`. Easiest way to wire it in
is to make `process_video_to_vertical()` accept an optional
`motion_graphics_timeline` argument and, if present, route the
silent-video output through the compositor before the audio merge.

#### Risks the pipeline analysis flagged

- **Re-encoding per overlay.** Mitigated by the compositor building a
  single `filter_complex` chain — the video is decoded and re-encoded
  exactly once regardless of how many effects are applied.
- **PNG-sequence disk usage.** Each effect writes its frames to a per-clip
  temp dir under `output/<job_id>/_mg/`; cleaned up after the final mux.

---

### Feature A — Background Soundtracks + SFX with Ducking

#### Why second

Logically independent of layouts. Needs the FFmpeg wrapper done so
the mixer can compose `amix` + `volume` + `silencedetect` chains cleanly.

#### Architecture

```
openshorts/audio/
├── mixer.py
│   def mix_audio_tracks(original_audio, music_track, sfx_cues, output, ducking_db=-18):
│       # 1. Detect speech intervals via Whisper word timings (already cached in metadata.json)
│       #    OR via FFmpeg silencedetect if no transcript available.
│       # 2. Build a `volume` filter on the music track with `enable=between(t,...)` per speech interval.
│       # 3. amix=inputs=2 (original + ducked music) + each SFX cue at its trigger time.
│       # 4. Funnel through openshorts.video.ffmpeg.run(...).
│
├── library.py
│   def list_tracks(genre=None, mood=None, length_sec=None) -> list[TrackMeta]
│       # Reads assets/music/manifest.json — committed file listing tracks under assets/music/
│
└── cues.py
    def generate_sfx_cues(transcript, gemini_key) -> list[SfxCue]
        # Gemini analyzes transcript to suggest SFX moments (zoom-ins, scene changes, hook delivery).
        # Prompt lives at openshorts/prompts/sfx_cues.md.
```

#### Files to add

- `openshorts/audio/mixer.py`
- `openshorts/audio/library.py`
- `openshorts/audio/cues.py`
- `openshorts/prompts/sfx_cues.md`
- `openshorts/routes/audio.py` — `POST /api/audio/apply`
- `openshorts/models/audio.py`
- `assets/music/manifest.json` + a small set of CC-licensed tracks (or stub manifest + user uploads in v1)

#### Integration

Inside `openshorts/video/pipeline.py:process_video_to_vertical()` at the
existing audio-mux step (today around the `merge_command` block). The
audio mixer takes the original audio from `temp_audio_output`, mixes in
the soundtrack + cues, and writes the mixed audio back over the
intermediate file before the final mux. The video side never sees this.

#### Risks

- **Speech-detection accuracy.** When word timings are unreliable
  (background noise, music in the source), fall back to FFmpeg
  `silencedetect=n=-30dB:d=0.5` to bracket speech intervals.
- **Music licensing.** v1 ships with placeholder royalty-free files
  under `assets/music/`. v2 can swap in an Epidemic Sound / Artlist
  client behind `openshorts/integrations/`.

---

### Feature B — Layout Templates

#### Why last

Touches the per-frame loop in `openshorts/video/pipeline.py`. The other
two features add new boxes alongside the loop; this one rewrites how the
loop branches. Biggest blast radius — best to land it after C and A are
shipped and the test suite has shaken out any edge cases.

#### Architecture

```
openshorts/layouts/
├── base.py
│   class Layout(ABC):
│       def __init__(self, output_w, output_h, video_w, video_h, fps): ...
│       def render_frame(self, frame, detections, frame_number) -> np.ndarray
│       def on_scene_change(self, scene_index): ...    # for cameramen / trackers to snap
│
├── vertical_panorama.py     class VerticalPanoramaLayout    # today's TRACK / GENERAL behavior, polymorphic
├── educational.py           class EducationalLayout         # top half = source content, bottom = presenter headshot
└── side_by_side.py          class SideBySideLayout          # stub for the next variant
```

#### Files to add

- `openshorts/layouts/base.py`, `vertical_panorama.py`, `educational.py`, `side_by_side.py`
- `openshorts/routes/layouts.py` — `layout` field accepted on `POST /api/process`; later `POST /api/layout/reapply` to swap layout on an existing job's clips without re-transcribing
- `openshorts/models/layouts.py`

#### Pipeline change

The branching at the heart of `process_video_to_vertical()` (the
`if current_strategy == 'GENERAL': ... else: ...` block) becomes:

```python
layout: Layout = layout_registry.get(request.layout)  # default: VerticalPanoramaLayout
# ... in the frame loop:
output_frame = layout.render_frame(frame, detections, frame_number)
```

`VerticalPanoramaLayout` wraps today's `SmoothedCameraman` +
`SpeakerTracker` + `create_general_frame()` exactly as they are — the
restructure already kept those in their own modules precisely to
support this.

`EducationalLayout` owns *two* cameramen — one for the source content
(top half, treated as a screencast crop) and one for the presenter face
(bottom half, tight headshot crop using `detect_face_candidates`).
At each frame, both crops are computed and stacked vertically. If no
face is detected for the presenter slot, falls back to vertical panorama
for that segment.

#### Risks

- **Per-frame cost.** Two cameramen + two crops doubles the
  detection / transform cost. Mitigation: detect once per frame; both
  cameramen consume the same `detections` list.
- **Layout-change-mid-clip.** Out of scope for v1 — layout is fixed for
  the whole clip. v2 could allow per-scene layout swaps.

---

## Deferred refactors (Phase 1 leftovers)

| Refactor | Why deferred | Plan |
| --- | --- | --- |
| Full router split of `app.py` | 2256 lines / 32 routes; doing it as one pass would have been risky given the test suite mocks heavy ML deps at the module-import boundary. | Split per the plan: 11 routers under `openshorts/routes/` + `create_app()` factory in `openshorts/app.py`. One router per commit. The OpenAPI snapshot in `tests/snapshots/baseline.openapi.json` is the gate — it must stay byte-identical except when a route is deliberately changed. |
| Migrate every `subprocess.run(['ffmpeg', ...])` to `openshorts/video/ffmpeg.py` | Many call sites (app.py, video/pipeline.py, overlays/*, editing/ai_filters.py, saas/pipeline.py). Migrating all of them in one pass would have ballooned the restructure commit set. | One caller per commit. Tests between. The hook overlay in `overlays/hooks.py:add_hook_to_video()` is a good first migration — small, well-tested. |
| Internal split of `openshorts/saas/pipeline.py` | 1474-line file. No direct test coverage (only via the OpenAPI contract). Splitting it carries risk without the safety net of tests. | Per the original plan: `saas/research.py` (scraping + analyze), `saas/scripting.py`, `saas/media.py` (fal.ai + ElevenLabs TTS), `saas/compositing.py`, `saas/pipeline.py` (orchestrator). Add focused unit tests for the research + scripting + compositing layers as you split them. |
| `openshorts/core/job_store.py` + `api_keys.py` resolver | Today the job-state dicts (`jobs`, `thumbnail_sessions`, `publish_jobs`, `saas_jobs`) live as globals in `app.py`. The router split is a natural place to extract them. | Land alongside the router split, not before — extracting them prematurely just shifts where the globals live without delivering value. |
| Frontend restructure | Done in the 4-phase UI overhaul (commits 667a88e → 95ca831): `App.jsx` is now 47 lines, state lives in `frontend/src/state/`, the wizards / shell / Settings VS-Code layout are all in place. Remaining frontend work is feature-level, not structural. | n/a — superseded. |

---

## What landed in this restructure

For posterity. Phase 0 + Phase 1 + Phases 2-5 + the 4-phase UI overhaul +
the smoke-test fix commit produced these on `chore/restructure-and-docs`
(newest first):

- `fix(smoke-test): runtime bugs + Codex H1/H2/M3 remediation` — backend `run_job` subprocess + LongForm StrictMode timer + JSX `>` warnings + Codex H1 (MP4/MOV signature validation) + H2 (wizard reset on File loss) + M3 (polling AbortController + terminal-status guard).
- `feat(ui): phase 4 — long-form 4-step wizard + Dashboard`.
- `feat(ui): phase 3 — short-form 4-step wizard + UI primitives`.
- `feat(ui): phase 2 — Settings VS-Code layout + notifications + tooltips`.
- `feat(ui): phase 1 — shell + theme + routing skeleton`.
- `feat(brand-kit): brand kit settings + font upload + port refresh`.
- `chore(restructure): split repo into backend/ + frontend/ + renderer/ + assets/`.
- `docs(roadmap): design future features + document deferred refactors`.
- `docs(claude.md): add per-folder sub-CLAUDE.md stubs for high-rule areas`.
- `docs(claude.md): rewrite with structured guidance + auto-managed sections`.
- `chore(tooling): add CLAUDE.md auto-updater + pre-commit hook`.
- `docs(env): expand .env.example to match what the code actually reads`.
- `chore(restructure): Dockerfile CMD points at openshorts.app:app`.
- `chore(restructure): add openshorts/video/ffmpeg.py wrapper scaffold`.
- `chore(restructure): add openshorts/app.py re-export for Docker entrypoint`.
- `chore(restructure): move saasshorts -> openshorts/saas/pipeline.py`.
- `chore(restructure): split main.py -> video/* + ml/* + ingest/youtube.py`.
- `chore(restructure): split thumbnail -> thumbnails/{titles,images,descriptions}.py`.
- `chore(restructure): split editor -> editing/ai_filters + editing/prompts + utils/filters`.
- `chore(restructure): split subtitles -> overlays/subtitles_{generate,render}.py`.
- `chore(restructure): move hooks -> openshorts/overlays/hooks.py`.
- `chore(restructure): move translate -> openshorts/integrations/elevenlabs.py`.
- `chore(restructure): move s3_uploader -> openshorts/integrations/s3.py`.
- `chore(restructure): scaffold empty openshorts/ package + extend pyproject`.
- `test: add Phase 0 safety net before restructure`.

The revert point: `git tag pre-restructure-20260519-1526`. `git reset --hard
pre-restructure-20260519-1526` returns the tree to its pre-restructure state.
