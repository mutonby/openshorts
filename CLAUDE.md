# CLAUDE.md

Guidance for Claude Code (and humans) when working with the OpenShorts repo.

## Project

OpenShorts is an AI-powered vertical short-video generator. It transforms
YouTube videos and local uploads into 9:16 viral clips for TikTok, Reels,
and Shorts. The pipeline uses Google Gemini for viral-moment detection and
title generation, faster-whisper for transcription, PySceneDetect for scene
boundaries, MediaPipe + YOLOv8 for face/person tracking, and FFmpeg for all
encoding/overlay/mux work.

## Quick start

```bash
# Full stack (backend + frontend + Remotion renderer)
docker compose up --build

# Backend only (local dev — needs Python 3.11+ and FFmpeg on PATH)
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pytest -m "not e2e"                    # unit + API contract suite (~0.6s)
uvicorn openshorts.app:app --host 0.0.0.0 --port 8000

# Frontend only
cd dashboard && npm install && npm run dev
```

Install the CLAUDE.md auto-updater hook once after cloning:

```bash
bash scripts/install_hooks.sh
```

## Where things go (decision table)

When you want to **add** something, this is where it lands:

| If you want to add… | Drop it in… | Notes |
| --- | --- | --- |
| A new HTTP endpoint | `openshorts/routes/<domain>.py` + register in `openshorts/app.py` | The router split from `app.py` is in flight; until it ships, edit `app.py` directly. |
| A new FFmpeg operation | `openshorts/video/ffmpeg.py` | Never call `subprocess.run(['ffmpeg', ...])` outside this module. |
| A new external service client | `openshorts/integrations/<service>.py` | Each one exposes a typed Python client. |
| A new AI model / inference call | `openshorts/ml/<purpose>.py` | Detection, transcription, viral extraction, etc. |
| A new layout template | `openshorts/layouts/<name>.py` | Subclass `Layout` (see [ROADMAP.md](ROADMAP.md) feature B). |
| A new motion-graphic effect | `openshorts/motion_graphics/library/<name>.py` | Subclass `MotionGraphicEffect` (see [ROADMAP.md](ROADMAP.md) feature C). |
| A new audio mixer / SFX | `openshorts/audio/<concern>.py` | See [ROADMAP.md](ROADMAP.md) feature A. |
| A new Gemini prompt | `openshorts/prompts/<name>.md` or `openshorts/editing/prompts.py` | Externalize prompts; don't bury them in handler code. |
| A new Pydantic schema | `openshorts/models/<domain>.py` | One file per request/response domain. |
| A new shared FFmpeg / filter helper | `openshorts/utils/filters.py` | Already used by editing + future motion-graphics compositor. |
| A new core infrastructure piece | `openshorts/core/<concern>.py` | Job queue, job store, api-key resolver, logging. |

When you want to **remove** something:

1. Delete the route file (or the function within it).
2. `grep -r <removed_name>` to find dead imports.
3. Delete the corresponding Pydantic model in `openshorts/models/` if any.
4. Delete or update tests that reference it.
5. Run `python scripts/update_claude_md.py` (the pre-commit hook will do this for you).

## Repo layout

The top-level folders. **The table below is auto-managed by `scripts/update_claude_md.py`** — never edit it by hand.

<!-- AUTO:REPO-MAP:START -->
| Folder | What it is |
| --- | --- |
| `dashboard/` | React + Vite frontend (out of scope for the current restructure). |
| `fonts/` | Committed TTFs (Noto Serif Bold) used by hook overlays. |
| `openshorts/` | Python package — all backend code lives here. |
| `output/` | Runtime: generated clips and thumbnails (gitignored). |
| `remotion/` | Remotion compositions (TypeScript) consumed by the render-service. |
| `render-service/` | Standalone TypeScript microservice that bundles + renders Remotion compositions. |
| `screenshots/` | Repo screenshots used in README.md. |
| `scripts/` | Developer tooling (update_claude_md.py, install_hooks.sh). |
| `tests/` | Pytest suite — unit, API contract, and e2e smoke. |
| `uploads/` | Runtime: incoming video uploads (gitignored). |
<!-- AUTO:REPO-MAP:END -->

### Backend package (`openshorts/`)

The Python package follows classical layered conventions. Each subfolder
has a one-line purpose statement in its `__init__.py`.

| Folder | Rule |
| --- | --- |
| `openshorts/core/` | Cross-cutting infra: job queue, job store, API-key resolver, logging. |
| `openshorts/routes/` | FastAPI routers, one module per API domain. |
| `openshorts/video/` | All video work goes here. **FFmpeg only via `video/ffmpeg.py`.** |
| `openshorts/ml/` | AI inference: face/person detection, transcription, viral extraction. |
| `openshorts/audio/` | Future feature A — soundtracks + ducking. |
| `openshorts/layouts/` | Future feature B — layout templates (panorama, educational, etc.). |
| `openshorts/motion_graphics/` | Future feature C — animated overlays + multi-effect compositor. |
| `openshorts/editing/` | AI-generated FFmpeg filter pipeline. |
| `openshorts/overlays/` | Hook cards + subtitle generation / burn-in. |
| `openshorts/ingest/` | YouTube downloads + local upload handling. |
| `openshorts/saas/` | SaaSShorts UGC pipeline (research → script → media → composite). |
| `openshorts/integrations/` | External-service clients (S3, ElevenLabs, fal.ai, Upload-Post). |
| `openshorts/thumbnails/` | YouTube thumbnail workflow (titles, images, descriptions). |
| `openshorts/prompts/` | Externalized Gemini prompt templates. |
| `openshorts/models/` | Pydantic request/response schemas grouped by domain. |
| `openshorts/utils/` | Shared helpers: filter sanitization, path utilities. |

## Module map

Every Python module under `openshorts/` and its public surface. **Auto-managed** — regenerated by the pre-commit hook from each file's docstring.

<!-- AUTO:MODULE-MAP:START -->
| Module | Purpose | Public surface |
| --- | --- | --- |
| `openshorts/app.py` | FastAPI application entrypoint for the openshorts package. | _(none)_ |
| `openshorts/editing/ai_filters.py` | VideoEditor: Gemini-driven FFmpeg filter generation and application. | `VideoEditor` |
| `openshorts/editing/prompts.py` | Gemini prompt templates for AI video-effect generation. | `build_ffmpeg_filter_prompt`, `build_effects_config_prompt` |
| `openshorts/ingest/youtube.py` | YouTube downloader with bot-detection workarounds (yt-dlp + cookies + alt clients). | `sanitize_filename`, `download_youtube_video` |
| `openshorts/integrations/elevenlabs.py` | ElevenLabs Dubbing API client: AI voice translation across 30+ languages. | `create_dubbing_project`, `get_dubbing_status`, `download_dubbed_video`, `translate_video`, `get_supported_languages` |
| `openshorts/integrations/s3.py` | AWS S3 client: clip uploads, actor gallery, UGC video gallery, presigned URLs. | `upload_file_to_s3`, `get_s3_client`, `generate_presigned_url`, `list_all_clips`, `upload_actor_to_s3`, `list_actor_gallery`, `upload_video_to_gallery`, `list_video_gallery`, `upload_job_artifacts` |
| `openshorts/ml/detection.py` | Face and person detection: MediaPipe BlazeFace (primary) + YOLOv8 (fallback). | `detect_face_candidates`, `detect_person_yolo` |
| `openshorts/ml/transcription.py` | faster-whisper transcription: CPU-optimized (INT8 quantization) with word timestamps. | `transcribe_video` |
| `openshorts/ml/viral_extraction.py` | Gemini 2.5 Flash viral-moment extraction: picks 3-15 short clips from a transcript. | `get_viral_clips` |
| `openshorts/overlays/hooks.py` | Hook text overlays: PIL-rendered cards (PNG) burned onto video via FFmpeg. | `download_font_if_needed`, `create_hook_image`, `add_hook_to_video` |
| `openshorts/overlays/subtitles_generate.py` | SRT subtitle generation: transcription and word-level grouping into short lines. | `transcribe_audio`, `generate_srt_from_video`, `generate_srt`, `format_srt_block` |
| `openshorts/overlays/subtitles_render.py` | Subtitle burn-in: FFmpeg subtitles filter + ASS color/style conversion. | `hex_to_ass_color`, `burn_subtitles` |
| `openshorts/saas/pipeline.py` | SaaSShorts: AI-powered UGC video generator for SaaS products. | `research_saas_online`, `scrape_website`, `analyze_saas`, `generate_scripts`, `generate_actor_images`, `generate_actor_image`, `generate_voiceover`, `get_elevenlabs_voices`, `generate_talking_head`, `generate_talking_head_lowcost`, `generate_broll`, `transcribe_audio_for_subs`, `generate_tiktok_subs`, `generate_srt_from_script`, `composite_video`, `generate_full_video` |
| `openshorts/thumbnails/descriptions.py` | YouTube description + chapter-marker generation from transcript segments. | `generate_youtube_description` |
| `openshorts/thumbnails/images.py` | Thumbnail image generation via Gemini multimodal image preview model. | `generate_thumbnail` |
| `openshorts/thumbnails/titles.py` | Gemini-driven viral title generation and conversational refinement. | `analyze_video_for_titles`, `refine_titles` |
| `openshorts/utils/filters.py` | Shared FFmpeg filter helpers: chain splitting, sanitization, zoompan size enforcement. | `split_filter_chain`, `enforce_zoompan_output_size`, `sanitize_filter_string` |
| `openshorts/video/ffmpeg.py` | Single FFmpeg wrapper for the entire codebase. | `FFmpegError`, `run`, `probe_resolution`, `probe_duration`, `cut`, `extract_audio`, `mux_video_audio`, `overlay_png`, `build_filter_complex` |
| `openshorts/video/pipeline.py` | process_video_to_vertical orchestrator: scenes -> strategy -> per-frame crop -> mux. | `process_video_to_vertical` |
| `openshorts/video/reframing.py` | Vertical reframing helpers: blurred-background 'General Shot' composite. | `create_general_frame` |
| `openshorts/video/scene_analysis.py` | PySceneDetect scene boundaries + per-scene TRACK/GENERAL strategy analysis. | `detect_scenes`, `get_video_resolution`, `analyze_scenes_strategy` |
| `openshorts/video/tracking.py` | SmoothedCameraman and SpeakerTracker: the heart of stabilized vertical reframing. | `SmoothedCameraman`, `SpeakerTracker` |
<!-- AUTO:MODULE-MAP:END -->

## Processing pipeline

1. **Ingest** — `openshorts/ingest/youtube.py:download_youtube_video()` or a local upload.
2. **Transcribe** — `openshorts/ml/transcription.py:transcribe_video()` (faster-whisper, word timestamps).
3. **Scene-detect** — `openshorts/video/scene_analysis.py:detect_scenes()` (PySceneDetect).
4. **Viral extraction** — `openshorts/ml/viral_extraction.py:get_viral_clips()` (Gemini 2.5 Flash picks 3–15 clips, 15–60 s each).
5. **Cut clips** — FFmpeg `-ss`/`-to` per clip.
6. **Strategy** — `openshorts/video/scene_analysis.py:analyze_scenes_strategy()` decides TRACK vs GENERAL per scene.
7. **Reframe** — `openshorts/video/pipeline.py:process_video_to_vertical()` runs the per-frame loop.
8. **Effects** (optional) — `openshorts/editing/ai_filters.py:VideoEditor` injects Gemini-generated FFmpeg filters.
9. **Hooks + subtitles** (optional) — `openshorts/overlays/`.
10. **Translate** (optional) — `openshorts/integrations/elevenlabs.py:translate_video()` dubs into 30+ languages.
11. **Backup + distribute** — `openshorts/integrations/s3.py` + `openshorts/integrations/upload_post.py` (planned).

## API surface

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/api/process` | Submit a video (URL or upload) for processing. |
| GET | `/api/status/{job_id}` | Poll status + logs. |
| POST | `/api/edit` | Apply Gemini-generated FFmpeg filters to a clip. |
| POST | `/api/effects/generate` | Get a structured EffectsConfig for Remotion. |
| POST | `/api/render/{render_id}` | Render via the Remotion microservice. |
| POST | `/api/subtitle` | Generate + burn subtitles. Auto-transcribes dubbed videos. |
| POST | `/api/hook` | Burn a text-hook PNG onto a clip. |
| POST | `/api/translate` | AI voice dubbing via ElevenLabs. |
| GET | `/api/translate/languages` | List supported languages. |
| POST | `/api/social/post` | Distribute via Upload-Post. |
| POST | `/api/thumbnail/*` | YouTube thumbnail workflow (titles, images, descriptions). |
| POST | `/api/saasshorts/*` | SaaS UGC pipeline. |

The full route inventory (32 endpoints) is locked in `tests/snapshots/baseline.openapi.json`.

## Environment

Server-side env vars the code actually reads. **Auto-managed** — generated from `.env.example`.

<!-- AUTO:ENV:START -->
| Variable | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | `_(empty — must set)_` | Required (server-side reads via os.getenv) |
| `AWS_ACCESS_KEY_ID` | `_(empty — must set)_` | Optional: AWS S3 (clip backup + public gallery) |
| `AWS_SECRET_ACCESS_KEY` | `_(empty — must set)_` | Optional: AWS S3 (clip backup + public gallery) |
| `AWS_REGION` | `eu-west-3` | Optional: AWS S3 (clip backup + public gallery) |
| `AWS_S3_BUCKET` | `_(empty — must set)_` | Optional: AWS S3 (clip backup + public gallery) |
| `AWS_S3_PUBLIC_BUCKET` | `_(empty — must set)_` | Optional: AWS S3 (clip backup + public gallery) |
| `DISABLE_YOUTUBE_URL` | `false` | Optional: YouTube ingestion |
| `YOUTUBE_COOKIES` | _(unset)_ | Optional: YouTube ingestion (commented — optional) |
| `RENDER_SERVICE_URL` | `http://renderer:3100` | Optional: Remotion render service |
| `MAX_CONCURRENT_JOBS` | `5` | Tuning |
| `VITE_API_URL` | `http://localhost:8000` | Tuning |
| `VITE_ENCRYPTION_KEY` | _(unset)_ | Tuning (commented — optional) |
| `ELEVENLABS_API_KEY` | _(unset)_ | Tuning (commented — optional) |
| `UPLOAD_POST_API_KEY` | _(unset)_ | Tuning (commented — optional) |
| `FAL_KEY` | _(unset)_ | Tuning (commented — optional) |
<!-- AUTO:ENV:END -->

ElevenLabs / Upload-Post / fal.ai keys are **client-side** (encrypted in browser localStorage, sent per-request via headers). They are NOT read from `.env`.

## Conventions

1. **Single FFmpeg wrapper.** Every `subprocess.run(['ffmpeg', ...])` call should funnel through `openshorts/video/ffmpeg.py`. Migration of existing callers is incremental — but new code must use the wrapper.
2. **API keys via headers, not env.** Client-side keys (Gemini, ElevenLabs, Upload-Post, fal.ai) arrive on each request as `X-...-Key`. The resolver helper for these lives in `openshorts/core/api_keys.py` (planned). Do NOT call `request.headers.get('X-...')` outside that file.
3. **Prompts as files.** New Gemini prompts go in `openshorts/prompts/<name>.md` and are loaded by name. Editing-domain prompts may stay inline in `openshorts/editing/prompts.py`.
4. **Every module starts with a docstring.** The pre-commit hook (`scripts/update_claude_md.py`) fails the commit if any `.py` file under `openshorts/` lacks one. Use a single line — it becomes the row in the auto-managed module map.
5. **Tests first.** A characterization test suite (`tests/`) was written *before* the restructure. Anything that touches behavior should keep `pytest -m "not e2e"` 100% green. The OpenAPI snapshot in `tests/snapshots/baseline.openapi.json` pins the public API.
6. **No new global dicts in routers.** Job state goes through `openshorts/core/job_store.py` (planned). Today, `app.py` still owns these dicts — keep them centralized there until the routers are split out.

## Pointers

- `ROADMAP.md` — designs for the three upcoming features (motion graphics, soundtracks, layouts) and deferred refactors (router split, FFmpeg-wrapper migration, saasshorts internal split).
- `scripts/update_claude_md.py` — what regenerates the auto-managed sections of this file.
- `scripts/install_hooks.sh` — one-liner to wire up the pre-commit hook.
- `tests/snapshots/baseline.openapi.json` — the contract that any backend change must keep green.
- `dashboard/` — the React/Vite frontend (deliberately out of scope for the current restructure).

## Tech stack

- **Backend:** Python 3.11, FastAPI, google-genai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg, httpx.
- **Frontend:** React 18, Vite 4, Tailwind CSS 3.4.
- **External:** Google Gemini, ElevenLabs Dubbing, Upload-Post, fal.ai (Flux + Kling), Remotion.
- **Infra:** Docker + Docker Compose, AWS S3.
