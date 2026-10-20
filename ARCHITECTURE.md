# OpenShorts Architecture Overview

This document provides a comprehensive overview of the OpenShorts platform's architecture, feature mapping, data flow, and logic execution.

## 1. Platform Features & File Mapping

The platform provides 3 main tools:
1. **Clip Generator**: Converts long-form videos into viral 9:16 shorts.
2. **AI Shorts (UGC)**: Generates complete marketing videos from scratch using AI actors and voiceovers.
3. **YouTube Studio**: AI-assisted toolkit for thumbnails, titles, and descriptions.

### Feature to File Mapping

| On-Screen Feature | Core Logic File(s) | Description |
|-------------------|--------------------|-------------|
| **App Routing & Endpoints** | `app.py` | FastAPI application, job queuing, cloud billing, and REST API definitions. |
| **Job Execution Pipeline** | `main.py` | Orchestrates the primary clip extraction pipeline (download -> transcribe -> analyze -> reframe -> composite). |
| **Video Download** | `yt_clients.py` | Handles `yt-dlp` integration for downloading source videos. |
| **Audio Transcription** | `transcribe_backends.py` | Generates word-level timestamps using `faster-whisper` or `parakeet`. |
| **Scene Detection** | `scene_detection.py` | Uses PySceneDetect to find natural breaks (cuts) in the video. |
| **AI Viral Moment Selection** | `clip_selection.py`, `gemini_worker.py`, `llm_backend.py` | Sends transcript to Gemini (or local LLM) to identify the 3-15 most viral moments based on hook and payoff. |
| **Video Cutting & Rendering** | `ffmpeg_utils.py` | Helper functions for precise FFmpeg cuts and encoding configurations. |
| **Smart 9:16 Cropping & Tracking** | `main.py` (`SmoothedCameraman`, `SpeakerTracker`), `reframe_v2.py` | Uses ultralytics YOLOv8 and MediaPipe to detect faces and bodies, moving a virtual camera smoothly to track the active speaker. |
| **Multi-Speaker Layouts (Split Screen)** | `layout_picker.py`, `layout_ranges.py`, `split_layout.py`, `screencast_layout.py`, `panel_layout.py` | Decides per-scene whether to use a tracking crop, blurred background (GENERAL), split-screen (SPLIT), or SCREENCAST layout. |
| **Subtitles / Captions** | `subtitles.py` | Generates stylized ASS subtitles and burns them into the video. |
| **Hooks / Text Overlays** | `hooks.py`, `hook_grounding.py` | Generates and composites attention-grabbing text overlays over the clips. |
| **AI Shorts Generation (UGC)** | `saasshorts.py` | Scrapes target URL, uses Gemini to write scripts (Hook-Problem-Solution-CTA), uses `fal.ai` for AI actors (Flux 2 Pro / Kling) and `ElevenLabs` for voiceovers. |
| **YouTube Studio (Thumbnails)** | `thumbnail.py` | AI-generated thumbnails using face overlays and background compositing. |
| **Cloud Publishing & Gallery** | `s3_uploader.py` | Uploads generated assets to S3/R2 and integrates with Upload-Post for TikTok, IG, and YouTube publishing. |

---

## 2. Data and Logic Flow

### A. Clip Generator Flow
When a user submits a long-form video URL (or uploads a file):
1. **Ingest (`app.py` -> `main.py`)**: Request is received in `app.py` (`/api/process`), quota is checked (if cloud billing is enabled), and a job is queued in an `asyncio.PriorityQueue`.
2. **Download (`yt_clients.py`)**: A worker thread in `main.py` downloads the video (if a URL) in optimal quality formats.
3. **Transcription (`transcribe_backends.py`)**: Audio is extracted and transcribed to produce precise word-level JSON arrays.
4: **Scene Analysis (`scene_detection.py`)**: Video is split into scenes to ensure cuts and camera tracks don't bleed across visual changes.
5. **Moment Detection (`gemini_worker.py` & `clip_selection.py`)**: Gemini analyzes the transcript to select optimal clips (15-60s) with viral potential, outputting JSON metadata with titles and hooks.
6. **Reframe & Tracking (`reframe_v2.py` & `main.py`)**: 
    - The `layout_picker` decides the strategy (TRACK, GENERAL, SPLIT, etc.).
    - `detect_face_candidates` and `SmoothedCameraman` apply hysteresis and panning rules to create the vertical crop coordinates.
7. **Effects Composition (`ffmpeg_utils.py` & `subtitles.py`)**: Subtitles and visual hooks are burned into the cropped video.
8. **Completion**: Files are saved to `output/<job_id>`, uploaded to S3 (if configured), and a webhook / status update notifies the frontend.

### B. AI Shorts (SaaS UGC) Flow
When a user describes a product or pastes a website URL:
1. **Scrape & Analyze (`saasshorts.py`)**: `scrape_website` extracts text/meta, and `research_saas_online` searches Google for reviews and pain points. `analyze_saas` merges this into a structured JSON analysis.
2. **Script Generation**: `generate_scripts` takes the analysis and produces a rigid 5-segment script (Hook -> B-Roll -> Actor -> Demo -> CTA).
3. **Asset Generation**:
    - **Actor**: `generate_actor_images` calls `fal.ai` (Flux 2 Pro) for portrait generation.
    - **Audio**: `generate_voiceover` calls `ElevenLabs` to generate TTS for the script segments.
4. **Video Generation**: Uses `fal.ai` (Hailuo or Kling) for lipsync and b-roll generation.
5. **Assembly (`saasshorts.py` & `ffmpeg_utils.py`)**: Assets are stitched sequentially via FFmpeg, with ASS subtitles overlayed on top.

---

## 3. Frontend / UI Relationship
The UI is a React application styled via Tailwind CSS. 

- **Styling constraints**: Strictly adheres to `design.md` (Lumen/Night Foundry theme). Colors heavily rely on CSS tokens (`--color-paper`, `--color-brass`, etc.).
- **Visible Elements**: 
  - The "Smart Tracking" crop visualized on screen is purely the result of `SmoothedCameraman` in `main.py`.
  - The "Split" layout toggle in the dashboard maps directly to the logic in `split_layout.py`.
  - Hook texts and subtitles edited in the UI are sent to the backend and re-rendered via `subtitles.py` and `hooks.py`.
  - All processing statuses poll `/api/status` in `app.py`.

## 4. Key Architectural Patterns
- **Asynchronous Queueing**: Job concurrency is strictly controlled via `asyncio.Semaphore` in `app.py` to prevent CPU/GPU exhaustion (specifically for FFmpeg/Whisper).
- **Subprocess Execution**: FFmpeg encoding runs in isolated subprocesses using `subprocess.run()`, ensuring main thread event loops are unblocked.
- **Fail-Safe Deduplication**: Features like `_strip_burned_captions` in `app.py` manage file naming conventions (`subtitled_<ts>_clean.mp4`) allowing non-destructive re-edits from the clean original.
- **Environment Isolation**: A `CHILD_ENV_DENYLIST` prevents critical secrets (Stripe, DB, JWT) from leaking into FFmpeg/yt-dlp child processes.
