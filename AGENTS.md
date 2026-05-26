# AGENTS.md

## Project Overview
OpenShorts is an AI-powered video platform designed to create viral short-form content. It features three primary tools:
- **Clip Generator**: Transforms long-form videos into viral 9:16 shorts using AI for moment detection and smart reframing.
- **AI Shorts (UGC Creator)**: Generates marketing videos with AI actors, lip-sync, and voiceovers from a product description or URL.
- **YouTube Studio**: A toolkit for generating AI-powered thumbnails, viral titles, and descriptions.

## Setup & Launch

### Full Stack (Recommended)
Run the entire platform using Docker:
```bash
docker compose up --build
```
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:5175

### Backend Only
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend Only
```bash
cd dashboard
npm install
npm run dev
```

## Tech Stack
- **Backend**: Python 3.11, FastAPI, openai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg, httpx.
- **Frontend**: React 18, Vite 4, Tailwind CSS 3.4.
- **AI APIs**: Google Gemini (Analysis/Scripts), fal.ai (Actors/Video), ElevenLabs (Voice/Dubbing).
- **Infrastructure**: Docker, Docker Compose, AWS S3 (Cloud Backup).
- **Publishing**: Upload-Post API (TikTok, Instagram, YouTube).

## Core Processing Pipeline
The video processing follows this sequence:
1. **Ingest**: YouTube download (yt-dlp) or local upload.
2. **Transcription**: faster-whisper for word-level timestamps.
3. **Scene Detection**: PySceneDetect for segment boundaries.
4. **AI Analysis**: Gemini identifies viral moments (15-60s).
5. **Extraction**: FFmpeg precise clip cutting.
6. **AI Cropping**: Vertical reframing (TRACK mode via MediaPipe/YOLOv8 or GENERAL mode).
7. **Effects/Subtitles**: AI-generated FFmpeg filters and burned-in subtitles.
8. **Hook Overlay**: Styled text overlays for attention.
9. **Voice Dubbing**: Optional ElevenLabs translation.
10. **S3 Backup**: Silent background upload to AWS S3.
11. **Social Distribution**: Async upload via Upload-Post API.

## Key Files & Architecture
| File | Purpose |
|------|---------|
| `main.py` | Core video processing: transcription, scene detection, extraction, reframing. |
| `app.py` | FastAPI server with async job queue and REST endpoints. |
| `editor.py` | Gemini AI integration for dynamic video effects (FFmpeg filter generation). |
| `hooks.py` | Hook text overlay generation and font rendering. |
| `s3_uploader.py` | AWS S3 upload and caching logic. |
| `subtitles.py` | SRT generation and FFmpeg subtitle burning. |
| `translate.py` | ElevenLabs dubbing API integration. |
| `dashboard/` | React-based frontend for managing jobs and settings. |

## Code Style & Conventions
- **Async First**: Use `async`/`await` for all I/O-bound tasks and API calls to prevent blocking the FastAPI event loop.
- **Security**: API keys must **never** be stored on the server. They are encrypted in the browser's `localStorage` and sent via request headers only when needed.
- **Concurrency**: Use the semaphore-based job queue. The concurrency limit is controlled by the `MAX_CONCURRENT_JOBS` environment variable (default: 5).
- **Video Processing**: Always use FFmpeg for precise cuts and composites. Ensure temp files are cleaned up.

## Testing & Validation
- **Frontend Linting**: Run `npm run lint` inside the `dashboard/` directory.
- **Verification Scripts**: Use the provided verification scripts to test specific features:
  - `python verify_hooks.py` (Test hook overlays)
  - `python verify_custom_hook.py` (Test custom hook logic)
  - `python verify_aesthetic.py` (Test aesthetic verification)

## MCP & Tooling Best Practices

### ⚠️ MANDATORY WORKFLOW
You MUST follow this exact workflow for EVERY coding task. Do not skip any step.

- **Core File System**: `read`, `write`, `edit`, `glob`, `grep` (precise code manipulation).
- **System Shell**: `bash` (commands, tests, environment checks).
- **Agent Orchestration**: `task` (complex, autonomous sub-tasks to keep main context clean).
- **Web Intelligence**: `webfetch` (external documentation and data).
- **Codebase Analysis**: `semantic_search` (intent-based search), `kilo_local_recall` (past session context).
- **Reasoning & Planning**: Sequential Thinking (complex problem solving), `todowrite` (task tracking).
- **Browser Automation**: `playwright_browser_*` (testing and scraping).
- **External Knowledge**: `context7_*` (up-to-date library documentation).
- **Memory**: `agentmemory_*` (long-term persistence of decisions and patterns).

### Development Guidelines
- **Core File System**: `read`, `write`, `edit`, `glob`, `grep` (precise code manipulation).
- **System Shell**: `bash` (commands, tests, environment checks).
- **Agent Orchestration**: `task` (complex, autonomous sub-tasks to keep main context clean).
- **Web Intelligence**: `webfetch` (external documentation and data).
- **Codebase Analysis**: `semantic_search` (intent-based search), `kilo_local_recall` (past session context).
- **Reasoning & Planning**: `sequential-thinking` (complex problem solving), `todowrite` (task tracking).
- **Browser Automation**: `playwright_browser_*` (testing and scraping).
- **External Knowledge**: `context7_*` (up-to-date library documentation).
- **Memory**: `agentmemory_*` (long-term persistence of decisions and patterns).

### Development Guidelines
1. **Parallel Execution**: Batch independent tool calls (e.g., `git status` and `git diff`) in a single response to reduce latency.
2. **Narrow-to-Broad Exploration**: Use `glob` $\rightarrow$ `grep` $\rightarrow$ `read` to find and analyze code. Avoid reading large files blindly.
3. **Context Management**: Use the `task` tool for multi-step implementations or deep research to prevent main context bloat.
4. **Iterative Verification**:
   - **Lint & Typecheck**: Always run project-specific linting (e.g., `npm run lint`) and typechecking after code changes.
   - **Test-Driven**: Run existing tests before and after changes to ensure no regressions.
5. **Safe Editing**: Always `read` a file before using `edit` or `write` to preserve indentation and context.
6. **Semantic Search First**: Use `semantic_search` to find relevant snippets by meaning when exploring unfamiliar areas.
7. **Structured Thinking**: Use `sequential-thinking` to plan the approach and verify hypotheses before writing code for non-trivial tasks.
8. **Memory Persistence**: Proactively use `agentmemory_memory_save` to record critical architectural decisions, library swaps (e.g., google-genai to openai), or tricky bug fixes.
9. **Tool Synergy**: Combine `context7_*` for library documentation, `webfetch` for general web data, and `kilo_local_recall` to maintain continuity across sessions.

## Environment Variables

### Server-side (`.env`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`: S3 backup configuration.
- `MAX_CONCURRENT_JOBS`: Max parallel processing tasks (default: 5).
- `VITE_API_URL`: Override for production API URL.

### Client-side (Encrypted in LocalStorage)
- `GEMINI_API_KEY`: Google Gemini API key (Required).
- `FAL_KEY`: fal.ai API key (Required for AI Shorts).
- `ELEVENLABS_API_KEY`: ElevenLabs API key (Optional).
- `UPLOAD_POST_API_KEY`: Upload-Post API key (Optional).

**⚠️ MANDATORY: ALWAYS follow this workflow for EVERY coding task:**

1. **LOAD CONTEXT** → `agentmemory_memory_recall` or `agentmemory_memory_smart_search`
2. **THINK** → Sequential Thinking for analysis
3. **LOOKUP** → Context7 if external libraries involved
4. **SEARCH** → `semantic_search` for code patterns and implementations
5. **EXECUTE** → tools for code operations
6. **VERIFY** → Check results
7. **SAVE** → `agentmemory_memory_save` to Agent Memory after completing work

**Memory Rules (MANDATORY):**

- ALWAYS load memories BEFORE starting any task
- ALWAYS save memories AFTER completing significant work
- Use `semantic_search` to find code by intent and meaning before using grep/glob
