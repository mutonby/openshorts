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

## Tool Reference

### File Operations
| Tool | When to Use |
|------|------------|
| `glob` | Find files by pattern (e.g., `**/*.py`, `src/components/**/*.tsx`) |
| `grep` | Search file contents by regex pattern |
| `read` | Read a file or directory listing (always read before editing) |
| `edit` | Precise string replacements in existing files |
| `write` | Create or overwrite a file (read first if the file exists) |
| `bash` | Shell commands — git, npm, python, tests, linting, docker |

### Codebase Intelligence
| Tool | When to Use |
|------|------------|
| `codebase_peek` | **FIRST** — quick metadata-only lookup. Find WHERE code lives. Saves ~90% tokens vs `codebase_search`. Use for: discovery, navigation, finding multiple locations. |
| `codebase_search` | **SECOND** — full code content search by intent/meaning. Use when you need to see the actual implementation after peeking. |
| `implementation_lookup` | Jump to the authoritative definition of a function, class, or method. Prefers real impl files over tests/fixtures. |
| `call_graph` | Trace callers or callees of a function. Understand code flow and dependencies. |
| `find_similar` | Find code semantically similar to a given snippet. Use for: duplicate detection, pattern discovery, refactoring. |

### Codebase Indexing
| Tool | When to Use |
|------|------------|
| `index_codebase` | Run at the start of a session if the codebase has changed. Creates/updates vector embeddings for semantic search. Incremental — re-indexes only changed files. |
| `index_status` | Check if codebase is indexed, how many chunks exist. |
| `index_health_check` | Remove stale entries from deleted files. |
| `index_metrics` | Performance stats for the index (search timings, cache rates). |

### Reasoning & Execution
| Tool | When to Use |
|------|------------|
| `sequential-thinking_sequentialthinking` | Break complex problems into steps. Use for non-trivial tasks requiring planning, analysis, or multi-step solutions. |
| `todowrite` | Track complex tasks (3+ steps). Mark progress in real-time. |
| `task` | Launch autonomous sub-agents for complex multi-step work. Keeps main context clean. Two types: `explore` (codebase search) and `general` (research/implementation). |
| `question` | Ask the user for preferences, clarifications, or decisions during execution. |

### External Knowledge
| Tool | When to Use |
|------|------------|
| `context7_resolve-library-id` + `context7_query-docs` | Fetch up-to-date documentation for ANY library, framework, SDK, or API. Use even for well-known libraries (React, Next.js, FastAPI, etc.). Always resolve ID first, then query with the user's full question. |
| `webfetch` | Fetch and parse web content (docs, articles, API responses). Use when Context7 doesn't cover the topic. |

### Memory & Session Management (via `skill` tool)
| Skill | When to Use |
|-------|------------|
| `recall` | Search past sessions for observations, decisions, or context about a topic. |
| `remember` | Save an insight, decision, or learning for future sessions. |
| `handoff` | Resume the most recent agent session ("where were we?") |
| `session-history` | List recent past sessions on this project. |
| `recap` | Summarize recent sessions grouped by date. |
| `commit-context` | Trace a file/function back to the session that produced its commit. |
| `commit-history` | List recent agent-linked commits. |

## Development Workflow

### Narrow-to-Broad Exploration
Use the pyramid approach — start cheap, go deep only when needed:

1. **`codebase_peek`** — find where code lives (metadata only, fast)
2. **`glob` / `grep`** — narrow to specific files by name or pattern
3. **`read`** — examine the actual code in context
4. **`codebase_search`** — full semantic search for implementation details (expensive, use last)

### Mandatory Workflow for EVERY Task
1. **RECALL CONTEXT** → `skill` (recall) or `skill` (handoff) to load past decisions
2. **INDEX** → `index_codebase` if code has changed since last session
3. **PLAN** → `sequential-thinking_sequentialthinking` for non-trivial problems
4. **SEARCH** → `codebase_peek` first, then `codebase_search` if needed
5. **EXECUTE** → file tools + `bash` for code changes
6. **VERIFY** → run lint, typecheck, and tests
7. **SAVE** → `skill` (remember) to persist key decisions/learnings

### When to Use Each Search Tool
```
Need to find something by intent/meaning?
  → codebase_peek (fast, metadata only)
  → Not enough? codebase_search (full code)

Need to find a definition?
  → implementation_lookup

Need to understand callers/dependencies?
  → call_graph

Need to find duplicate code?
  → find_similar

Need to find files by name/pattern?
  → glob (then grep if needed)

Need to search for a regex in files?
  → grep
```

### Parallel Execution
- Batch independent tool calls in a single response (e.g., `bash git status` + `bash git diff` simultaneously).
- Use `task` to launch multiple sub-agents in parallel for independent research.

### Iterative Verification
- Always run project-specific linting after code changes (e.g., `npm run lint`).
- Run existing tests before and after changes to ensure no regressions.
- Never commit unless explicitly asked.

### Memory Rules
- ALWAYS recall past context before starting a task (`skill` recall)
- ALWAYS save key decisions after completing significant work (`skill` remember)
- Use `codebase_peek` to find code by intent before using `grep`/`glob`

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
