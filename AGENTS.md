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

### Codebase Knowledge Graph (codebase-memory-mcp)

This project uses codebase-memory-mcp to maintain a knowledge graph of the codebase.
ALWAYS prefer MCP graph tools over grep/glob/file-search for code discovery.

#### Priority Order
1. `search_graph` — find functions, classes, routes, variables by BM25 / name pattern / semantic query
2. `trace_path` — trace who calls a function or what it calls (modes: calls / data_flow / cross_service)
3. `get_code_snippet` — read specific function/class source code with complexity metrics
4. `query_graph` — run Cypher queries for complex multi-hop patterns
5. `get_architecture` — high-level project summary (layers, clusters, boundaries, hotspots, packages)

#### When to fall back to grep/glob
- Searching for string literals, error messages, config values
- Searching non-code files (Dockerfiles, shell scripts, configs)
- When MCP tools return insufficient results

#### Quick Reference
| Tool | What it replaces from `opencode-codebase-index` | Use case |
|------|------|----------|
| `search_graph` | `codebase_peek` + `codebase_search` | Find anything: BM25 full-text, regex name patterns, semantic vector search |
| `search_code` | `codebase_search` (grep) | Graph-augmented grep over indexed files only (mode: compact/full/files) |
| `get_code_snippet` | `implementation_lookup` | Read function/class/method source + complexity metrics |
| `trace_path` | `call_graph` | Trace callers/callees with depth control, data flow, cross-service |
| `query_graph` | `find_similar` | Cypher: `MATCH (f:Function)-[:CALLS]->(g) WHERE f.name = 'main' RETURN g.name` |
| `get_architecture` | — (new) | Architecture overview: packages, routes, hotspots, layers, Louvain clusters |
| `index_repository` | `index_codebase` | Index project (mode: full/moderate/fast) |
| `index_status` | `index_status` | Check node/edge count and ready status |
| `detect_changes` | `index_health_check` | Map uncommitted changes to affected symbols with risk classification |
| `manage_adr` | — (new) | Persist architectural decisions across sessions |

### Codebase Indexing
| Tool | When to Use |
|------|------------|
| `index_repository` | Run at the start of a session if the codebase has changed. Creates/updates graph with full/moderate/fast modes. Incremental — re-indexes only changed files. |
| `index_status` | Check if codebase is indexed, how many nodes/edges exist. |
| `detect_changes` | Map uncommitted changes to affected symbols with risk classification. |
| `manage_adr` | Persist architectural decisions across sessions. |

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

1. **`search_graph`** — find where code lives (metadata only, fast)
2. **`glob` / `grep`** — narrow to specific files by name or pattern
3. **`read`** — examine the actual code in context
4. **`search_code`** — graph-augmented grep for implementation details (expensive, use last)

### Mandatory Workflow for EVERY Task
1. **RECALL** → `skill` (recall) scoped to the task at hand (not blanket dump)
2. **PRE-FLIGHT** → `index_repository` (always run, fast incremental check)
3. **CLARIFY** → `question` if intent is ambiguous; skip if clear
4. **SCOPE** → `search_graph` to find relevant files (metadata only, ~90% token savings vs full search)
5. **DEEP DIVE** → `read` + `search_code` / `get_code_snippet` only on files identified in step 4
6. **PLAN** → `sequential-thinking_sequentialthinking` for non-trivial problems
7. **EXECUTE** → file tools + `bash` for code changes
8. **VERIFY** → run lint, typecheck, and tests
9. **SAVE** → `skill` (remember) to persist key decisions/learnings

**When to go deep vs. abort early:**
- **Always do steps 1-2** (Recall + Index) — cheap, high-value context
- **Steps 3-9 scale with complexity** — simple questions ("what port is the server on?") can abort after step 2; complex features ("implement clip generator reframing") go all the way
- If unsure, err on the side of going deeper

### When to Use Each Search Tool (codebase-memory-mcp)
```
Need to find something by intent/meaning?
  → search_graph (BM25 + semantic, metadata fast)
  → Not enough? search_code (graph-augmented grep, full code)

Need to find a definition/source?
  → search_graph (find qualified_name) → get_code_snippet

Need to understand callers/dependencies?
  → trace_path (mode: calls / data_flow / cross_service)

Need to find similar/duplicate code?
  → search_graph with semantic_query (array of keywords)

Need complex multi-hop analysis?
  → query_graph (Cypher)

Need a high-level project overview?
  → get_architecture (layers, clusters, routes, hotspots)

Need to find files by name/pattern?
  → glob (then grep if needed)

Need to search for a regex in files?
  → search_code or grep
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
- Use `search_graph` to find code by intent before using `grep`/`glob`

## Environment Variables

### Server-side (`.env`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET`: S3 backup configuration.
- `CLOUDFLARE_R2_ACCOUNT_ID`, `CLOUDFLARE_R2_ACCESS_KEY_ID`, `CLOUDFLARE_R2_SECRET_ACCESS_KEY`, `CLOUDFLARE_R2_BUCKET_NAME`, `CLOUDFLARE_R2_PUBLIC_URL`: Cloudflare R2 storage for Buffer video uploads.
- `MAX_CONCURRENT_JOBS`: Max parallel processing tasks (default: 5).
- `VITE_API_URL`: Override for production API URL.

### Client-side (Encrypted in LocalStorage)
- `GEMINI_API_KEY`: Google Gemini API key (Required).
- `FAL_KEY`: fal.ai API key (Required for AI Shorts).
- `ELEVENLABS_API_KEY`: ElevenLabs API key (Optional).
- `UPLOAD_POST_API_KEY`: Upload-Post API key (Optional).
- `BUFFER_API_KEY`: Buffer API key (Optional, for Buffer publishing).
