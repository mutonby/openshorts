# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenShorts is an AI-powered vertical video generator that turns long YouTube videos or local uploads into short 9:16 clips for TikTok, Instagram Reels and YouTube Shorts. Google Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`, overridable with `GEMINI_MODEL`) picks the viral moments and writes the titles. It ships as self-host (BYOK) and as a Cloud edition (`BILLING_ENABLED`, code under `cloud/`).

## Development Commands

```bash
docker compose up --build        # full stack: backend :8000 (FastAPI), frontend :5175 (Vite proxies the API)

cd dashboard && npm install
npm run dev                      # dev server with HMR (:5173)
npm run build                    # production build (CI runs this)
npm run lint                     # ESLint, --max-warnings 0

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
pytest tests/ -v                 # CI runs this without the heavy ML stack
```

## Pipeline

Ingest (yt-dlp or upload) → transcription (word timestamps) → scene detection → Gemini picks moments (15-60 s) → FFmpeg cut → vertical reframing → effects/subtitles → hook overlay → optional ElevenLabs dubbing → S3/R2 backup → optional social posting via Upload-Post.

## Key Files

- `main.py`: core processing (transcription, scene detection, clip extraction, reframing). Runs as a child process per job.
- `app.py`: FastAPI server, job queue, REST endpoints, deploy handover/drain.
- `clip_selection.py`, `layout_picker.py`, `reframe_v2.py`, `split_layout.py`, `screencast_layout.py`, `camera_inset.py`: moment picking and layouts.
- `editor.py` (Gemini FFmpeg effects), `hooks.py` (hook overlays), `subtitles.py` (SRT/ASS, burning), `translate.py` (ElevenLabs dubbing), `thumbnail.py`.
- `mcp_server.py`, `mcp_stdio.py`: MCP over HTTP and stdio.
- `cloud/`: auth, API keys, metering, billing, lifecycle emails, account erasure, autopilot, proxy ledger, alerts.
- `dashboard/src/App.jsx`: main React component. `dashboard/vite-plugin-seo.js` + `dashboard/seo/`: build-time SEO pages, sitemap, llms.txt. `dashboard/seo/data.js`: single source of truth for pricing/pipeline/competitor facts.

## Subsystem reference (read before touching)

- SEO pages, free tools, attribution → `docs/architecture/seo-and-tools.md`
- Layout picker, reframing modes, hook grounding → `docs/architecture/layouts-and-reframing.md`
- Clip count, silent-footage fallback, local LLM → `docs/architecture/clip-selection.md`
- Thumbnail Studio → `docs/architecture/thumbnails.md`
- API endpoints, API keys, MCP, OAuth, webhooks → `docs/architecture/agent-access.md`
- Autopilot → `docs/architecture/autopilot.md`
- Free-plan watermark (served `wm_` copy, unmark on upgrade) → `docs/architecture/watermark.md`
- Quota wall, free overflow, lifecycle emails, cancel flow, survey, GDPR erasure → `docs/architecture/billing-and-account.md`
- Concurrency/VRAM, auto-retry, deploy handover and drain → `docs/architecture/jobs-and-deploys.md`
- Download routing, yt-dlp clients, paid proxy ledger → `docs/architecture/proxy-routing.md`

## Hard rules

**SEO / site**
- Keep `seo/landing-fallback.js` in sync with `Landing.jsx`. Never add a static `public/sitemap.xml` back: `vite-plugin-seo.js` generates it.
- Standalone pages are emitted as flat `.html` files (directories make nginx 301 and break canonicals).
- When editing pricing anywhere, edit `seo/data.js` too. Never say "OpenShorts is free" without the Cloud price next to it.
- `/gta-5-clips` states the silent-footage thresholds: change `dashboard/seo/pages.js` if that behaviour changes.

**Free tools**
- The youtube-transcript tool uses `STATIC_PROXY_URLS` (or direct) only, **never** `PROXY_URL`. A static route returning zero formats and zero captions is degraded, not "no captions".

**Layouts and rendering**
- `layout_picker` sends 12 frames at 1024px and asks for a closed choice, never the video and never a measurement.
- `layout_picker.apply()` only adds options: an explicit user choice is never turned off by the model.
- A source that is already vertical skips the picker and classifier and renders TRACK.
- SPLIT captions need ASS with `{\an5}` per word event on the seam; SRT cannot do per-stretch alignment. Preserve `layout_ranges` through recut (`layout_ranges.remap`).
- Mouth activity is normalised per speaker before comparing (`normalise_activity`).
- Canonical files are always clean; the free-plan mark exists only on the served `wm_` copy (`main.mark_delivery` / `app._deliver`). While a job has `.marked`, `/videos` refuses its clean deliverables.

**Clip selection**
- The scoring pass ranks every window (global top N), never "choose up to K per batch". The `min_clips` floor is enforced in code.
- `get_visual_clips` is the only stage that sends Gemini the video; nothing guards its length.

**Jobs, GPU, deploys**
- Call `transcribe_backends.release_models()` after any ASR (job processes and in-API transcriptions alike). Size `MAX_CONCURRENT_JOBS` by free VRAM first.
- `DRAIN_TIMEOUT_SECONDS` must stay below the deployment's (Coolify) stop grace period. The image must ship `curl` for the platform health check.
- A drain-cancelled job keeps its manifest and reservation and is resumed, never marked failed or completed.

**Download proxies**
- Route order is direct → static → paid (per-GB `PROXY_URL`). Never use the paid proxy for non-YouTube URLs, nor for failures another IP cannot fix (`static_failure_warrants_paid`).
- The probe also carries `YOUTUBE_COOKIES`, like the download does; first attempt with cookies, second anonymous, on probe and download alike.
- Keep the explicit `yt_clients.py` client list; never put `player_skip: webpage` back. `noplaylist` always; non-video URLs refused by an allowlist.
- `DOWNLOAD_SKIP_STATICS` only when the probe saw every static bot-checked. `PAID_PROXY_DAILY_MB` is the hard daily ceiling. Never keep `PROXY_URL` in a local `.env`.

**Agent access**
- MCP stdio: swap `sys.stdout` for stderr **before importing `app`**, and enter the app lifespan.
- Webhook URLs go through `security_utils.assert_public_url` at submit AND at delivery.
- API-key auth can never manage keys or delete the account. OAuth tokens are ordinary `osk_` keys.

**Billing, account, privacy**
- GDPR erasure order is fixed: Stripe cancel (abort on failure) → R2 purge → one DB transaction over `USER_OWNED_TABLES`. Never via API key. Every new table referencing `users.id` joins `USER_OWNED_TABLES` (a test enforces it).
- Coupon ids and promotion codes live only in env / Stripe, never in this repo.
- Cancel and churn alerts carry the reason and rating, never the user's free text. Deletion reasons are a closed list.
- `reserve_process_minutes` never reserves more than the balance, whatever the client asks; the source cut travels in the resume manifest.
- Lifecycle/commercial emails go through `emails.send_commercial_email` (opt-out + unsubscribe headers), once per account per kind.

**Public repo**
- This repository is public: no hosts, IPs, server paths, production metrics, business figures or customer data in code, comments, docs or commit messages.

## CI: a green pipeline closes the task, not the push

- Before pushing, run locally what CI runs: `pytest tests/ -v` and `npm run build` (+ `npm run lint`) in `dashboard/`.
- After every `git push`, wait for the commit's workflow and confirm it is green: `ci-wait` if available, or `gh run watch $(gh run list -c $(git rev-parse HEAD) -L1 --json databaseId -q ".[0].databaseId") --exit-status`.
- If it is red: fix, push, check again. Never report the task as done with a red CI.
- Every push to `main` redeploys; batch small commits (tests, docs) with the next real change when possible.
