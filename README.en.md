# OpenShorts.app

[Deutsch](README.md) | **English**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Free & open source AI video platform** with 3 tools in one: **Clip Generator**, **AI Shorts (UGC videos with AI actors)**, and **YouTube Studio**. Self-hosted — your data, your keys, your rules. Fully configurable watermarking (or none at all), no artificial limits.

![OpenShorts Demo](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

### Video Tutorial: How it works
[![OpenShorts Tutorial](https://img.youtube.com/vi/xlyjD1qCaX0/maxresdefault.jpg)](https://www.youtube.com/watch?v=xlyjD1qCaX0 "Click to watch the video on YouTube")

*Click the image above to watch the full walkthrough.*

---

## 3 Tools in 1 Platform

### 1. Clip Generator
Turn long YouTube videos or local uploads into viral-ready shorts for TikTok, Instagram Reels, and YouTube Shorts — in the format you choose: **Auto (smart)**, **9:16**, **16:9**, or **1:1**.

![Clip Generator](screenshots/clip-generator.png)

![Clip Results](screenshots/clip-results.png)

### 2. AI Shorts (UGC Video Creator)
Generate marketing videos with AI actors for **any product or business**. No camera, no studio, no influencer budget. Just describe your product or paste a URL.

![AI Shorts Setup](screenshots/ai-shorts.png)

- **Two cost modes**: Low Cost (~$0.65/video) and Premium (~$2/video)
- Works for any business: SaaS, restaurants, e-commerce, coaching, local businesses
- AI-generated actors with lip-sync, voiceover, b-roll, and TikTok-style subtitles
- Choose from a shared avatar gallery or upload your own photo
- Publish directly to TikTok, Instagram, and YouTube

### 3. YouTube Studio
Complete free AI YouTube toolkit: thumbnails, titles, descriptions, and direct publishing.

![YouTube Studio](screenshots/youtube-studio.png)

- AI thumbnail generator with face overlay
- 10 viral title suggestions with refinement chat
- Auto-generated descriptions with chapter timestamps
- One-click publish to YouTube

### UGC Video Gallery
All generated videos and avatars are saved to a public gallery with SEO pages for each video.

![UGC Gallery](screenshots/ugc-gallery.png)

- Public gallery page with hover-to-play (`/gallery`)
- Individual SEO video pages with og:video meta tags (`/video/{id}`)
- JSON-LD structured data for search engines
- Avatar gallery with prompt history

---

## Key Features

### Clip Generator
- **Viral Moment Detection**: two-stage Gemini analysis (score → detail) finds 3–15 high-potential moments, with word-accurate cut snapping and per-task model selection via `.env`
- **Output Formats**: Auto (smart source detection), 9:16, 16:9 original, or 1:1 square — chosen per job, like Opus Clip
- **Smart Reframing**: dual-mode AI cropping — TRACK mode (MediaPipe + YOLOv8 speaker tracking with "heavy tripod" stabilization) and GENERAL mode (blurred background); sources already in the target aspect are passed through untouched
- **Karaoke Subtitles**: word-level highlighting, 11 preset looks (TikTok, Gold Glow, Neon, Beast …), glow/pop/box effects, bulk apply to all clips, ZIP download
- **Auto-Edit v2**: Gemini plans an edit decision list (zooms, punch-ins, color pops, B&W moments, flashes, vignettes); a deterministic builder renders it with hard safety limits — burned-in captions are never cropped by zooms
- **Watermark**: subtle centered watermark on every clip (your text, your logo, or off) so nobody re-uploads your work as theirs
- **Quality Gate**: pre-flight probe warns you *before* processing if YouTube only offers low resolution, with cookie-refresh instructions
- **Self-learning ETA**: realistic time estimates calibrated to your machine, with an upfront total estimate per job
- **AI Voice Dubbing**: ElevenLabs integration for 30+ languages with voice cloning
- **Hook Text Overlays**: AI-generated attention-grabbing text overlays with emoji support

### AI Shorts Pipeline
1. **Analyze**: Scrape website URL + web research, or generate from manual description
2. **Script**: AI writes viral scripts (hook - problem - solution - CTA format)
3. **Actor**: Generate AI actors with Flux 2 Pro or select from shared gallery
4. **Voice**: ElevenLabs TTS voiceover (English/Spanish, male/female)
5. **Video**: Talking head generation (Hailuo 2.3 Fast img2video + VEED Lipsync)
6. **B-roll**: AI-generated visuals with Ken Burns effect
7. **Composite**: FFmpeg final assembly with subtitles and hook overlays
8. **Publish**: Direct posting to TikTok, Instagram Reels, YouTube Shorts via Upload-Post

### YouTube Studio
- AI-powered title generation with 10 viral options
- Interactive refinement chat for titles
- AI thumbnail generation with custom face + background
- Auto descriptions with chapter timestamps from Whisper transcript
- Direct YouTube publishing via Upload-Post

### Social Publishing
- One-click posting to TikTok, Instagram Reels, and YouTube Shorts
- Schedule posts for later
- Upload-Post integration with async uploads

### Infrastructure
- S3 cloud backup (private bucket for clips, public bucket for gallery/avatars)
- SEO gallery pages served by FastAPI with JSON-LD structured data
- Async job queue with concurrency control, clean cancel, crash resume, keepalive heartbeat and Windows standby prevention
- 63 automated tests + GitHub Actions CI (backend, frontend, Docker)

---

## Requirements

- **Google Gemini API Key** ([Free — get it here](https://aistudio.google.com/app/apikey)) — required for all AI features
- **fal.ai API Key** ([Pay-per-use](https://fal.ai)) — required for AI Shorts (actor generation, video, lip-sync)
- **ElevenLabs API Key** ([Free tier](https://elevenlabs.io)) — required for voiceover/dubbing
- **Upload-Post API Key** (Optional, [free tier](https://upload-post.com)) — for direct social posting
- **Docker & Docker Compose** — or Python 3.11+ / Node 18+ / FFmpeg for a local install

---

## Getting Started

### 1. Clone
```bash
git clone https://github.com/Themegaindex/openshorts.git
cd openshorts
```

### 2. Configure (optional)
```bash
cp .env.example .env
# Edit .env: AWS keys for S3 backup, Gemini models, watermark, quality gate …
```

### 3. Launch

**Option A — Docker:**
```bash
docker compose up --build
```

**Option B — Local without Docker (Windows):**
```bash
pip install -r requirements.txt
cd dashboard && npm install && cd ..
start.bat
```
(Linux/macOS: run `uvicorn app:app --host 0.0.0.0 --port 8000` and `npm run dev` in `dashboard/`.)

### 4. Open Dashboard
Navigate to **`http://localhost:5175`**

1. Go to **Settings** and enter your API keys (Gemini, fal.ai, ElevenLabs, Upload-Post)
2. **Clip Generator**: Paste a YouTube URL or upload a video, pick an output format, generate viral shorts
3. **AI Shorts**: Describe your product or paste a URL to generate UGC marketing videos
4. **YouTube Studio**: Generate thumbnails, titles, and descriptions for YouTube
5. **UGC Gallery**: Browse all generated videos and avatars

---

## Technical Pipeline

### Clip Generator
1. **Ingest** — YouTube download (yt-dlp, HD-capable default clients) or local upload
2. **Quality Gate** — pre-flight resolution probe with user confirmation on low quality
3. **Transcribe** — faster-whisper with word-level timestamps
4. **Detect** — PySceneDetect for scene boundaries
5. **Analyze** — two-stage Gemini analysis identifies 3–15 viral moments (15–60s each)
6. **Extract** — FFmpeg precise clip cutting
7. **Reframe** — format-aware rendering: AI vertical/square cropping with subject tracking, or original passthrough; watermark blended in-frame
8. **Effects** — karaoke subtitles, hooks, Auto-Edit v2
9. **Publish** — S3 backup + Upload-Post social distribution

### AI Shorts
1. **Analyze** — Website scraping + Gemini web research (or manual description)
2. **Script** — Gemini generates viral scripts with segments
3. **Actor** — Flux 2 Pro portrait generation (or gallery/upload)
4. **Voice** — ElevenLabs TTS voiceover
5. **Video** — Hailuo 2.3 Fast img2video + VEED Lipsync (Low Cost) or Kling Avatar v2 (Premium)
6. **B-roll** — Flux 2 Pro image generation + Ken Burns effect
7. **Composite** — FFmpeg assembly with ASS subtitles and hook overlays
8. **Gallery** — Upload to public S3 with metadata for SEO pages
9. **Publish** — Upload-Post to TikTok, Instagram, YouTube

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, google-genai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg, httpx |
| Frontend | React 18, Vite 4, Tailwind CSS 3.4 |
| AI APIs | Google Gemini, fal.ai (Flux, Hailuo, VEED, Kling), ElevenLabs |
| Infrastructure | Docker + Docker Compose, AWS S3, GitHub Actions CI |
| Publishing | Upload-Post API (TikTok, Instagram, YouTube) |

---

## Environment Variables

**Server-side (.env)** — see `.env.example` for the full annotated list:
| Variable | Description |
|----------|------------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` / `AWS_S3_BUCKET` / `AWS_S3_PUBLIC_BUCKET` | S3 backup & public gallery |
| `MAX_CONCURRENT_JOBS` | Concurrent processing limit (default: 5) |
| `GEMINI_MODEL` / `GEMINI_MODEL_ANALYSIS` / `GEMINI_MODEL_EDITOR` / … | Gemini model per task |
| `GEMINI_THINKING_SCORE` | Optional thinking budget for clip scoring |
| `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE` | Transcription quality/speed |
| `QUALITY_GATE_MIN_HEIGHT` | Ask before processing below this height (default: 720) |
| `WATERMARK_ENABLED` / `WATERMARK_TEXT` / `WATERMARK_IMAGE` / `WATERMARK_OPACITY` | Watermark configuration |

**Client-side (encrypted in localStorage):**
| Key | Description |
|-----|------------|
| `GEMINI_API_KEY` | Google Gemini — required |
| `FAL_KEY` | fal.ai — required for AI Shorts |
| `ELEVENLABS_API_KEY` | ElevenLabs — required for voiceover/dubbing |
| `UPLOAD_POST_API_KEY` | Upload-Post — optional, for social posting |

---

## Security & Performance

- **Non-Root Execution**: Containers run as dedicated `appuser`
- **Concurrency Control**: Semaphore-based job queue (`MAX_CONCURRENT_JOBS`)
- **Auto-Cleanup**: Automatic purging of old jobs (1h retention)
- **Encrypted Keys**: API keys encrypted client-side, never stored server-side
- **Injection Hardening**: fonts, colors and numbers sanitized before entering ASS/FFmpeg
- **Upload Validation**: Image uploads validated for format and minimum size
- **File Limits**: 2GB upload limit protection
- **Fast Frontend**: code splitting, GZip API responses, faststart MP4s

---

## Social Media Setup (Upload-Post)

1. **Register**: [app.upload-post.com/login](https://app.upload-post.com/login)
2. **Create Profile**: Go to [Manage Users](https://app.upload-post.com/manage-users)
3. **Connect Accounts**: Link TikTok, Instagram, and/or YouTube
4. **Get API Key**: Navigate to [API Keys](https://app.upload-post.com/api-keys)
5. **Use in OpenShorts**: Paste the key in Settings

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

## Contributions

Contributions are welcome! Whether it's adding new AI models, improving the lip-sync pipeline, or building new features — feel free to open a PR.

## License

MIT License. OpenShorts is yours to use, modify, and scale.
