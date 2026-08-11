# Deploying OpenShorts+ for $0

The zero-budget edition is designed to run on free hosting. Two pieces:

| Piece | What it is | Where it runs free |
|---|---|---|
| **Dashboard** | React/Vite frontend (this folder) | Vercel / Netlify / Cloudflare Pages / GitHub Pages |
| **Backend** | FastAPI + ffmpeg + Whisper + yt-dlp | Render (free), Fly.io (free tier), Railway (trial), Oracle Cloud Always-Free VM, or your own machine |

> **Why not Vercel for the backend?** The backend shells out to `ffmpeg`,
> `yt-dlp` and downloads Whisper models (~150MB+). Serverless functions time
> out and have a 250MB limit. So: Vercel serves the UI, a small always-on host
> runs the API. That combo is still $0/month.

---

## 1. Backend — Render (free tier, simplest)

1. Fork/push this repo to GitHub.
2. On [render.com](https://render.com) → **New → Web Service** → pick the repo.
3. Settings:
   - **Root directory:** leave empty
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** Free
4. Add a **Disk** (Render free plans get 1GB) mounted at `/opt/data` and set:
   - `UPLOAD_DIR=/opt/data/uploads`
   - `OUTPUT_DIR=/opt/data/output`
5. Add at least one **free AI key** env var (see below). Save → deploy.

Render free instances sleep after 15 min of inactivity. The first request after
a sleep takes ~30s to wake. Fine for personal use.

## 2. Dashboard — Vercel (free)

1. In Vercel: **Add New → Project** → import this repo → set **Root
   Directory** to `dashboard`.
2. Build settings auto-detect Vite. Add the env var:
   - `VITE_API_URL=https://your-backend.onrender.com` (the Render URL from step 1)
3. Deploy. The dashboard calls the backend directly (CORS is enabled on the
   backend for this origin — add your domain to `ALLOWED_ORIGINS` on Render if
   you see CORS errors).

## 3. Other free options

- **Fly.io** — free allowance per month; `fly launch` with the included
  `Dockerfile`, add a volume for `uploads/` and `output/`.
- **Railway** — trial credits; `railway up` with the `Dockerfile`.
- **Oracle Cloud Always-Free VM** (4 OCPU / 24GB RAM) — run the `Dockerfile`
  with `docker compose up`; this is the most powerful free option and never
  sleeps.
- **Coolify / your own VPS** — `docker compose up -d` works out of the box.

## Free AI providers — two ways to set them

**Way 1 — in the app UI (easiest, works from a phone):** after deploying, open
the dashboard → **Settings → Free AI keys (server)** → paste one or more keys →
**Save to server**. Keys are stored on the backend's disk (`DATA_DIR`), merged
into the environment immediately, and inherited by processing jobs — no env
edits, no redeploys. The same page sets the **default caption theme** for every
new clip.

**Way 2 — backend env vars (survive redeploys):** set **any one** of these on
Render/Vercel. The gateway uses everything you give it, with automatic
fallback — it also auto-fetches OpenRouter's catalog and uses **free models
only**, and puts rate-limited providers in cooldown so traffic shifts to the
next healthy key/model automatically. Keys set in the app UI take precedence
while they exist.

| Env var | Provider | Free tier |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter | many `:free` models, no credits needed |
| `GEMINI_API_KEY` | Google AI Studio | Gemini free tier (generous daily limits) |
| `GROQ_API_KEY` | Groq | llama-3.3-70b, fast |
| `DEEPSEEK_API_KEY` | DeepSeek | new-user credits |
| `ZHIPU_API_KEY` | Zhipu GLM | glm-4.5-air / glm-4-flash |
| `DASHSCOPE_API_KEY` | Alibaba Qwen | new-user free quota |
| `MOONSHOT_API_KEY` | Moonshot Kimi | free tier quota |

Optional: `EDGE_TTS_VOICE` (default `en-US-JennyNeural`) for free AI-Shorts
voiceovers. `UPLOAD_POST_API_KEY` (free tier) enables publishing to TikTok /
Instagram / YouTube.

## Sources: Kick, YouTube & beyond

The URL box accepts any site yt-dlp supports: **Kick** (live streams and VODs —
`https://kick.com/<channel>/videos/<id>`), **YouTube**, Twitch, TikTok,
Facebook, Vimeo, Dailymotion, Reddit and more. There is **no length limit** —
paste any multi-hour podcast VOD and clip it freely.

## Subtitles: themed by default

Every clip burns stylish captions automatically (karaoke word-highlight look).
Pick a **default theme** in Settings (TikTok, Reels, Shorts Pop, Gold Glow,
Neon, Cyber, Karaoke, Minimal, Beast, Boxed, Classic), and restyle any clip
individually from its **subtitles** modal with a live preview.

## Publish kits (manual posting, no auto-upload)

Every clip gets a **publish kit**: a viral title, a description, and hashtags
mixing niche tags with **today's trending hashtags** for a region of your
choice (US, IN, GB, BD, …). Trends refresh automatically every day — the
backend pulls free public trend data with an AI fallback, cached per day, so
hashtags stay current with zero maintenance.

Nothing is posted automatically (and the short-lived "direct YouTube upload"
feature was removed for the same reason): you play the clip, copy
title + description + hashtags, and paste them into YouTube/TikTok/Instagram
yourself.

## Text summaries

Any processed job can produce a **chaptered written digest** — timestamps, key
points, best quotes, clip hooks — via the **text summary** button on the
results screen. Copy it or download as `.md` for show notes / newsletters /
LinkedIn threads.

## What stays free forever

- Clip Generator: unlimited YouTube/Kick/upload processing — **no watermark, no
  limits, no 20-min/month quota**.
- YouTube Studio: titles, descriptions, thumbnails (free image gen or local
  typographic fallback).
- AI Shorts: free script generation, free Edge TTS voiceover, free Ken Burns
  motion. (Optional fal.ai/ElevenLabs keys upgrade quality but are never
  required.)
- Social publishing via Upload-Post's own free tier (your key, billed by them
  if you exceed it).
