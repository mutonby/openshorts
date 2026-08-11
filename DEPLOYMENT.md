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

## Free AI providers (backend env vars)

Set **any one** of these — the gateway uses everything you give it, with
automatic fallback:

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

## What stays free forever

- Clip Generator: unlimited YouTube/upload processing — **no watermark, no
  limits, no 20-min/month quota**.
- YouTube Studio: titles, descriptions, thumbnails (free image gen or local
  typographic fallback).
- AI Shorts: free script generation, free Edge TTS voiceover, free Ken Burns
  motion. (Optional fal.ai/ElevenLabs keys upgrade quality but are never
  required.)
- Social publishing via Upload-Post's own free tier (your key, billed by them
  if you exceed it).
