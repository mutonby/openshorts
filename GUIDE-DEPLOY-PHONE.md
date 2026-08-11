# 🚀 Deploy OpenShorts+ from your phone — step by step

> Written for someone brand new to this. You only need a phone browser — no
> computer, no terminal. Total time: ~20–30 minutes, all free.

## Before you start (5 min)

1. **GitHub account** — you already forked the repo, so you have one. ✅
2. **Vercel account** — download the Vercel app or open vercel.com, sign up
   with **Continue with GitHub** (free).
3. **Render account** — open render.com, sign up with **Continue with GitHub**
   (free).
4. Get one **free AI key** (any of these — OpenRouter is easiest):
   - **OpenRouter** → openrouter.ai/keys → “Create key” → copy `sk-or-v1-...`
     (free models need no credits). Our app auto-fetches OpenRouter's catalog
     and **uses only the FREE models** — paid ones are never touched.
   - or **Google AI Studio** → aistudio.google.com/apikey → `AIzaSy...`
   - Get as many as you like — the app tries them all and **switches
     automatically** when one is rate-limited.

---

## Step 1 — Put the new code on your `main` branch (2 min)

The work lives on the branch `arena/019feea4-openshort`. Merge it into
`main` with the dedicated step-by-step guide:

> 📖 **`GUIDE-BRANCH-MERGE.md`** — merge via the GitHub app or browser,
> verify, and what to do if something looks off.

Short version: GitHub app → **Branches** → `arena/019feea4-openshort` →
**New pull request** → base `main` ← compare `arena/019feea4-openshort` →
**Create pull request** → **Merge pull request** → **Confirm merge**.

Now your `main` has everything.

## Step 2 — Deploy the backend on Render (10 min)

1. On render.com → **New +** → **Web Service** → **Build and deploy from a Git
   repository** → pick `openshort-`.
2. Settings:
   - **Root directory:** leave empty
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** **Free**
3. Scroll down → **Add Disk** → name `data`, **Size 1 GB**, **Mount path**
   `/opt/data`. (This keeps your clips + saved keys.)
4. Click **Advanced** → **Add Environment Variable**:
   - `DATA_DIR` = `/opt/data`
   - *(optional)* `MAX_FILE_SIZE_MB` = `8192`
5. **Create Web Service.** The first build takes **5–15 minutes** (it installs
   Whisper/ffmpeg dependencies). Wait until the URL `https://yourapp.onrender.com`
   shows **Live**.
6. Note your backend URL — you'll need it in the next step. It looks like
   `https://openshorts-abc123.onrender.com`.

> 📱 You don't need to set AI keys here! You'll paste them in the app UI in
> Step 4 — works from your phone, no env edits.

## Step 3 — Deploy the dashboard on Vercel (5 min)

1. On vercel.com → **Add New…** → **Project** → import `openshort-`.
2. In **Root Directory** choose **`dashboard`**.
3. Click **Environment Variables** → add one:
   - `VITE_API_URL` = `https://yourapp.onrender.com` (your Step 2 URL, no
     trailing slash)
4. Click **Deploy**. In ~1 minute you get `https://yourproject.vercel.app`.
5. Open that URL — you should see the OpenShorts+ landing page.

## Step 4 — Add your AI keys inside the app (2 min, no env vars!)

1. In the app → **Settings** (gear icon, top right).
2. You'll see **“Free AI keys (server)”** with one empty row.
3. Paste your first **OpenRouter** key → tap **+ add new** → paste another
   key (Gemini/Groq/DeepSeek/GLM/Qwen/Kimi, or even a second OpenRouter key)
   — add as many rows as you want, like Vercel's env section.
4. Tap **Save to server** → it turns green: **Saved & active**.
5. The panel shows you:
   - **Auto-switching** status — if a provider gets rate-limited it's put in
     cooldown and the app routes around it to the next key/model automatically.
   - **OpenRouter free models** count — only free models are ever used.
   - **Fallback order** — the providers your app will try in order.
6. While you're there, pick a **default caption theme** (TikTok, Neon, Beast…)
   — every new clip burns that look automatically.

That's it — no Vercel/Render env edits for keys. ✅

## Step 5 — Test it (2 min)

1. In the app, switch to **Video URL** and paste any link:
   - a **Kick** VOD or live stream → `https://kick.com/<channel>/videos/<id>`
   - a **YouTube** video
   - Twitch / TikTok / Facebook / any site
2. Tick the ownership checkbox → **Generate clips**.
3. Wait for the job (5–8 min per 8-min video on a free CPU; free Render
   sleeps after inactivity, so the first request may wake it — give it 30s).
4. Every clip comes out **9:16 with stylish captions burned in**, no watermark.
5. Want a different subtitle look per clip? Open a clip → **subtitles** →
   pick a preset (TikTok, Reels, Gold Glow, Neon, Boxed…) → **apply to this
   clip** — live preview included.
6. **Text summary:** on the results screen, tap **text summary** — the app
   turns the whole video into a chaptered written digest (timestamps, key
   points, best quotes, clip hooks). Copy or download as .md for show notes,
   newsletters or LinkedIn threads.

## Step 6 — Publish kit (manual posting, YouTube-safe)

No auto-posting: you always review and post yourself (that's also the
YouTube-safe way).

1. On any clip, tap **publish kit**.
2. The app writes a **viral title**, a **description**, and **hashtags** for
   that clip — mixing relevant niche tags with **today's trending hashtags**
   for the region you pick (US, India, UK, Bangladesh, …).
3. The trend list **refreshes automatically every day** (live public trends,
   with an AI fallback), so the hashtags stay current with no work from you.
4. Edit anything you like, toggle hashtags on/off, then:
   - **copy title + description + hashtags** and paste into YouTube's upload
     page (hashtags go at the end of the description), or
   - **download .txt** and keep it for later.
5. Play the clip once more, post it manually. That's it — nothing is posted
   automatically, ever.

The old "post straight to my channel" auto-upload is removed on purpose:
posting into YouTube yourself keeps you in control of what goes on your
channel.

## Troubleshooting (quick)

| Problem | Fix |
|---|---|
| App loads but says “No AI provider configured” | Go to Settings → Free AI keys → save a key, or refresh the page |
| First job takes 30s+ before starting | Render free tier slept — normal, it wakes up |
| Clip download fails on a Kick URL | Make sure the stream is public (not subscriber-only); Kick live VODs need the `/videos/<id>` link |
| CORS error in the console | `VITE_API_URL` must be exactly your Render URL (no `/`, no trailing slash) |
| Want keys that survive every redeploy | Add the same keys as env vars on Render instead of (or in addition to) the app UI |

## Optional upgrades (still $0)

- Add more free keys (Groq, DeepSeek, GLM, Qwen, Kimi…) for extra fallback power.
- Upload-Post free tier (`UPLOAD_POST_API_KEY`) unlocks one-click publishing to
  TikTok/Instagram/YouTube.
- Attach the free Render disk (Step 2.3) before generating — your clips and
  saved settings persist across redeploys.
