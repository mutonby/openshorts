# Open-Dispatch

> **One API to dispatch your content anywhere.**
> Twitter / X · Instagram · Telegram · Bluesky · LinkedIn · Threads · YouTube Shorts
> — self-host forever free · MIT license

<p align="center">
  <a href="https://github.com/Matthew-Selvam/Open-Dispatch/stargazers"><img src="https://img.shields.io/github/stars/Matthew-Selvam/Open-Dispatch?style=for-the-badge&logo=github&color=4ade80&labelColor=141418" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-4ade80?style=for-the-badge&labelColor=141418" alt="MIT License"></a>
  <a href="https://github.com/Matthew-Selvam/Open-Dispatch/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Matthew-Selvam/Open-Dispatch/ci.yml?style=for-the-badge&logo=github&label=CI&labelColor=141418" alt="CI status"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=141418" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white&labelColor=141418" alt="Docker ready">
</p>

Open-Dispatch is the **infrastructure layer** for content distribution. Like Stripe is to payments, this is to posting. One HTTP call, ten platforms, zero vendor lock-in. Integrate from any app, n8n workflow, cron job, or AI agent.

<p align="center">
  <!-- Record a 30s demo (see docs/DEMO_SCRIPT.md) and drop it here as docs/demo.gif -->
  <img src="docs/demo.gif" alt="Open-Dispatch — compose once, dispatch to every platform" width="760">
</p>

> ⭐ **If self-hosting your social posting sounds good, [star the repo](https://github.com/Matthew-Selvam/Open-Dispatch) to follow development** — it's the fastest way to support an independent MIT project.

```bash
curl -X POST http://localhost:8000/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "targets": ["twitter", "bluesky", "telegram", "threads"],
    "formats": {
      "twitter_thread":   {"tweets": ["shipped v0.4", "thanks for reading"]},
      "bluesky_post":     {"text": "shipped v0.4"},
      "telegram_message": {"text": "shipped v0.4"},
      "threads_post":     {"text": "shipped v0.4"}
    }
  }'
```

Or use the built-in dark dashboard at `http://localhost:8000/` — compose, schedule, retry, watch the queue live.

---

## Why self-host?

Hosted scheduling services charge **$3–6 per connected account**, gate analytics and inbox management behind paid tiers, and keep your OAuth tokens on their servers. Open-Dispatch runs on your own machine, stores nothing externally, and costs nothing beyond electricity.

| | Hosted tools | Open-Dispatch |
|---|---|---|
| Pricing | Per-account fees that stack up | Self-host free — $0 forever |
| Credentials | Live on their servers | Never leave your machine |
| Errors | Fail silently or stay vague | Visible + retryable in the dashboard |
| Inbox / analytics / X | Extra paid add-ons | All platforms included, no tiers |
| Your data | Lingers after you cancel | You own the queue end-to-end |
| Lock-in | Proprietary | MIT licensed — fork it any time |

---

## Install

Five ways to run Open-Dispatch. Pick what fits your stack.

| Method | Setup | Needs | Best for |
|---|---|---|---|
| 🍺 Homebrew | ~2 min | macOS + Homebrew | macOS devs, auto-start at login |
| ⬇️ install.sh | ~90s | bash + Python 3.11+ | Linux servers, scripts, CI |
| 🐳 Docker | ~60s | Docker | Self-hosters, Linux VMs, zero Python |
| 🐍 pip | instant | Python 3.11+ | Python devs, embedding in apps |
| 🍎 macOS App | 10s | macOS 13+ | Non-technical users, menubar control |

---

### 🍺 Homebrew (macOS)

```bash
# Add the tap (formula lives in the main repo)
brew tap matthew-selvam/open-dispatch \
  https://github.com/Matthew-Selvam/Open-Dispatch
brew install open-dispatch

# Set up credentials
$EDITOR ~/.open-dispatch/.env

# Start the server (foreground)
open-dispatch

# Or run as a background service — auto-starts on login
brew services start open-dispatch
```

Installs three commands: `dispatch` (CLI), `open-dispatch` (server), `open-dispatch-worker` (background worker).
`brew services` wires a launchd plist so the server starts at login automatically.

---

### ⬇️ install.sh (macOS & Linux)

```bash
curl -fsSL \
  https://raw.githubusercontent.com/Matthew-Selvam/Open-Dispatch/main/install.sh \
  | bash

# Set up credentials
$EDITOR ~/.open-dispatch/.env

# Start the server
open-dispatch

# macOS: start at login via launchd
launchctl load ~/Library/LaunchAgents/dev.open-dispatch.plist

# Linux: start via systemd user session
systemctl --user enable --now open-dispatch
```

The script auto-detects your OS, finds Python 3.11+, creates an isolated virtualenv, and writes
a launchd plist (macOS) or systemd user unit (Linux). No sudo required.

Advanced usage:

```bash
bash install.sh \
  --prefix   /opt/open-dispatch \   # install location (default: ~/.local)
  --version  v0.4.0 \               # specific tag (default: main)
  --data-dir /var/open-dispatch \   # data / .env location
  --no-service                      # skip auto-start setup
```

---

### 🐳 Docker Compose (zero Python required)

```bash
git clone https://github.com/Matthew-Selvam/Open-Dispatch
cd Open-Dispatch
cp .env.example .env   # fill in your platform credentials
docker compose up -d

# Verify
curl http://localhost:8000/healthz   # → {"status":"ok"}

# Open the dashboard
open http://localhost:8000

# Optional: add Redis for multi-worker throughput
docker compose --profile redis up -d
```

Multi-arch image (amd64 + arm64). Non-root `dispatch` user. No compiler tools in the final layer.

---

### 🐍 pip (Python 3.11+)

```bash
# Install from GitHub (PyPI publish pending)
pip install git+https://github.com/Matthew-Selvam/Open-Dispatch.git

# Optional extras
pip install "open-dispatch[redis]"     # Redis queue backend
pip install "open-dispatch[postgres]"  # Postgres queue backend

# Start the API + UI
uvicorn api.app:app --reload

# Worker (separate terminal)
python -m scheduler.worker

# CLI
dispatch send --platforms bluesky --text "hello world"
```

Best for Python developers who want to embed Open-Dispatch in an existing codebase.

---

### 🍎 macOS menubar app

1. Download `Open-Dispatch-0.4.0.dmg` from [GitHub Releases](https://github.com/Matthew-Selvam/Open-Dispatch/releases)
2. Drag `Open-Dispatch.app` to `/Applications`
3. **First launch:** right-click → **Open** → click **Open** in the dialog (unsigned build — bypasses Gatekeeper once)
4. A status icon appears in your menu bar
5. Click → **Edit .env** → add your platform credentials
6. Click → **Start Server**
7. Click → **Open Dashboard** → `http://localhost:8000`

> **Apple Silicon & Intel** — this release is native arm64. Rosetta 2 runs it on Intel Macs without any extra steps.

SwiftUI menubar app (macOS 13+). Bundles the Python server — no separate Python install. Supports launch-at-login via `SMAppService`.

Build it yourself:

```bash
bash scripts/make-dmg.sh
# → dist/Open-Dispatch-0.4.0.dmg

# With code-signing + notarization:
bash scripts/make-dmg.sh \
  --sign "Developer ID Application: Your Name (TEAMID)" \
  --notarize
```

---

## Configuration

All config lives in `.env` (or `~/.open-dispatch/.env` for Homebrew / install.sh installs).

```bash
cp .env.example .env
$EDITOR .env
```

**Where do I get the API keys?** [`CREDENTIALS_GUIDE.html`](CREDENTIALS_GUIDE.html) walks through all ten platforms one at a time, with the scope each one needs and what to paste where. Open it in a browser. [`.env.example`](.env.example) is the same information as a copy-paste template: Telegram, Twitter / X, Instagram (Graph API, needs public media URLs), Bluesky (App Password, not your login password), LinkedIn (3-legged OAuth with `w_member_social`), Threads (Meta Graph API), YouTube Shorts (Data API v3), TikTok (Content Posting API v2), Facebook (Meta Graph API, Page token), and Discord (webhook, no bot required).

Only fill in the platforms you actually dispatch to. The API does not complain about targets whose env vars are unset.

```env
# ── Twitter / X ──────────────────────────────────────────────────────────────
TWITTER_API_KEY=...
TWITTER_API_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_SECRET=...

# ── Bluesky ──────────────────────────────────────────────────────────────────
BLUESKY_HANDLE=user.bsky.social
BLUESKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=123456:ABC-...
TELEGRAM_CHAT_ID=@channel_or_id

# ── Instagram / Threads / LinkedIn / YouTube: see .env.example for full list ─

# ── Queue backend (optional) ──────────────────────────────────────────────────
# REDIS_URL=redis://localhost:6379        # Redis backend
# DATABASE_URL=postgresql://...          # Postgres backend

# ── API access control (optional, but read this) ───────────────────────────
# When set, EVERY route except /healthz requires `Authorization: Bearer <token>`.
# When unset, the API is completely open — fine for 127.0.0.1, NOT fine if the
# container port is published to a public interface. The Docker image binds
# 0.0.0.0:8000, so set this before exposing the port.
# OPEN_DISPATCH_API_TOKEN=change-me-to-a-long-random-string

# ── AI caption adapter (optional) ────────────────────────────────────────────
# OPENROUTER_API_KEY=...
# OLLAMA_HOST=http://localhost:11434
```

Only configure the platforms you actually use — missing vars are silently skipped.

### API access control

`OPEN_DISPATCH_API_TOKEN` is **off by default**. With it unset, anyone who can
reach the port can dispatch content to your connected accounts, so treat it as
required for anything beyond a local, firewalled bind.

```bash
# .env
OPEN_DISPATCH_API_TOKEN=$(openssl rand -hex 32)
```

With the token set, every route except `/healthz` (kept open for Docker health
checks and uptime monitors) requires:

```
Authorization: Bearer <your token>
```

Requests without a valid token get `401`.

All three bundled clients send it automatically once the variable is set in
> their environment:
>
> - **CLI** — reads `OPEN_DISPATCH_API_TOKEN`, or pass `--token <value>`
> - **MCP server** (`mcp_server.py`) — reads `OPEN_DISPATCH_API_TOKEN`
> - **n8n node** — paste it into the credential's **API Key Header** field
>
> A `401` or `403` from any of them names this variable rather than dumping a
> bare HTTP error.

State-changing requests (`POST`/`PUT`/`PATCH`/`DELETE`) are additionally checked
against the `Origin` header, and must be sent **without** an `Authorization`
header from a browser form — i.e. use the dashboard on the same origin.

> **Behind a reverse proxy:** the origin check compares against the request's own
> base URL. If your proxy rewrites `Host` or `X-Forwarded-*` in a way that
> doesn't match the browser's `Origin`, legitimate dashboard mutations will
> return `403`. Either serve on a single origin, or pass the original host
> through unchanged.

---

## Web UI

```
http://localhost:8000/         # dashboard — queue, filters, live auto-refresh
http://localhost:8000/compose  # post composer with AI caption adapt
http://localhost:8000/queue/<id>  # row detail + retry
```

Dark terminal aesthetic. HTMX-driven (no JS build step). 51 KB vendored `htmx.min.js`. Runs on the same port as the API.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness probe — JSON for monitors, HTML dashboard for browsers |
| POST | `/dispatch` | Enqueue a ContentUnit for one or many platforms |
| GET | `/queue?status=…` | List rows (`queued / publishing / published / failed / dead / canceled`) |
| GET | `/campaign/{unit_id}` | Show every platform row for one dispatch |
| POST | `/campaign/{unit_id}/cancel` | Cancel that campaign's queued rows |
| GET | `/queue/{id}` | One row — JSON or HTML (content-negotiated) |
| POST | `/queue/{id}/retry` | Reset an errored / dead row to `queued` |
| DELETE | `/queue/{id}` | Delete a single queue row permanently |
| POST | `/_purge?status=…` | Bulk-delete rows by status (`published` or `dead`) |
| POST | `/ai/adapt` | Rewrite a caption per target platform |
| POST | `/media/transcode` | Resize image to a platform spec |
| GET | `/media/specs` | List all 10 platform image specs |

### ContentUnit shape

```json
{
  "category": "general",
  "targets": ["twitter:work", "bluesky", "threads"],
  "scheduled_for": "2026-05-19T18:00:00+00:00",
  "formats": {
    "twitter_thread":   { "tweets": ["t1", "t2"], "media_paths": [] },
    "bluesky_post":     { "text": "…", "images": [{"path": "…", "alt": "…"}] },
    "telegram_message": { "text": "…", "photo_path": "…", "parse_mode": "HTML" },
    "instagram_post":   { "caption": "…", "image_url": "https://…" },
    "linkedin_post":    { "text": "…", "asset_urn": "urn:li:digitalmediaAsset:…" },
    "threads_post":     { "text": "…", "image_url": "https://…" }
  },
  "webhook_url": "https://example.com/dispatch-callback"
}
```

**Target syntax**: `platform[:account]`. Per-account credentials: `<PLATFORM>_<FIELD>_<ACCOUNT>` (uppercase). E.g. `telegram:broadcast` resolves to `TELEGRAM_CHAT_ID_BROADCAST`.

### Webhooks

When `webhook_url` is set the worker fires `POST {url}` after each publish or failure:

```json
{"event":"published","id":"<row-id>","post_id":"<platform-id>","platform":"twitter:work"}
{"event":"failed",   "id":"<row-id>","error":"…","platform":"…","dead":false}
```

---

## CLI

After any install, three commands are available:

```bash
# Send content now
dispatch send --platforms twitter,bluesky --text "hello world"

# Schedule for later
dispatch send --platforms telegram --text "scheduled post" \
  --scheduled-for "2026-06-01T09:00:00Z"

# View queue
dispatch queue --status queued
dispatch queue --status failed

# View or cancel one dispatch campaign
# (use --local to bypass HTTP and access the configured local queue)
dispatch campaign <unit-id> [--local]
dispatch campaign <unit-id> --cancel [--local]

# Run the worker in-process
dispatch worker

# Quick connectivity test (Telegram ping)
dispatch quick-test
```

---

## Adapter contract

Each `adapters/<platform>.py` exposes one function:

```python
def publish(unit: ContentUnit, account: str | None) -> tuple[bool, str, str]:
    """Returns (ok, post_id, error_message)."""
```

Add a platform in three steps:

1. Write `adapters/myplatform.py` — implement `publish`
2. Add it to the `ADAPTERS` dict in `adapters/__init__.py`
3. Document the format key in this README

That's the whole integration surface. The worker handles retry, backoff, and webhooks automatically.

---

## Architecture

```
POST /dispatch
   │
   ▼
Queue (JSONL | Redis | Postgres)
   │
   ▼
scheduler/worker.py
   │
   ├─▶ adapters/twitter.py
   ├─▶ adapters/bluesky.py
   ├─▶ adapters/instagram.py
   ├─▶ adapters/telegram.py
   ├─▶ adapters/linkedin.py
   ├─▶ adapters/threads.py
   └─▶ adapters/youtube.py
          │
          ▼  on publish / fail
      webhook_url
```

- **Queue**: JSONL on disk (default), Redis (`REDIS_URL`), or Postgres (`DATABASE_URL`). All three implement `QueueProtocol` — swap with a single env var, no code changes.
- **Retry**: exponential backoff (`WORKER_BACKOFF_BASE × 2^(attempts-1)` + jitter), gives up after `WORKER_MAX_ATTEMPTS`.
- **State machine**: `queued → publishing → published | failed | dead`
- **No-cloud option**: everything runs from a single `python` process — `python cli.py worker` alongside `uvicorn`.

---

## n8n integration

Community node lives in [`n8n-node/`](./n8n-node). One node, five operations:

| Operation | Endpoint | Purpose |
|---|---|---|
| Dispatch | `POST /dispatch` | Send content (now or scheduled) |
| Adapt Caption with AI | `POST /ai/adapt` | Per-platform rewrite |
| Get Queue Row | `GET /queue/{id}` | Fetch one row |
| Retry Queue Row | `POST /queue/{id}/retry` | Re-queue a failed row |
| List Queue | `GET /queue` | List rows by status |

```bash
cd n8n-node
npm install --ignore-scripts && npm run build
# In your n8n install:
npm link n8n-nodes-open-dispatch
```

The credential just needs your Open-Dispatch base URL (plus an optional bearer if you proxy with auth).

---

## MCP server (Claude Desktop, Cursor, AI agents)

`mcp_server.py` wraps the entire Open-Dispatch API as an MCP server — 10 tools usable from Claude Desktop, Cursor, or any MCP-compatible AI agent via natural language.

```bash
pip install "open-dispatch[mcp]"
```

**Claude Desktop** — add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "open-dispatch": {
      "command": "python",
      "args": ["/path/to/open-dispatch/mcp_server.py"],
      "env": { "OPEN_DISPATCH_URL": "http://localhost:8000" }
    }
  }
}
```

Then ask Claude things like:
- *"Post this to Twitter and Bluesky"*
- *"Check if my last post went through"*
- *"Retry the failed LinkedIn post"*
- *"Adapt this caption for Twitter, Instagram and LinkedIn"*

**Available tools:** `health`, `dispatch`, `get_queue`, `get_row`, `retry_row`, `delete_row`, `adapt_caption`, `list_platforms`

**SSE transport** (for remote/browser use):
```bash
python mcp_server.py --transport sse --port 8001
```

---

## Media transcoding

Per-platform image resize via Pillow — 10 platform specs out of the box. Honors EXIF orientation, flattens RGBA to JPEG, never upscales.

```bash
# Transcode a Twitter card from a phone photo
curl -X POST "http://localhost:8000/media/transcode?platform=twitter" \
  -H "Content-Type: image/jpeg" \
  --data-binary @photo.jpg --output photo.twitter.jpg

# List all specs
curl http://localhost:8000/media/specs | jq .twitter
```

Python:

```python
from media import transcode_image
transcode_image("photo.jpg", "instagram")   # → photo.instagram.jpg
```

---

## AI caption adapter

One source text → per-platform rewrites. Respects character limits and style conventions (Twitter punchy, LinkedIn formal, Instagram hashtag-heavy). Falls back to a heuristic when no LLM is configured — the endpoint never 500s on missing credentials.

```bash
curl -X POST http://localhost:8000/ai/adapt \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Open-Dispatch v0.4: 5 install methods, macOS app, Homebrew tap.",
    "platforms": ["twitter", "bluesky", "linkedin", "instagram", "threads"]
  }'
```

Provider priority: **Ollama** (free, local) → **OpenRouter** (any cloud model) → **heuristic** (no LLM). Set `OPENROUTER_API_KEY` or `OLLAMA_HOST` in `.env` to enable AI rewrites.

---

## Accessing the API

Once running, the API is at `http://localhost:8000` — no auth by default (designed for trusted self-hosting).

**Check it's live:**
```bash
curl http://localhost:8000/healthz
# → {"status": "ok"}
```

**Post to Twitter:**
```bash
curl -X POST http://localhost:8000/dispatch \
  -H "Content-Type: application/json" \
  -d '{
    "targets": ["twitter"],
    "formats": {
      "twitter_thread": { "tweets": ["your tweet here"] }
    }
  }'
```

**Check the queue:**
```bash
curl "http://localhost:8000/queue?status=published"
```

**Retry a failed post:**
```bash
curl -X POST http://localhost:8000/queue/<id>/retry
```

Full endpoint reference: [open-dispatch.vercel.app/api-reference](https://open-dispatch.vercel.app/api-reference) · or open `http://localhost:8000/` for the built-in dashboard.

> **Exposing to the internet?** Put it behind Cloudflare Access, Tailscale, or basic auth first. See [SECURITY.md](SECURITY.md).

---

## Tests

```bash
pytest -q
# 318 tests — schema, queue, API, media, profiles, adapters, and docs coverage — no network, no real credentials
```

---

## Roadmap

- [x] Twitter / X, Instagram, Telegram, Bluesky, LinkedIn, Threads, YouTube Shorts adapters
- [x] JSONL queue, exponential retry, webhooks
- [x] Docker Compose self-host
- [x] Web UI (HTMX + Jinja, dark theme — dashboard, composer, retry, row detail, live refresh)
- [x] AI caption-adaptation per platform (OpenRouter / Ollama / heuristic fallback)
- [x] YouTube Shorts adapter (Data API v3 resumable upload, OAuth2 refresh-token flow)
- [x] Redis queue backend (multi-worker safe)
- [x] Postgres queue backend (ACID + SKIP LOCKED for cross-region workers)
- [x] Media transcoding (10 platform specs, REST + Python API)
- [x] n8n community node (Dispatch / Adapt / Get Row / Retry / List Queue)
- [x] Homebrew tap (`brew install open-dispatch`)
- [x] Universal install.sh (macOS + Linux, launchd + systemd)
- [x] macOS menubar app + DMG
- [x] **TikTok adapter** (Content Posting API v2 — PULL_FROM_URL)
- [x] **Facebook adapter** (Meta Graph API v19 — text, photo, video)
- [x] **Bulk CSV import** (`POST /dispatch/bulk` — 1 MiB cap, validated columns)
- [x] **MCP server** (`mcp_server.py` — 10 tools, works with Claude Desktop, Cursor, any MCP client;
      compatible with mcp 1.x and 2.x)
- [ ] Video transcoding (ffmpeg-backed)
- [ ] Calendar view in dashboard
- [ ] Analytics (fetch engagement metrics per post)
- [ ] PyPI publish

## Third-party adapter backends

Some adapters support an alternative backend via an env var. These backends route your content **and credentials** through a third-party service instead of the platform's official API. They are fully opt-in — the default backend never changes.

| Adapter | Env var | Default | Alternative | Routes through |
|---------|---------|---------|-------------|----------------|
| Twitter | `TWITTER_BACKEND` | `tweepy` (official API) | `xquik` | [xquik.com](https://xquik.com) |

**Before enabling a third-party backend:**

- Review that service's privacy policy and ToS.
- Understand that your tweet content and API key will transit their servers.
- Use a scoped/rotated API key wherever possible.
- See [SECURITY.md](SECURITY.md) for the full threat model.

---

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). The adapter contract is one function — adding a platform is ~80 LOC.

## Security

See [SECURITY.md](./SECURITY.md). Report vulnerabilities via GitHub Security Advisories (private disclosure).

## Install methods reference

Full documentation for all five install methods: [INSTALL_METHODS.md](./INSTALL_METHODS.md).

## Migrating from a hosted scheduler

Moving off Buffer, Hootsuite, or a similar paid tool? The step-by-step guide is in
[docs/migrating-from-zernio.md](./docs/migrating-from-zernio.md) — export your accounts,
map them to `.env`, and dispatch your first post in about ten minutes.

## Community

- 💬 **[Discussions](https://github.com/Matthew-Selvam/Open-Dispatch/discussions)** — questions, ideas, show-and-tell
- 🐛 **[Issues](https://github.com/Matthew-Selvam/Open-Dispatch/issues)** — bugs and feature requests
- ⭐ **Star the repo** to follow development — every star helps an independent MIT project reach more self-hosters

## Star History

<a href="https://star-history.com/#Matthew-Selvam/Open-Dispatch&Date">
  <img src="https://api.star-history.com/svg?repos=Matthew-Selvam/Open-Dispatch&type=Date" alt="Star History Chart" width="600">
</a>

## License

MIT © [Matthew Selvam](https://github.com/Matthew-Selvam). Fork it, ship it, never pay per-account fees again.
