# OPENDISPATCH_EXPLAINED.md

> **For AI agents and future contributors.** A deep technical reference for the Open-Dispatch codebase (`v0.4.0`). Read top to bottom before writing any code. Every section is drawn from actual source — no assumptions.

## What OpenShorts calls

The supported entry point from this repo is the package `opendispatch` (`publish_now`, `schedule`, `run_due`). It posts a finished short to **YouTube** (local file, resumable upload) and **Instagram** (the file is staged on the public R2 bucket, then the Graph API gets that URL). Scheduling lives in `schedule_times.py` and `local.py`. Credentials are environment variables (`IG_USER_ID`, `IG_TOKEN`, `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN`, and the `R2_*` names in `cloud/config.py`). The HTTP app, HTMX UI, and the other platform adapters described below stay in the tree and are not that integration.

---

## 1. What Open-Dispatch Is

Open-Dispatch is a **self-hosted, Python-based content cross-poster**. Its core job: accept one structured payload (`ContentUnit`) and publish it to up to 10 social platforms simultaneously, either immediately or at a scheduled time.

Two interfaces:
- **JSON REST API** — for scripts, SDKs, AI agents, n8n, cron jobs
- **HTMX web UI** — a browser dashboard for humans (compose, queue view, health, profiles)

A background **worker process** polls a pluggable queue and calls platform adapters. AI caption adaptation degrades gracefully: Ollama → OpenRouter → rule-based heuristic. Never hard-fails if AI is offline.

**Stack:** Python 3.11+, FastAPI, Jinja2, HTMX, httpx. No ORM. No message broker required.

---

## 2. Directory Map

```
opendispatch/
├── api/
│   ├── app.py          # FastAPI app — all routes (JSON + HTMX). ~973 lines.
│   ├── queue.py        # Pluggable queue backends (JSONL / Redis / Postgres). ~763 lines.
│   └── schema.py       # ContentUnit dataclass + validate() + parse_target(). ~139 lines.
├── adapters/
│   ├── __init__.py     # ADAPTERS dict + Adapter Protocol.
│   ├── errors.py       # Shared redaction + retry classification helpers.
│   ├── bluesky.py      # AT Protocol via atproto SDK
│   ├── discord.py      # Discord webhooks
│   ├── facebook.py     # Meta Graph API
│   ├── instagram.py    # Meta Graph API (image/video/carousel)
│   ├── linkedin.py     # LinkedIn REST API
│   ├── telegram.py     # Telegram Bot API
│   ├── threads.py      # Meta Threads API
│   ├── tiktok.py       # TikTok Content Posting API
│   ├── twitter.py      # Twitter/X v2 via tweepy
│   └── youtube.py      # YouTube Data API v3 resumable upload
├── scheduler/
│   └── worker.py       # Poll loop; claim rows; call adapters; retry/backoff. ~187 lines.
├── ai/
│   └── caption_adapter.py  # LLM/heuristic per-platform caption rewriting. ~420 lines.
├── web/
│   ├── templates/      # Jinja2 HTML templates (10 files)
│   └── static/         # CSS + JS assets
├── profiles.py         # Named per-platform credential sets. ~250 lines.
├── cli.py              # CLI entry points (send, queue, campaign, worker). ~226 lines.
├── mcp_server.py       # MCP server — exposes API as tools for AI agents. ~317 lines.
├── pyproject.toml      # Package config, entry points, optional deps.
├── Dockerfile
├── docker-compose.yml
└── tests/              # 21 pytest files, one per subsystem.
```

---

## 3. Core Data Structure: ContentUnit

**File:** `api/schema.py`

This is the single payload that everything — every adapter, every queue row, every API call — revolves around.

```python
@dataclass
class ContentUnit:
    id: str             # UUID, auto-generated
    created_at: str     # ISO-8601 UTC, auto-generated
    category: str       # Arbitrary label, default "general"
    targets: list[str]  # e.g. ["twitter", "bluesky:work", "telegram:news"]
    scheduled_for: str | None   # ISO-8601 or None (means "now")
    formats: dict[str, Any]     # Per-platform payload blocks
    webhook_url: str | None     # HTTPS URL for publish/fail callbacks
    profile_id: str | None      # Which named credential profile to use
```

### Target Syntax

Targets follow `platform` or `platform:account` — validated by regex `^(?P<platform>[a-z]+)(?::(?P<account>[a-z0-9._-]+))?$`.

```python
parse_target("twitter")        # -> ("twitter", None)
parse_target("twitter:work")   # -> ("twitter", "work")
parse_target("telegram:news")  # -> ("telegram", "news")
```

The `account` component maps to `_<ACCOUNT>` env var suffixes (uppercase). Example: `twitter:work` reads `TWITTER_ACCESS_TOKEN_WORK`.

### Formats Dict

Each platform needs its own key in `formats`. Multiple platforms can share one `ContentUnit`:

```python
{
  "twitter_thread":   {"tweets": ["First tweet", "Second tweet"]},
  "bluesky_post":     {"text": "My post text"},
  "telegram_message": {"text": "Long message here", "parse_mode": "HTML"},
  "instagram_post":   {"caption": "...", "image_url": "https://..."},
  "linkedin_post":    {"text": "Professional post..."},
  "threads_post":     {"text": "Short take..."},
  "facebook_post":    {"text": "...", "image_url": "https://..."},
  "discord_message":  {"content": "Alert message", "embeds": [...]},
  "tiktok_post":      {"video_url": "https://...", "caption": "..."},
  "youtube_short":    {"video_path": "media/clip.mp4", "title": "My Short"}
}
```

The worker picks only the format key matching the adapter for the enqueued platform — extra keys are ignored.

### Validation

`validate(unit) -> list[str]` returns a list of error strings (empty = valid). Checks:
- `targets` must be non-empty, at most 20
- Each target must match the regex
- `formats` must be non-empty
- Media paths (`*_path`, `*_paths` keys) must be relative and no `..`
- `webhook_url` must be HTTPS, not localhost, not a private/loopback IP
- `scheduled_for` must parse as ISO-8601 if present

### Caption Limits

```python
CAPTION_LIMITS = {
    "instagram": 2200,
    "twitter": 280,
    "telegram": 4096,
    "youtube": 5000,
    "tiktok": 2200,
    "bluesky": 300,
    "linkedin": 3000,
    "threads": 500,
    "facebook": 63206,
}
```

These are enforced by adapters and the AI adapter. They are not enforced by `validate()` itself.

---

## 4. The Queue System

**File:** `api/queue.py`

### Queue Row Schema

Every queued item is a dict:

```python
{
  "id": "uuid",
  "unit": {...ContentUnit.to_dict()...},
  "platform": "twitter:work",   # platform_key = "platform:account_or_default"
  "scheduled_for": "2026-07-01T09:00:00+00:00",
  "status": "queued",           # see Status Lifecycle below
  "attempts": 0,
  "post_id": None,              # platform's post ID once published
  "last_error": None,           # error string, max 500 chars
  "created_at": "...",
  "updated_at": "..."
}
```

### Status Lifecycle

```
queued -> publishing -> published
       |           |
       |           +-> (PUBLISHED class: still marks published even on error)
       |
       +-> queued (retry with backoff)
       |
       +-> dead
             |
             +-> (manual retry -> queued)

Any status -> canceled  (via cancel_campaign)
```

Status values: `queued`, `publishing`, `published`, `failed`, `dead`, `canceled`

Note: `failed` is referenced in the retry endpoint but is not a state the worker sets. The worker uses `queued` (for retryable) and `dead` (exhausted). `failed` appears in the API for filtering legacy rows.

### Backend Selection (Priority Order)

```
DATABASE_URL set -> PostgresQueue
REDIS_URL set    -> RedisQueue
(else)           -> JsonlQueue  (default, zero-infra)
```

`get_queue()` returns a singleton that implements `QueueProtocol`. All three backends expose identical methods.

### QueueProtocol Methods

```python
class QueueProtocol(Protocol):
    def enqueue(self, unit_dict, platform_key, scheduled_for) -> str  # returns row_id
    def list_all(self, status=None) -> list[dict]
    def get(self, row_id) -> dict | None
    def list_due(self, now_iso) -> list[dict]          # queued rows due <= now
    def mark_publishing(self, row_id) -> bool          # atomic claim
    def mark_published(self, row_id, post_id) -> bool
    def mark_failed(self, row_id, err, dead=False, retry_at=None) -> bool
    def retry(self, row_id) -> bool
    def delete(self, row_id) -> bool
    def cancel_campaign(self, unit_id) -> list[dict]   # cancel by ContentUnit.id
```

### Backend Implementation Notes

**JsonlQueue** (`data/queue.jsonl`):
- Single-process only (threading lock, no cross-process safety)
- Write-atomic: rewrites via `.tmp` file + `rename`
- `enqueue()` appends; all other writes rewrite the full file
- Corrupt lines are skipped with a warning — the queue never goes down
- Good for: dev, single-server self-hosting

**RedisQueue**:
- Key layout: `dispatch:row:<id>` (JSON string), `dispatch:all_ids` (set), `dispatch:due` (sorted set with epoch scores)
- `mark_publishing` uses `WATCH/MULTI/EXEC` for optimistic concurrency
- `list_due` uses `ZRANGEBYSCORE 0 <epoch>` — very fast
- Multi-worker safe (but has a TOCTOU window in cancel_campaign)
- Good for: multi-worker, single-region

**PostgresQueue**:
- Auto-creates table `open_dispatch_queue` on first use
- `mark_publishing` uses a single `UPDATE ... WHERE status='queued' AND scheduled_for <= now()` — truly atomic
- `cancel_campaign` uses `UPDATE ... RETURNING *` — atomic across workers
- `list_due` backed by a partial index on `(status, scheduled_for) WHERE status='queued'`
- Uses a connection factory (new connection per operation — psycopg2 is not thread-safe)
- Good for: multi-worker, multi-region production

### Critical Invariant: Backoff in Same Write

When a row fails, the `scheduled_for` reschedule happens **in the same write as the status flip**. This prevents a race where another worker sees a `queued` row with a past `scheduled_for` and claims it before the backoff lands. All three backends implement this:

```python
# Worker computes retry_at before calling mark_failed
retry_at = datetime.fromtimestamp(...).isoformat()
q.mark_failed(rid, err, dead=dead, retry_at=retry_at)
# mark_failed updates status AND scheduled_for atomically
```

---

## 5. The Worker

**File:** `scheduler/worker.py`

### Configuration (Env Vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `WORKER_POLL_INTERVAL` | `5` | Seconds between poll loops |
| `WORKER_MAX_ATTEMPTS` | `3` | Failures before dead-lettering |
| `WORKER_BACKOFF_BASE` | `30` | Base seconds for exponential backoff |

### Main Loop

```python
def main():
    while True:
        _write_heartbeat()   # writes data/.worker_heartbeat (ISO timestamp)
        run_once()           # fetch due rows, publish each
        time.sleep(POLL_INTERVAL)
```

`run_once()` is the core unit of work — it returns the count of rows processed and is directly testable.

### Per-Row Publish Flow (run_once)

```
1. q.list_due(now_iso)          — get all queued rows due <= now
2. for each row:
   a. q.mark_publishing(rid)    — atomic claim; skip if another worker won
   b. _publish(row)             — calls the platform adapter
   c. on success: q.mark_published(rid, post_id)
   d. on failure: classify error -> q.mark_failed(rid, err, dead, retry_at)
   e. fire webhook if unit.webhook_url is set
```

### Backoff Formula

```python
def _backoff_seconds(attempts: int) -> int:
    return BACKOFF_BASE * (2 ** max(0, attempts - 1)) + random.randint(0, 10)
```

With default `BACKOFF_BASE=30`: attempt 1 -> ~30s, attempt 2 -> ~60s, attempt 3 -> dead. Jitter (`+0-10s`) prevents thundering herd.

### Error Classification

The worker parses the **prefix** of the error string returned by adapters:

| Prefix | Action |
|--------|--------|
| `retryable:` | Back off, retry up to MAX_ATTEMPTS |
| `auth:` | Dead-letter immediately (no retries) |
| `permanent:` | Dead-letter immediately (no retries) |
| `published:` | Mark as published with empty/partial post_id — **never retry** |
| (none / anything else) | Treated as retryable |

The `published` class is critical: if a post goes live but we can't read the returned ID, retrying would create a duplicate post. Used by Twitter (mid-thread failures), Telegram (mid-chunk failures), and Bluesky (mid-thread failures).

### Heartbeat

Every loop writes the current UTC ISO timestamp to `data/.worker_heartbeat`. The `/healthz` endpoint reads this file and reports the worker as:
- `running`: last beat <= `WORKER_POLL_INTERVAL * 3` seconds ago
- `stale`: file exists but is old
- `not running`: file missing

### Credential Injection (Profiles)

```python
def _publish(row):
    ...
    profile = ProfileStore().get(unit.profile_id) if unit.profile_id else None
    with profile_env(profile):   # temporarily overrides os.environ
        return adapter.publish(unit, account)
```

`profile_env()` uses a threading lock so concurrent rows with different profiles don't corrupt env state.

### Log Redaction

The worker sets `httpx`, `httpcore`, `tweepy`, `atproto` loggers to WARNING level. These log full URLs at INFO level, which would expose tokens in paths/query strings (e.g., Telegram's `bot<TOKEN>/sendMessage`).

---

## 6. The Adapter System

**Files:** `adapters/__init__.py` + each platform file

### The Adapter Contract

Every adapter is a Python module with one function:

```python
def publish(unit: ContentUnit, account: str | None = None) -> tuple[bool, str, str]:
    # Returns: (ok, post_id, error_message)
    # ok=True  -> published, post_id is the platform's post identifier
    # ok=False -> failed, error_message classifies the failure
```

### The ADAPTERS Registry

```python
ADAPTERS: dict[str, Adapter] = {
    "telegram": telegram,
    "twitter": twitter,
    "instagram": instagram,
    "bluesky": bluesky,
    "linkedin": linkedin,
    "threads": threads,
    "youtube": youtube,
    "tiktok": tiktok,
    "facebook": facebook,
    "discord": discord,
}
```

To add a new adapter: create `adapters/<platform>.py` and add it here.

### Adapter Template Pattern

Adapters follow a consistent structure (~80-130 lines each):

```python
# 1. Module docstring: env vars, format key, fields
# 2. _creds(account) -> reads env vars with optional _<ACCOUNT> suffix
# 3. publish(unit, account) -> the one public function
#    a. Read formats block: fmt = unit.formats.get("platform_key") or {}
#    b. Validate required fields; return (False, "", "message") on missing creds/fields
#    c. Lazy-import SDK; return friendly error if not installed
#    d. Call platform API; handle errors using adapters.errors helpers
#    e. Return (True, post_id, "") on success
```

The best template to copy is `adapters/twitter.py`.

### Multi-Account Credential Pattern

Every adapter uses this pattern for account-specific credentials:

```python
def _creds(account: str | None) -> tuple[str, str]:
    suffix = f"_{account.upper()}" if account else ""
    token = os.getenv(f"TWITTER_ACCESS_TOKEN{suffix}") or os.getenv("TWITTER_ACCESS_TOKEN", "")
    secret = os.getenv(f"TWITTER_ACCESS_SECRET{suffix}") or os.getenv("TWITTER_ACCESS_SECRET", "")
    return token, secret
```

So `twitter:work` reads `TWITTER_ACCESS_TOKEN_WORK` falling back to `TWITTER_ACCESS_TOKEN`.

### Error Helpers (adapters/errors.py)

**`redact(text)`** — strips credentials from error messages before they reach logs, the API response, or webhooks:
- Telegram bot tokens: `bot123456:ABCxxx` -> `bot***`
- OAuth tokens in query strings: `access_token=abc123` -> `access_token=***`
- Bearer headers: `Bearer abc123` -> `Bearer ***`

**`clip(text, limit=500)`** — calls `redact()` then truncates. Always use `clip(e)` instead of `str(e)` in error returns.

**`classify(http_status)`** -> retry class string:
- 401, 403 -> `AUTH`
- 429, 5xx -> `RETRYABLE`
- 4xx (other) -> `PERMANENT`
- None -> `RETRYABLE`

**`prefix_error(cls, message)`** — formats `"auth: token expired"` so the worker can parse the class prefix.

**Error class constants:** `RETRYABLE`, `AUTH`, `PERMANENT`, `PUBLISHED`

### Platform-Specific Notes

**Twitter** (`twitter_thread`):
- Uses tweepy v2 Client for text, v1 API for media upload
- Thread = list of tweets posted as a reply chain
- If some tweets posted before a failure -> `PUBLISHED` (no retry to avoid duplicates)
- Media: `media_paths: list[str]`, max 4, applied to first tweet only

**Bluesky** (`bluesky_post`):
- Uses atproto SDK with App Password (not main password)
- Single post: `{"text": "..."}`, max 300 chars
- Thread: `{"thread": ["post1", "post2"]}` — reply chain
- Images: `{"images": [{"path": "...", "alt": "..."}]}`, max 4
- Mid-thread failure -> `PUBLISHED`

**Telegram** (`telegram_message`):
- Text messages chunked automatically at 4096 chars
- Photo/video: local file via `photo_path` or `video_path`
- Caption capped at 1024 for media; overflow sent as follow-up messages
- Telegram returns HTTP 200 even on failure; adapter checks `{"ok": false}` in body
- Mid-chunk or mid-caption failure -> `PUBLISHED`
- Bot token is in the URL path — `clip()` is essential

**Instagram** (`instagram_post`):
- Requires media at a **public HTTPS URL** (Graph API limitation — no local file upload)
- Three post types: single image (`image_url`), reel (`video_url`), carousel (`carousel_image_urls`)
- Two-step: create container -> poll until `FINISHED` -> publish
- Carousel: max 10 items (silently publishes wrong images if sliced; adapter rejects instead)
- Access token in query strings — use `clip()` carefully

**YouTube** (`youtube_short`):
- OAuth2 flow: exchanges `YOUTUBE_REFRESH_TOKEN` for short-lived access token on every call
- Two-step resumable upload: POST initiate -> PUT file bytes to session URL
- Streams file bytes (not `f.read()`) to avoid loading large videos into RAM
- Validates Location header host is `*.googleapis.com` before uploading
- Auto-adds `#Shorts` to description if not already present
- Privacy validation: rejects unknown values rather than defaulting to "public" (visibility escalation)

**LinkedIn** (`linkedin_post`):
- Uses `LINKEDIN_AUTHOR_URN` (`urn:li:person:...` or `urn:li:organization:...`)
- Text post or post with asset: `{"text": "...", "asset_urn": "..."}`

**Threads** (`threads_post`):
- Meta's separate Threads API (not Instagram)
- Two-step: create container -> publish

**TikTok** (`tiktok_post`):
- Video from public URL: `{"video_url": "https://...", "caption": "...", "privacy": "PUBLIC_TO_EVERYONE"}`

**Facebook** (`facebook_post`):
- Page post via Graph API: text, image_url, video_url, or link
- Requires `FACEBOOK_PAGE_ID` (numeric) + `FACEBOOK_ACCESS_TOKEN`

**Discord** (`discord_message`):
- Sends to a webhook URL: `{"content": "...", "username": "...", "embeds": [...]}`
- `DISCORD_WEBHOOK_URL` — treated as a password field in the UI

---

## 7. The REST API

**File:** `api/app.py`

### Authentication

- Set `OPEN_DISPATCH_API_TOKEN` env var to require authentication
- If unset: all requests allowed (local dev / trusted networks)
- Header: `Authorization: Bearer <token>`
- `/healthz` is exempt from auth
- CSRF protection: mutating requests (`POST/PUT/PATCH/DELETE`) from a browser origin that differs from the server's own origin are rejected (403)

### Content Negotiation

Routes serve HTML to browsers and JSON to API clients based on `Accept` header and presence of `HX-Request` header (HTMX):

```python
def _wants_html(request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept or "hx-request" in {k.lower() for k in request.headers}
```

### Key Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Health check (JSON or HTML dashboard) |
| `POST` | `/dispatch` | Enqueue a ContentUnit |
| `POST` | `/dispatch/bulk` | Bulk import from CSV |
| `GET` | `/queue` | List rows (paginated, filterable by status) |
| `GET` | `/queue/{row_id}` | Get single row (HTML or JSON) |
| `GET` | `/queue/{row_id}/json` | Always returns JSON |
| `POST` | `/queue/{row_id}/retry` | Re-queue a failed/dead row |
| `POST` | `/queue/{row_id}/delete` | Delete a row (HTMX-friendly alias) |
| `DELETE` | `/queue/{row_id}` | Delete a row |
| `GET` | `/campaign/{unit_id}` | All rows for one ContentUnit |
| `POST` | `/campaign/{unit_id}/cancel` | Cancel queued rows |
| `POST` | `/_retry-all` | Re-queue all errored rows |
| `POST` | `/_purge` | Delete all published or dead rows |
| `POST` | `/ai/adapt` | AI caption adaptation |
| `GET` | `/media/specs` | Per-platform image transcoding specs |
| `POST` | `/media/transcode` | Transcode an image for a platform |
| `GET` | `/profiles` | List credential profiles (HTML) |
| `GET` | `/profiles/new` | New profile form |
| `POST` | `/profiles` | Create profile |
| `POST` | `/profiles/{id}` | Update profile |
| `POST` | `/profiles/{id}/delete` | Delete profile |
| `GET` | `/api/profiles` | JSON list of profiles |
| `GET` | `/compose` | Composer UI |
| `POST` | `/_compose` | Submit compose form |
| `POST` | `/_compose-adapt` | HTMX: AI-adapt caption for composer |
| `GET` | `/_queue-fragment` | HTMX partial: queue table |
| `GET` | `/` | Dashboard (queue view) |

### Dispatch Flow (POST /dispatch)

```
body -> ContentUnit.from_dict() -> validate()
-> for each target:
    platform, account = parse_target(target)
    key = f"{platform}:{account or 'default'}"
    row_id = q.enqueue(unit.to_dict(), key, scheduled_for)
-> return 202 { "unit_id": ..., "enqueued": [{id, target, scheduled_for}, ...] }
```

One row is enqueued per target. All share the same `ContentUnit` (and thus the same `unit.id`). This lets you group them as a "campaign".

### Bulk CSV Import (POST /dispatch/bulk)

Format: `targets, format_key, text_or_json, scheduled_for`

```
"twitter,bluesky","twitter_thread","Hello world!","2026-07-01T09:00:00+00:00"
"telegram","telegram_message","{\"text\": \"Rich post\"}",""
```

- `text_or_json`: plain text is auto-wrapped using `_TEXT_FIELD_BY_FORMAT`
- Max 1000 rows, max 1 MiB
- Errors are per-row; valid rows are still imported
- Returns `{"imported": N, "errors": M, "results": [...], "error_details": [...]}`

### Webhook Events

When a `webhook_url` is set on a ContentUnit, the worker fires an HTTPS POST after each outcome:

```json
// On publish:
{"event": "published", "id": "<row_id>", "post_id": "<platform_post_id>", "platform": "twitter:work"}

// On failure:
{"event": "failed", "id": "<row_id>", "error": "<clipped_error>", "platform": "...", "dead": false}

// Unconfirmed publish (PUBLISHED error class):
{"event": "published", "id": "<row_id>", "post_id": "", "unconfirmed": true, "platform": "..."}
```

Webhook URLs are validated at enqueue time (HTTPS only, no private IPs) and at fire time (DNS resolution checked for private addresses).

---

## 8. Profiles System

**File:** `profiles.py`

Profiles are named sets of platform credentials, stored in `$OPEN_DISPATCH_DATA/profiles.json` (mode 0600). The directory is set to 0700.

```python
@dataclass
class Profile:
    id: str          # 8-char UUID prefix
    name: str        # "My Work Account"
    emoji: str       # "💼"
    color: str       # "#4ade80"
    platforms: dict[str, dict[str, str]]
    # e.g. {"twitter": {"access_token": "...", "access_secret": "..."}}
```

`profile.to_env()` converts stored credentials to `{ENV_VAR: value}` dict using `PLATFORM_CRED_MAP`.

`profile_env(profile)` is a context manager that temporarily injects profile credentials into `os.environ`, restoring originals on exit. Protected by a threading lock.

`ProfileStore` provides CRUD: `list()`, `get(id)`, `save(profile)`, `delete(id)`. No database — reads/writes the JSON file each call.

`PLATFORM_CRED_MAP` defines the credential field -> env var mapping for all 10 platforms.

---

## 9. AI Caption Adaptation

**File:** `ai/caption_adapter.py`

### Public Interface

```python
# Async (use inside FastAPI/async context):
formats = await adapt_caption_async(source_text, ["twitter", "linkedin", "instagram"])

# Sync (use in CLI, scripts, worker threads):
formats = adapt_caption(source_text, ["twitter", "linkedin", "instagram"])
# Note: sync version raises RuntimeError if called from inside a running event loop
```

Returns a dict shaped like `ContentUnit.formats`, ready to use directly.

### Provider Selection

```
OLLAMA_HOST set          -> Ollama (local, no API key)
OPENROUTER_API_KEY set   -> OpenRouter (cloud, any model)
(neither)                -> heuristic (no LLM, always works)
```

Provider can be overridden by passing `provider="ollama"|"openrouter"|"heuristic"`.
On any LLM error, falls back to heuristic silently.

### Platform Specs

Each platform has a spec defining the format key, character limit, style guidance, and expected JSON output shape. The LLM receives all of this in a structured prompt.

Supported platforms for adaptation: `twitter`, `bluesky`, `telegram`, `instagram`, `linkedin`, `threads`, `youtube`.

### Heuristic Fallback

Per-platform, without any LLM:
- **Twitter**: splits on sentence boundaries into <=280 char tweets; caps at 25
- **Bluesky/Telegram/Threads**: simple truncation at word boundary
- **LinkedIn**: strips hashtags then truncates
- **Instagram**: keeps existing hashtags, strips from body, appends as tag line
- **YouTube**: uses first sentence as title, appends `#Shorts`

### Belt-and-Suspenders: _enforce_limits()

After the LLM responds, `_enforce_limits()` clips every field to the spec's `max_chars`. If the LLM ignores limits, the output is still safe.

### LLM JSON Robustness

`_parse_llm_json()` handles:
1. Tries `json.loads()` on the cleaned text
2. Strips ` ```json ... ``` ` fences with regex
3. Falls back to finding the first `{...}` block in the output

---

## 10. The MCP Server

**File:** `mcp_server.py`

Exposes the API as MCP tools so AI agents (Claude, Cursor, etc.) can post content and manage queues via natural language.

**Transport:** stdio (default, for Claude Desktop / Cursor) or SSE (`--transport sse`).

**Tools exposed:**
- `health()` — check server status
- `dispatch(targets, formats, scheduled_for?, category?, webhook_url?)` — post content
- `get_queue(status)` — list rows by status
- `campaign_status(unit_id)` — all rows for a campaign
- `cancel_campaign(unit_id)` — cancel queued rows
- `get_row(row_id)` — single row details
- `retry_row(row_id)` — re-queue failed row
- `delete_row(row_id)` — permanent delete
- `adapt_caption(text, platforms)` — AI-adapt caption
- `list_platforms()` — list all platforms + format keys

**Auth:** reads `OPEN_DISPATCH_API_TOKEN` from env, passes as Bearer header. `OPEN_DISPATCH_URL` sets the API base URL.

**Claude Desktop config:**
```json
{
  "mcpServers": {
    "open-dispatch": {
      "command": "python",
      "args": ["/path/to/opendispatch/mcp_server.py"],
      "env": { "OPEN_DISPATCH_URL": "http://localhost:8000" }
    }
  }
}
```

---

## 11. The CLI

**File:** `cli.py`

Entry points defined in `pyproject.toml`:
- `dispatch` -> `cli:main` (subcommand CLI)
- `open-dispatch` -> `cli:server` (start uvicorn server)
- `open-dispatch-worker` -> `cli:worker` (start scheduler worker)

### Subcommands

```bash
# Send a post via HTTP to a running server
dispatch send --platforms twitter,bluesky --text "Hello" --at "2026-07-01T09:00:00Z"

# Send directly into queue (bypass HTTP — useful for scripting)
dispatch send --platforms twitter --text "Hello" --local

# Load a full ContentUnit JSON file
dispatch send --file unit.json

# View queue
dispatch queue --status queued

# Inspect/cancel a campaign
dispatch campaign <unit-id>
dispatch campaign <unit-id> --cancel --local

# Run the worker in-process
dispatch worker

# Quick connectivity test (sends a Telegram ping)
dispatch quick-test
```

`--local` flag on `send` and `campaign` bypasses HTTP and directly calls the queue.

---

## 12. Security Properties

| Concern | Implementation |
|---------|---------------|
| API auth | Bearer token via `OPEN_DISPATCH_API_TOKEN`; constant-time compare |
| CSRF | Same-origin check on mutating requests from browser origins |
| Webhook SSRF | URL validated for HTTPS + public IP at enqueue; DNS re-checked at fire time |
| YouTube upload SSRF | Location header host pinned to `*.googleapis.com` |
| Credential leakage in logs | `httpx`/`tweepy`/`atproto` loggers set to WARNING; `redact()` on all error strings |
| Media path traversal | `_validate_media_paths()` rejects absolute paths and `..` |
| Payload size | `/dispatch` capped at 100KB; `/dispatch/bulk` at 1 MiB / 1000 rows; `/media/transcode` at 10 MiB |
| Profile credentials at rest | Stored as plaintext JSON in 0600 file; directory 0700 |
| Profile env injection | threading lock prevents concurrent profile env corruption |
| Instagram carousel over-limit | Reject rather than silently slice (wrong images would be posted) |
| YouTube privacy unknown value | Reject rather than default to "public" (visibility escalation) |

---

## 13. Running the System

### Minimal (dev / single-server)

```bash
cp .env.example .env
# Fill in credentials for whichever platforms you want
pip install -r requirements.txt
uvicorn api.app:app --reload    # API + dashboard on :8000
python -m scheduler.worker       # in a separate terminal
```

### Docker

```bash
docker-compose up  # starts api + worker services
```

### Environment Variables

| Var | Required | Purpose |
|-----|----------|---------|
| `OPEN_DISPATCH_DATA` | No | Data directory path (default: `<repo>/data`) |
| `OPEN_DISPATCH_API_TOKEN` | No | API bearer token (if unset: no auth) |
| `DATABASE_URL` | No | Postgres DSN (enables PostgresQueue) |
| `REDIS_URL` | No | Redis URL (enables RedisQueue) |
| `WORKER_POLL_INTERVAL` | No | Poll seconds, default 5 |
| `WORKER_MAX_ATTEMPTS` | No | Max retries before dead, default 3 |
| `WORKER_BACKOFF_BASE` | No | Backoff base seconds, default 30 |
| `OPENROUTER_API_KEY` | No | OpenRouter LLM for caption adaptation |
| `OPENROUTER_MODEL` | No | Default: `openai/gpt-4o-mini` |
| `OLLAMA_HOST` | No | Local Ollama endpoint for caption adaptation |
| `OLLAMA_MODEL` | No | Default: `qwen2.5:latest` |
| `PORT` | No | Server port for `open-dispatch` entry, default 8000 |

Each platform's env vars follow the pattern `PLATFORM_FIELD[_ACCOUNT]` — see `.env.example` for the full list.

---

## 14. Testing

**Directory:** `tests/` — 21 pytest files

```bash
pytest                       # run all tests
pytest tests/test_queue.py   # single file
pytest -k "redis"            # by keyword
```

Key test files:
- `test_queue.py` — JsonlQueue, including the tearline / torn-write safety
- `test_redis_queue.py` — RedisQueue with fakeredis
- `test_postgres_queue.py` — PostgresQueue with a real Postgres (or fakepsycopg2)
- `test_adapters.py` — all adapters with mocked HTTP
- `test_adapter_errors.py` — redaction, classify, PUBLISHED error class
- `test_adapter_limits.py` — per-platform limits enforcement
- `test_adapter_fixes.py` — regression tests for specific bugs
- `test_api.py` — HTTP routes via TestClient
- `test_ai_caption.py` — caption adaptation heuristic and LLM paths
- `test_worker_backoff.py` — backoff formula, dead-lettering, webhook firing
- `test_partial_publish.py` — PUBLISHED error class prevents duplicate posts
- `test_profiles_credentials.py` — profile store + env injection
- `test_schema.py` — ContentUnit validation
- `conftest.py` — shared fixtures

The test suite uses `pytest-asyncio`. Tests mock `httpx` and platform SDKs — no real API calls are made.

---

## 15. How to Extract and Reuse Parts

### Reuse: Queue Backend Standalone

The queue system is self-contained. To use it in another project:

```python
# Copy api/queue.py (and optionally set DATABASE_URL / REDIS_URL)
from api.queue import get_queue

q = get_queue()
row_id = q.enqueue({"my": "data"}, "myplatform:default", "2026-07-01T09:00:00Z")
due = q.list_due(datetime.now(tz=timezone.utc).isoformat())
q.mark_publishing(row_id)
q.mark_published(row_id, "external_post_id_123")
```

The queue stores arbitrary dicts in the `unit` field — it doesn't require `ContentUnit` specifically.

### Reuse: A Single Adapter

Each adapter is standalone once `ContentUnit` and `adapters/errors.py` are available:

```python
from api.schema import ContentUnit
from adapters.telegram import publish

unit = ContentUnit(
    targets=["telegram"],
    formats={"telegram_message": {"text": "Hello from my script!"}}
)
ok, post_id, err = publish(unit)
```

### Reuse: AI Caption Adaptation

```python
from ai.caption_adapter import adapt_caption  # sync
# or: from ai.caption_adapter import adapt_caption_async  # async

formats = adapt_caption("My announcement post!", ["twitter", "linkedin", "instagram"])
# formats is ready to use as ContentUnit.formats
```

### Add a New Platform Adapter

1. Create `adapters/<platform>.py` following the template in `adapters/twitter.py`
2. Add format key to `_TEXT_FIELD_BY_FORMAT` in `api/app.py`
3. Add to `ADAPTERS` dict in `adapters/__init__.py`
4. Add credential map to `PLATFORM_CRED_MAP` in `profiles.py`
5. Add `CAPTION_LIMITS["<platform>"]` in `api/schema.py`
6. Add PLATFORM_SPECS entry in `ai/caption_adapter.py` if you want AI adaptation
7. Add emoji to `PLATFORM_EMOJI` in `profiles.py`
8. Update `_platforms_configured()` in `api/app.py`
9. Add test in `tests/test_adapters.py`

### Add a New Queue Backend

Implement the `QueueProtocol` interface (all 9 methods) and add backend selection logic to `get_queue()` in `api/queue.py`.

---

## 16. Known Limitations and Non-Features

- **No concurrent-safe single-process worker**: The JSONL backend is single-process. If two `run_once()` calls run concurrently on the same JSONL file, behavior is undefined. Use Redis or Postgres for multi-worker.
- **No database migration system**: Postgres schema is created on first use with `CREATE TABLE IF NOT EXISTS`. Schema changes require manual migration.
- **No webhook retry**: If the webhook POST fails, it is logged as a warning and dropped. The post itself is still marked as published.
- **No media upload to Instagram/TikTok from local files**: These platforms' APIs require public HTTPS URLs. Local video upload works only for Telegram (direct bot file), Bluesky (blob upload via atproto), and YouTube (resumable upload).
- **Profile credentials stored in plaintext**: `profiles.json` is protected only by filesystem permissions (0600). No encryption at rest.
- **ProfileStore is not thread-safe for concurrent writes**: It reads-then-writes with no lock. Two simultaneous profile saves could lose one update.
- **No scheduled recurring posts**: `scheduled_for` is a one-time datetime. Recurring posting requires external cron or re-enqueue logic.
- **No fanout parallelism**: The worker processes rows serially in `run_once()`. High-volume deployments should run multiple worker processes.
- **cancel_campaign for Redis has a TOCTOU window**: The Redis implementation reads then writes separately (unlike Postgres's single UPDATE). A concurrent worker could claim a row between the read and the cancel write.
- **TikTok adapter**: Uses URL-based upload (content must be at a public URL). No local file upload path.
- **YouTube is the only adapter that handles OAuth token refresh automatically**: Other adapters rely on long-lived tokens set in env vars.

---

## 17. Guardrails for Future Development

From `docs/HANDOFF.md` — enforced patterns:

1. **Keep pytest green** — add tests with every feature
2. **Adapters never crash the worker** — always return `(False, "", msg)`, never raise
3. **Adapters lazy-import their SDK** — return `(False, "", "sdk not installed (pip install ...)")` if missing
4. **Do not change HTML behavior when touching JSON error shapes** — HTMX routes are separate from JSON API routes
5. **Type hints + `from __future__ import annotations`** on every module
6. **Small focused functions** — adapters ~80-130 lines, functions <=30 lines
7. **`clip(e)` not `str(e)`** — all adapter error strings go through `clip()` for redaction
8. **Same-write backoff** — when failing a row, the `scheduled_for` reschedule and status flip must be atomic

---

*Generated from source at `v0.4.0`. Verify against active code before relying on any implementation detail.*
