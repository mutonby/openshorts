# Contributing to OpenShorts

Thanks for taking the time to contribute! OpenShorts is MIT-licensed and
open to PRs of all sizes — bug fixes, new features, documentation, tests,
and integration examples are all welcome.

## Setup

```bash
git clone https://github.com/mutonby/openshorts.git
cd openshorts
cp .env.example .env   # add your API keys

# Full stack (backend :8000, dashboard :5175)
docker compose up --build

# Or run each piece separately for faster iteration:
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

cd dashboard && npm install && npm run dev
```

## Running the checks

**CI runs these two jobs on every PR — both must be green before merge.**

```bash
# Backend tests (no heavy ML stack needed)
pip install pytest pillow httpx pydantic sqlalchemy greenlet numpy \
            opencv-python-headless python-dotenv google-genai fastapi boto3 \
            PyJWT email-validator python-multipart asyncpg
pytest tests/ -v

# Frontend (from the dashboard/ directory)
npm ci
npm run lint    # ESLint, --max-warnings 0
npm run build
```

A green CI is the merge gate — not the push. After every `git push`,
confirm the workflow is green before marking the task done.

## Branch and PR conventions

- Branch names: `feat/<short-description>`, `fix/<short-description>`,
  `test/<short-description>`, `docs/<short-description>`
- Commit messages: short imperative first line, body explaining *why*
  (not *what*) when it isn't obvious
- One logical change per PR; split large changes into stacked PRs
- Include (or update) tests for every behavior change
- Keep changes small: ~100-300 lines changed is reviewable in one sitting

## Architecture pointers

Before touching a subsystem, read its architecture doc in `docs/architecture/`:

| File | Covers |
|------|--------|
| `clip-selection.md` | Viral moment picking, local LLM, silent footage |
| `layouts-and-reframing.md` | TRACK / SPLIT / SCREENCAST / INSET / WIDE |
| `agent-access.md` | MCP server, API keys, OAuth, webhooks |
| `jobs-and-deploys.md` | Job queue, GPU/VRAM, deploy handover |
| `seo-and-tools.md` | Free tools, SEO pages, `seo/data.js` |

**Hard rules** (see `CLAUDE.md` for the full list):
- Never add a static `public/sitemap.xml` — the Vite plugin generates it.
- When editing pricing anywhere, edit `dashboard/seo/data.js` too.
- New MCP tools calling new REST endpoints must add those endpoints to
  `cloud/scopes.py:_MCP_ROUTES`. A test (`tests/test_oauth_scopes.py`)
  reads the tool source and fails if any endpoint is missing.
- Canonical files are always clean; the free-plan watermark lives only
  on the served `wm_` copy.

## What is and isn't open source

The core application (`app.py`, `main.py`, pipeline modules, dashboard,
CLI, MCP server) is **MIT**. The `cloud/` directory (billing, managed keys,
hosted-service infrastructure) is **source-available** under the
OpenShorts Commercial License — you can read and modify it for personal/
internal use, but you can't offer it as a paid hosted service. PRs that
touch only `cloud/` require more context; open an issue first.

## Questions?

Open a GitHub issue or start a Discussion. If you're unsure whether a
change is in scope, ask before writing the code — saves everyone time.
