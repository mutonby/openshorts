# Minimal instructions and docs for the OpenRouter integration

This directory contains a lightweight OpenRouter "model rotator" and adapter designed to run alongside OpenShorts.

Files added:
- backend/openrouter/manager.py  -- model roster/cache/rotation logic
- backend/openrouter/adapter.py  -- HTTP adapter that selects a model and sends chat completions
- backend/openrouter/config.py   -- env var configuration
- backend/openrouter/run_demo.py -- small FastAPI demo exposing /api/openrouter/health and /api/openrouter/chat
- backend/openrouter/roster.json (created at runtime)

Quick start (local)
1. Create a virtualenv and install dependencies:
   pip install fastapi uvicorn httpx pydantic python-dotenv

2. Set your OpenRouter API key in the env:
   export OPENROUTER_API_KEY="your_key_here"

3. Run the demo app:
   uvicorn backend.openrouter.run_demo:app --reload --port 8001

4. Health endpoint:
   GET http://localhost:8001/api/openrouter/health

5. Chat endpoint (POST JSON):
   POST http://localhost:8001/api/openrouter/chat
   Body: {"messages": [{"role":"user","content":"Hello"}]}

Notes
- The rotator uses heuristics to filter models returned by OpenRouter. You may need to adjust filtering in manager._fetch_models depending on the exact API response shape.
- For serverless hosts (Vercel), run short-lived requests that call the adapter directly; the adapter will refresh the roster on demand (TTL-based) so it works without a background process.
- For heavy video workers, run the full OpenShorts Docker Compose stack on an always-on host (Oracle Cloud Always Free, a local machine, or another provider) and point the frontend/API on Vercel to it.
