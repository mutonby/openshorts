# OpenRouter integration notes (short)

This change adds a model rotator and adapter that automatically fetches models from OpenRouter and rotates between them. It is designed to be usable from both serverless functions (on-demand TTL refresh) and persistent workers (background poller via manager.refresh()).

Environment variables (required/optional):
- OPENROUTER_API_KEY (required)  -- your OpenRouter API key
- OPENROUTER_POLL_INTERVAL (optional) -- seconds; default 300
- MODEL_ROTATION_STRATEGY (optional) -- round_robin | random ; default round_robin
- OPENROUTER_CACHE_FILE (optional) -- path to roster json; default backend/openrouter/roster.json
- OPENROUTER_RETRY_ATTEMPTS (optional) -- default 3

Vercel deployment notes:
- Add OPENROUTER_API_KEY to your Vercel project environment variables.
- If you expose a serverless function to call chat completions, call the adapter send_chat_completion() from the function; the adapter will pick and rotate models automatically.
- Keep video processing and long-running jobs off Vercel — use an always-on worker host.
