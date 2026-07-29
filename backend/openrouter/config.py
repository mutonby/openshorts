import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_POLL_INTERVAL = int(os.getenv("OPENROUTER_POLL_INTERVAL", "300"))
OPENROUTER_CACHE_FILE = os.getenv("OPENROUTER_CACHE_FILE", "backend/openrouter/roster.json")
MODEL_ROTATION_STRATEGY = os.getenv("MODEL_ROTATION_STRATEGY", "round_robin")
OPENROUTER_RETRY_ATTEMPTS = int(os.getenv("OPENROUTER_RETRY_ATTEMPTS", "3"))

# Ensure API key exists when running actions that require it
if not OPENROUTER_API_KEY:
    # We don't raise here; the code will fail later when trying to call.
    pass
