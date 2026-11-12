import os
import sys

# Make the repo root importable so tests can import the app modules directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests always run the app in self-host (BYOK) mode. app.py freezes
# BILLING_ENABLED at import time and load_dotenv never overrides an existing
# variable, so this must be set here — before any test module imports app — or
# the suite's behavior would depend on the developer's personal .env.
os.environ["BILLING_ENABLED"] = "0"

# The shared YouTube-account state (yt_account) lives in files: give the suite
# its own directory and no pacing delay, so a test never waits on, or leaves a
# cooldown for, another test or the developer's own output/ directory.
import tempfile  # noqa: E402

os.environ["YT_ACCOUNT_STATE_DIR"] = tempfile.mkdtemp(prefix="yt-account-tests-")
os.environ.setdefault("YT_ACCOUNT_MIN_INTERVAL", "0")
