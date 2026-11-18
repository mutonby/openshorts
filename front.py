"""A front process of the split API (docs/architecture/multiprocess-api.md).

``serve.py`` runs API_FRONT_WORKERS of these on the public port and one
engine (``app.py`` with API_ROLE=engine) on a local socket. Python runs one
thread at a time per process, so the single API process used to cap the
whole service at one core while the host sat mostly idle; now every request
that needs none of the engine's memory is served here, on as many cores as
there are fronts, and only the rest reaches the engine:

* ``serves_locally`` lists those routes. Each one reads only the database,
  the shared disk and env (sign-in, the account, billing pages, config, the
  remote worker's file transfers, CORS preflights), never ``app.jobs`` and
  the other per-process state of the engine. tests/test_multiprocess_api.py
  checks the app.py ones, and a route not listed goes to the engine, so a
  new route is safe until someone lists it on purpose.
* /api/status of a job still waiting or running comes from the snapshot the
  engine publishes (``app.front_status_view``); without a usable one the
  poll goes to the engine like any other request.
* Everything else is passed to the engine as it came, streamed both ways
  (uploads and downloads never sit in memory here), headers untouched, so
  the engine sees the same client address, scheme and host it always did.
"""
import os

os.environ["API_ROLE"] = "front"   # before app: its lifespan and DB pool read it

import app as _app  # noqa: E402
from front_proxy import Front, serves_locally  # noqa: E402,F401

app = Front(_app.app, status_view=_app.front_status_view,
            status_key=_app._STATUS_VIEW_KEY)
