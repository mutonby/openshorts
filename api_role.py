"""Which part of the API this process is.

``serve.py`` can run the API as several processes (docs/architecture/
multiprocess-api.md):

* ``engine``: the one process that owns the queue, the jobs in memory, the
  remote worker's claims, deliveries, sweeps and the deploy handover. It
  listens on a local socket only.
* ``front``: N processes on the public port. Each serves the routes that need
  none of the engine's memory (``front.serves_locally``) and passes every
  other request to the engine.
* ``all`` (the default): one process doing both, as the API always ran.
  Self-host, the tests and ``API_FRONT_WORKERS=0`` keep it.
"""
import os

ENGINE = "engine"
FRONT = "front"
ALL = "all"


def role() -> str:
    value = (os.environ.get("API_ROLE") or ALL).strip().lower()
    return value if value in (ENGINE, FRONT, ALL) else ALL


def is_front() -> bool:
    return role() == FRONT


def is_engine() -> bool:
    """The engine of a split API (not the single all-in-one process)."""
    return role() == ENGINE
