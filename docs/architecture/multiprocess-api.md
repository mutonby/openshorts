# Multiprocess API (engine + fronts)

Python runs one thread at a time per process. The API was one uvicorn
process, so however many cores the host had, every request, every status
poll and all the job orchestration shared one: under heavy traffic that core
sat at 100 % (login and submissions answering 500 as the DB pool timed out
behind a busy event loop, jobs starting late, the GPU idle) while most of the
host was free.

`serve.py` (the image's CMD) can now run the API as several processes:

```
            :8000 (Traefik, health check)
   ┌──────────────┬──────────────┬──────────────┐
   front 1        front 2   ...  front N          uvicorn --workers N (front.py)
   └──────────────┴──────┬───────┴──────────────┘
                         │ local socket (ENGINE_SOCKET), only what they can't serve
                      engine                        app.py, API_ROLE=engine
       queue · jobs · remote-worker claims · deliveries · sweeps · handover
```

- **Engine** (one): `app.py` exactly as before (lifespan, queue, jobs in
  memory, deploy handover, background loops), listening on a unix socket.
- **Fronts** (`API_FRONT_WORKERS`): the same `app.py` imported with
  `API_ROLE=front`, whose lifespan only opens a small DB pool. Each one serves
  the routes in `front_proxy.serves_locally` itself, answers CORS
  preflights, answers `/api/status` from the engine's snapshots, and streams
  every other request to the engine unchanged (headers, client address,
  body, response).
- `API_FRONT_WORKERS=0` (the default, self-host, tests) runs the single
  process as it always ran. Rollback = set it to 0 and redeploy.

## What a front may serve

Only routes that need none of the engine's per-process memory: the database,
the shared disk and env. Today: sign-in (`/api/auth/*`, Google), `/api/me`,
config, billing pages and checkout, the onboarding survey, `/api/media-token`,
`/api/social/user`, and the remote worker's calls after its claim (they only
read and write the job directory under file locks, as either container of a
deploy already could). Not the Stripe webhook nor end-trial (a plan change
moves the user's queued jobs), not `/internal/worker/claim` (the queue) nor
`/source` (one bandwidth budget for all transfers).

`tests/test_multiprocess_api.py` walks every `app.py` route the fronts serve
and fails if it reaches a module-level mutable container of `app.py`. A route
that is not listed goes to the engine, so a new route is safe by default;
listing it is a decision.

## Status polls without the engine

The engine publishes, at most every `STATUS_PUBLISH_SECONDS` (1 s):

- `<job>/.status.json` for each job still queued or running: what a poll
  would show (presented status, friendly logs, partial result, owner), only
  rewritten when that changes, removed when the job ends;
- `output/.queue_view.json`: the queue position and ETA of every waiting job,
  with the publishing instance and time.

A front answers a poll from them only when the queue view is fresh
(`STATUS_SNAPSHOT_FRESH_SECONDS`, 10 s) and the snapshot was written by the
same instance that publishes the queue view; anything else (finished job,
stale publisher, a draining or another instance's snapshot) goes to the
engine, which answers from memory or disk as before. The owner check and the
signed download URLs are done in the front exactly as the engine does them.
Only the active instance (not draining, holding the `.instance` marker)
publishes; a draining one never removes the snapshots the next one wrote.

## Caches shared across processes

Each process keeps its own short-lived memory of who a credential is
(`cloud/owner_cache.py`, `cloud/auth.py`). A forget (ban, revoked key,
erased account, any Stripe event) happens in one process, usually the engine,
so it touches `CACHE_EPOCH_FILE` (`output/.cache_epoch`, on the disk both
containers share); every process checks it at most every 0.5 s and drops what
it remembers when it changed.

## Stopping and deploys

Docker's SIGTERM reaches `serve.py`, which passes it to the engine only. The
engine drains exactly as before (finishes its jobs, `/health/ready` answers
503 through the fronts so Traefik stops routing here, keeps serving for
`PROXY_DRAIN_SECONDS`) while the fronts keep carrying its requests; when the
engine has exited the fronts are stopped. If either side dies on its own the
other is stopped and the container exits, so the platform restarts it whole.

## Connections to the engine

A front keeps one pool of connections to the engine's socket
(`front_proxy.ENGINE_LIMITS`, 1000). A streamed response holds its connection
until the client has read it or left, and a player that buffers a long video
reads slowly on purpose, so the pool has to be far larger than the number of
requests per second suggests. The limits are set on the transport: httpx
ignores the ones given to a client that is handed a transport and falls back
to 100, which one viewer seeking through a long source used up, and from then
on that front answered every proxied request, the health check included, with
a 503 after its 10 s pool wait. In the log that is
`Front: engine unreachable (PoolTimeout)`, on one front and not the others.

## Database connections

A front keeps `FRONT_DB_POOL_SIZE` (3) connections and opens up to
`FRONT_DB_MAX_OVERFLOW` (7) more while busy; the engine keeps
`DB_POOL_SIZE`/`DB_MAX_OVERFLOW`. Both containers of a deploy are up at once,
so Postgres' `max_connections` must hold
`2 x (engine pool + API_FRONT_WORKERS x front pool)`; lower the engine's pool
(it serves far fewer requests now) or raise `max_connections` before raising
`API_FRONT_WORKERS`.

Every instance runs the additive schema steps at boot
(`cloud/database._ADDITIVE_COLUMNS`). An `ALTER TABLE ... IF NOT EXISTS` still
takes the table's exclusive lock, which during a deploy queued every read of
the live container behind it and deadlocked some: a step whose column or index
already exists is now skipped after a catalog look, and a needed one waits at
most `ADDITIVE_LOCK_TIMEOUT` for its lock.

## Event loop and HTTP parser

The image ships `uvloop` and `httptools` (in the locks); uvicorn picks them
up on its own in the engine and in every front.
