"""Static file serving that can bring a job's files back before answering 404.

``OUTPUT_DIR`` is not durable: a redeploy or the hourly sweep wipes a
finished job, while the user's library still points at
``/videos/<job_id>/<file>``. The API endpoints re-pull a project from R2 on
demand (``app._ensure_job_files``), but ``/videos`` was a plain
``StaticFiles`` mount, so when the dashboard reopened a project it fired the
15 transcript requests (which restored the job, ~25 s for 2 GB) and the 15
``<video>`` loads at the same instant, and every player got a 404 that only
a reload fixed (job cff3ad6c, 6-sep-2026 00:37 UTC).

This subclass keeps everything StaticFiles does (Range requests, ETags, HEAD)
and adds one step: on a miss it hands the first path segment (the job id) to
``restorer``; when that reports the files may now be there, it looks once
more. The restorer is awaited on the request, so a player that arrives while
a restore is in flight simply waits for it (the per-job lock lives in the
restorer) instead of failing.
"""
from typing import Awaitable, Callable, Optional

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

Restorer = Callable[[str], Awaitable[bool]]
Guard = Callable[[str], bool]
# Raises (HTTPException) when the caller may not have this path. Gets the
# ASGI scope so it can read the query string, cookies and headers.
Authorizer = Callable[[str, dict], Awaitable[None]]


class RestoringStaticFiles(StaticFiles):
    """StaticFiles that can restore a missing job, and refuses non-deliverables.

    ``guard`` is the allowlist (``media_auth.is_servable``). Without it this
    mount hands out the whole working directory to anyone holding the job id:
    ``.owner`` (a user uuid), ``.resume.json`` (the customer's own
    ``webhook_secret``), ``.transcript_checkpoint.json``, and the
    ``*_metadata.json`` carrying the full transcript of the user's video.
    Verified served, unauthenticated, on 7-sep-2026.

    ``authorize`` is the caller half (``app._authorize_media``): a signed
    link, the owner's media token, or the owner's own credentials. It runs
    after the guard and before the filesystem and the restorer, so a stranger
    holding a job id can neither read its files nor make us pull them back
    from storage.
    """

    def __init__(self, *args, restorer: Optional[Restorer] = None,
                 guard: Optional[Guard] = None,
                 authorize: Optional[Authorizer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.restorer = restorer
        self.guard = guard
        self.authorize = authorize

    async def get_response(self, path: str, scope):
        # Before the filesystem: a refused path must look exactly like a
        # missing one, or the 404-vs-403 difference confirms the file is there.
        if self.guard is not None and not self.guard(path):
            raise HTTPException(status_code=404)
        if self.authorize is not None:
            await self.authorize(path, scope)
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or self.restorer is None:
                raise
            # Starlette may pass the path with either "/" or "\" as separator
            # depending on platform; normalise before splitting. Only proceed
            # for paths of the form "<job_id>/<file>" — bare names like
            # "robots.txt" have no separator and must not reach the restorer.
            norm = path.replace("\\", "/")
            if "/" not in norm:
                raise
            job_id = norm.split("/", 1)[0]
            if not job_id:
                raise
            try:
                restored = await self.restorer(job_id)
            except Exception:
                restored = False
            if not restored:
                raise
            # The guard may read the job directory (a free job's clean twins
            # are refused by a marker file that only exists once the files
            # are back), so the verdict before the restore is not final.
            if self.guard is not None and not self.guard(path):
                raise HTTPException(status_code=404)
            return await super().get_response(path, scope)
