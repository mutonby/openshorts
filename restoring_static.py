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
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

Restorer = Callable[[str], Awaitable[bool]]
Guard = Callable[[str], bool]
# Raises (HTTPException) when the caller may not have this path. Gets the
# ASGI scope so it can read the query string, cookies and headers. May return
# a word for what let the caller in, which the redirector is handed.
Authorizer = Callable[[str, dict], Awaitable[Optional[str]]]
# A link where the caller can fetch this very file from somewhere better than
# this process, or None to serve it here. Asked only once the caller is allowed.
Redirector = Callable[[str, dict, Optional[str]], Awaitable[Optional[str]]]


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

    ``redirect`` (``app._media_redirect``) runs last of the three: an allowed
    caller is sent to the archived copy when there is one, so neither the
    bytes nor a restore of the whole job go through this process.
    """

    def __init__(self, *args, restorer: Optional[Restorer] = None,
                 guard: Optional[Guard] = None,
                 authorize: Optional[Authorizer] = None,
                 redirect: Optional[Redirector] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.restorer = restorer
        self.guard = guard
        self.authorize = authorize
        self.redirect = redirect

    async def get_response(self, path: str, scope):
        # Before the filesystem: a refused path must look exactly like a
        # missing one, or the 404-vs-403 difference confirms the file is there.
        if self.guard is not None and not self.guard(path):
            raise HTTPException(status_code=404)
        granted = None
        if self.authorize is not None:
            granted = await self.authorize(path, scope)
        if self.redirect is not None:
            try:
                target = await self.redirect(path, scope, granted)
            except Exception:
                target = None
            if target:
                # Never stored: the answer depends on who asks and how (a
                # script's fetch is served here), and on the job's state.
                return RedirectResponse(target, status_code=307,
                                        headers={"Cache-Control": "no-store"})
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or self.restorer is None:
                raise
            job_id = path.split("/", 1)[0] if "/" in path else ""
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
