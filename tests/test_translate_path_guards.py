"""Security guards on the /api/translate endpoint.

Two classes of bugs are tested here:

1. ``target_language`` path traversal (MEDIUM)
   The language code is embedded in the output filename with no validation.
   A value like ``"../victim-job/clip"`` would cause the dubbed video to be
   written outside the caller's job directory, overwriting another tenant's
   file.  Fix: validate against the SUPPORTED_LANGUAGES allowlist before
   constructing the filename, and apply ``os.path.basename`` as defence-in-depth.

2. ``clip_index < 0`` not checked (LOW)
   The original upper-bound-only check ``clip_index >= len(clips)`` allowed
   negative Python indices (e.g. ``-1`` → last clip), so the caller could
   silently operate on a clip they did not specify.  Fix: add the missing
   ``clip_index < 0`` guard to match every other edit endpoint.
"""
import asyncio
import json
import os
import uuid

import httpx
import pytest

app_module = pytest.importorskip("app")

OWNER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


class _User:
    def __init__(self, uid):
        self.id = uid
        self.plan = "pro"


def _call(method, path, **kwargs):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await getattr(client, method)(path, **kwargs)
    return asyncio.run(_do())


def _auth(uid):
    return {"Authorization": f"Bearer {uid}"}


@pytest.fixture()
def translate_env(tmp_path, monkeypatch):
    """Set up a minimal in-memory job + on-disk metadata for translate tests."""
    out = tmp_path / "output"
    out.mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)

    # Fake identity: every request is OWNER
    async def fake_user(request):
        auth = request.headers.get("authorization", "")
        return _User(auth.split()[-1]) if auth else None
    monkeypatch.setattr(app_module, "_user_from_request", fake_user)

    # Silence the _media_project lookup used by some owner-checks
    async def no_project(job_id):
        return None, None
    monkeypatch.setattr(app_module, "_media_project", no_project)

    # Build a real job directory with two dummy clips
    job_id = str(uuid.uuid4())
    job_dir = out / job_id
    job_dir.mkdir()

    clip1 = job_dir / "clip_1.mp4"
    clip2 = job_dir / "clip_2.mp4"
    clip1.write_bytes(b"fake-video-1")
    clip2.write_bytes(b"fake-video-2")

    metadata = {
        "shorts": [
            {"video_url": f"/videos/{job_id}/clip_1.mp4"},
            {"video_url": f"/videos/{job_id}/clip_2.mp4"},
        ]
    }
    (job_dir / f"{job_id}_metadata.json").write_text(json.dumps(metadata))

    # Register the job in the in-memory store
    monkeypatch.setattr(app_module, "jobs", {
        job_id: {"user_id": OWNER, "result": {"clips": [
            {"video_url": f"/videos/{job_id}/clip_1.mp4"},
            {"video_url": f"/videos/{job_id}/clip_2.mp4"},
        ]}}
    })

    return job_id, job_dir, out


# ---------------------------------------------------------------------------
# Finding 1 — target_language path traversal
# ---------------------------------------------------------------------------

class TestTargetLanguagePathTraversal:
    """Reject any target_language not in the SUPPORTED_LANGUAGES allowlist."""

    def _post(self, job_id, target_language, clip_index=0):
        return _call(
            "post", "/api/translate",
            headers={**_auth(OWNER), "X-ElevenLabs-Key": "test-key"},
            json={
                "job_id": job_id,
                "clip_index": clip_index,
                "target_language": target_language,
            },
        )

    def test_path_traversal_with_dotdot_is_rejected(self, translate_env):
        job_id, job_dir, out = translate_env
        resp = self._post(job_id, "../other-job/clip_1")
        assert resp.status_code == 400
        # No file must have been written outside the job directory
        for f in out.rglob("*"):
            assert str(f).startswith(str(job_dir)), (
                f"File written outside job dir: {f}"
            )

    def test_absolute_path_injection_is_rejected(self, translate_env):
        job_id, _, _ = translate_env
        resp = self._post(job_id, "/etc/passwd")
        assert resp.status_code == 400

    def test_null_byte_in_language_code_is_rejected(self, translate_env):
        job_id, _, _ = translate_env
        resp = self._post(job_id, "es\x00../secret")
        assert resp.status_code == 400

    def test_empty_language_code_is_rejected(self, translate_env):
        job_id, _, _ = translate_env
        resp = self._post(job_id, "")
        assert resp.status_code == 400

    def test_unsupported_but_safe_code_is_rejected(self, translate_env):
        """A plausible but unlisted code like 'xx' must be rejected."""
        job_id, _, _ = translate_env
        resp = self._post(job_id, "xx")
        assert resp.status_code == 400

    def test_valid_language_code_passes_validation(self, translate_env, monkeypatch):
        """A code in SUPPORTED_LANGUAGES must NOT be rejected by the guard.

        We short-circuit the actual ElevenLabs call by patching translate_video
        to avoid making real HTTP requests.  The test only verifies the guard
        does not fire for a known-good language code.
        """
        job_id, job_dir, _ = translate_env

        def fake_translate(video_path, output_path, target_language, api_key,
                           source_language=None, **kwargs):
            # Write a dummy output file so the endpoint can proceed
            with open(output_path, "wb") as f:
                f.write(b"dubbed")
            return output_path

        monkeypatch.setattr(app_module, "translate_video", fake_translate)

        # Also stub mark_ai_generated to avoid ffmpeg dependency
        try:
            import ffmpeg_utils
            monkeypatch.setattr(ffmpeg_utils, "mark_ai_generated", lambda *a, **k: None)
        except ImportError:
            pass

        resp = self._post(job_id, "es")
        # Must NOT be a 400; the guard let it through (200 or any non-400 is fine)
        assert resp.status_code != 400, (
            f"Valid language 'es' was incorrectly rejected: {resp.json()}"
        )


# ---------------------------------------------------------------------------
# Finding 2 — clip_index < 0 not checked
# ---------------------------------------------------------------------------

class TestClipIndexNegative:
    """Negative clip_index must be rejected with 404, not silently wrap around."""

    def _post(self, job_id, clip_index):
        return _call(
            "post", "/api/translate",
            headers={**_auth(OWNER), "X-ElevenLabs-Key": "test-key"},
            json={
                "job_id": job_id,
                "clip_index": clip_index,
                "target_language": "es",
            },
        )

    def test_negative_one_is_rejected(self, translate_env):
        job_id, _, _ = translate_env
        resp = self._post(job_id, -1)
        assert resp.status_code == 404

    def test_large_negative_is_rejected(self, translate_env):
        job_id, _, _ = translate_env
        resp = self._post(job_id, -999)
        assert resp.status_code == 404

    def test_zero_is_accepted_by_bounds_check(self, translate_env, monkeypatch):
        """Index 0 is a valid lower bound; guard must not reject it."""
        job_id, job_dir, _ = translate_env

        def fake_translate(video_path, output_path, target_language, api_key,
                           source_language=None, **kwargs):
            with open(output_path, "wb") as f:
                f.write(b"dubbed")
            return output_path

        monkeypatch.setattr(app_module, "translate_video", fake_translate)
        try:
            import ffmpeg_utils
            monkeypatch.setattr(ffmpeg_utils, "mark_ai_generated", lambda *a, **k: None)
        except ImportError:
            pass

        resp = self._post(job_id, 0)
        assert resp.status_code != 404, (
            f"clip_index=0 was incorrectly rejected: {resp.json()}"
        )
