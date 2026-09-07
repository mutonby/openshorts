"""Caption presets at the API surface: /api/config lists them, /api/subtitle
and /api/process validate them, and a preset fills only the fields the caller
left out.

Self-host mode (tests/conftest.py sets BILLING_ENABLED=0), no jobs on disk.
"""
import asyncio

import httpx
import pytest
from fastapi import HTTPException

import app as app_module
from app import SubtitleRequest, _apply_caption_preset, app
from subtitles import CAPTION_PRESETS, caption_presets_for_api


def _request(method, path, **kwargs):
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


class TestConfig:
    def test_lists_the_presets_in_order(self):
        resp = _request("GET", "/api/config")
        assert resp.status_code == 200
        assert resp.json()["captionPresets"] == caption_presets_for_api()
        assert [p["id"] for p in resp.json()["captionPresets"]] == list(CAPTION_PRESETS)


class TestSubtitlePreset:
    def test_preset_fills_what_the_caller_left_out(self):
        req = SubtitleRequest(job_id="j", clip_index=0, preset="hormozi")
        _apply_caption_preset(req)
        assert req.style == "karaoke"
        assert req.font_name == "Montserrat ExtraBold"
        assert req.highlight_color == "#00FFFF"
        assert req.uppercase is True

    def test_explicit_fields_win_over_the_preset(self):
        req = SubtitleRequest(job_id="j", clip_index=0, preset="hormozi",
                              highlight_color="#FF0000", uppercase=False)
        _apply_caption_preset(req)
        assert req.font_name == "Montserrat ExtraBold"  # from the preset
        assert req.highlight_color == "#FF0000"     # kept
        assert req.uppercase is False               # kept, even though falsy

    def test_no_preset_leaves_the_request_alone(self):
        req = SubtitleRequest(job_id="j", clip_index=0)
        before = req.model_dump()
        _apply_caption_preset(req)
        assert req.model_dump() == before

    def test_unknown_preset_is_a_400_naming_the_valid_ones(self):
        req = SubtitleRequest(job_id="j", clip_index=0, preset="nope")
        with pytest.raises(HTTPException) as exc:
            _apply_caption_preset(req)
        assert exc.value.status_code == 400
        assert "hormozi" in exc.value.detail

    def test_endpoint_rejects_unknown_preset_before_looking_for_the_job(self):
        resp = _request("POST", "/api/subtitle",
                        json={"job_id": "no-such-job", "clip_index": 0, "preset": "nope"})
        assert resp.status_code == 400
        assert "Unknown caption preset" in resp.json()["detail"]


class TestProcessPreset:
    def test_unknown_caption_preset_is_rejected_up_front(self, monkeypatch):
        # Self-host BYOK: the Gemini header satisfies the key check, and the
        # preset is validated before any download or probe runs.
        resp = _request("POST", "/api/process",
                        headers={"X-Gemini-Key": "test-key"},
                        json={"url": "https://example.com/video", "caption_preset": "nope"})
        assert resp.status_code == 400
        assert "Unknown caption_preset" in resp.json()["detail"]
