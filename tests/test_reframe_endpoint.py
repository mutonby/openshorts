"""Tests for PR #70's manual per-scene reframing: the /api/clip/reframe
endpoint contract (validation, metering seam, caption reapply, persistence,
interplay with whole-clip framing) and the pure apply_crop_overrides math.

Same harness as test_rerender_endpoint.py: BILLING_ENABLED=0 self-host round
trips with the render stubbed at recut.perform_recut.
"""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
import recut  # noqa: E402

JOB_ID = "reframe-endpoint-test-job"

TRANSCRIPT = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 60.0, "text": "hello world",
         "words": [{"word": "hello", "start": 12.0, "end": 12.5},
                   {"word": "world", "start": 20.0, "end": 20.4}]},
    ],
}


def _request(method, path, json_body=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, json=json_body)
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))

    clip = {
        "start": 10.0, "end": 40.0,
        "video_title_for_youtube_short": "test clip",
        "video_url": f"/videos/{JOB_ID}/mytitle_clip_1.mp4",
    }
    meta = {"shorts": [clip], "transcript": TRANSCRIPT,
            "source_video": "src.mp4", "output_format": "auto",
            "cost_analysis": {}}
    (job_dir / "mytitle_metadata.json").write_text(json.dumps(meta))
    (job_dir / "mytitle_clip_1.mp4").write_bytes(b"canonical")
    (job_dir / "src.mp4").write_bytes(b"source")

    app_module.jobs[JOB_ID] = {
        "status": "completed", "logs": [],
        "result": {"clips": [dict(clip)], "cost_analysis": {}},
        "user_id": None, "watermark": False,
    }
    try:
        yield {"dir": job_dir, "meta_path": job_dir / "mytitle_metadata.json"}
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def fake_recut(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        name = f"recut_1_{kwargs['clean_name']}"
        with open(os.path.join(kwargs["output_dir"], name), "wb") as f:
            f.write(b"recut")
        return name, name

    monkeypatch.setattr(recut, "perform_recut", fake)
    return calls


class TestReframeValidation:
    def test_no_usable_overrides_400(self, job, fake_recut):
        for overrides in ({}, {"notanumber": "x"}, {"0": "garbage"}):
            resp = _request("POST", "/api/clip/reframe", {
                "job_id": JOB_ID, "clip_index": 0,
                "crop_overrides": overrides})
            assert resp.status_code == 400, overrides
        assert fake_recut == []

    def test_horizontal_clips_400(self, job, fake_recut):
        meta = json.loads(job["meta_path"].read_text())
        meta["output_format"] = "horizontal"
        job["meta_path"].write_text(json.dumps(meta))
        resp = _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0, "crop_overrides": {"0": 0.5}})
        assert resp.status_code == 400
        assert fake_recut == []

    def test_gone_source_409(self, job, fake_recut):
        os.remove(job["dir"] / "src.mp4")
        resp = _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0, "crop_overrides": {"0": 0.5}})
        assert resp.status_code == 409
        assert fake_recut == []


class TestReframeRender:
    def test_renders_from_source_with_overrides_and_captions(self, job, fake_recut):
        resp = _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0,
            "crop_overrides": {"0": 0.2, "1": {"top": 0.6, "bottom": {"x": 0.3, "y": 0.4}}}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["framed_scenes"] == [0, 1]

        call = fake_recut[0]
        assert call["input_path"].endswith("src.mp4")
        assert call["reframe"] is True
        assert call["crop_overrides"][0] == 0.2
        assert call["crop_overrides"][1]["top"] == {"x": 0.6, "y": 0.5}
        assert call["crop_overrides"][1]["bottom"] == {"x": 0.3, "y": 0.4}
        # Captions come back by default — every default clip ships with them.
        assert call["captions_transcript"]["segments"][0]["words"]

        # Overrides persisted (string keys, JSON-style) for /scenes to serve.
        meta = json.loads(job["meta_path"].read_text())
        assert meta["shorts"][0]["crop_overrides"]["0"] == 0.2

    def test_reapply_captions_false(self, job, fake_recut):
        resp = _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0, "reapply_captions": False,
            "crop_overrides": {"0": 0.5}})
        assert resp.status_code == 200
        assert fake_recut[0]["captions_transcript"] is None

    def test_whole_clip_framing_carries_into_the_render(self, job, fake_recut):
        # recipe.framing='full' (the clip editor's selector) must keep forcing
        # WIDE on the scenes the user did not hand-position.
        meta = json.loads(job["meta_path"].read_text())
        meta["shorts"][0]["recipe"] = {
            "v": 1, "segments": [{"start": 10.0, "end": 40.0}],
            "canonical_range": {"start": 10.0, "end": 40.0},
            "framing": "full"}
        job["meta_path"].write_text(json.dumps(meta))
        resp = _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0, "crop_overrides": {"0": 0.5}})
        assert resp.status_code == 200
        assert fake_recut[0]["force_strategy"] == "WIDE"
        assert resp.json()["recipe"]["framing"] == "full"


class TestRerenderClearsSceneOverrides:
    def test_trim_clears_stale_overrides(self, job, fake_recut):
        # Hand-frame first…
        assert _request("POST", "/api/clip/reframe", {
            "job_id": JOB_ID, "clip_index": 0,
            "crop_overrides": {"0": 0.2}}).status_code == 200
        # …then trim: scene indices no longer mean anything against the new
        # cut, so the stale overrides must not survive for /scenes to serve.
        assert _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 12, "end": 22}]}).status_code == 200
        meta = json.loads(job["meta_path"].read_text())
        assert not meta["shorts"][0].get("crop_overrides")


class TestApplyCropOverrides:
    """Pure math on the trajectory — no ffmpeg, no models."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.rf = pytest.importorskip("reframe_v2")

    def test_single_crop_centres_the_window(self):
        xs = [None] * 100
        strategies = ["GENERAL"]
        xs, strategies = self.rf.apply_crop_overrides(
            xs, strategies, [(0, 100)], {"0": 0.5}, crop_w=608, orig_w=1920)
        assert strategies == ["TRACK"]
        assert xs[0] == xs[99] == 656  # 1920*0.5 - 608/2

    def test_clamps_at_both_edges(self):
        xs = [0] * 10
        xs, _ = self.rf.apply_crop_overrides(
            xs, ["TRACK"], [(0, 10)], {"0": 0.0}, crop_w=608, orig_w=1920)
        assert xs[0] == 0
        xs, _ = self.rf.apply_crop_overrides(
            xs, ["TRACK"], [(0, 10)], {"0": 1.0}, crop_w=608, orig_w=1920)
        assert xs[0] == 1920 - 608

    def test_untouched_scenes_stay_untouched(self):
        xs = [111] * 20
        strategies = ["TRACK", "GENERAL"]
        xs, strategies = self.rf.apply_crop_overrides(
            xs, strategies, [(0, 10), (10, 20)], {"1": 0.5},
            crop_w=608, orig_w=1920)
        assert xs[:10] == [111] * 10          # scene 0 keeps the auto camera
        assert strategies[0] == "TRACK"
        assert strategies[1] == "TRACK"        # scene 1 forced back from GENERAL

    def test_malformed_and_unknown_indices_are_skipped(self):
        xs = [7] * 10
        out_xs, strategies = self.rf.apply_crop_overrides(
            xs, ["TRACK"], [(0, 10)],
            {"nope": 0.5, "9": 0.5, "0": "garbage"}, crop_w=608, orig_w=1920)
        assert out_xs == [7] * 10
        assert strategies == ["TRACK"]

    def test_split_writes_into_splits_keyed_by_start_frame(self):
        splits = {}
        xs, strategies = self.rf.apply_crop_overrides(
            [None] * 10, ["GENERAL"], [(0, 10)],
            {"0": {"top": {"x": 0.6, "y": 0.3}, "bottom": 0.2}},
            crop_w=608, orig_w=1920, orig_h=1080, splits=splits)
        assert strategies == ["SPLIT"]
        top, bottom = splits[0]
        assert top == (0.6 * 1920, 0.3 * 1080)
        assert bottom == (0.2 * 1920, 0.5 * 1080)

    def test_custom_writes_into_custom_layouts(self):
        custom_layouts = {}
        xs, strategies = self.rf.apply_crop_overrides(
            [None] * 10, ["GENERAL"], [(0, 10)],
            {"0": {"custom": [{"crop": {"x": 0.1, "w": 0.5}, "dest": {"y": 0.5}}]}},
            crop_w=608, orig_w=1920, orig_h=1080, custom_layouts=custom_layouts)
        assert strategies == ["CUSTOM"]
        panels = custom_layouts[0]
        assert len(panels) == 1
        assert panels[0]["crop"]["x"] == 0.1
        assert panels[0]["dest"]["y"] == 0.5
