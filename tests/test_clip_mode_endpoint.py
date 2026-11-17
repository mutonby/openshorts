"""HTTP tests for POST /api/clip/mode: the per-clip switch between the normal
render and pro mode (pro_mode.py).

Same conventions as test_rerender_endpoint.py: a real ASGI round-trip against
the imported app (self-host branch), with the render stubbed at the
pro_mode.run seam. These tests own the endpoint contract: what the metadata
says after each switch, which layers go back on, and that going back to normal
is really going back."""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
pytest.importorskip("numpy")
import pro_mode  # noqa: E402
import recut  # noqa: E402

JOB_ID = "clip-mode-endpoint-test-job"

TRANSCRIPT = {"language": "en", "segments": [{
    "start": 0.0, "end": 60.0, "text": "hello world again",
    "words": [{"word": "hello", "start": 12.0, "end": 12.5},
              {"word": "world", "start": 20.0, "end": 20.4},
              {"word": "again", "start": 50.0, "end": 50.5}]}]}


def _request(method, path, json_body=None):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.request(method, path, json=json_body)
    return asyncio.run(_do())


def _switch(mode):
    return _request("POST", "/api/clip/mode",
                    {"job_id": JOB_ID, "clip_index": 0, "mode": mode})


def _make_job(tmp_path, monkeypatch, served="mytitle_clip_1.mp4", **clip_extra):
    out_root = tmp_path / "output"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    clip = {"start": 10.0, "end": 40.0, "video_title_for_youtube_short": "test clip",
            "video_url": f"/videos/{JOB_ID}/{served}", **clip_extra}
    meta = {"shorts": [clip], "transcript": TRANSCRIPT, "output_format": "auto"}
    meta_path = job_dir / "mytitle_metadata.json"
    meta_path.write_text(json.dumps(meta))
    (job_dir / "mytitle_clip_1.mp4").write_bytes(b"canonical")
    app_module.jobs[JOB_ID] = {
        "status": "completed", "logs": [],
        "result": {"clips": [dict(clip)], "cost_analysis": {}},
        "user_id": None, "watermark": False}
    return job_dir, meta_path


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """No test here talks to a model, whatever key the environment holds."""
    monkeypatch.setattr(pro_mode, "_ask_model", lambda prompt, key: {"titles": []})


@pytest.fixture()
def job(tmp_path, monkeypatch):
    job_dir, meta_path = _make_job(tmp_path, monkeypatch)
    try:
        yield {"dir": job_dir, "meta_path": meta_path}
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def fake_render(monkeypatch):
    """Stub the render seam: writes the output and keeps two stretches."""
    calls = []

    def fake_run(clip_path, out_path, words=(), ranges=(), wait=0.0, **kw):
        calls.append({"clip": os.path.basename(clip_path), "wait": wait})
        with open(out_path, "wb") as f:
            f.write(b"pro")
        return {"keep": [[0.5, 10.0], [12.0, 29.0]], "duration": 26.5,
                "shots": 2, "framed": 2}
    monkeypatch.setattr(pro_mode, "run", fake_run)
    return calls


def _clip(meta_path):
    return json.loads(meta_path.read_text())["shorts"][0]


class TestSwitchToPro:
    def test_the_clip_becomes_a_pro_recut(self, job, fake_render):
        resp = _switch("pro")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["mode"] == "pro"
        name = data["new_video_url"].split("/")[-1]
        assert name.startswith("recut_") and name.endswith("_mytitle_clip_1.mp4")
        assert (job["dir"] / name).read_bytes() == b"pro"

        clip = _clip(job["meta_path"])
        assert clip["mode"] == "pro"
        assert clip["video_url"] == data["new_video_url"]
        assert clip["recipe"]["segments"] == [
            {"start": 10.5, "end": 20.0}, {"start": 22.0, "end": 39.0}]
        assert (clip["start"], clip["end"]) == (10.5, 39.0)
        assert clip["pro_before"]["base"] == "mytitle_clip_1.mp4"
        # The dashboard reads the in-memory result while the tab is open.
        mem = app_module.jobs[JOB_ID]["result"]["clips"][0]
        assert mem["mode"] == "pro" and mem["video_url"] == data["new_video_url"]
        assert fake_render[0]["clip"] == "mytitle_clip_1.mp4"
        assert fake_render[0]["wait"] == app_module.PRO_WAIT_SECONDS

    def test_a_second_click_renders_nothing(self, job, fake_render):
        first = _switch("pro").json()
        again = _switch("pro")
        assert again.status_code == 200
        assert again.json()["unchanged"] is True
        assert again.json()["new_video_url"] == first["new_video_url"]
        assert len(fake_render) == 1

    def test_a_busy_host_says_so_and_changes_nothing(self, job, monkeypatch):
        def busy(*a, **kw):
            raise pro_mode.Busy("taken")
        monkeypatch.setattr(pro_mode, "run", busy)
        before = job["meta_path"].read_text()
        resp = _switch("pro")
        assert resp.status_code == 503
        assert resp.headers["retry-after"] == "60"
        assert job["meta_path"].read_text() == before

    def test_a_clip_that_is_not_vertical_is_refused(self, job, monkeypatch):
        def flat(*a, **kw):
            raise pro_mode.NotApplicable("pro mode needs a vertical clip")
        monkeypatch.setattr(pro_mode, "run", flat)
        assert _switch("pro").status_code == 400
        assert not _clip(job["meta_path"]).get("mode")

    def test_a_failed_render_leaves_the_clip_as_it_was(self, job, monkeypatch):
        def broken(*a, **kw):
            raise RuntimeError("ffmpeg said no /some/server/path")
        monkeypatch.setattr(pro_mode, "run", broken)
        before = job["meta_path"].read_text()
        resp = _switch("pro")
        assert resp.status_code == 500
        assert "/some/server/path" not in resp.text
        assert job["meta_path"].read_text() == before

    def test_an_unknown_mode_is_a_400(self, job):
        assert _switch("cinema").status_code == 400

    def test_an_unknown_clip_is_a_404(self, job, fake_render):
        resp = _request("POST", "/api/clip/mode",
                        {"job_id": JOB_ID, "clip_index": 5, "mode": "pro"})
        assert resp.status_code == 404


class TestLayersGoBackOn:
    @pytest.fixture()
    def layered_job(self, tmp_path, monkeypatch):
        """A clip served with a burned hook and captions on top of it."""
        served = "subtitled_5_hooked_4_mytitle_clip_1.mp4"
        job_dir, meta_path = _make_job(
            tmp_path, monkeypatch, served=served,
            auto_hook={"text": "Wait for it", "style": "pill", "position": "auto",
                       "duration_seconds": 5.0})
        (job_dir / "hooked_4_mytitle_clip_1.mp4").write_bytes(b"hooked")
        (job_dir / served).write_bytes(b"hooked+captions")

        hooks, captions = [], []

        def fake_hook(video_path, text, output_path, **kw):
            hooks.append({"on": os.path.basename(video_path), "text": text, **kw})
            with open(output_path, "wb") as f:
                f.write(b"hooked")

        def fake_captions(job_id, clip_index, video_path):
            captions.append(os.path.basename(video_path))
            out = os.path.join(os.path.dirname(video_path),
                               f"subtitled_9_{os.path.basename(video_path)}")
            with open(out, "wb") as f:
                f.write(b"captioned")
            return out
        monkeypatch.setattr(app_module, "add_hook_to_video", fake_hook)
        monkeypatch.setattr(app_module, "_reapply_captions", fake_captions)
        try:
            yield {"dir": job_dir, "meta_path": meta_path, "served": served,
                   "hooks": hooks, "captions": captions}
        finally:
            app_module.jobs.pop(JOB_ID, None)

    def test_hook_and_captions_are_burned_on_the_pro_file(self, layered_job, fake_render):
        resp = _switch("pro")
        assert resp.status_code == 200, resp.text
        name = resp.json()["new_video_url"].split("/")[-1]
        # Captions last, on the hooked pro file; the hook on the clean pro file.
        assert name.startswith("subtitled_9_hooked_")
        assert fake_render[0]["clip"] == "mytitle_clip_1.mp4"
        hook = layered_job["hooks"][0]
        assert hook["on"].startswith("recut_") and hook["text"] == "Wait for it"
        assert hook["position"] == "auto" and hook["has_captions"] is True
        assert layered_job["captions"][0].startswith("hooked_")

    def test_pro_mode_brings_its_caption_style_and_takes_it_back(self, layered_job, fake_render):
        _switch("pro")
        clip = _clip(layered_job["meta_path"])
        assert clip["caption_style"] == app_module.CAPTION_PRESETS["cine"]
        _switch("normal")
        assert not _clip(layered_job["meta_path"]).get("caption_style")

    def test_titles_stay_off_the_hook(self, layered_job, monkeypatch):
        seen = {}

        def fake_run(clip_path, out_path, words=(), ranges=(), wait=0.0, **kw):
            seen.update(kw)
            with open(out_path, "wb") as f:
                f.write(b"pro")
            return {"keep": [[0.0, 30.0]], "duration": 30.0, "shots": 1, "framed": 1}
        monkeypatch.setattr(pro_mode, "run", fake_run)
        assert _switch("pro").status_code == 200
        assert seen["hook_seconds"] == 5.0

    def test_back_to_normal_serves_the_file_from_before(self, layered_job, fake_render):
        _switch("pro")
        burned = (len(layered_job["hooks"]), len(layered_job["captions"]))
        resp = _switch("normal")
        assert resp.status_code == 200, resp.text
        assert resp.json()["mode"] == "normal"
        assert resp.json()["new_video_url"].endswith("/" + layered_job["served"])
        # Nothing changed on the clip meanwhile, so nothing is burned again.
        assert (len(layered_job["hooks"]), len(layered_job["captions"])) == burned
        clip = _clip(layered_job["meta_path"])
        assert clip["mode"] == "normal" and clip["pro_before"] is None
        assert (clip["start"], clip["end"]) == (10.0, 40.0)
        assert not clip.get("recipe")

    def test_a_hook_changed_in_pro_mode_is_burned_again_on_the_way_back(
            self, layered_job, fake_render):
        _switch("pro")
        data = json.loads(layered_job["meta_path"].read_text())
        data["shorts"][0]["auto_hook"]["text"] = "New hook"
        layered_job["meta_path"].write_text(json.dumps(data))
        resp = _switch("normal")
        assert resp.status_code == 200, resp.text
        assert layered_job["hooks"][-1] == {
            **layered_job["hooks"][-1], "on": "mytitle_clip_1.mp4", "text": "New hook"}
        assert resp.json()["new_video_url"].split("/")[-1].startswith("subtitled_9_hooked_")


class TestBackToNormal:
    def test_a_clip_without_layers_points_at_its_clean_file_again(self, job, fake_render):
        _switch("pro")
        resp = _switch("normal")
        assert resp.status_code == 200, resp.text
        assert resp.json()["new_video_url"] == f"/videos/{JOB_ID}/mytitle_clip_1.mp4"
        assert app_module.jobs[JOB_ID]["result"]["clips"][0]["mode"] == "normal"

    def test_normal_on_a_normal_clip_is_a_no_op(self, job):
        resp = _switch("normal")
        assert resp.status_code == 200 and resp.json()["unchanged"] is True


class TestOtherEditsLeaveProMode:
    def test_a_rerender_is_the_normal_look_again(self, job, fake_render, monkeypatch):
        _switch("pro")

        def fake_recut(**kwargs):
            name = f"recut_1_{kwargs['clean_name']}"
            with open(os.path.join(kwargs["output_dir"], name), "wb") as f:
                f.write(b"recut")
            return name, name
        monkeypatch.setattr(recut, "perform_recut", fake_recut)
        resp = _request("POST", "/api/clip/rerender", {
            "job_id": JOB_ID, "clip_index": 0,
            "segments": [{"start": 12.0, "end": 30.0}]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["mode"] == "normal"
        clip = _clip(job["meta_path"])
        assert clip["mode"] == "normal" and clip["pro_before"] is None


class TestCleanFileGone:
    def test_a_clip_left_with_only_its_captioned_file_is_not_converted(
            self, tmp_path, monkeypatch, fake_render):
        """A restored project keeps the served file only: reframing it would
        crop the captions burned into the picture."""
        served = "subtitled_5_mytitle_clip_1.mp4"
        job_dir, meta_path = _make_job(tmp_path, monkeypatch, served=served)
        (job_dir / served).write_bytes(b"captioned")
        (job_dir / "mytitle_clip_1.mp4").unlink()
        try:
            resp = _switch("pro")
            assert resp.status_code == 409
            assert fake_render == []
            assert not _clip(meta_path).get("mode")
        finally:
            app_module.jobs.pop(JOB_ID, None)


class TestServedFileOfAFreshJob:
    """The job writes no ``video_url`` into the metadata: until the first edit
    only the in-memory result, the dashboard and the disk know which file a
    clip is served from. Converting the clean reframe instead dropped the
    captions and the hook."""

    @pytest.fixture()
    def fresh_job(self, tmp_path, monkeypatch):
        served = "subtitled_5_mytitle_clip_1.mp4"
        job_dir, meta_path = _make_job(tmp_path, monkeypatch, served=served)
        (job_dir / served).write_bytes(b"captioned")
        data = json.loads(meta_path.read_text())
        del data["shorts"][0]["video_url"]
        meta_path.write_text(json.dumps(data))
        captions = []

        def fake_captions(job_id, clip_index, video_path):
            captions.append(os.path.basename(video_path))
            out = os.path.join(os.path.dirname(video_path),
                               f"subtitled_9_{os.path.basename(video_path)}")
            with open(out, "wb") as f:
                f.write(b"captioned")
            return out
        monkeypatch.setattr(app_module, "_reapply_captions", fake_captions)
        try:
            yield {"dir": job_dir, "meta_path": meta_path, "served": served,
                   "captions": captions}
        finally:
            app_module.jobs.pop(JOB_ID, None)

    def test_captions_go_back_on_from_the_in_memory_result(self, fresh_job, fake_render):
        resp = _switch("pro")
        assert resp.status_code == 200, resp.text
        assert resp.json()["new_video_url"].split("/")[-1].startswith("subtitled_9_recut_")
        assert fake_render[0]["clip"] == "mytitle_clip_1.mp4"
        assert len(fresh_job["captions"]) == 1

    def test_the_disk_answers_after_a_restart_lost_the_result(self, fresh_job, fake_render):
        app_module.jobs[JOB_ID]["result"]["clips"][0].pop("video_url")
        resp = _switch("pro")
        assert resp.status_code == 200, resp.text
        assert resp.json()["new_video_url"].split("/")[-1].startswith("subtitled_9_recut_")

    def test_the_dashboard_can_say_which_file_it_shows(self, fresh_job, fake_render):
        app_module.jobs[JOB_ID]["result"]["clips"][0]["video_url"] = (
            f"/videos/{JOB_ID}/mytitle_clip_1.mp4")
        resp = _request("POST", "/api/clip/mode", {
            "job_id": JOB_ID, "clip_index": 0, "mode": "pro",
            "input_filename": fresh_job["served"]})
        assert resp.status_code == 200, resp.text
        assert len(fresh_job["captions"]) == 1

    def test_another_clips_file_is_not_taken_for_this_one(self, fresh_job, fake_render):
        (fresh_job["dir"] / "subtitled_5_mytitle_clip_2.mp4").write_bytes(b"other clip")
        resp = _request("POST", "/api/clip/mode", {
            "job_id": JOB_ID, "clip_index": 0, "mode": "pro",
            "input_filename": "subtitled_5_mytitle_clip_2.mp4"})
        assert resp.status_code == 200, resp.text
        assert fake_render[0]["clip"] == "mytitle_clip_1.mp4"

    def test_back_to_normal_brings_the_captioned_file_back(self, fresh_job, fake_render):
        _switch("pro")
        resp = _switch("normal")
        assert resp.status_code == 200, resp.text
        assert resp.json()["new_video_url"].endswith("/" + fresh_job["served"])


class TestWhileTheJobRuns:
    def test_a_running_job_is_not_switched(self, job, fake_render):
        """The job rewrites the metadata when it ends: the switch would be
        undone under the user's eyes."""
        app_module.jobs[JOB_ID]["status"] = "processing"
        resp = _switch("pro")
        assert resp.status_code == 409
        assert fake_render == []
