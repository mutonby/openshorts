"""gpu_render: filtergraph -> layers, sendcmd timing, slots and fallbacks (no GPU needed)."""
import os
import subprocess

import pytest

import camera_inset
import gpu_render
import reframe_v2
import screencast_layout
import split_layout

W, H, OW, OH = 1920, 1080, 1080, 1920


def _layers(graph):
    return gpu_render.layers_from_graph(graph, W, H, OW, OH)


def _inside(layer):
    x, y, w, h = layer["dst"]
    return w > 0 and h > 0 and x >= -1 and y >= -1 and x + w <= OW + 1 and y + h <= OH + 1


class TestLayersFromGraph:
    def test_general_is_a_blurred_backdrop_under_the_centred_picture(self):
        bg, fg = _layers(reframe_v2.general_filtergraph(OW, OH, orig_w=W, orig_h=H))
        assert bg["dst"] == [0, 0, OW, OH] and bg["blur"]["w"] == 270 and bg["blur"]["h"] == 480
        assert "blur" not in fg
        # scale=-2:806 gives 1432 wide; crop keeps the middle 1080 of it.
        assert fg["dst"][2:] == [1080, 806]
        assert fg["src"][2] == pytest.approx(1080 * W / 1432, abs=0.01)
        # overlay puts 4:2:0 pictures on even rows: (1920 - 806) / 2 = 557 -> 556.
        assert fg["dst"][1] == 556

    def test_wide_keeps_the_whole_width(self):
        _bg, fg = _layers(reframe_v2.general_filtergraph(
            OW, OH, reframe_v2.full_width_content_height(W, H, OW)))
        assert fg["src"] == [0.0, 0.0, float(W), float(H)] and fg["dst"][2] == OW

    def test_split_stacks_two_crops(self):
        top, bottom = _layers(split_layout.split_filtergraph(W, H, OW, OH, (500, 450), (1400, 500)))
        assert top["dst"][1] == 0 and bottom["dst"][1] == top["dst"][3]
        assert top["src"][0] < bottom["src"][0]

    @pytest.mark.parametrize("graph", [
        screencast_layout.screencast_filtergraph(W, H, OW, OH, (1500, 400), (0, 1250)),
        screencast_layout.focus_filtergraph(W, H, OW, OH, (300, 1200)),
        camera_inset.inset_filtergraph(W, H, OW, OH, (1500, 760, 380, 280)),
    ])
    def test_other_layouts_are_readable(self, graph):
        layers = _layers(graph)
        assert layers[0].get("blur") and all(_inside(l) for l in layers)

    @pytest.mark.parametrize("graph", [
        "[0:v]drawtext=text=x[v]",
        "[0:v]crop=100:100,eq=contrast=2[v]",
        "[0:v]scale=1080:1920,overlay=enable='between(t,0,1)'[v]",
        "[0:v]pad=1080:1920:0:0:color=white[v]",
        "[0:v]scale=540:960[v]",  # wrong output size
        "[0:v]split=2[a][b];[a]scale=1080:1920[v]",  # dangling stream
    ])
    def test_anything_else_is_refused(self, graph):
        with pytest.raises(gpu_render.Unsupported):
            _layers(graph)

    def test_scale_keeps_aspect_with_minus_one_and_minus_two(self):
        (layer,) = _layers("[0:v]scale=-2:1920,crop=1080:1920[v]")
        assert layer["dst"] == [0, 0, 1080, 1920]


class TestTrack:
    def test_sendcmd_order_follows_rounded_command_times(self):
        # 0.0667 > 2/30, so frame 2 still shows box 1 (as ffmpeg does).
        assert gpu_render.sendcmd_order(5, 30) == [0, 1, 1, 3, 4]

    def test_boxes_are_clamped_and_even(self):
        layer = gpu_render.track_layer([(607, 1081, -5, 3), (608, 1080, 1500, 0)],
                                       W, H, OW, OH, fps=1)
        assert layer["boxes"] == [[0, 0, 606, 1080], [1312, 0, 608, 1080]]
        assert layer["dst"] == [0, 0, OW, OH]


def test_encoder_options_mirror_the_ffmpeg_tier():
    opts = dict(gpu_render.encoder_options())
    assert opts["preset"] == "p3" and opts["cq"] == "25" and opts["b"] == "0"
    assert "c:v" not in opts and "pix_fmt" not in opts


def test_unusable_without_the_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu_render, "BIN", str(tmp_path / "missing"))
    assert gpu_render.usable("x.mp4") is False


def test_unusable_when_disabled(monkeypatch):
    monkeypatch.setattr(gpu_render, "ENABLED", False)
    assert gpu_render.usable("x.mp4") is False


class _Done:
    def __init__(self, rc, stderr):
        self.returncode, self.stderr = rc, stderr


def _scene():
    return [(0.0, 2.0, reframe_v2.general_filtergraph(OW, OH, orig_w=W, orig_h=H), None)]


def test_render_runs_the_binary_with_a_plan(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        import json
        seen["plan"] = json.load(open(cmd[2]))
        return _Done(0, "osrender: 60 frames\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert gpu_render.render("in.mp4", _scene(), W, H, OW, OH, 30, str(tmp_path / "o.mp4"))
    plan = seen["plan"]
    assert (plan["out_w"], plan["out_h"], plan["fps"]) == (OW, OH, 30)
    assert len(plan["scenes"]) == 1 and len(plan["scenes"][0]["layers"]) == 2


@pytest.mark.parametrize("result", [_Done(1, "osrender: CUDA launch failed\n"),
                                    _Done(0, "osrender: 12 frames\n")])
def test_render_reports_failure_and_short_output(monkeypatch, tmp_path, result):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
    assert not gpu_render.render("in.mp4", _scene(), W, H, OW, OH, 30, str(tmp_path / "o.mp4"))


def test_render_refuses_unreadable_graphs_without_running(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran osrender"))
    scenes = [(0.0, 1.0, "[0:v]drawtext=text=x[v]", None)]
    assert not gpu_render.render("in.mp4", scenes, W, H, OW, OH, 30, str(tmp_path / "o.mp4"))


def test_render_timeout_falls_back(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("osrender", 1)
    monkeypatch.setattr(subprocess, "run", boom)
    assert not gpu_render.render("in.mp4", _scene(), W, H, OW, OH, 30, str(tmp_path / "o.mp4"))


def test_nvenc_slot_takes_a_free_lock(monkeypatch, tmp_path):
    import fcntl
    monkeypatch.setenv("NVENC_SLOTS", "2")
    monkeypatch.setenv("NVENC_LOCK_DIR", str(tmp_path))
    busy = open(tmp_path / ".nvenc-0.lock", "a")
    fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with gpu_render.nvenc_slot():
        probe = open(tmp_path / ".nvenc-1.lock", "a")
        with pytest.raises(OSError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe.close()
    probe = open(tmp_path / ".nvenc-1.lock", "a")
    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released after the block
    probe.close()
    busy.close()


def test_nvenc_slot_wait_goes_to_the_job_log(monkeypatch, tmp_path):
    """Same line as bin/ffmpeg: seconds waited, for the job's JOB_CPU."""
    import fcntl
    import threading
    locks = tmp_path / "locks"
    locks.mkdir()
    log = tmp_path / "wait.log"
    monkeypatch.setenv("NVENC_SLOTS", "1")
    monkeypatch.setenv("NVENC_LOCK_DIR", str(locks))
    monkeypatch.setenv("NVENC_WAIT_LOG", str(log))
    with gpu_render.nvenc_slot():
        pass
    assert not log.exists()                      # no wait, no line
    busy = open(locks / ".nvenc-0.lock", "a")
    fcntl.flock(busy, fcntl.LOCK_EX | fcntl.LOCK_NB)
    threading.Timer(1.5, busy.close).start()
    with gpu_render.nvenc_slot():
        pass
    waits = [int(x) for x in log.read_text().split()]
    assert len(waits) == 1 and waits[0] >= 1


def test_nvenc_slot_is_a_no_op_without_slots(monkeypatch):
    monkeypatch.delenv("NVENC_SLOTS", raising=False)
    with gpu_render.nvenc_slot():
        pass


class TestEncodeScenes:
    """reframe_v2.encode_scenes picks the GPU render and falls back to segments."""

    def _setup(self, monkeypatch, usable, rendered):
        calls = {"segments": 0, "mux": None}
        monkeypatch.setattr(gpu_render, "usable", lambda path: usable)
        monkeypatch.setattr(gpu_render, "render", lambda *a, **k: rendered)
        monkeypatch.setattr(reframe_v2, "_run_segment",
                            lambda *a: calls.__setitem__("segments", calls["segments"] + 1))
        monkeypatch.setattr(reframe_v2, "_run", lambda cmd: calls.__setitem__("mux", cmd))
        return calls

    def _encode(self, tmp_path):
        specs = _scene() * 3
        reframe_v2.encode_scenes("clip.mp4", specs, W, H, OW, OH, 30, str(tmp_path), None,
                                 str(tmp_path / "final.mp4"))

    def test_gpu_render_replaces_the_segments(self, monkeypatch, tmp_path):
        calls = self._setup(monkeypatch, True, True)
        self._encode(tmp_path)
        assert calls["segments"] == 0
        mux = calls["mux"]
        assert mux[mux.index("-i") + 1] == os.path.join(str(tmp_path), "gpu.mp4")
        assert "clip.mp4" in mux and "+faststart" in mux

    @pytest.mark.parametrize("usable,rendered", [(False, True), (True, False)])
    def test_falls_back_to_segments(self, monkeypatch, tmp_path, usable, rendered):
        calls = self._setup(monkeypatch, usable, rendered)
        self._encode(tmp_path)
        assert calls["segments"] == 3
        assert "concat" in calls["mux"]
