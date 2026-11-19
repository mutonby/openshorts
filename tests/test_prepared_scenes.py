"""Scene probabilities prepared on the GPU host for a remote worker's clips
(main.prepare_scenes, scene_detection.use_prepared, remote_worker.*scenes*)."""
import base64
import json
import os

import numpy as np
import pytest

import remote_worker as rw


def _sd():
    return pytest.importorskip("scene_detection")     # cv2/scenedetect, absent in minimal CI


def _payload(n=300, fps=30.0, name="V.mp4", duration=10.0, spike=None):
    """What main.prepare_scenes writes (scene_detection.scenes_payload)."""
    probs = np.zeros(n, dtype="<f4")
    if spike is not None:
        probs[spike] = 0.97
    return {"source": {"name": name, "duration": round(duration, 1)}, "fps": fps,
            "frames": n, "probs": base64.b64encode(probs.tobytes()).decode()}


def test_round_trip_keeps_every_cut():
    sd = _sd()
    assert sd.scenes_payload("/x/V.mp4", 10.0, np.zeros(300, np.float32), 30.0) == _payload()
    p = np.array([0.0, 0.49, 0.5, 0.51, 1.0, 0.2], dtype=np.float32)
    back = sd.decode_probabilities(sd.encode_probabilities(p), len(p))
    assert (back == p).all()                     # the network's own numbers
    nan = sd.encode_probabilities(np.array([np.nan], np.float32))
    with pytest.raises(ValueError):
        sd.decode_probabilities(nan, 1)
    with pytest.raises(ValueError):
        sd.decode_probabilities(sd.encode_probabilities(p), len(p) + 1)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(fps=0), lambda d: d.update(fps=True), lambda d: d.update(fps=500),
    lambda d: d.update(frames=0), lambda d: d.update(frames=rw.MAX_SCENES_FRAMES + 1),
    lambda d: d.update(frames="300"), lambda d: d.update(probs=d["probs"][:-4]),
    lambda d: d.update(probs="<" * len(d["probs"])), lambda d: d.update(probs=None),
    lambda d: d["source"].update(name="../V.mp4"), lambda d: d["source"].update(name=""),
    lambda d: d["source"].update(duration="10"), lambda d: d.update(source=[]),
])
def test_a_payload_of_the_wrong_shape_is_refused(mutate):
    good = _payload()
    assert rw.valid_scenes(good)
    bad = json.loads(json.dumps(good))
    mutate(bad)
    assert not rw.valid_scenes(bad)


@pytest.fixture
def prepared(tmp_path):
    global sd
    sd = _sd()
    yield tmp_path
    sd._prepared.update(probs=None, fps=None)
    sd._clip_starts.clear()


def test_clips_map_the_source_by_time(prepared):
    d = str(prepared)
    # A cut at source frame 300 (10 s at 30 fps).
    json.dump(_payload(n=900, duration=30.0, spike=300), open(os.path.join(d, sd.SCENES_CHECKPOINT), "w"))
    assert sd.use_prepared(d, "/somewhere/V.mp4", 30.0)
    clip = os.path.join(d, "temp_V_clip_1.mp4")
    sd.register_clip(clip, 8.0)
    # The clip was re-timed to 25 fps: the cut lands 2 s in, at frame 50.
    p = sd._prepared_for(clip, 250, 25.0)
    assert len(p) == 250 and int(np.argmax(p)) == 50 and p[50] > 0.9
    assert sd._prepared_for(os.path.join(d, "other.mp4"), 250, 25.0) is None   # never registered
    sd.register_clip(clip, 29.0)
    assert sd._prepared_for(clip, 250, 25.0) is None                           # runs past the source


def test_another_sources_probabilities_are_ignored(prepared):
    d = str(prepared)
    json.dump(_payload(name="V.mp4", duration=10.0), open(os.path.join(d, sd.SCENES_CHECKPOINT), "w"))
    assert not sd.use_prepared(d, "/x/W.mp4", 10.0)
    assert not sd.use_prepared(d, "/x/V.mp4", 25.0)
    open(os.path.join(d, sd.SCENES_CHECKPOINT), "w").write("{nope")
    assert not sd.use_prepared(d, "/x/V.mp4", 10.0)
    assert sd._prepared["probs"] is None


def test_a_registered_clip_skips_the_network(prepared, monkeypatch):
    d = str(prepared)
    json.dump(_payload(n=600, duration=20.0, spike=150), open(os.path.join(d, sd.SCENES_CHECKPOINT), "w"))
    assert sd.use_prepared(d, "/x/V.mp4", 20.0)
    clip = os.path.join(d, "clip.mp4")
    sd.register_clip(clip, 2.0)

    class Cap:
        def __init__(self, *_):
            pass

        def get(self, prop):
            return 30.0 if prop == sd.cv2.CAP_PROP_FPS else 300

        def release(self):
            pass

    class Model:
        def predictions_to_scenes(self, p, threshold):
            cuts = [i for i in range(len(p)) if p[i] > threshold]
            return [[0, cuts[0] - 1], [cuts[0], len(p) - 1]] if cuts else [[0, len(p) - 1]]

    monkeypatch.setattr(sd.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(sd, "_extract_frames_small", lambda _p: np.zeros((300, 27, 48, 3), np.uint8))
    monkeypatch.setattr(sd, "_get_tn2_model", lambda: Model())
    monkeypatch.setattr(sd, "_predict", lambda _f: (_ for _ in ()).throw(AssertionError("ran the network")))
    scenes, fps = sd._detect_transnetv2(clip)
    # The cut at source frame 150 (5 s) is 3 s into a clip that starts at 2 s.
    assert [(s.get_frames(), e.get_frames()) for s, e in scenes] == [(0, 90), (90, 300)]


def test_the_api_host_announces_and_serves_them_as_data(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w1")
    src = os.path.join(d, "V.mp4")
    open(src, "wb").write(b"source")
    rw.set_ready(d, src, [], {})
    assert rw.read(d)["scenes_size"] == 0 and rw.read_scenes(d) is None
    path = os.path.join(d, rw.SCENES_FILE)
    json.dump(dict(_payload(), extra="not sent"), open(path, "w"))
    rw.set_ready(d, src, [], {})
    assert rw.read(d)["scenes_size"] == os.path.getsize(path)
    assert json.loads(rw.read_scenes(d)) == _payload()
    os.remove(path)
    os.symlink(src, path)
    assert rw.scenes_path(d) is None
    rw.drop_transcript(d)
    assert not os.path.lexists(path)


def test_the_file_name_is_scene_detections_own():
    sd = _sd()
    assert rw.SCENES_FILE == sd.SCENES_CHECKPOINT
    assert rw.MAX_SCENES_FRAMES == sd.MAX_PREPARED_FRAMES


def test_prepare_does_nothing_off_the_card(tmp_path, monkeypatch):
    sd = _sd()
    main = pytest.importorskip("main")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main.prepare_scenes(str(tmp_path / "V.mp4"), str(tmp_path)) is False
    assert not os.path.exists(tmp_path / sd.SCENES_CHECKPOINT)


def test_scenes_take_their_own_slots_not_the_transcriptions(tmp_path, monkeypatch):
    sd = _sd()
    main = pytest.importorskip("main")
    torch = pytest.importorskip("torch")
    import transcribe_backends as tb
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tb, "ASR_LOCK_DIR", str(tmp_path))
    seen = {}

    class Cap:
        def __init__(self, *_):
            pass

        def get(self, prop):
            return 30.0 if prop == main.cv2.CAP_PROP_FPS else 300

        def release(self):
            pass

    def probs(_path):
        seen["locks"] = sorted(os.listdir(tmp_path / ".scene-gpu"))
        return np.zeros(300, np.float32), 30.0
    monkeypatch.setattr(main.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(sd, "source_probabilities", probs)
    monkeypatch.setenv("SCENE_HOST_SLOTS", "3")
    assert main.prepare_scenes(str(tmp_path / "V.mp4"), str(tmp_path))
    assert seen["locks"] == [".asr-gpu-0.lock"]                 # its own pool
    assert not any(n.startswith(".asr-gpu") for n in os.listdir(tmp_path))   # never ASR's
    assert json.load(open(tmp_path / sd.SCENES_CHECKPOINT))["frames"] == 300
