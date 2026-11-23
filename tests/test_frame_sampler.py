import cv2
import numpy as np
import pytest

import frame_sampler

N_FRAMES = 60


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A short clip whose frames all differ, with keyframes far apart."""
    path = str(tmp_path_factory.mktemp("fs") / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    if not writer.isOpened():
        pytest.skip("no video writer in this OpenCV build")
    rng = np.random.default_rng(1)
    for i in range(N_FRAMES):
        frame = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


def _seek_read(path, indices):
    """What the classifiers did before: one seek + read per sample."""
    cap = cv2.VideoCapture(path)
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        out.append(frame if ok else None)
    cap.release()
    return out


def _forward(path, indices, **kw):
    cap = cv2.VideoCapture(path)
    out = list(frame_sampler.read_at(cap, indices, **kw))
    cap.release()
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x is None) == (y is None)
        if x is not None:
            assert np.array_equal(x, y)


def test_forward_pass_returns_the_frames_a_seek_returns(video):
    idx = [0, 3, 4, 10, 11, 12, 30, 45, 59]
    _same(_forward(video, idx), _seek_read(video, idx))


def test_repeated_index_yields_the_same_frame_again(video):
    idx = [5, 5, 5, 6]
    got = _forward(video, idx)
    _same(got, _seek_read(video, idx))
    assert got[0] is got[1] is got[2]


def test_past_the_end_is_none_like_a_failed_read(video):
    idx = [58, 59, 60, 75]
    got = _forward(video, idx)
    assert got[2] is None and got[3] is None
    _same(got, _seek_read(video, idx))


def test_long_jumps_and_steps_back_seek_instead(video):
    idx = [2, 50, 20, 21, 57]
    _same(_forward(video, idx, max_forward=10), _seek_read(video, idx))


def test_empty_request_reads_nothing(video):
    assert _forward(video, []) == []


def test_it_seeks_once_the_target_is_a_gop_away(monkeypatch):
    """Decoding forward only pays up to about the cut's keyframe spacing."""
    import importlib

    import ffmpeg_utils
    assert frame_sampler.MAX_FORWARD == ffmpeg_utils.cut_gop() + 16
    monkeypatch.setenv("CUT_GOP", "0")                      # the encoder's ~250 frames
    try:
        assert importlib.reload(frame_sampler).MAX_FORWARD == 300
    finally:
        monkeypatch.undo()
        importlib.reload(frame_sampler)
