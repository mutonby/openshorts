"""A probe VideoCapture is handed back even when a cap.get() raises mid-read.

``_source_duration_seconds`` (and the sibling probes in the /scenes and
restyle handlers) released the capture only on the success line, so a
``cap.get()`` that throws on a corrupt or truncated file skipped the release
and leaked the handle for the life of the long-running API process. The
release now lives in a ``finally``.
"""
import pytest

app = pytest.importorskip("app")
cv2 = pytest.importorskip("cv2")


class _RaisingCap:
    def __init__(self):
        self.released = False

    def get(self, *_a):
        raise RuntimeError("corrupt file")

    def release(self):
        self.released = True


class _OkCap:
    def __init__(self):
        self.released = False

    def get(self, prop):
        return 30.0 if prop == cv2.CAP_PROP_FPS else 300.0

    def release(self):
        self.released = True


def test_source_duration_releases_capture_on_error(monkeypatch):
    caps = []
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda *_a: caps.append(_RaisingCap()) or caps[-1])
    assert app._source_duration_seconds("whatever.mp4") is None
    assert caps and caps[0].released, "capture leaked when the probe raised"


def test_source_duration_releases_capture_on_success(monkeypatch):
    caps = []
    monkeypatch.setattr(cv2, "VideoCapture",
                        lambda *_a: caps.append(_OkCap()) or caps[-1])
    assert app._source_duration_seconds("ok.mp4") == 10.0  # 300 frames / 30 fps
    assert caps and caps[0].released
