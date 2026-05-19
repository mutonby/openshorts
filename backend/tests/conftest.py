"""
Shared pytest fixtures + heavy-module stubbing.

The production code imports `mediapipe`, `ultralytics`, `torch`, `cv2`,
`scenedetect`, `yt_dlp`, `google.genai`, and `faster_whisper` at module
load time. None of those are available in the test environment by
default — and several can't even install on every Python version.

To keep the safety net runnable on a stock laptop, we stub these out
in `sys.modules` BEFORE any production module is imported. The classes
under test (SmoothedCameraman, SpeakerTracker, _sanitize_filter_string,
generate_srt, create_hook_image, SUPPORTED_LANGUAGES) are pure Python
and never touch the stubbed surfaces, so the tests still exercise real
production logic.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make `backend/` importable so `import app.*` resolves.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# --- Heavy module stubbing -----------------------------------------------

def _stub(name):
    """Register a MagicMock for a module name if it is not already importable."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        sys.modules[name] = MagicMock(name=name)


# Mock ML / video deps that are heavy or platform-restricted.
# Order matters for submodules: parent first, then attributes.
for _m in (
    "cv2",
    "scenedetect",
    "scenedetect.detectors",
    "ultralytics",
    "torch",
    "torchvision",
    "mediapipe",
    "yt_dlp",
    "tqdm",
    "faster_whisper",
    "google",
    "google.genai",
    "google.genai.types",
    "google.genai.errors",
    "google.protobuf",
):
    _stub(_m)

# scenedetect's `open_video`, `SceneManager`, and `ContentDetector` are
# imported by name from main.py. Wire them onto the mock so the import
# doesn't ImportError.
if isinstance(sys.modules.get("scenedetect"), MagicMock):
    sys.modules["scenedetect"].open_video = MagicMock(name="open_video")
    sys.modules["scenedetect"].SceneManager = MagicMock(name="SceneManager")
if isinstance(sys.modules.get("scenedetect.detectors"), MagicMock):
    sys.modules["scenedetect.detectors"].ContentDetector = MagicMock(
        name="ContentDetector"
    )

# ultralytics.YOLO('yolov8n.pt') runs at main.py import time. Make it
# return a harmless MagicMock instead of trying to download weights.
if isinstance(sys.modules.get("ultralytics"), MagicMock):
    sys.modules["ultralytics"].YOLO = MagicMock(name="YOLO")

# mediapipe.solutions.face_detection.FaceDetection(...) runs at import.
if isinstance(sys.modules.get("mediapipe"), MagicMock):
    mp_mock = sys.modules["mediapipe"]
    mp_mock.solutions = MagicMock()
    mp_mock.solutions.face_detection = MagicMock()
    mp_mock.solutions.face_detection.FaceDetection = MagicMock()

# google.genai is referenced as `from google import genai`. Make the
# `genai` attribute return our stub.
if isinstance(sys.modules.get("google"), MagicMock):
    sys.modules["google"].genai = sys.modules.get("google.genai", MagicMock())

# Make boto3 importable even without it installed (s3_uploader uses it
# at module load via `import boto3`). It's tiny so dev-requirements
# pulls it in, but stub as a safety net.
_stub("boto3")
_stub("botocore")
_stub("botocore.exceptions")
if isinstance(sys.modules.get("botocore.exceptions"), MagicMock):
    sys.modules["botocore.exceptions"].ClientError = type(
        "ClientError", (Exception,), {}
    )


# --- Fixtures ------------------------------------------------------------

import pytest  # noqa: E402  (after sys.modules stubbing)


@pytest.fixture
def tmp_output_dir(tmp_path, monkeypatch):
    """Per-test temporary OUTPUT_DIR / UPLOAD_DIR pair."""
    output = tmp_path / "output"
    uploads = tmp_path / "uploads"
    output.mkdir()
    uploads.mkdir()
    monkeypatch.chdir(tmp_path)
    yield output, uploads


@pytest.fixture
def fake_transcript():
    """A synthetic faster-whisper transcript with word-level timing.

    Shape matches what main.transcribe_video returns:
        {"text": str, "language": str,
         "segments": [{"start": float, "end": float, "text": str,
                       "words": [{"start": float, "end": float, "word": str}, ...]}]}
    """
    return {
        "text": "Hello world this is a test",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 6.0,
                "text": "Hello world this is a test",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "Hello"},
                    {"start": 0.6, "end": 1.1, "word": "world"},
                    {"start": 1.2, "end": 1.5, "word": "this"},
                    {"start": 1.6, "end": 1.8, "word": "is"},
                    {"start": 1.9, "end": 2.0, "word": "a"},
                    {"start": 2.1, "end": 2.6, "word": "test"},
                ],
            }
        ],
    }
