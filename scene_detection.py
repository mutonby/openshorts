"""Scene detection engines for the reframe pipeline.

Primary engine: TransNetV2, a small neural shot-boundary detector that is
markedly more accurate than threshold-based detection (handles fast camera
motion, flashes and gradual transitions such as fades/dissolves). Runs on
48x27 frames, faster than realtime even on CPU.

Fallback/legacy engine: PySceneDetect ContentDetector, byte-for-byte the
pre-existing behavior. Any TransNetV2 failure (missing package, corrupt
weights, decode error) falls back automatically so a scene-detection edge
case can never kill a job.

Environment variables:
  SCENE_ENGINE          "transnetv2" (default) | "pyscenedetect" (legacy)
  SCENE_MIN_SEC         minimum scene length in seconds; shorter scenes are
                        merged into a neighbor (default 0.4, TransNetV2 path
                        only — the legacy path stays untouched)
  TRANSNETV2_THRESHOLD  shot-boundary probability threshold (default 0.5)
  TRANSNETV2_DEVICE     torch device: "auto" (default) | "cpu" | "cuda" | "mps"
"""

import os
import subprocess
import threading

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector

import gpu_decode

# TransNetV2 input size (width x height), fixed by the trained model.
_TN2_W, _TN2_H = 48, 27

# One shared model instance; clips can process in parallel (CLIP_WORKERS) so
# inference is serialized like the other detectors in main.py.
_TN2_LOCK = threading.Lock()
_tn2_model = None


def detect_scenes(video_path):
    """Detect scenes. Returns (scene_list, fps) where scene_list is a list of
    (FrameTimecode, FrameTimecode) pairs — the same contract PySceneDetect's
    SceneManager.get_scene_list() has always given callers."""
    engine = os.environ.get("SCENE_ENGINE", "transnetv2").strip().lower()
    if engine != "pyscenedetect":
        try:
            return _detect_transnetv2(video_path)
        except Exception as e:
            print(f"   ⚠️ TransNetV2 scene detection failed "
                  f"({type(e).__name__}: {e}) — falling back to PySceneDetect")
    return _detect_pyscenedetect(video_path)


# --- legacy engine ----------------------------------------------------------

def _detect_pyscenedetect(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps


# --- TransNetV2 engine ------------------------------------------------------

def _get_tn2_model():
    global _tn2_model
    if _tn2_model is None:
        from transnetv2_pytorch import TransNetV2
        device = os.environ.get("TRANSNETV2_DEVICE", "auto")
        model = TransNetV2(device=device)
        model.eval()
        _tn2_model = model
    return _tn2_model


def _extract_frames_small(video_path):
    """Decode the whole clip as 48x27 RGB frames via ffmpeg (~4KB/frame).

    Decoded on the card when it can (gpu_decode: same frames). The number of
    frames decoded is remembered for the trajectory pass
    (``gpu_decode.decoded_frames``)."""
    def build(on_gpu):
        return ["ffmpeg", "-nostdin", *gpu_decode.input_args(on_gpu), "-i", video_path,
                "-vf", gpu_decode.vf(f"scale={_TN2_W}:{_TN2_H}", on_gpu),
                "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]

    def runner(cmd):
        return subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=900)

    proc = gpu_decode.run(build, runner, ok=lambda r: r.returncode == 0 and r.stdout,
                          path=video_path, render=False)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    frame_bytes = _TN2_H * _TN2_W * 3
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError("ffmpeg produced no frames")
    if gpu_decode.one_frame_per_decoded_frame(proc.stderr):
        gpu_decode.remember_count(video_path, n)
    return np.frombuffer(proc.stdout[:n * frame_bytes],
                         dtype=np.uint8).reshape(n, _TN2_H, _TN2_W, 3)


def _predict(frames):
    """TransNetV2's per-frame shot-boundary probabilities for ``frames``."""
    import torch
    model = _get_tn2_model()
    with _TN2_LOCK, torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(model.device)
        single_frame_pred, _ = model.predict_frames(tensor, quiet=True)
        single_frame_pred = single_frame_pred.cpu()
        del tensor
        # torch's caching allocator keeps the peak of this pass reserved for
        # the rest of the process; eight jobs doing that filled a 20 GB card.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return single_frame_pred.numpy().ravel()


# --- Prepared probabilities (a remote worker's clips) --------------------------
# The host with the GPU runs TransNetV2 once over the whole source (main.py
# --prepare-only, PREPARE_SCENES) and leaves its per-frame probabilities in the
# job directory; a CPU worker that renders the job reads them instead of
# running the network on every clip, which was about half of its CPU. A clip
# is cut out of the source at a known start, so its frames map onto the
# source's by time (cut_rate_filter may change the frame rate). Measured over
# 24 clips of 8 corpus sources (19-nov-2026): the 72 cuts found clip by clip
# were all found from the sliced probabilities within 1.5 frames, plus one
# extra. The clip is still decoded small: its exact frame count drives the
# sparse trajectory pass (gpu_decode.remember_count).
SCENES_CHECKPOINT = ".scene_probs.json"
# Longest source the probabilities are kept for, in frames (2 h at 60 fps;
# remote_worker.MAX_SCENES_FRAMES checks the same on the wire).
MAX_PREPARED_FRAMES = 432_000
_prepared = {"probs": None, "fps": None}
_clip_starts = {}
_clip_lock = threading.Lock()


def _source_key(video_path, duration):
    return {"name": os.path.basename(video_path), "duration": round(float(duration), 1)}


def source_probabilities(video_path):
    """(probabilities, fps) of the whole source, for main.prepare_scenes."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    frames = _extract_frames_small(video_path)
    return _predict(frames), float(fps)


def encode_probabilities(probs):
    """Probabilities as base64 of their float32 bytes (little-endian): the
    very numbers the network gave, so no cut moves on the way."""
    import base64
    return base64.b64encode(np.asarray(probs, dtype="<f4").tobytes()).decode("ascii")


def decode_probabilities(text, n):
    import base64
    raw = base64.b64decode(text, validate=True)
    if len(raw) != 4 * n:
        raise ValueError("probabilities do not match their frame count")
    probs = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if not np.isfinite(probs).all():
        raise ValueError("probabilities are not finite")
    return probs


def scenes_payload(video_path, duration, probs, fps):
    return {"source": _source_key(video_path, duration), "fps": round(float(fps), 6),
            "frames": int(len(probs)), "probs": encode_probabilities(probs)}


def valid_scenes_payload(payload):
    """The same shape check the API host makes (remote_worker.valid_scenes)."""
    import remote_worker
    return remote_worker.valid_scenes(payload)


def use_prepared(output_dir, video_path, duration):
    """Load the job's prepared probabilities when they belong to THIS source.
    True when the clips will use them. Any problem: False, and every clip
    runs the network as it always did."""
    import json
    path = os.path.join(output_dir, SCENES_CHECKPOINT)
    if not os.path.isfile(path) or os.path.islink(path):
        return False
    try:
        with open(path) as f:
            payload = json.load(f)
        if not valid_scenes_payload(payload):
            raise ValueError("bad shape")
        expected = _source_key(video_path, duration)
        if payload["source"]["name"] != expected["name"] \
                or abs(float(payload["source"]["duration"]) - expected["duration"]) > 0.5:
            print("⏭️ Prepared scenes belong to another source — ignoring them.")
            return False
        _prepared["probs"] = decode_probabilities(payload["probs"], payload["frames"])
        _prepared["fps"] = float(payload["fps"])
        print(f"⏩ Scene probabilities prepared for this source "
              f"({payload['frames']} frames): the clips skip TransNetV2.")
        return True
    except Exception as e:
        print(f"⚠️ Ignoring unusable prepared scenes ({type(e).__name__}: {e}).")
        return False


def register_clip(clip_path, start):
    """``clip_path`` was cut out of the source from ``start`` seconds."""
    with _clip_lock:
        _clip_starts[os.path.abspath(clip_path)] = float(start)


def _prepared_for(video_path, n, fps):
    """The source's probabilities mapped onto this clip's ``n`` frames, or
    None when there are none for it."""
    probs, src_fps = _prepared["probs"], _prepared["fps"]
    with _clip_lock:
        start = _clip_starts.get(os.path.abspath(video_path))
    if probs is None or start is None or not fps or n <= 0:
        return None
    t = start + np.arange(n) / float(fps)
    idx = np.rint(t * src_fps).astype(np.int64)
    if idx[-1] >= len(probs) + 2:
        return None  # the clip runs past the prepared source: not this one
    return probs[np.clip(idx, 0, len(probs) - 1)]


def _detect_transnetv2(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames = _extract_frames_small(video_path)
    model = _get_tn2_model()
    threshold = float(os.environ.get("TRANSNETV2_THRESHOLD", "0.5"))

    single_frame_pred = _prepared_for(video_path, len(frames), fps)
    if single_frame_pred is None:
        single_frame_pred = _predict(frames)

    # predictions_to_scenes returns [[start, end], ...] with INCLUSIVE ends;
    # downstream expects PySceneDetect's exclusive ends.
    raw = model.predictions_to_scenes(single_frame_pred, threshold=threshold)
    bounds = [(int(s), int(e) + 1) for s, e in raw]

    # cv2's frame count can differ by a few frames from what ffmpeg decodes;
    # downstream loops run to the decoder's count, so cover the gap.
    if total_frames > bounds[-1][1]:
        bounds[-1] = (bounds[-1][0], total_frames)

    min_sec = float(os.environ.get("SCENE_MIN_SEC", "0.4"))
    bounds = _merge_short_scenes(bounds, fps, min_sec)

    scene_list = [(FrameTimecode(s, fps), FrameTimecode(e, fps))
                  for s, e in bounds]
    print(f"   🎬 Scene engine: TransNetV2 — {len(scene_list)} scenes")
    return scene_list, fps


def _merge_short_scenes(bounds, fps, min_sec):
    """Absorb scenes shorter than min_sec into a neighbor. Ultra-short scenes
    cause camera-snap bursts and starve the per-scene strategy sampling."""
    if len(bounds) <= 1 or min_sec <= 0:
        return bounds
    min_frames = max(1, int(round(min_sec * float(fps))))

    merged = []
    for s, e in bounds:
        if merged and (e - s) < min_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    # The first scene can still be short — fold it into the one after it.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_frames:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged
