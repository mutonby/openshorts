"""Read sampled frames without a seek per sample.

The scene classifiers sample a handful of frames per scene with
``cap.set(CAP_PROP_POS_FRAMES, i); cap.read()``. On H.264 every such seek
decodes again from the previous keyframe, so a clip's ~100 samples decoded
the same GOPs over and over: 89 CPU-s for one 20 s clip, against 6 CPU-s for
a single forward pass that keeps the frames asked for (prod bench,
25-sep-2026). The frames are byte-identical either way (checked on 217
samples of NVENC-cut clips), so every downstream decision stays the same.
"""
import subprocess
import tempfile

import cv2
import numpy as np

import ffmpeg_utils
import gpu_decode

# Past this many frames ahead, one seek is cheaper than decoding forward to
# the target: OpenCV seeks 16 frames short of it and decodes from the keyframe
# before that, so about a GOP (ffmpeg_utils.cut_gop; 250 at the encoders'
# default spacing).
MAX_FORWARD = (ffmpeg_utils.cut_gop() + 16) if ffmpeg_utils.cut_gop() else 300


def read_at(cap, indices, max_forward=MAX_FORWARD):
    """Yield ``cap.read()``'s frame for each index in ``indices``, in order,
    or None where a seek + read would have failed (past the last frame).

    ``cap`` must be freshly opened (positioned at frame 0). Indices should be
    non-decreasing to get the forward-pass saving; a step backwards or a jump
    past ``max_forward`` seeks exactly like the callers used to. A repeated
    index yields the same frame object again: treat frames as read-only.
    """
    pos = 0            # index of the frame the next grab() decodes
    last = (None, None)
    for idx in indices:
        idx = int(idx)
        if idx == last[0]:
            yield last[1]
            continue
        if idx < pos or idx - pos > max_forward:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pos = idx
        frame = None
        while pos < idx:
            if not cap.grab():
                break
            pos += 1
        if pos == idx:
            ok, frame = cap.read()
            pos += 1
            if not ok:
                frame = None
        last = (idx, frame)
        yield frame


def read_at_path(video_path, cap, indices, max_forward=MAX_FORWARD):
    """``read_at`` for a file, decoding the samples on the card when it can.

    One ffmpeg pass decodes the clip with NVDEC and downloads only the asked
    frames, converted to BGR the way ``cap.read()`` converts them (see
    gpu_decode): the same frames for a fraction of the CPU. Falls back to
    ``read_at(cap, ...)`` for indices that are not non-decreasing, when the
    card is not usable, and for whatever is left if that ffmpeg fails.
    ``cap`` must be freshly opened, as for ``read_at``.
    """
    indices = [int(i) for i in indices]
    if (not indices or indices[0] < 0
            or any(b < a for a, b in zip(indices, indices[1:]))
            or not gpu_decode.usable(video_path)):
        yield from read_at(cap, indices, max_forward)
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        yield from read_at(cap, indices, max_forward)
        return
    unique = sorted(set(indices))
    frame_bytes = width * height * 3
    expr = gpu_decode.any_of(f"eq(n,{i})" for i in unique)
    errors = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", *gpu_decode.HW_ARGS, "-i", video_path,
         "-vf", f"select='{expr}',{gpu_decode.DOWNLOAD}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=errors, bufsize=frame_bytes)
    done = 0           # how many of ``indices`` were answered from the card
    last = (None, None)
    ended = False
    failed = False
    try:
        for idx in indices:
            if idx == last[0]:
                yield last[1]
                done += 1
                continue
            frame = None
            if not ended:
                buf = bytearray(frame_bytes)
                got = proc.stdout.readinto(buf)
                if got == frame_bytes:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                else:
                    ended = True
                    proc.stdout.close()
                    # Past the clip's last frame read_at gives None too; an
                    # ffmpeg that failed leaves the rest to the CPU.
                    failed = proc.wait() != 0
                    if failed:
                        break
            last = (idx, frame)
            done += 1
            yield frame
    finally:
        if not proc.stdout.closed:
            proc.stdout.close()
        proc.kill()
        proc.wait()
    gpu_decode.note(True, failed=failed)
    errors.seek(0)
    detail = errors.read()[-200:].decode(errors="replace").strip().replace("\n", " | ")
    errors.close()
    if failed:
        print(f"   ⚠️ GPU frame sampling failed (exit {proc.returncode}, "
              f"{done}/{len(indices)} read: {detail or 'no message'}), "
              f"reading the rest on the CPU")
        yield from read_at(cap, indices[done:], max_forward)
