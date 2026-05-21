# AI Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new sidebar product "AI Restyle" that uploads a ≤30s video, relights its first frame via Nano Banana (Gemini 2.5 Flash image preview), and uses that frame as a style reference for video-to-video restyling that preserves the original motion + audio.

**Architecture:** New backend route `/api/restyle` orchestrating a 7-step pipeline (validate → probe → extract first frame → relight → v2v → mux audio → persist). Frontend 3-step wizard (Upload → Configure → Review) plus a Settings tab for preset CRUD. Presets are 2-dimensional (background + lighting) and live in browser localStorage. Full design at `docs/superpowers/specs/2026-05-20-ai-restyle-design.md`.

**Tech Stack:** Python 3.11 + FastAPI + google-genai + fal_client + FFmpeg (backend); React 18 + Vite + Tailwind + lucide-react (frontend); pytest (backend tests); browser smoke test via chrome-devtools MCP (frontend gate).

---

## Phase 0 — Model selection spike (research, ~0.5 day)

### Task 0.1: Validate the video-to-video model

**Files:**
- Create: `docs/superpowers/specs/2026-05-20-ai-restyle-phase0-spike.md`

This phase produces a research artifact, not code. The spec defers model choice to this spike.

- [ ] **Step 1: Survey fal.ai catalog**

Open https://fal.ai/models and filter by "video-to-video". Document candidates that accept a source video AND a reference image (or that can be combined with a separate img2img reference). As of 2026-05 candidates to evaluate (verify each still exists):
- `fal-ai/wan/v2.5/turbo/video-to-video`
- `fal-ai/luma-photon` (Luma's photon family)
- `fal-ai/runway-gen3-alpha-turbo/video-to-video` (if hosted on fal.ai)
- `fal-ai/pixverse/v4/restyle` (if exists)

- [ ] **Step 2: Pick one candidate; run a 5s test gen**

Use `demo-openshorts.mp4` (already in repo root) — trim to 5s with `ffmpeg -i demo-openshorts.mp4 -t 5 -c copy /tmp/spike-5s.mp4`. Generate a reference frame by hand: run the first frame through https://aistudio.google.com/ with prompt "relight this image with Bahamas beach background and golden hour lighting; keep subject and pose unchanged". Save reference frame to `/tmp/spike-ref.png`.

Call the candidate model via `fal_client` (requires `FAL_KEY` env var):

```python
import fal_client
res = fal_client.run(
    "fal-ai/<chosen-model>",
    arguments={
        "video_url": <upload /tmp/spike-5s.mp4>,
        "image_url": <upload /tmp/spike-ref.png>,
        "prompt": "match the lighting and background of the reference image",
    },
)
print(res)
```

Time the call. Inspect the output video.

- [ ] **Step 3: Score against acceptance criteria**

Acceptance bar (from spec §3 `video_restyle.py`):
- Accepts source video AND reference image
- Restyles to match reference's lighting + background
- Preserves motion + content (subject still talks/moves the same)
- Cost: ≤$2 per 30s gen (linear-extrapolate from the 5s test)
- Latency: ≤5min per 30s gen

If the chosen candidate fails on any criterion, repeat Step 2 with the next candidate. If all fail, escalate to direct Runway API integration (write a new spike doc; pivot the plan).

- [ ] **Step 4: Write the spike doc**

Create `docs/superpowers/specs/2026-05-20-ai-restyle-phase0-spike.md` with:
- Models surveyed (table: name + URL + supports v2v + supports ref image)
- Test methodology (which clip, which reference, the prompt)
- Per-model results (quality screenshot, cost, latency)
- **Decision:** chosen model ID, justification
- Output payload shape (the exact response JSON from the chosen model) — required for Phase 2 Task 2.1.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-20-ai-restyle-phase0-spike.md
git commit -m "docs(ai-restyle): Phase 0 spike — picked <model> for video-to-video"
```

---

## Phase 1 — Backend ML: frame extract + relight (~1 day)

### Task 1.1: First-frame extractor

**Files:**
- Create: `backend/app/ml/frame_extract.py`
- Test:   `backend/tests/unit/test_frame_extract.py`
- Existing fixture: `demo-openshorts.mp4` (repo root, 5.3 MB) — copy a trimmed version into `backend/tests/fixtures/` (create the dir).

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_frame_extract.py
"""Tests for ml/frame_extract: extract first frame of a video to PNG."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.ml.frame_extract import extract_first_frame


FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "short-clip.mp4"


@pytest.fixture(scope="module", autouse=True)
def _ensure_fixture():
    """Trim demo-openshorts.mp4 to 5s on first run so the test is fast."""
    if FIXTURE.exists():
        return
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    src = repo_root / "demo-openshorts.mp4"
    if not src.exists():
        pytest.skip("demo-openshorts.mp4 fixture missing")
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", "5", "-c", "copy", str(FIXTURE)],
        check=True, capture_output=True,
    )


def test_extract_first_frame_writes_png(tmp_path):
    out = tmp_path / "frame.png"
    result = extract_first_frame(str(FIXTURE), str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 1000  # not empty


def test_extract_first_frame_missing_input(tmp_path):
    out = tmp_path / "frame.png"
    with pytest.raises(FileNotFoundError):
        extract_first_frame(str(tmp_path / "does-not-exist.mp4"), str(out))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/unit/test_frame_extract.py -v
```

Expected: `ImportError: cannot import name 'extract_first_frame' from 'app.ml.frame_extract'` (module doesn't exist yet)

- [ ] **Step 3: Implement extract_first_frame**

```python
# backend/app/ml/frame_extract.py
"""Extract the first video frame to a PNG file via FFmpeg."""
from __future__ import annotations

import os

from app.video.ffmpeg import run as ffmpeg_run, FFmpegError


def extract_first_frame(video_path: str, out_path: str) -> str:
    """Write the frame at t=0 of ``video_path`` to ``out_path`` as PNG.

    Returns ``out_path`` on success. Raises ``FileNotFoundError`` if the
    source is missing; ``FFmpegError`` if encoding fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ffmpeg_run(
        ["-y", "-ss", "0", "-i", video_path, "-frames:v", "1", "-update", "1", out_path],
    )
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && pytest tests/unit/test_frame_extract.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd "/Users/matissevansteenbergen/Downloads/AGENTIC WORKLFOWS/PERSONAL/Auto-shorts (TODO)/openshorts"
git add backend/app/ml/frame_extract.py backend/tests/unit/test_frame_extract.py
git commit -m "feat(ai-restyle): ml/frame_extract first-frame extractor"
```

---

### Task 1.2: Nano Banana relight

**Files:**
- Create: `backend/app/ml/frame_relight.py`
- Test:   `backend/tests/unit/test_frame_relight.py`
- Reference pattern: `backend/app/thumbnails/images.py:generate_thumbnail` (same model, similar shape)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_frame_relight.py
"""Tests for ml/frame_relight: Nano Banana relight call.

Gemini client is mocked — we don't want network calls or API costs in unit
tests. We do verify the prompt template structure, the model name, and
the file write contract.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ml.frame_relight import (
    SAFETY_CONSTRAINTS,
    build_relight_prompt,
    relight_frame,
)


def test_build_relight_prompt_contains_inputs():
    p = build_relight_prompt("bahamas beach", "golden hour")
    assert "bahamas beach" in p.lower()
    assert "golden hour" in p.lower()


def test_build_relight_prompt_contains_safety_constraints():
    p = build_relight_prompt("x", "y")
    for clause in SAFETY_CONSTRAINTS:
        assert clause in p


def test_relight_frame_calls_gemini_image_preview_model(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out.png"

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = MagicMock(data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    fake_resp.candidates = [MagicMock(content=MagicMock(parts=[fake_part]))]
    fake_client.models.generate_content.return_value = fake_resp

    with patch("app.ml.frame_relight.genai.Client", return_value=fake_client):
        result = relight_frame(
            api_key="fake-key",
            frame_path=str(src),
            background_prompt="bahamas beach",
            lighting_prompt="golden hour",
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash-image-preview"


def test_relight_frame_missing_input(tmp_path):
    out = tmp_path / "out.png"
    with pytest.raises(FileNotFoundError):
        relight_frame(
            api_key="x",
            frame_path=str(tmp_path / "missing.png"),
            background_prompt="x",
            lighting_prompt="y",
            out_path=str(out),
        )


def test_relight_frame_handles_no_inline_data(tmp_path):
    """Model sometimes returns text instead of image — must raise, not silently write nothing."""
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out.png"

    fake_client = MagicMock()
    fake_resp = MagicMock()
    fake_part = MagicMock(inline_data=None, text="sorry, can't comply")
    fake_resp.candidates = [MagicMock(content=MagicMock(parts=[fake_part]))]
    fake_client.models.generate_content.return_value = fake_resp

    with patch("app.ml.frame_relight.genai.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="no image"):
            relight_frame(
                api_key="x",
                frame_path=str(src),
                background_prompt="x",
                lighting_prompt="y",
                out_path=str(out),
            )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/unit/test_frame_relight.py -v
```

Expected: `ImportError: cannot import name 'relight_frame'` (module doesn't exist).

- [ ] **Step 3: Implement frame_relight**

```python
# backend/app/ml/frame_relight.py
"""Nano Banana relight: send a frame + relight prompts to Gemini's image
preview model and write the relit frame to disk.

Mirrors the call pattern in ``backend/app/thumbnails/images.py:generate_thumbnail``.
"""
from __future__ import annotations

import os
from typing import List

from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-image-preview"

SAFETY_CONSTRAINTS: List[str] = [
    "Keep the person, pose, clothing, and composition EXACTLY as in the source.",
    "Do not add or remove any people or objects.",
    "Do not change facial features or body proportions.",
    "Preserve the framing and camera angle.",
]


def build_relight_prompt(background_prompt: str, lighting_prompt: str) -> str:
    """Compose the Nano Banana prompt from user-controlled fragments + safety."""
    safety_block = "\n".join(f"- {c}" for c in SAFETY_CONSTRAINTS)
    return (
        "Relight this image with the following style. Only change the "
        "background and lighting.\n\n"
        f"Background: {background_prompt}\n"
        f"Lighting: {lighting_prompt}\n\n"
        "Constraints:\n"
        f"{safety_block}"
    )


def relight_frame(
    api_key: str,
    frame_path: str,
    background_prompt: str,
    lighting_prompt: str,
    out_path: str,
) -> str:
    """Call Nano Banana with the input frame + prompts. Writes relit PNG to ``out_path``.

    Returns ``out_path``. Raises ``FileNotFoundError`` if input missing,
    ``RuntimeError`` if the response carries no image data.
    """
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Input frame not found: {frame_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    client = genai.Client(api_key=api_key)
    prompt = build_relight_prompt(background_prompt, lighting_prompt)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    for part in response.candidates[0].content.parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            with open(out_path, "wb") as f:
                f.write(part.inline_data.data)
            return out_path

    raise RuntimeError("Nano Banana returned no image (likely content policy)")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && pytest tests/unit/test_frame_relight.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run the full backend gate**

```bash
cd backend && pytest -m "not e2e" -q
```

Expected: 157 passed (150 prior + 7 new) — or close to it. If anything else breaks, fix before continuing.

- [ ] **Step 6: Commit**

```bash
git add backend/app/ml/frame_relight.py backend/tests/unit/test_frame_relight.py
git commit -m "feat(ai-restyle): ml/frame_relight Nano Banana wrapper + safety constraints"
```

---

## Phase 2 — Backend ML: video restyle + pipeline (~1.5 days)

### Task 2.1: Video-to-video restyle module

**Files:**
- Create: `backend/app/ml/video_restyle.py`
- Test:   `backend/tests/unit/test_video_restyle.py`
- Reference pattern: `backend/app/saas/pipeline.py:generate_talking_head` (existing fal.ai integration)
- Depends on: Phase 0 spike — replace `<MODEL_NAME>` and `<PAYLOAD_KEY>` placeholders below with the values from the spike doc.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_video_restyle.py
"""Tests for ml/video_restyle: fal.ai video-to-video call.

fal_client is mocked. We assert the model id, payload keys, and that the
output file is written from the returned URL.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ml.video_restyle import MODEL_ID, restyle_video


def test_restyle_video_calls_chosen_model(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake video")
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"fake png")
    out = tmp_path / "out.mp4"

    fake_run = MagicMock(return_value={"video": {"url": "https://fake.fal.ai/out.mp4"}})
    fake_upload = MagicMock(side_effect=["https://fake/video.mp4", "https://fake/ref.png"])

    fake_resp = MagicMock()
    fake_resp.iter_bytes = MagicMock(return_value=iter([b"video-bytes"]))
    fake_resp.raise_for_status = MagicMock()
    fake_httpx = MagicMock()
    fake_httpx.stream.return_value.__enter__.return_value = fake_resp

    with patch("app.ml.video_restyle.fal_client.subscribe", fake_run), \
         patch("app.ml.video_restyle.fal_client.upload_file", fake_upload), \
         patch("app.ml.video_restyle.httpx.Client", return_value=fake_httpx):
        result = restyle_video(
            api_key="fake",
            video_path=str(video),
            reference_frame_path=str(ref),
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
    assert out.read_bytes() == b"video-bytes"
    fake_run.assert_called_once()
    assert fake_run.call_args.args[0] == MODEL_ID


def test_restyle_video_missing_inputs(tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError):
        restyle_video(
            api_key="x",
            video_path=str(tmp_path / "missing.mp4"),
            reference_frame_path=str(tmp_path / "missing.png"),
            out_path=str(out),
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/unit/test_video_restyle.py -v
```

Expected: `ImportError: cannot import name 'MODEL_ID' from 'app.ml.video_restyle'`.

- [ ] **Step 3: Implement video_restyle**

```python
# backend/app/ml/video_restyle.py
"""fal.ai video-to-video restyle: send a source video + reference frame,
download the restyled output.

Model and payload shape were chosen during Phase 0 spike. See
``docs/superpowers/specs/2026-05-20-ai-restyle-phase0-spike.md``.
"""
from __future__ import annotations

import os

import fal_client
import httpx

# Set during Phase 0 spike. Replace if the spike picked a different model.
MODEL_ID = "fal-ai/wan/v2.5/turbo/video-to-video"


def restyle_video(
    api_key: str,
    video_path: str,
    reference_frame_path: str,
    out_path: str,
) -> str:
    """Run fal.ai v2v with the source video and reference frame. Writes
    the restyled MP4 to ``out_path`` and returns it.

    Raises ``FileNotFoundError`` if either input is missing. fal_client
    raises its own errors for API / network failures — those propagate.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not os.path.exists(reference_frame_path):
        raise FileNotFoundError(f"Reference frame not found: {reference_frame_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    os.environ["FAL_KEY"] = api_key  # fal_client reads this env var

    video_url = fal_client.upload_file(video_path)
    ref_url = fal_client.upload_file(reference_frame_path)

    response = fal_client.subscribe(
        MODEL_ID,
        arguments={
            "video_url": video_url,
            "image_url": ref_url,
            "prompt": "Match the lighting and background of the reference image. Preserve all motion, subject, and camera angle from the source video.",
        },
        with_logs=False,
    )

    out_url = response["video"]["url"]
    with httpx.Client(timeout=300.0) as client:
        with client.stream("GET", out_url) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    return out_path
```

> ⚠️ **If Phase 0 picked a different model**, replace `MODEL_ID` and adjust the `arguments` keys (`video_url`, `image_url`, `prompt`) to match the model's documented input schema. Run the spike's recorded sample payload through this function manually to verify before continuing.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && pytest tests/unit/test_video_restyle.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml/video_restyle.py backend/tests/unit/test_video_restyle.py
git commit -m "feat(ai-restyle): ml/video_restyle fal.ai v2v wrapper"
```

---

### Task 2.2: Pipeline orchestrator

**Files:**
- Create: `backend/app/saas/restyle_pipeline.py`
- Test:   `backend/tests/unit/test_restyle_pipeline.py`
- Reference pattern: `backend/app/editing/auto_pipeline.py` (Short-form's auto-pipeline) — orchestrates multiple ML modules and writes progress to a job dict.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/unit/test_restyle_pipeline.py
"""Tests for saas/restyle_pipeline: the 7-step orchestrator.

Each ML module is mocked. We verify the pipeline:
- Calls the steps in order
- Writes status='processing' → 'completed' to the supplied jobs dict
- Captures the result video_url + duration
- Marks status='failed' if any step raises
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.saas.restyle_pipeline import run_restyle_job


@pytest.fixture
def fake_jobs(tmp_path):
    """A jobs dict primed with an in-progress entry, plus a fake input file."""
    job_id = "test-restyle-job"
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 200)
    return {
        job_id: {
            "status": "processing",
            "logs": [],
            "result": None,
            "progress_pct": 0,
        },
    }, job_id, input_file


def test_pipeline_happy_path(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    output_dir = tmp_path / "output" / job_id
    output_dir.mkdir(parents=True)

    monkeypatch.setattr("app.saas.restyle_pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.saas.restyle_pipeline.probe_duration", return_value=12.0), \
         patch("app.saas.restyle_pipeline.extract_first_frame") as fake_extract, \
         patch("app.saas.restyle_pipeline.relight_frame") as fake_relight, \
         patch("app.saas.restyle_pipeline.restyle_video") as fake_v2v, \
         patch("app.saas.restyle_pipeline.mux_video_audio") as fake_mux:
        # Each step "writes" its output by returning the path it was given
        fake_extract.side_effect = lambda src, dst: dst
        fake_relight.side_effect = lambda **kw: kw["out_path"]
        fake_v2v.side_effect = lambda **kw: kw["out_path"]
        fake_mux.side_effect = lambda video, audio_src, out: out

        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="bahamas",
            lighting_prompt="golden",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "completed"
    assert job["result"]["video_url"].endswith(".mp4")
    assert job["result"]["duration_sec"] == 12.0
    assert job["progress_pct"] == 100
    fake_extract.assert_called_once()
    fake_relight.assert_called_once()
    fake_v2v.assert_called_once()
    fake_mux.assert_called_once()


def test_pipeline_marks_failed_when_relight_raises(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.saas.restyle_pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.saas.restyle_pipeline.probe_duration", return_value=10.0), \
         patch("app.saas.restyle_pipeline.extract_first_frame", side_effect=lambda src, dst: dst), \
         patch("app.saas.restyle_pipeline.relight_frame", side_effect=RuntimeError("content policy")), \
         patch("app.saas.restyle_pipeline.restyle_video") as fake_v2v, \
         patch("app.saas.restyle_pipeline.mux_video_audio") as fake_mux:
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "failed"
    assert any("content policy" in line for line in job["logs"])
    fake_v2v.assert_not_called()
    fake_mux.assert_not_called()


def test_pipeline_rejects_videos_longer_than_30s(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.saas.restyle_pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.saas.restyle_pipeline.probe_duration", return_value=45.0):
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "failed"
    assert any("30s" in line or "duration" in line.lower() for line in job["logs"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && pytest tests/unit/test_restyle_pipeline.py -v
```

Expected: `ImportError: cannot import name 'run_restyle_job'`.

- [ ] **Step 3: Implement the orchestrator**

```python
# backend/app/saas/restyle_pipeline.py
"""AI Restyle pipeline orchestrator.

7-step pipeline:
  1. Validate (caller already enforces MIME + ftyp + size)
  2. Probe duration via ffprobe; reject if >30s
  3. Extract first frame to PNG
  4. Nano Banana relight of that frame
  5. fal.ai video-to-video with source + relit frame as reference
  6. Mux original audio back onto the restyled video
  7. Persist result_url to jobs dict; mark status=completed

Any step's exception marks the job 'failed' with the exception message
appended to logs. The job dict is mutated in place so the route handler
and frontend poll see progress.
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Dict

from app.ml.frame_extract import extract_first_frame
from app.ml.frame_relight import relight_frame
from app.ml.video_restyle import restyle_video
from app.video.ffmpeg import probe_duration, mux_video_audio

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MAX_DURATION_SEC = 30.0


def _log(jobs: Dict[str, Any], job_id: str, line: str, pct: int | None = None) -> None:
    job = jobs.get(job_id)
    if job is None:
        return
    job["logs"].append(line)
    if pct is not None:
        job["progress_pct"] = pct


async def run_restyle_job(
    jobs: Dict[str, Any],
    job_id: str,
    input_path: str,
    background_prompt: str,
    lighting_prompt: str,
    gemini_key: str,
    fal_key: str,
) -> None:
    """Run the full restyle pipeline for ``job_id``. Mutates ``jobs[job_id]`` in place."""
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    loop = asyncio.get_event_loop()

    try:
        # Step 2 — duration probe
        _log(jobs, job_id, "🔎 Probing video duration…", pct=5)
        duration = await loop.run_in_executor(None, partial(probe_duration, input_path))
        if duration > MAX_DURATION_SEC:
            raise ValueError(f"Video duration {duration:.1f}s exceeds 30s cap for AI Restyle v1")

        # Step 3 — extract first frame
        _log(jobs, job_id, "🎞️ Extracting first frame…", pct=10)
        frame_path = os.path.join(output_dir, f"{base}_frame.png")
        await loop.run_in_executor(None, partial(extract_first_frame, input_path, frame_path))

        # Step 4 — Nano Banana relight
        _log(jobs, job_id, "🪄 Relighting frame with Nano Banana…", pct=20)
        relit_path = os.path.join(output_dir, f"{base}_relit.png")
        await loop.run_in_executor(
            None,
            partial(
                relight_frame,
                api_key=gemini_key,
                frame_path=frame_path,
                background_prompt=background_prompt,
                lighting_prompt=lighting_prompt,
                out_path=relit_path,
            ),
        )
        _log(jobs, job_id, "💰 Nano Banana relight: ~$0.039 per call", pct=30)

        # Step 5 — fal.ai video-to-video
        _log(jobs, job_id, "🎬 Restyling video via fal.ai (~30-90s)…", pct=40)
        restyled_noaudio = os.path.join(output_dir, f"{base}_restyled_noaudio.mp4")
        await loop.run_in_executor(
            None,
            partial(
                restyle_video,
                api_key=fal_key,
                video_path=input_path,
                reference_frame_path=relit_path,
                out_path=restyled_noaudio,
            ),
        )
        cost_est = round(duration * 0.04, 2)
        _log(jobs, job_id, f"💰 fal.ai v2v: ~${cost_est:.2f} ({duration:.1f}s × $0.04/s)", pct=85)

        # Step 6 — mux original audio
        _log(jobs, job_id, "🔊 Muxing original audio back…", pct=90)
        final_out = os.path.join(output_dir, f"restyled_{os.path.basename(input_path)}")
        await loop.run_in_executor(
            None,
            partial(mux_video_audio, restyled_noaudio, input_path, final_out),
        )

        # Step 7 — persist result
        jobs[job_id]["result"] = {
            "video_url": f"/videos/{job_id}/{os.path.basename(final_out)}",
            "original_url": f"/videos/{job_id}/{os.path.basename(input_path)}",
            "duration_sec": duration,
        }
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress_pct"] = 100
        _log(jobs, job_id, "✅ AI Restyle complete.")

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["logs"].append(f"❌ {exc}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend && pytest tests/unit/test_restyle_pipeline.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Backend gate**

```bash
cd backend && pytest -m "not e2e" -q
```

Expected: all green, ~160 tests.

- [ ] **Step 6: Commit**

```bash
git add backend/app/saas/restyle_pipeline.py backend/tests/unit/test_restyle_pipeline.py
git commit -m "feat(ai-restyle): saas/restyle_pipeline orchestrator"
```

---

## Phase 3 — Backend routes (~0.5 day)

### Task 3.1: Routes scaffold + Pydantic + endpoints

**Files:**
- Create: `backend/app/routes/__init__.py` (empty)
- Create: `backend/app/routes/ai_restyle.py`
- Modify: `backend/app/main.py` (one-line router registration + share the jobs dict)
- Update:  `backend/tests/snapshots/baseline.openapi.json` (regen)
- Reference: `backend/app/main.py:1275` (existing color_grade route — same shape we want)

- [ ] **Step 1: Create the routes package**

```bash
echo '"""FastAPI routers, split out of main.py incrementally."""' > backend/app/routes/__init__.py
```

- [ ] **Step 2: Write the routes module**

Inspect how `main.py` exposes the shared `jobs` dict — currently it's a module-level dict. The router needs access. Pattern: expose `jobs` via `app.state` or import directly from `app.main`. Import directly to match the existing `_resolve_clip_input` pattern.

```python
# backend/app/routes/ai_restyle.py
"""AI Restyle FastAPI router.

Endpoints:
- POST /api/restyle          start a restyle job
- GET  /api/restyle/{job_id} poll status

Job state is the same in-memory dict as the rest of main.py (jobs[]).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()

# Reuse the same OUTPUT_DIR / jobs dict / upload guard as main.py.
# Imports are deferred to avoid a circular import at module load time.

MAX_DURATION_SEC = 30.0
MAX_PROMPT_LEN = 500


class RestyleStatus(BaseModel):
    status: str
    logs: list[str]
    progress_pct: int = Field(default=0, ge=0, le=100)
    result: Optional[dict] = None


@router.post("/api/restyle")
async def start_restyle(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background_prompt: str = Form(...),
    lighting_prompt: str = Form(...),
):
    """Start a restyle job. Returns ``{job_id}`` immediately; poll
    ``GET /api/restyle/{job_id}`` for status."""
    from app.main import jobs, _ensure_video_upload, OUTPUT_DIR, UPLOAD_DIR
    from app.saas.restyle_pipeline import run_restyle_job

    if len(background_prompt) > MAX_PROMPT_LEN or len(lighting_prompt) > MAX_PROMPT_LEN:
        raise HTTPException(status_code=413, detail=f"Prompt fragments must be ≤{MAX_PROMPT_LEN} chars each")

    gemini_key = request.headers.get("X-Gemini-Key")
    fal_key = request.headers.get("X-Fal-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    if not fal_key:
        raise HTTPException(status_code=401, detail="X-Fal-Key header required")

    job_id = str(uuid.uuid4())
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Persist the upload (re-using main.py's guard for MIME+ftyp)
    first_chunk = await file.read(4096)
    _ensure_video_upload(file.filename, first_chunk)
    input_path = os.path.join(output_dir, file.filename or f"{job_id}.mp4")
    with open(input_path, "wb") as out:
        out.write(first_chunk)
        shutil.copyfileobj(file.file, out)

    jobs[job_id] = {
        "status": "processing",
        "logs": [f"📥 Received {os.path.basename(input_path)}"],
        "progress_pct": 0,
        "result": None,
        "product": "ai-restyle",  # tag so /api/status doesn't get confused
    }

    background_tasks.add_task(
        asyncio.create_task,
        run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=input_path,
            background_prompt=background_prompt,
            lighting_prompt=lighting_prompt,
            gemini_key=gemini_key,
            fal_key=fal_key,
        ),
    )

    return {"job_id": job_id}


@router.get("/api/restyle/{job_id}", response_model=RestyleStatus)
async def restyle_status(job_id: str):
    from app.main import jobs
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return RestyleStatus(
        status=job["status"],
        logs=job.get("logs", []),
        progress_pct=job.get("progress_pct", 0),
        result=job.get("result"),
    )
```

- [ ] **Step 3: Write the route test**

```python
# backend/tests/api/test_ai_restyle.py
"""Contract tests for /api/restyle and /api/restyle/{job_id}."""
from __future__ import annotations

import io
from pathlib import Path

import pytest


@pytest.fixture
def restyle_client(tmp_path, monkeypatch):
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def _mp4_bytes() -> bytes:
    """Minimum-viable MP4 header to pass _ensure_video_upload's ftyp check."""
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 200


def test_post_restyle_requires_gemini_key(restyle_client):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"background_prompt": "beach", "lighting_prompt": "golden"},
        headers={"X-Fal-Key": "fake"},
    )
    assert res.status_code == 401
    assert "Gemini" in res.json()["detail"]


def test_post_restyle_requires_fal_key(restyle_client):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"background_prompt": "beach", "lighting_prompt": "golden"},
        headers={"X-Gemini-Key": "fake"},
    )
    assert res.status_code == 401
    assert "Fal" in res.json()["detail"]


def test_post_restyle_rejects_long_prompt(restyle_client):
    res = restyle_client.post(
        "/api/restyle",
        files={"file": ("clip.mp4", io.BytesIO(_mp4_bytes()), "video/mp4")},
        data={"background_prompt": "x" * 600, "lighting_prompt": "y"},
        headers={"X-Gemini-Key": "g", "X-Fal-Key": "f"},
    )
    assert res.status_code == 413


def test_get_restyle_status_not_found(restyle_client):
    res = restyle_client.get("/api/restyle/nonexistent")
    assert res.status_code == 404
```

- [ ] **Step 4: Run the new tests — expect ImportError**

```bash
cd backend && pytest tests/api/test_ai_restyle.py -v
```

Expected: collection error / import error because the router isn't registered yet.

- [ ] **Step 5: Register the router in main.py**

Find the existing FastAPI app object (`app = FastAPI(...)` near the top of `main.py`). After it's instantiated, add:

```python
# main.py — after app = FastAPI(...)
from app.routes.ai_restyle import router as ai_restyle_router
app.include_router(ai_restyle_router)
```

- [ ] **Step 6: Run the route tests**

```bash
cd backend && pytest tests/api/test_ai_restyle.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Regenerate the OpenAPI snapshot**

```bash
cd backend && rm tests/snapshots/baseline.openapi.json && pytest tests/api/test_openapi_contract.py -v
```

Expected: the contract test re-baselines and passes. Inspect the diff:

```bash
git diff backend/tests/snapshots/baseline.openapi.json | grep -E '"/api/restyle' || echo "WARNING: new routes not in baseline"
```

Expected: matches show `"/api/restyle"` and `"/api/restyle/{job_id}"` paths are added.

- [ ] **Step 8: Backend gate**

```bash
cd backend && pytest -m "not e2e" -q
```

Expected: ~167 tests, all green.

- [ ] **Step 9: Commit**

```bash
git add backend/app/routes/ backend/app/main.py backend/tests/api/test_ai_restyle.py backend/tests/snapshots/baseline.openapi.json
git commit -m "feat(ai-restyle): /api/restyle + /api/restyle/{job_id} routes"
```

---

## Phase 4 — Frontend (~2 days, split across 4a + 4b)

### Task 4a.1: Preset store (shared dependency for wizard + Settings tab)

**Files:**
- Create: `frontend/src/state/aiRestylePresets.js`
- Reference pattern: `frontend/src/state/keysStore.js` (event-broadcasting localStorage store)

- [ ] **Step 1: Implement the preset store**

```javascript
// frontend/src/state/aiRestylePresets.js
// AI Restyle preset store. Two dimensions (backgrounds + lightings), each a
// list of { id, label, prompt } records with one marked as default via
// `defaultBackgroundId` / `defaultLightingId`. Persisted to localStorage and
// broadcast via a custom event so any subscribed component re-renders.
//
// Mirrors the keysStore.js + brandKit.js pattern. Seeded with 5 hand-tuned
// presets per dimension on first load.

import { useEffect, useState } from 'react';

const STORAGE_KEY = 'openshorts.aiRestyle.presets';
const EVENT = 'openshorts:ai-restyle-presets-changed';

const SEED = {
  backgrounds: [
    { id: 'studio-white',     label: 'Studio white',     prompt: 'clean white seamless backdrop, minimalist photo studio, no clutter, perfect color separation' },
    { id: 'sunlit-office',    label: 'Sunlit office',    prompt: 'bright modern office interior with floor-to-ceiling windows, soft natural light, plants, wooden desk' },
    { id: 'bahamas-beach',    label: 'Bahamas beach',    prompt: 'tropical beach with palm trees, turquoise ocean water in the distance, soft white sand' },
    { id: 'cyberpunk-neon',   label: 'Cyberpunk neon',   prompt: 'nighttime city street with vivid neon signs, pink-and-cyan color palette, light fog' },
    { id: 'cinematic-forest', label: 'Cinematic forest', prompt: 'deep forest with dappled sunlight through tall pine trees, mossy ground, atmospheric haze' },
  ],
  lightings: [
    { id: 'studio-softbox',   label: 'Studio softbox',   prompt: 'soft diffused studio softbox lighting from camera-left, gentle fill on the right, no harsh shadows' },
    { id: 'sunlit-office',    label: 'Sunlit office',    prompt: 'bright daylight pouring through large windows, soft fill on subject\'s face' },
    { id: 'golden-hour',      label: 'Golden hour',      prompt: 'warm golden-hour sun low and to the side, long shadows, amber and rose tones' },
    { id: 'cinematic-moody',  label: 'Cinematic moody',  prompt: 'low-key cinematic lighting with strong directional key, deep shadows, single soft fill' },
    { id: 'neon-nighttime',   label: 'Neon nighttime',   prompt: 'colored neon spill lighting (pink and cyan accents), low ambient, subject lit from multiple sides' },
  ],
  defaultBackgroundId: 'studio-white',
  defaultLightingId: 'studio-softbox',
};

function read() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return seedOnce();
    const data = JSON.parse(raw);
    if (!data.backgrounds || !data.lightings) return seedOnce();
    return data;
  } catch {
    return seedOnce();
  }
}

function seedOnce() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(SEED)); } catch {/* ignore */}
  return SEED;
}

function write(next) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); } catch {/* ignore */}
  window.dispatchEvent(new CustomEvent(EVENT, { detail: next }));
}

export function getPresets() { return read(); }

export function setDefault(dimension, id) {
  const cur = read();
  const key = dimension === 'background' ? 'defaultBackgroundId' : 'defaultLightingId';
  write({ ...cur, [key]: id });
}

export function upsertPreset(dimension, preset) {
  const cur = read();
  const list = dimension === 'background' ? cur.backgrounds : cur.lightings;
  const next = list.some(p => p.id === preset.id)
    ? list.map(p => p.id === preset.id ? preset : p)
    : [...list, preset];
  const dimKey = dimension === 'background' ? 'backgrounds' : 'lightings';
  write({ ...cur, [dimKey]: next });
}

export function deletePreset(dimension, id) {
  const cur = read();
  const list = dimension === 'background' ? cur.backgrounds : cur.lightings;
  const defaultKey = dimension === 'background' ? 'defaultBackgroundId' : 'defaultLightingId';
  if (cur[defaultKey] === id) return; // can't delete the default
  const next = list.filter(p => p.id !== id);
  const dimKey = dimension === 'background' ? 'backgrounds' : 'lightings';
  write({ ...cur, [dimKey]: next });
}

export function useAIRestylePresets() {
  const [state, setState] = useState(() => read());
  useEffect(() => {
    const onChange = (e) => setState(e.detail || read());
    window.addEventListener(EVENT, onChange);
    return () => window.removeEventListener(EVENT, onChange);
  }, []);
  return state;
}
```

- [ ] **Step 2: Smoke-check the store in the browser**

```bash
cd "/Users/matissevansteenbergen/Downloads/AGENTIC WORKLFOWS/PERSONAL/Auto-shorts (TODO)/openshorts/frontend" && npm run build
```

Expected: 0 errors. (No unit tests in this repo for frontend stores — pattern matches `keysStore.js` which is also untested. Smoke-tested via the wizard in Phase 6.)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/state/aiRestylePresets.js
git commit -m "feat(ai-restyle): preset store (localStorage + event broadcast)"
```

---

### Task 4b.1: AIRestyle pages folder + Wizard.jsx

**Files:**
- Create: `frontend/src/pages/AIRestyle/index.jsx`
- Create: `frontend/src/pages/AIRestyle/Wizard.jsx`
- Create: `frontend/src/pages/AIRestyle/History.jsx` (read-only stub — past job list, deferred to follow-up)
- Reference pattern: `frontend/src/pages/ShortForm/index.jsx` + `Wizard.jsx`

- [ ] **Step 1: Implement index.jsx (routing wrapper)**

```jsx
// frontend/src/pages/AIRestyle/index.jsx
// AI Restyle page — wizard + history tabs. Mirrors pages/ShortForm/index.jsx.
import { NavLink, Route, Routes } from 'react-router-dom';
import Wizard from './Wizard.jsx';
import History from './History.jsx';

export default function AIRestyle() {
  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-3 pb-2 flex items-center gap-4 border-b border-border bg-background shrink-0">
        <h1 className="text-[18px] font-semibold">AI Restyle</h1>
        <nav className="flex items-center gap-2 ml-4">
          <NavLink
            to=""
            end
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-md text-[12px] ${isActive ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`
            }
          >
            Wizard
          </NavLink>
          <NavLink
            to="history"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded-md text-[12px] ${isActive ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`
            }
          >
            History
          </NavLink>
        </nav>
      </div>
      <div className="flex-1 overflow-hidden">
        <Routes>
          <Route index element={<Wizard />} />
          <Route path="history" element={<History />} />
        </Routes>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Implement Wizard.jsx (3-step wrapper)**

```jsx
// frontend/src/pages/AIRestyle/Wizard.jsx
// 3-step AI Restyle wizard. Same useWizard pattern as ShortForm/Wizard.jsx.
import { Check } from 'lucide-react';
import { useWizard } from '../../hooks/useWizard.js';
import Upload from './steps/Upload.jsx';
import Configure from './steps/Configure.jsx';
import Review from './steps/Review.jsx';

const STEPS = [
  { id: 'upload',    label: 'Upload' },
  { id: 'configure', label: 'Configure' },
  { id: 'review',    label: 'Review', lock: false },
];

const INITIAL = {
  file: null,
  selection: {
    backgroundPresetId: null,
    lightingPresetId: null,
    backgroundPromptOverride: null,
    lightingPromptOverride: null,
  },
  job: null,
};

const STORAGE_KEY = 'openshorts.aiRestyle.wizard';

function needsFreshUpload(data) {
  return data?.file && !(data.file.file instanceof File);
}

export default function Wizard() {
  const w = useWizard({
    steps: STEPS,
    initialData: INITIAL,
    storageKey: STORAGE_KEY,
    resetOnRehydrate: needsFreshUpload,
  });

  return (
    <div className="h-full flex flex-col">
      <StepIndicator wizard={w} />
      <div className="flex-1 overflow-hidden">
        {w.currentStep.id === 'upload'    && <Upload wizard={w} />}
        {w.currentStep.id === 'configure' && <Configure wizard={w} />}
        {w.currentStep.id === 'review'    && <Review wizard={w} />}
      </div>
    </div>
  );
}

function StepIndicator({ wizard }) {
  return (
    <div className="px-6 py-4 border-b border-border bg-background shrink-0">
      <div className="flex items-center gap-3">
        {wizard.steps.map((s, i) => {
          const active = i === wizard.step;
          const done = i < wizard.step;
          const reachable = i <= wizard.step;
          return (
            <div key={s.id} className="flex items-center gap-3 flex-1">
              <button
                onClick={() => reachable && wizard.goto(i)}
                disabled={!reachable}
                className={`flex items-center gap-2 disabled:cursor-not-allowed ${
                  active ? 'text-white' : done ? 'text-zinc-300' : 'text-zinc-600'
                }`}
              >
                <span className={`w-6 h-6 flex items-center justify-center rounded-full text-[11px] font-medium ${
                  active ? 'bg-primary text-white' :
                  done  ? 'bg-success/20 text-success border border-success/40' :
                          'bg-white/5 text-zinc-500 border border-border'
                }`}>
                  {done ? <Check size={12} /> : i + 1}
                </span>
                <span className="text-[12px]">{s.label}</span>
              </button>
              {i < wizard.steps.length - 1 && (
                <div className={`flex-1 h-px ${done ? 'bg-success/40' : 'bg-border'}`} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Implement History.jsx (placeholder)**

```jsx
// frontend/src/pages/AIRestyle/History.jsx
export default function History() {
  return (
    <div className="h-full flex items-center justify-center text-zinc-500 text-[12px]">
      Past AI Restyle jobs will appear here. (Tracking lands in a follow-up.)
    </div>
  );
}
```

- [ ] **Step 4: Commit (pre-step files)**

```bash
git add frontend/src/pages/AIRestyle/
git commit -m "feat(ai-restyle): scaffold AIRestyle page + Wizard.jsx (no steps yet)"
```

---

### Task 4b.2: Upload step

**Files:**
- Create: `frontend/src/pages/AIRestyle/steps/Upload.jsx`
- Reference pattern: `frontend/src/pages/ShortForm/steps/Upload.jsx`

- [ ] **Step 1: Implement Upload step**

```jsx
// frontend/src/pages/AIRestyle/steps/Upload.jsx
// AI Restyle Upload step. Single file, MP4/MOV, ≤30s. Probes duration on
// client (HTMLVideoElement) before allowing Continue.
import { useRef, useState } from 'react';
import { Upload as UploadIcon, X } from 'lucide-react';

const MAX_SEC = 30;
const ACCEPT = 'video/mp4,video/quicktime,.mp4,.mov';

async function probeDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement('video');
    v.preload = 'metadata';
    v.onloadedmetadata = () => { URL.revokeObjectURL(url); resolve(v.duration); };
    v.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
    v.src = url;
  });
}

export default function Upload({ wizard }) {
  const inputRef = useRef(null);
  const [error, setError] = useState(null);
  const data = wizard.data.file;

  async function onChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setError(null);

    const ext = f.name.toLowerCase().match(/\.(mp4|mov)$/);
    if (!ext) { setError('File must be MP4 or MOV.'); return; }
    const dur = await probeDuration(f);
    if (dur == null) { setError('Could not read video duration.'); return; }
    if (dur > MAX_SEC) {
      setError(`AI Restyle v1 caps at 30s. Your file is ${dur.toFixed(1)}s. Trim it first or use Short-form.`);
      return;
    }

    wizard.setData({ file: { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, name: f.name, size: f.size, durationSec: dur, file: f } });
  }

  function clearFile() {
    wizard.setData({ file: null });
    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Upload a video</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          MP4 or MOV, up to 30 seconds. We'll relight the lighting and replace
          the background while keeping your motion and audio.
        </p>

        {!data ? (
          <button
            onClick={() => inputRef.current?.click()}
            className="w-full border-2 border-dashed border-border rounded-lg p-12 flex flex-col items-center gap-3 hover:bg-white/5 transition"
          >
            <UploadIcon size={24} className="text-zinc-500" />
            <div className="text-[13px] text-zinc-300">Drop a video here or click to browse</div>
            <div className="text-[11px] text-zinc-500">MP4 / MOV · ≤30 seconds · ≤2 GB</div>
          </button>
        ) : (
          <div className="rounded-lg border border-border bg-surface p-4 flex items-center justify-between">
            <div>
              <div className="text-[13px] text-white font-medium truncate">{data.name}</div>
              <div className="text-[11px] text-zinc-500 mt-0.5">
                {(data.size / 1024 / 1024).toFixed(1)} MB · {data.durationSec.toFixed(1)}s
              </div>
            </div>
            <button onClick={clearFile} className="p-1.5 hover:bg-white/10 rounded text-zinc-400" aria-label="Remove">
              <X size={14} />
            </button>
          </div>
        )}

        <input ref={inputRef} type="file" accept={ACCEPT} onChange={onChange} className="hidden" />

        {error && (
          <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>
        )}

        <div className="mt-6 flex justify-end">
          <button
            onClick={wizard.next}
            disabled={!data}
            className="btn-primary px-4 py-2 text-[13px] disabled:opacity-40"
          >
            Continue →
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/AIRestyle/steps/Upload.jsx
git commit -m "feat(ai-restyle): wizard Upload step (client-side duration probe)"
```

---

### Task 4b.3: Configure step

**Files:**
- Create: `frontend/src/pages/AIRestyle/steps/Configure.jsx`

- [ ] **Step 1: Implement Configure step**

```jsx
// frontend/src/pages/AIRestyle/steps/Configure.jsx
// Pick a Background preset + a Lighting preset. The effective prompt
// (preset.prompt joined by " • ") is shown in an editable textarea —
// editing overrides for this job only. POSTs /api/restyle on submit.
import { useEffect, useMemo, useState } from 'react';
import { useAIRestylePresets } from '../../../state/aiRestylePresets.js';
import { useKeys } from '../../../state/keysStore.js';
import { getApiUrl } from '../../../config';

export default function Configure({ wizard }) {
  const presets = useAIRestylePresets();
  const keys = useKeys();
  const sel = wizard.data.selection;

  // Initialize selection from defaults on first render
  useEffect(() => {
    if (sel.backgroundPresetId && sel.lightingPresetId) return;
    wizard.setData({
      selection: {
        ...sel,
        backgroundPresetId: sel.backgroundPresetId || presets.defaultBackgroundId,
        lightingPresetId:   sel.lightingPresetId   || presets.defaultLightingId,
      },
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presets.defaultBackgroundId, presets.defaultLightingId]);

  const bgPreset = presets.backgrounds.find((p) => p.id === sel.backgroundPresetId);
  const ltPreset = presets.lightings.find((p) => p.id === sel.lightingPresetId);

  const effectivePrompt = useMemo(() => {
    const bg = sel.backgroundPromptOverride ?? bgPreset?.prompt ?? '';
    const lt = sel.lightingPromptOverride   ?? ltPreset?.prompt ?? '';
    return `${bg}\n${lt}`;
  }, [sel, bgPreset, ltPreset]);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  function setBg(id) {
    wizard.setData({ selection: { ...sel, backgroundPresetId: id, backgroundPromptOverride: null } });
  }
  function setLt(id) {
    wizard.setData({ selection: { ...sel, lightingPresetId: id, lightingPromptOverride: null } });
  }
  function setOverride(text) {
    const [bg, ...rest] = text.split('\n');
    wizard.setData({
      selection: { ...sel, backgroundPromptOverride: bg, lightingPromptOverride: rest.join('\n') },
    });
  }

  async function start() {
    setError(null);
    if (!keys.gemini) { setError('Set your Gemini key in Settings first.'); return; }
    if (!keys.fal)    { setError('Set your fal.ai key in Settings first.'); return; }

    const fd = new FormData();
    fd.append('file', wizard.data.file.file);
    const [bgLine, ...rest] = effectivePrompt.split('\n');
    fd.append('background_prompt', bgLine.slice(0, 500));
    fd.append('lighting_prompt', rest.join('\n').slice(0, 500));

    setSubmitting(true);
    try {
      const res = await fetch(getApiUrl('/api/restyle'), {
        method: 'POST',
        headers: { 'X-Gemini-Key': keys.gemini, 'X-Fal-Key': keys.fal },
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      const { job_id } = await res.json();
      wizard.setData({ job: { jobId: job_id, status: 'processing', result: null, progressPct: 0, logs: [] } });
      wizard.next();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-8">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-[24px] font-semibold mb-2">Configure restyle</h1>
        <p className="text-[13px] text-zinc-400 mb-6">
          Pick a background and lighting preset. Tweak the prompt below if you want.
        </p>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <PresetSelect
            label="Background"
            value={sel.backgroundPresetId || ''}
            onChange={setBg}
            options={presets.backgrounds}
            defaultId={presets.defaultBackgroundId}
          />
          <PresetSelect
            label="Lighting"
            value={sel.lightingPresetId || ''}
            onChange={setLt}
            options={presets.lightings}
            defaultId={presets.defaultLightingId}
          />
        </div>

        <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mb-2">
          Effective prompt (editable for this job)
        </label>
        <textarea
          value={effectivePrompt}
          onChange={(e) => setOverride(e.target.value)}
          rows={5}
          className="w-full bg-surface border border-border rounded-md p-3 text-[12px] text-zinc-200 font-mono leading-relaxed"
        />

        {error && <div className="mt-3 text-[12px] text-red-400" role="alert">{error}</div>}

        <div className="mt-6 flex justify-between">
          <button onClick={wizard.back} className="px-4 py-2 text-[13px] text-zinc-400 hover:text-white">← Back</button>
          <button
            onClick={start}
            disabled={submitting}
            className="btn-primary px-4 py-2 text-[13px] disabled:opacity-50"
          >
            {submitting ? 'Starting…' : 'Start restyle →'}
          </button>
        </div>
      </div>
    </div>
  );
}

function PresetSelect({ label, value, onChange, options, defaultId }) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-wider text-zinc-500 mb-2">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-surface border border-border rounded-md px-3 py-2 text-[13px] text-zinc-200"
      >
        {options.map((p) => (
          <option key={p.id} value={p.id}>
            {p.label}{p.id === defaultId ? '  ★' : ''}
          </option>
        ))}
      </select>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/AIRestyle/steps/Configure.jsx
git commit -m "feat(ai-restyle): wizard Configure step (preset dropdowns + override)"
```

---

### Task 4b.4: Review step

**Files:**
- Create: `frontend/src/pages/AIRestyle/steps/Review.jsx`

- [ ] **Step 1: Implement Review step**

```jsx
// frontend/src/pages/AIRestyle/steps/Review.jsx
// Polls /api/restyle/{job_id} until terminal. Shows progress bar + log tail
// during processing. On completion: Before/After preview + Download + Send
// to Short-form CTA.
import { useEffect, useState } from 'react';
import { Download, Eye } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import PhoneFrame from '../../../components/ui/PhoneFrame.jsx';
import { getApiUrl } from '../../../config';

export default function Review({ wizard }) {
  const job = wizard.data.job;
  const file = wizard.data.file;
  const [showOriginal, setShowOriginal] = useState(false);
  const [sourceUrl, setSourceUrl] = useState(null);
  const navigate = useNavigate();

  // Blob URL for the original (Before view)
  useEffect(() => {
    if (!file?.file) { setSourceUrl(null); return; }
    const u = URL.createObjectURL(file.file);
    setSourceUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file?.file]);

  // Poll status until terminal
  useEffect(() => {
    if (!job?.jobId || job.status === 'completed' || job.status === 'failed') return;
    let alive = true;
    const tick = async () => {
      try {
        const res = await fetch(getApiUrl(`/api/restyle/${job.jobId}`));
        if (!res.ok) throw new Error(`status ${res.status}`);
        const data = await res.json();
        if (!alive) return;
        wizard.setData({ job: { ...job, ...data } });
      } catch (e) { /* swallow transient */ }
    };
    const i = setInterval(tick, 2000);
    tick();
    return () => { alive = false; clearInterval(i); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.jobId, job?.status]);

  const url = job?.result?.video_url ? getApiUrl(job.result.video_url) : null;
  const status = job?.status || 'idle';

  function sendToShortForm() {
    if (!url) return;
    // Stash the restyled URL in sessionStorage so ShortForm picks it up.
    sessionStorage.setItem('openshorts.shortForm.handoff', JSON.stringify({ url, name: `restyled-${file?.name || 'video.mp4'}` }));
    navigate('/short-form');
  }

  if (status === 'processing') {
    return (
      <div className="h-full flex items-center justify-center p-12">
        <div className="max-w-md w-full">
          <div className="text-[14px] text-white font-medium mb-3">Restyling…</div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div className="h-full bg-primary transition-all" style={{ width: `${job?.progressPct || 5}%` }} />
          </div>
          <div className="mt-4 text-[11px] text-zinc-500 font-mono leading-relaxed max-h-40 overflow-y-auto">
            {(job?.logs || []).slice(-8).map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="h-full flex items-center justify-center p-12 text-center">
        <div className="max-w-md">
          <div className="text-[14px] text-red-400 font-medium mb-2">Restyle failed</div>
          <div className="text-[12px] text-zinc-500 font-mono whitespace-pre-line">
            {(job?.logs || []).slice(-6).join('\n')}
          </div>
          <div className="mt-4 flex gap-3 justify-center">
            <button onClick={wizard.back} className="px-3 py-1.5 text-[12px] border border-border rounded-md text-zinc-300 hover:bg-white/5">Try again</button>
            <button onClick={wizard.reset} className="px-3 py-1.5 text-[12px] btn-primary">Start over</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-8">
      <div className="flex-1 flex flex-col items-center gap-4">
        <div className="flex items-center gap-2 text-[12px]">
          <button onClick={() => setShowOriginal(false)} className={`px-3 py-1.5 rounded-md ${!showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}>After</button>
          <button onClick={() => setShowOriginal(true)} disabled={!sourceUrl} className={`px-3 py-1.5 rounded-md disabled:opacity-30 ${showOriginal ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-white'}`}>
            <Eye size={12} className="inline mr-1" /> Before
          </button>
        </div>

        <PhoneFrame size="md">
          {showOriginal && sourceUrl ? (
            <video key="src" src={sourceUrl} controls className="w-full h-full object-contain" />
          ) : url ? (
            <video key="rst" src={url} controls className="w-full h-full object-contain" />
          ) : (
            <div className="text-zinc-600 text-[12px] p-4 text-center">No preview available.</div>
          )}
        </PhoneFrame>
      </div>

      <div className="border-t border-border pt-4 flex items-center gap-3">
        <a href={url || '#'} download className={`btn-primary px-3 py-2 text-[12px] flex items-center gap-2 ${!url ? 'opacity-40 pointer-events-none' : ''}`}>
          <Download size={12} /> Download
        </a>
        <button onClick={sendToShortForm} disabled={!url} className="px-3 py-2 text-[12px] border border-primary/40 text-primary rounded-md hover:bg-primary/10 disabled:opacity-40">
          Send to Short-form →
        </button>
        <button onClick={wizard.reset} className="ml-auto px-3 py-2 text-[12px] text-zinc-400 hover:text-white">
          Start another
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend && npm run build
```

Expected: 0 errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/AIRestyle/steps/Review.jsx
git commit -m "feat(ai-restyle): wizard Review step (poll + Before/After + Send-to-Short-form)"
```

---

### Task 4b.5: Sidebar entry + App route

**Files:**
- Modify: `frontend/src/layouts/Sidebar.jsx`
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Add sidebar entry**

Replace the existing `NAV` array in `frontend/src/layouts/Sidebar.jsx`:

```jsx
import { LayoutDashboard, Smartphone, Video, Scissors, Settings as SettingsIcon, Wand2 } from 'lucide-react';

const NAV = [
  { to: '/dashboard',      label: 'Dashboard',     icon: LayoutDashboard },
  { to: '/long-form',      label: 'Long-form',     icon: Video },
  { to: '/ai-restyle',     label: 'AI Restyle',    icon: Wand2 },
  { to: '/short-form',     label: 'Short-form',    icon: Smartphone },
  { to: '/clip-generator', label: 'Clip Generator', icon: Scissors },
  { to: '/settings',       label: 'Settings',      icon: SettingsIcon },
];
```

- [ ] **Step 2: Wire the App route**

Locate the existing `<Routes>` block in `frontend/src/App.jsx`. Add:

```jsx
import AIRestyle from './pages/AIRestyle/index.jsx';

// inside <Routes>, after the LongForm route:
<Route path="/ai-restyle/*" element={<AIRestyle />} />
```

- [ ] **Step 3: Verify build**

```bash
cd frontend && npm run build
```

Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/layouts/Sidebar.jsx frontend/src/App.jsx
git commit -m "feat(ai-restyle): sidebar entry + /ai-restyle route"
```

---

## Phase 5 — Settings tab (~1 day)

### Task 5.1: AIRestylePresetsSection.jsx + tab registration

**Files:**
- Create: `frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx`
- Modify: `frontend/src/pages/Settings/index.jsx` — register the new section.
- Reference pattern: `frontend/src/pages/Settings/sections/BrandKitSection.jsx`

- [ ] **Step 1: Read the existing Settings shape**

```bash
cd "/Users/matissevansteenbergen/Downloads/AGENTIC WORKLFOWS/PERSONAL/Auto-shorts (TODO)/openshorts" && cat frontend/src/pages/Settings/index.jsx | head -40
cat frontend/src/pages/Settings/sections/BrandKitSection.jsx | head -50
```

Match the export shape and props convention (probably `({ ... })` with no specific contract).

- [ ] **Step 2: Implement AIRestylePresetsSection**

```jsx
// frontend/src/pages/Settings/sections/AIRestylePresetsSection.jsx
// Edit / delete / star presets for the AI Restyle wizard.
import { useState } from 'react';
import { Star, Pencil, Trash2, Plus } from 'lucide-react';
import SectionHeader from './SectionHeader.jsx';
import {
  useAIRestylePresets,
  upsertPreset,
  deletePreset,
  setDefault,
} from '../../../state/aiRestylePresets.js';

export default function AIRestylePresetsSection() {
  const presets = useAIRestylePresets();
  const [editing, setEditing] = useState(null); // { dimension, preset } | null

  return (
    <section className="space-y-6">
      <SectionHeader
        title="AI Restyle presets"
        description="Edit the prompts used to relight the first frame. Star marks the recommended default."
      />

      <Dimension
        title="Backgrounds"
        items={presets.backgrounds}
        defaultId={presets.defaultBackgroundId}
        onEdit={(p) => setEditing({ dimension: 'background', preset: p })}
        onAdd={() => setEditing({ dimension: 'background', preset: { id: '', label: '', prompt: '' } })}
        onStar={(id) => setDefault('background', id)}
        onDelete={(id) => deletePreset('background', id)}
      />

      <Dimension
        title="Lightings"
        items={presets.lightings}
        defaultId={presets.defaultLightingId}
        onEdit={(p) => setEditing({ dimension: 'lighting', preset: p })}
        onAdd={() => setEditing({ dimension: 'lighting', preset: { id: '', label: '', prompt: '' } })}
        onStar={(id) => setDefault('lighting', id)}
        onDelete={(id) => deletePreset('lighting', id)}
      />

      {editing && (
        <EditModal
          dimension={editing.dimension}
          preset={editing.preset}
          onClose={() => setEditing(null)}
          onSave={(p) => { upsertPreset(editing.dimension, p); setEditing(null); }}
        />
      )}
    </section>
  );
}

function Dimension({ title, items, defaultId, onEdit, onAdd, onStar, onDelete }) {
  return (
    <div>
      <h3 className="text-[13px] font-medium text-white mb-2">{title}</h3>
      <div className="space-y-1 rounded-lg border border-border overflow-hidden">
        {items.map((p) => {
          const isDefault = p.id === defaultId;
          return (
            <div key={p.id} className="p-3 hover:bg-white/5 flex items-start gap-3">
              <button
                onClick={() => onStar(p.id)}
                title={isDefault ? 'Default' : 'Set as default'}
                className={`mt-0.5 ${isDefault ? 'text-yellow-400' : 'text-zinc-600 hover:text-zinc-300'}`}
              >
                <Star size={14} fill={isDefault ? 'currentColor' : 'none'} />
              </button>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] text-white font-medium">{p.label}</div>
                <div className="text-[11px] text-zinc-500 mt-0.5 leading-snug">{p.prompt}</div>
              </div>
              <button onClick={() => onEdit(p)} className="p-1.5 text-zinc-500 hover:text-white" title="Edit"><Pencil size={12} /></button>
              <button
                onClick={() => onDelete(p.id)}
                disabled={isDefault}
                className="p-1.5 text-zinc-500 hover:text-red-400 disabled:opacity-30 disabled:cursor-not-allowed"
                title={isDefault ? 'Cannot delete the default preset' : 'Delete'}
              >
                <Trash2 size={12} />
              </button>
            </div>
          );
        })}
        <button onClick={onAdd} className="w-full p-3 text-[12px] text-zinc-400 hover:text-white hover:bg-white/5 border-t border-border flex items-center justify-center gap-2">
          <Plus size={12} /> Add {title.toLowerCase().slice(0, -1)} preset
        </button>
      </div>
    </div>
  );
}

function EditModal({ dimension, preset, onClose, onSave }) {
  const [label, setLabel] = useState(preset.label || '');
  const [prompt, setPrompt] = useState(preset.prompt || '');

  function save() {
    if (!label.trim() || !prompt.trim()) return;
    const id = preset.id || label.toLowerCase().replace(/\s+/g, '-').slice(0, 40);
    onSave({ id, label: label.slice(0, 40), prompt: prompt.slice(0, 500) });
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-surface border border-border rounded-lg p-5 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-[14px] font-medium text-white mb-3">
          {preset.id ? 'Edit' : 'Add'} {dimension} preset
        </h3>
        <label className="block text-[11px] text-zinc-500 uppercase mb-1">Name</label>
        <input value={label} onChange={(e) => setLabel(e.target.value)} maxLength={40}
               className="w-full bg-background border border-border rounded px-3 py-1.5 text-[13px] text-white mb-3" />
        <label className="block text-[11px] text-zinc-500 uppercase mb-1">Prompt</label>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={500} rows={4}
                  className="w-full bg-background border border-border rounded px-3 py-2 text-[12px] text-zinc-200 font-mono" />
        <div className="text-[10px] text-zinc-500 text-right mt-1">{prompt.length}/500</div>
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onClose} className="px-3 py-1.5 text-[12px] text-zinc-400 hover:text-white">Cancel</button>
          <button onClick={save} disabled={!label.trim() || !prompt.trim()} className="btn-primary px-3 py-1.5 text-[12px] disabled:opacity-40">Save</button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Register the section in Settings/index.jsx**

Open `frontend/src/pages/Settings/index.jsx`. Add an import and one entry to whatever array/object defines the tab list (the existing pattern will be visible — match it exactly):

```jsx
import AIRestylePresetsSection from './sections/AIRestylePresetsSection.jsx';

// inside the tab list (alongside BrandKit, ApiKeys, etc.):
{ id: 'ai-restyle', label: 'AI Restyle', component: AIRestylePresetsSection },
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: 0 errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Settings/
git commit -m "feat(ai-restyle): Settings tab with preset CRUD (star/edit/delete)"
```

---

## Phase 6 — Smoke test + Codex + ship (~0.5 day)

### Task 6.1: Browser smoke test (chrome-devtools MCP)

**No files.** Walk the user through these steps in a real browser. Per HANDOFF.md §6 rule 6 + Convention #5, UI features need browser verification, not just `npm run build`.

- [ ] **Step 1: Restart backend container**

```bash
docker restart openshorts-backend
```

- [ ] **Step 2: Open the app**

Navigate the browser to `http://localhost:3001/#/ai-restyle`. Confirm:
- Sidebar shows "AI Restyle" between Long-form and Short-form with the Wand2 icon.
- Upload step renders with the drop zone.

- [ ] **Step 3: Upload + duration check**

Upload `demo-openshorts.mp4` (42s) → expect the rejection "AI Restyle v1 caps at 30s". Trim a copy to 10s and upload → expect Continue button enables.

- [ ] **Step 4: Configure step**

Verify two dropdowns render the 5 default backgrounds + 5 default lightings, each marked with ★ on the default. Pick "Bahamas beach" + "Golden hour". Effective prompt textarea updates. Type a custom override to confirm it doesn't blow away the saved preset.

- [ ] **Step 5: Start the job**

Click `Start restyle →`. Review step opens with the progress bar. Watch the logs cycle through:
- 🔎 Probing video duration
- 🎞️ Extracting first frame
- 🪄 Relighting frame with Nano Banana
- 💰 Nano Banana relight: ~$0.039
- 🎬 Restyling video via fal.ai
- 💰 fal.ai v2v: ~$0.40
- 🔊 Muxing original audio
- ✅ AI Restyle complete

Total wall-clock should be ≤5 minutes. If it stalls past 10 minutes, kill the job, inspect `docker logs openshorts-backend`, and debug. Most common cause: wrong fal.ai model ID (revisit Phase 0 spike).

- [ ] **Step 6: Verify output**

After completion:
- Phone preview plays the restyled clip with new background/lighting.
- Before/After toggle works.
- Download button downloads the file.
- `Send to Short-form →` navigates to `/short-form` (initial sessionStorage handoff payload behavior — full integration with ShortForm's Upload step is documented as follow-up if not already present).

- [ ] **Step 7: Settings tab smoke**

Navigate to Settings → AI Restyle. Verify the two preset lists render. Click ★ on a non-default — it should move. Click `Edit` on a preset, change the label, save, verify the wizard's dropdown updates without a reload (event broadcast works). Click `+ Add background preset`, create one with custom name + prompt, save, confirm it appears in the wizard dropdown.

- [ ] **Step 8: Take screenshots for the commit**

Save 3 screenshots to `.compact-ultra/`:
- `ai-restyle-upload.png`
- `ai-restyle-configure.png`
- `ai-restyle-review-completed.png`

These are session-local — gitignored.

---

### Task 6.2: Codex adversarial review

Per global CLAUDE.md "Codex Adversarial Review" rule: this phase introduces new HTTP endpoints calling LLMs + external services. Codex must review.

- [ ] **Step 1: Trigger Codex**

```bash
/codex:rescue --background "deep security audit of AI Restyle (new sidebar product): /api/restyle + /api/restyle/{job_id} routes, frame_extract / frame_relight / video_restyle ML modules, restyle_pipeline orchestrator. Focus on: input validation (prompt length, duration cap), command injection via filenames, FFmpeg argument injection, fal.ai key leakage, prompt-injection of background_prompt / lighting_prompt into the Gemini call, race conditions on shared jobs[] dict, output file collisions, missing auth/rate-limit/timeout (acknowledged as opt-out per HANDOFF.md §5 but flag any NEW gaps not already covered by the existing opt-outs)."
```

- [ ] **Step 2: Address Codex findings**

Triage each finding. For each:
- BLOCKER (auth bypass, RCE, secret leak): fix before merge.
- HIGH (input validation gap, injection vector): fix in this PR.
- MEDIUM (defense-in-depth, missing log): add a follow-up issue.
- LOW (style nit): ignore.

Apply fixes via Edit and re-run pytest + npm run build.

- [ ] **Step 3: Commit Codex fixes**

```bash
git add -A
git commit -m "fix(ai-restyle): address Codex review findings (<short summary>)"
```

---

### Task 6.3: Final gates + roadmap update

**Files:**
- Modify: `ROADMAP.md` — promote AI Restyle to "Stubbed in v1"
- Modify: `~/.claude/CLAUDE.md` (auto-managed sections — run the script)

- [ ] **Step 1: Run backend + frontend gates**

```bash
cd backend && pytest -m "not e2e" -q
cd ../frontend && npm run build
```

Expected: pytest fully green (~170 tests), build 0 errors.

- [ ] **Step 2: Update ROADMAP.md**

Add (or replace if already a placeholder) under a new "### AI Restyle" section:

```markdown
### AI Restyle

**Shipped**
- Sidebar entry between Long-form and Short-form (icon: Wand2).
- 3-step wizard: Upload → Configure → Review.
- Two preset dimensions (Background + Lighting), 5 hand-tuned seed presets each.
- Per-job prompt override via editable textarea.
- Settings tab with full preset CRUD (star/edit/delete).
- 30s duration cap (client + server enforced).
- Original audio preserved bit-for-bit.

**Stubbed in v1**
- History tab is a placeholder ("Past AI Restyle jobs will appear here").
- Send-to-Short-form CTA stashes a session payload; full wire-through into ShortForm's Upload step is follow-up work.

**Later**
- Lift the 30s cap via chunked v2v with shared reference frame (Approach B from design).
- Bridge from Short-form Review's stage selector ("+ AI Restyle" stage).
- Auto-suggest preset based on the source frame (Gemini-driven).
- Backend-stored preset sharing / team marketplace.
```

- [ ] **Step 3: Regenerate CLAUDE.md auto-managed sections**

```bash
python3 scripts/update_claude_md.py
```

Expected: the module-map table gains entries for `ml/frame_extract.py`, `ml/frame_relight.py`, `ml/video_restyle.py`, `saas/restyle_pipeline.py`, `routes/ai_restyle.py`. ENV table is unchanged (no new env vars).

- [ ] **Step 4: Final commit**

```bash
git add ROADMAP.md ~/.claude/CLAUDE.md
git commit -m "docs(ai-restyle): roadmap entry + CLAUDE.md module-map refresh"
```

- [ ] **Step 5: Verify on the branch**

```bash
git log --oneline | head -20
git status
```

Expected: clean working tree, AI Restyle commits sit on top of the polish-plan commits, ready for PR.

---

## Self-Review (post-write)

**Spec coverage:**
- ✓ §1 goal — Phase 2 (pipeline) delivers.
- ✓ §2 user flow Upload — Task 4b.2.
- ✓ §2 user flow Configure — Task 4b.3.
- ✓ §2 user flow Review — Task 4b.4.
- ✓ §3 routes — Task 3.1.
- ✓ §3 pipeline orchestrator — Task 2.2.
- ✓ §3 ML modules (3) — Tasks 1.1, 1.2, 2.1.
- ✓ §4 frontend pages — Task 4b.1 onwards.
- ✓ §4 sidebar — Task 4b.5.
- ✓ §4 preset store — Task 4a.1.
- ✓ §4 Settings tab — Task 5.1.
- ✓ §4 seed presets — embedded in Task 4a.1.
- ✓ §5 security baseline C3 + C9 — covered in route Pydantic + cost log lines (Task 2.2 + 3.1).
- ✓ §5 cost telemetry — Task 2.2 log lines.
- ✓ §5 failure handling — Task 2.2 except block + Task 4b.4 failed-state UI.
- ✓ §5 tests — Tasks 1.1, 1.2, 2.1, 2.2, 3.1.
- ✓ §6 files — every added/modified file maps to a task.
- ✓ §7 milestones — Phases 0-6 mirror §7 exactly.
- ✓ §8 roadmap entry — Task 6.3 Step 2.
- ✓ §9 decisions — all referenced in code/tests where they apply.

**Placeholder scan:** Phase 0 outputs (model ID, payload shape) are explicit dependencies of Phase 2 Task 2.1, with a clear ⚠️ callout. No "TBD" left in implementation steps.

**Type consistency:** `relight_frame` signature matches between tests + impl + pipeline (api_key, frame_path, background_prompt, lighting_prompt, out_path). `restyle_video` signature matches (api_key, video_path, reference_frame_path, out_path). `run_restyle_job` signature matches (jobs, job_id, input_path, background_prompt, lighting_prompt, gemini_key, fal_key). Preset store API (`upsertPreset`, `deletePreset`, `setDefault`, `useAIRestylePresets`) used consistently across wizard + Settings.

**Ambiguity:** No requirement in the spec is left to interpretation — Phase 0 picks the model, Phase 2 Task 2.1 has the contract for replacing the placeholder if Phase 0 picks differently.

No issues found. Plan is ready for execution.

---

*End of plan.*
