# Indonesian Podcast Comedy Multi-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace global `podcast_comedy` selection with a DeepSeek-compatible two-pass Indonesian scout-and-judge pipeline that sends one transcript representation and falls back safely.

**Architecture:** `main.py` will shape Whisper output into timestamped segments, scout candidates in bounded windows, consolidate and contextualize them, then ask a second AI pass to rank them globally. Deterministic helpers enforce candidate identity, evidence, duration, overlap, lane balance, and downstream metadata before existing boundary snapping and extraction.

**Tech Stack:** Python 3.11, OpenAI-compatible chat completions client, pytest, existing Whisper transcript structures.

---

### Task 1: Canonical Transcript And Bounded Windows

**Files:**
- Modify: `tests/test_clip_quality.py`
- Modify: `main.py`

- [x] **Step 1: Write failing tests**

Add tests asserting `_build_timestamped_transcript()` emits segment IDs, timestamps, text, and average confidence, and `_segments_for_window()` excludes segments entirely outside `[window_start, window_end]`.

- [x] **Step 2: Verify red**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: import failures for the new helper names.

- [x] **Step 3: Implement minimal helpers**

Add `_build_timestamped_transcript(transcript_result)` and `_segments_for_window(segments, start, end)`. Preserve word timestamps outside these structures.

- [x] **Step 4: Verify green**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: transcript/window tests pass.

### Task 2: Scout Prompt And Candidate Consolidation

**Files:**
- Modify: `tests/test_clip_quality.py`
- Modify: `main.py`

- [x] **Step 1: Write failing tests**

Test that `_build_podcast_comedy_scout_prompt()` contains `TIMESTAMPED_TRANSCRIPT` exactly once and omits `WORDS_JSON`. Test `_consolidate_scout_candidates()` rejects missing evidence/invalid timestamps and retains the higher-scored overlapping duplicate.

- [x] **Step 2: Verify red**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: failures because scout and consolidation helpers do not exist.

- [x] **Step 3: Implement minimal helpers**

Add lane normalization, evidence validation, hard 15-70 second bounds, stable candidate IDs, overlap deduplication, and a default 24-candidate cap.

- [x] **Step 4: Verify green**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: scout/consolidation tests pass.

### Task 3: Judge Context, Validation, And Balanced Ranking

**Files:**
- Modify: `tests/test_clip_quality.py`
- Modify: `main.py`

- [x] **Step 1: Write failing tests**

Test candidate context padding, judge prompt compactness, rejection of unknown IDs and out-of-context timestamps, lane-aware duration tolerance, six-word hook normalization, and quality-weighted lane balance.

- [x] **Step 2: Verify red**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: failures for missing judge helpers.

- [x] **Step 3: Implement minimal helpers**

Add `_attach_candidate_context()`, `_build_podcast_comedy_judge_prompt()`, `_validate_judge_output()`, and `_rank_balanced_clips()`. Seed a lane only when its best score is within 0.12 of the best remaining quality and fill remaining positions by score.

- [x] **Step 4: Verify green**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: judge validation and ranking tests pass.

### Task 4: Two-Pass Orchestration And Fallback

**Files:**
- Modify: `tests/test_clip_quality.py`
- Modify: `main.py`

- [x] **Step 1: Write failing orchestration tests**

Use a callable fake analysis function to assert `_run_podcast_comedy_multipass()` makes a scout call then a judge call, and returns validated scout clips when the judge raises or returns invalid JSON.

- [x] **Step 2: Verify red**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: failures because orchestration is absent.

- [x] **Step 3: Implement and route**

Add `_run_podcast_comedy_multipass()` and call it from `get_viral_clips()` only for `category == "podcast_comedy"`. Keep generic categories on the existing path. Use configurable temperatures and enable flags from environment variables.

- [x] **Step 4: Verify focused suite**

Run: `python3 -m pytest tests/test_clip_quality.py -q`

Expected: all tests pass.

### Task 5: Final Verification

**Files:**
- Verify: `main.py`
- Verify: `tests/test_clip_quality.py`

- [x] **Step 1: Compile Python sources**

Run: `python3 -m py_compile main.py subtitles.py hooks.py`

Expected: exit code 0.

- [x] **Step 2: Inspect diff**

Run: `git diff --check && git diff -- main.py tests/test_clip_quality.py`

Expected: no whitespace errors; changes remain within approved analysis and test boundaries.

- [x] **Step 3: Do not commit**

The repository instructions require explicit user approval before any commit.
