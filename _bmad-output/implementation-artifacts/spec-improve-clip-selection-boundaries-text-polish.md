---
title: 'Improve Clip Selection, Boundaries, and Text Polish'
type: 'feature'
created: '2026-06-09'
status: 'in-review'
baseline_commit: '3f58ccb086bc123a67ab136d3f570303e45ff59a'
context:
  - '{project-root}/AGENTS.md'
  - '{project-root}/CLAUDE.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Clip Generator results need stronger viral moment selection, cleaner start/end timing, and more polished subtitles/hooks. Current logic asks Gemini for engaging clips and has basic post-processing, but it can still keep weak or overlapping moments, snap cuts in a way that trims speech awkwardly, and generate text metadata that is not normalized enough for downstream overlays.

**Approach:** Add focused, testable helpers around clip quality normalization, overlap/ranking, boundary snapping, and text sanitization. Keep the existing Gemini/FFmpeg pipeline intact while making generated clips more self-contained, accurately timed, and ready for readable subtitle/hook rendering.

## Boundaries & Constraints

**Always:** Preserve the existing `shorts` metadata schema used by `app.py` and the dashboard. Keep API keys out of server persistence. Use FFmpeg-based output paths already in the project. Add tests before production changes for deterministic helper behavior.

**Ask First:** Any change requiring a new third-party package, changing Gemini model defaults, changing crop/framing behavior, or modifying public API request/response shapes.

**Never:** Do not refactor the full video pipeline, rewrite the frontend editor, change social publishing behavior, or alter user-owned dirty worktree changes outside this scope.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Better selected moments | Multiple candidate clips with scores, durations, repeated viral pattern types, and partial overlap | Keep higher-scoring, valid-duration, less-duplicative clips; preserve enough diversity without promoting weaker clips above strong ones | If all candidates fail duration/confidence filters, keep the best fallback rather than returning empty results |
| Cleaner boundaries | A model-selected clip starts/ends inside words or near scene boundaries | Snap to a natural word or scene edge while preserving complete speech and minimum duration | If snapping would invert or overly shrink the clip, keep the original boundaries |
| Text polish | Hook/title/caption fields are missing, too long, oddly spaced, or not normalized | Populate stable fallback fields, limit hook text length, normalize whitespace/casing, and keep captions grounded in existing fields | Do not fabricate topic-specific captions when source text is missing |

</frozen-after-approval>

## Code Map

- `main.py` -- Builds Gemini prompts, parses/merges clip candidates, post-processes clip metadata, snaps boundaries before extraction, and writes clip metadata.
- `subtitles.py` -- Generates SRT blocks and burns subtitles; relevant for readability and timing polish.
- `hooks.py` -- Renders hook overlay images and applies them with FFmpeg; relevant for hook text display.
- `app.py` -- Reads and updates clip metadata for subtitle/hook endpoints; must keep current schema stable.
- `tests/` -- Add deterministic unit tests for helper behavior without requiring Gemini, Whisper, FFmpeg, or video files.

## Tasks & Acceptance

**Execution:**
- [x] `tests/test_clip_quality.py` -- Add failing tests for clip post-processing, boundary snapping, and metadata text normalization -- Locks expected behavior before editing production code.
- [x] `main.py` -- Add small helper functions for duration/confidence normalization, overlap scoring, hook/title/caption cleanup, and safer boundary snapping -- Improves selected moments and cut points without changing the external pipeline.
- [x] `main.py` -- Strengthen Gemini prompt quality gates for evidence, weak-start rejection, complete thoughts, and self-contained clips -- Improves model-side selection before deterministic post-processing.
- [x] `main.py` -- Use helpers in single-call and chunked clip analysis paths before metadata is saved -- Ensures both normal and long-video flows benefit.
- [x] `subtitles.py` -- Tighten SRT timing/text edge cases only if tests or inspection show subtitle blocks can be empty, inverted, or overflow-prone -- Keeps subtitle polish scoped.
- [x] `hooks.py` -- Tighten hook text handling only if main metadata normalization is insufficient for overlay readability -- Avoids duplicate presentation logic.

**Acceptance Criteria:**
- Given weak, duplicate, and high-quality candidate clips, when post-processing runs, then output clips are ranked by quality, invalid durations are removed when possible, and near-duplicates keep the strongest candidate.
- Given model clip boundaries inside word timestamps, when snapping runs, then the adjusted clip starts before the first relevant word and ends after the final relevant word without violating minimum duration.
- Given missing or messy `viral_hook_text`, title, and caption fields, when post-processing runs, then downstream-compatible fields are populated with normalized, bounded text.
- Given all candidates are below ideal thresholds, when post-processing runs, then it keeps a small best-effort fallback instead of returning no clips.
- Given no Gemini, video file, or FFmpeg binary, when unit tests run, then helper behavior is still verifiable.

## Spec Change Log

## Design Notes

Prefer deterministic safeguards after Gemini over prompt-only improvements. Prompt edits can help, but helper-level normalization gives repeatable behavior across models, categories, and retry paths. Keep these helpers side-effect-light so tests can exercise them without the full processing stack.

## Verification

**Commands:**
- `python3 -m pytest tests/test_clip_quality.py` -- expected: focused helper tests pass.
- `python3 -m py_compile main.py subtitles.py hooks.py` -- expected: touched Python modules compile.
