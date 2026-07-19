# Acceptance Auditor Review Prompt

Check this implementation against:
- `_bmad-output/implementation-artifacts/spec-improve-clip-selection-boundaries-text-polish.md`
- `AGENTS.md`
- `CLAUDE.md`

Review with project read access. Verify whether the implementation satisfies the spec acceptance criteria and does not violate the frozen constraints.

Relevant implementation files:
- `main.py`
- `tests/test_clip_quality.py`

Verification already run:
- `python3 -m pytest tests/test_clip_quality.py` passed.
- `python3 -m py_compile main.py subtitles.py hooks.py` passed.

Return findings only. Classify each as:
- `intent_gap`
- `bad_spec`
- `patch`
- `defer`
- `reject`

## Acceptance Criteria To Audit

- Given weak, duplicate, and high-quality candidate clips, when post-processing runs, then output clips are ranked by quality, invalid durations are removed when possible, and near-duplicates keep the strongest candidate.
- Given model clip boundaries inside word timestamps, when snapping runs, then the adjusted clip starts before the first relevant word and ends after the final relevant word without violating minimum duration.
- Given missing or messy `viral_hook_text`, title, and caption fields, when post-processing runs, then downstream-compatible fields are populated with normalized, bounded text.
- Given all candidates are below ideal thresholds, when post-processing runs, then it keeps a small best-effort fallback instead of returning no clips.
- Given no Gemini, video file, or FFmpeg binary, when unit tests run, then helper behavior is still verifiable.

## Diff Summary

- Added metadata cleanup helpers in `main.py`.
- Added prompt quality gates for evidence-based selection, weak-start rejection, complete thought boundaries, self-contained clips, and clear endings.
- Routed post-processing text metadata through the cleanup helper.
- Changed boundary snapping to expand around words instead of trimming through words.
- Added focused unit tests with import stubs for optional video packages.
