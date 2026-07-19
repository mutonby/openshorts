# Edge Case Hunter Review Prompt

Use the `bmad-review-edge-case-hunter` skill.

Review this project with read access. Focus only on edge cases introduced by the current change to `main.py` and `tests/test_clip_quality.py`. Check boundary snapping, metadata normalization, compatibility with downstream dashboard/API fields, and test reliability when optional video dependencies are unavailable.

Relevant files:
- `main.py`
- `tests/test_clip_quality.py`
- `app.py`
- `hooks.py`
- `subtitles.py`

Return only actionable edge cases caused or exposed by this patch. For each finding, include file/line, why it can happen, user-visible effect, and a minimal fix.

## Diff Summary

- Added `_clean_text`, `_clean_hook_text`, and `_clip_text_metadata` in `main.py`.
- Added prompt quality gates requiring `START_QUALITY_CHECK`, `COMPLETE_THOUGHT`, `QUOTE_EVIDENCE`, `SELF-CONTAINED CHECK`, and stronger ending quality.
- `post_process_clips` now normalizes `viral_hook_text`, `social_caption`, `video_title_for_youtube_short`, `video_description_for_tiktok`, and `video_description_for_instagram`.
- `snap_clip_to_boundaries` now accepts both `s/e` and `start/end` word timestamp keys.
- When a clip start lands inside a word, snapping expands backward to the word start minus padding, preferring a nearby scene boundary.
- When a clip end lands inside a word, snapping expands forward to the word end instead of trimming to word start.
- Added `tests/test_clip_quality.py` with stubs for heavy import dependencies.
