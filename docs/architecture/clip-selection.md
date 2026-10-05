# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived. `get_viral_clips` builds ~90 s scoring
windows over the transcript, scores them in batches, shortlists the best
`shortlist_target` (`min(10, duration//90 + 2)`, floor 3) and asks the detail pass
for `clip_count_targets(len(shortlist))` clips. The floor matters: jobs that
return only 1-3 clips retain far worse than jobs that return 4-9.

Two invariants keep that chain from leaking:

- **The scoring pass ranks, it does not select.** The prompt scores **every**
  window and the shortlist is the global top N. A prompt that says "choose up to
  3 windows from this batch" caps the shortlist at `3 * n_batches` instead of the
  target and starves every source under ~30 min. Batches are near-equal
  (`score_batches`): a trailing batch of one window would return that window
  whatever its score.
- **The clip floor is enforced in code, not just in the prompt.** When the detail
  pass comes back under `min_clips`, the shortlist windows that produced nothing
  get one more call for the difference. An empty answer is accepted: padding is
  worse than a short list.

`target_clips` on `/api/process` (dashboard: advanced options) pins both ends via
`CLIP_TARGET_MIN` / `CLIP_TARGET_MAX` when the user wants an exact number.

## Silent footage: the vision fallback (`main.get_visual_clips`)

`transcribe_video` raising `NoAudioError`, **or** `speech_is_sparse()` returning
true, both set `transcript = None` and route to `get_visual_clips`. Sparse means
under `MIN_SPEECH_WORDS` (8) in total or under `MIN_SPEECH_WORDS_PER_MIN` (5).
Music-only footage and mic-muted recordings transcribe to a handful of stray
words; without the second test the picker scores those words and cuts around
them.

The vision pass uploads the video, Gemini returns the same `{"shorts"}` shape
(`gemini_worker.VisualResponse`) in the same 15-60 s band, so every later stage
is identical. `CLIP_TARGET_MIN`/`MAX` apply directly (there are no scoring
windows). The transcript is stored as `{"language": "none", "segments": []}`, so
the clips have no subtitles, which is correct.

**This is the one stage that sends Gemini the video instead of frames, on
purpose**: frames can say what kind of video it is, not which 40 seconds to cut.
The ceiling: ~300 tokens/second, so an hour is ~1.08M tokens, past a 1M window,
and **nothing guards the length**. If that needs fixing, add a guard or segment
the source; do not port the frame trick. Gemini-only: a text-only `LLM_BASE_URL`
server cannot see footage, and without `GEMINI_API_KEY` the function logs one
line and returns None, which fails the job.

The public `/gta-5-clips` page states these thresholds and this ceiling; if the
behaviour changes, change `dashboard/seo/pages.js` too.

## Local LLM for the moment picker (`llm_backend.py`)

`LLM_BASE_URL` (+ `LLM_MODEL`, `LLM_API_KEY`) routes the two transcript passes of
`get_viral_clips` to any OpenAI-compatible `/chat/completions` instead of Gemini;
the response is validated with the same pydantic schemas Gemini enforces
server-side, so `main.py` sees one shape. `main.score_batch_size` drops to 3
windows per call there (local contexts are 4-8k; a truncated prompt scores garbage
silently). Self-host `/api/process` then accepts a request without
`X-Gemini-Key` and `/api/config.localLlm` tells the dashboard not to demand one.
Frame-based stages (`layout_picker`, `screencast_layout`, `get_visual_clips`)
stay on Gemini. Never wired in cloud mode: `BILLING_ENABLED` ignores it.
