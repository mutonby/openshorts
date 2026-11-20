# Clip selection

## How many clips a job returns (`clip_selection.py`, `main.get_viral_clips`)

The count is not a setting, it is derived. `get_viral_clips` builds ~90 s scoring
windows over the transcript, scores them in batches, shortlists the best
`shortlist_target` and asks the detail pass for `clip_count_targets` clips per
group of the shortlist (`detail_plan`). The floor matters: jobs that return
only 1-3 clips retain far worse than jobs that return 4-9.

**Up to half an hour** the shortlist is `min(10, duration//90 + 2)` (floor 3),
one detail call, 6 to 12 clips from 12 minutes on.

**Past half an hour the count follows the length** (`length_floor`): at least
one clip for every `CLIP_LONG_SECONDS_PER_CLIP` (300 s; 0 turns it off), at
most `CLIP_MAX_PER_JOB` (60, also what a worker's metadata may carry). The
shortlist grows to five windows for every three clips of that floor and is
read in groups of ten in video order (`detail_groups`), one detail call per
group, each asked for its own 6-12: an hour gives 12-24, two hours 24-48,
five hours 60. Until 20-nov-2026 the shortlist stopped at 10 whatever the
length, so a five-hour source came back with the 6-12 clips of a quarter of an
hour while it was charged for every minute. On a five-hour transcript the
count went from 6 clips to 58, from every hour of it, with ten detail calls
instead of one. A pinned count (`CLIP_TARGET_MIN`/`MAX`: the user's
own number, the free plan's cap) is never added to. The public pages that
state the count (`dashboard/seo/clip-tools.js`, the landing FAQ, the game and
stream pages) say the same: change them with it.

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
windows); without them it is the classic 3-15, and past half an hour the same
`length_floor` as the transcript path, spread over the stretches the model
watches (`clip_selection.vision_targets`). The transcript is stored as `{"language": "none", "segments": []}`, so
the clips have no subtitles, which is correct.

**This is the one stage that sends Gemini the video instead of frames, on
purpose**: frames can say what kind of video it is, not which 40 seconds to cut.
Gemini counts video at ~300 tokens/second, so an hour is ~1.08M tokens, past a 1M
window. A source longer than `VISION_SEGMENT_SECONDS` (2400) is therefore split by
`clip_selection.vision_segments` into near-equal stretches (no one-minute tail),
each re-encoded to a 360p / 1 fps proxy (`_vision_proxy`: re-encoded, not
stream-copied, or every timestamp would be off by up to a GOP), watched on its
own, offset back to source time, then merged with `dedupe_overlapping` +
`trim_to_best` to `CLIP_TARGET_MAX`. One failed stretch is logged and skipped.
`VISION_MAX_SEGMENTS` (8, ~5 h 20) bounds the spend of one job; past it only the
first stretches are watched. A source that fits one stretch is one stretch: it
goes through the same proxy, never as the raw file (a broken capture, such as an
HLS VOD whose audio will not decode, made Gemini answer 400 INVALID_ARGUMENT).
When the proxy's audio cannot be encoded it is retried without audio. Do not
port the frame trick here. Gemini-only: a
text-only `LLM_BASE_URL` server cannot see footage, and without `GEMINI_API_KEY`
the function logs one line and returns None, which fails the job.

The public `/gta-5-clips`, `/gta-6-clips`, `/wow-forever-clips` and `/platforms/twitch` pages state
these thresholds and the stretch sizes; if the behaviour changes, change
`dashboard/seo/pages.js` and `dashboard/seo/clip-tools.js` too.

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
