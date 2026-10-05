# Layout selection and vertical reframing

## Choosing the layout (`layout_picker.py`, `app.py:layout_env`)

`POST /api/process` accepts `layouts`: a JSON list or comma-separated string of
`auto`, `split`, `screencast`, `speaker_cut`, `punch_in` and `none`. Each name
turns on its env variable for **that** job (`app.py:layout_env`); `none` turns
the picker off even when the deployment runs with `AUTO_LAYOUT=1` (plain crop
and nothing else). Without `layouts` the deployment env decides. The dashboard
exposes it under advanced options ("vertical layout": auto / split / screencast
/ none, `MediaInput.jsx`, remembered in `localStorage.os_layout`).

`auto` enables `layout_picker.py`: **one** Gemini call per source video (not per
clip) choosing between `none` / `screencast` / `split`. On a hand-labelled
48-clip corpus it scores 92-96% across passes with 0-1 false positives on the
clips that must not be touched.

**It sends 12 frames at 1024px, not the video.** Gemini bills video at ~300
tokens per second: an hour of source is ~1.08M tokens (past a 1M window) and a
1-2 GB upload to get one word back. Twelve frames cost ~3k tokens **whatever the
source length**. Resolution matters and frame count does not: at 640px a
spreadsheet is illegible, 1024px is clearly better, and 24 frames is worse. At
1024px the gap to sending the whole video is within the video mode's own
variance, at ~2 s per clip instead of ~15 s.

What makes it work, and should not be undone: the model is asked for a
**decision between closed options**, not a measurement. Earlier attempts (Canny,
MSER, temporal coverage, width) asked for a number and none separated a
spreadsheet from a corner ticker.

`layout_picker.apply()` only **adds**: an explicit user choice is never turned
off because the model said `none`.

## Hook grounding for on-screen clips (`hook_grounding.py`)

The hook and title come from the detail pass, which only reads the transcript, so
on a clip whose meaning is on screen they summarise the video's topic instead of
naming what is shown. After the render, if the `<clip>.layout.json` sidecar says
at least 25% of the clip is `screencast` / `wide` / `inset` (plus `general` when
the layout picker called the video a screencast), three frames from those
stretches at 1024px plus the clip's own words go to Gemini (`GroundedHook`) and
`viral_hook_text` / `video_title_for_youtube_short` are rewritten in place before
`auto_hook_clip` burns them; the originals stay under `hook_grounding.before`.
Gemini-only (frames): with just a local LLM it logs one line and keeps the
transcript hook. `HOOK_GROUNDING=0` disables it. The detail prompt also carries
the rule "about this moment, not the video".

## Reframing modes

**A source already shot vertical is passed through untouched.**
`reframe_v2.source_already_fits()` gates it: every layout below reorganises the
frame to buy back width the crop threw away, and on a 9:16 upload there is none.
GENERAL's 0.42 height ratio would shrink a portrait source to a sliver over a
blurred copy of itself. So the picker and the scene classifier are skipped and
every scene renders TRACK, whose crop is the whole frame. `general_filtergraph`
also floors the foreground at the height where the source fills the output
width, so an explicit GENERAL override on a portrait clip cannot shrink it.

- **TRACK** (single subject): MediaPipe face detection + YOLOv8 fallback with
  "Heavy Tripod" stabilization.
- **GENERAL** (groups/landscapes): blurred-background layout preserving full width.
- **SPLIT** (two-shot conversation, `split_layout.py`, `SPLIT_LAYOUT=1`): both
  speakers stacked in half-frames. v2 engine only; a fallback to the v1 loop
  silently renders GENERAL. It upgrades scenes the classifier already sent to
  GENERAL, never TRACK ones, and needs both faces visible **in the same frame**
  for at least half the sampled frames (that separates a real two-shot from
  shot/reverse-shot, where stacking would show the same person twice).
  `SPLIT_TIGHTNESS` (default 0.8) trades a little upscale for keeping the other
  speaker out of each half. Captions on a SPLIT stretch sit on the seam between
  the halves (`{\an5}` per word event in `subtitles.generate_ass`); the render
  records stacked stretches in a `<clip>.layout.json` sidecar (`layout_ranges.py`)
  and every metadata writer copies it into the clip's `layout_ranges`, so
  `/api/subtitle` finds it after a restyle too. The fast rerender (cut without
  reframe) carries the ranges through the new cut (`layout_ranges.remap`, in
  `recut.perform_recut`). Only the ASS path can do this; SRT burns keep one
  alignment for the whole file.
- **SCREENCAST / WIDE** (`screencast_layout.py`, `SCREENCAST_LAYOUT=1`): for
  scenes whose meaning lives outside the centre. Gemini reports each range's
  **width_fraction**, and that is the gate (coverage did not separate a
  spreadsheet from a corner ticker; width does). Content narrower than 0.5 moves
  nothing. Between 0.5 and 0.85 SCREENCAST stacks it over the presenter. Above
  0.85 the presenter is composited on top of the content and stacking would show
  it twice, so those scenes get WIDE: the GENERAL layout with side-cropping off.
- **INSET** (`camera_inset.py`): full-width screen on top, the enlarged webcam
  box below, for a single source with the camera composited in a corner (OBS,
  stream VODs). Chained after the `screencast` decision, **not** asked of Gemini
  (offered as a fourth option it answered `screencast` and overall accuracy
  dropped). The geometric detector needs three filters: a **small** subject,
  **horizontally off-centre** (a talking-head face is centred even when high),
  and **still between samples**.
- **ALTERNATE** (`active_speaker.py`, `SPEAKER_SIGNAL=1` + `SPEAKER_CUT=1`):
  hard cuts to whoever is talking, rendered through the TRACK path as a
  trajectory with jumps. `SPEAKER_SIGNAL=1` alone just gates SPLIT on both people
  actually speaking. Mouth activity **must** be normalised per speaker before
  comparing (`normalise_activity`): raw frame-difference magnitude scales with
  local contrast and lighting.
- **Punch-in** (`punch_in.py`, `PUNCH_IN=1`): not a layout. A ~12% push on the
  clip's beats, riding the TRACK path by widening its per-frame crop command from
  x-only to w/h/x/y. Beats come from the audio envelope; `emphasis_times` is a
  plain list of seconds so transcript hook words can replace it.

## Key classes

- `SmoothedCameraman`: stabilized camera movement with safe-zone logic (prevents jitter).
- `SpeakerTracker`: prevents rapid speaker switching, handles temporary occlusions.
