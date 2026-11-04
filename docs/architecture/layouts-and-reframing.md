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

With the picker on, **the SPLIT check runs on every video**, whatever the model
answers (`PER_SCENE_FLAGS`: `split_layout` + `active_speaker`). The video-level
`split` answer was the wrong gate: an edited podcast cuts between close-ups and
a wide two-shot, the 12 stills are mostly close-ups, the model rightly says
`none`, and every two-shot rendered GENERAL (two small people in a blurred
strip). `split_layout`'s own per-scene guards (below) decide each scene.

## Subtitles already burned into the source (`burned_subtitles.py`)

A re-uploaded TV show often carries its own subtitles in the picture; burning
ours on top gave two text layers, theirs cut off at the sides by the 9:16 crop.
When the job captions (`AUTO_CAPTIONS` not `0`) and the source is not vertical,
one Gemini call on the picker's 12 stills asks, per frame, for the top of the
subtitle text or null (logos, tickers, lower-thirds and scene text excluded).
Present in at least 40% of the frames and starting below 75% of the height →
`source_subtitle_band` in the job metadata, and `main.render_clip` trims that
band off the bottom of the cut clip before reframing (one extra fast encode;
cropping from the bottom keeps every face coordinate). The recut and the
editor's re-render pass the same band, but only when captions go back on.
`SUBTITLE_BAND_CHECK=0` disables it; any failure renders as before.

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

## Where the hook is drawn (`hook_placement.py`)

The hook used to sit at a fixed 20% of the height, which is where a TRACK
speaker's eyes are and where the top SPLIT speaker's face is. `position="auto"`
(the job pipeline's auto-hook, `/api/hook` without a position, and the editor's
default) samples 6 frames of the hook's window, detects faces with MediaPipe
(boxes grown to the whole head), and picks the height that covers the least
face, inside the platform-safe area (6%-80%) and never on a caption band: the
seam on SPLIT stretches, the bottom elsewhere (`layout_ranges`). A WIDE
stretch records its picture's band in the sidecar (`content`), and the hook
stays in the blur above or below it when it fits (`content_bands`), else the
caption-only rule applies. Ties go to
the old 20% spot, so a clip with no face renders as before; the text shrinks
(0.85, 0.72) only when every height still covers a face. No frames or no
MediaPipe means the old spot. An explicit top / center / bottom is drawn
exactly there. The browser preview cannot see faces and draws auto at the top.

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
  scenes whose meaning lives outside the centre. **Until 6-oct-2026 nothing
  asked which scenes those were**: the flag was set (by `layouts=["screencast"]`
  or the picker) but `reframe_v2.render` never got any content ranges, so every
  screen tutorial rendered GENERAL/TRACK, even when forced. Now each clip asks
  (`reframe_v2.screen_ranges` → `screencast_layout.detect_content_ranges`): one
  1024px still per shot, one Gemini call per clip, a closed choice per shot
  (`screen` / `beside` / `camera`, `gemini_worker.SHOT_CONTENT_PROMPT`) plus the
  horizontal **reading area** of a screen. `camera` shots keep the face
  classifier's verdict. `beside` (width 0.7) stacks **only the content
  columns** (the side away from the presenter's body, `content_side`) over the
  face-framed presenter as SCREENCAST; stacking the whole frame showed the
  presenter twice, and a presenter who leaves less than 30% of the width
  beside them is not stacked. A `screen` or `beside` scene with a person
  standing in it at full size (YOLO body at least 60% of the frame height,
  `_filmed_person`) is a filmed room, a whiteboard or TV behind the presenter,
  and keeps the face classifier's verdict: Gemini answers `beside` or
  `screen` for that shot whatever the prompt says (measured 17-oct-2026), while
  a presenter laid over a screen measures 22-40% of the height. `screen` (width 1.0: the presenter, if any, is
  composited on top of it, and stacking would show it twice) gets WIDE, cropped
  to the reading area when that is narrower than 90% of the frame
  (`focus_crop`, never past 60% of the output height, full source height kept);
  otherwise a moderate zoom (`wide_crop`): the whole width of a 16:9 screen
  was a 608 px strip, 32% of the frame, illegible on a phone (gameplay chases
  and pileups land here). The picture now fills the window between the hook's
  default spot and the bottom captions (`WIDE_CONTENT_HEIGHT_RATIO` 0.41: x1.3,
  77% of the width kept, at most ~11% off each side; capped at x1.4 for
  ultrawide, untouched for 4:3), and the crop slides toward the shot's motion
  (`wide_action_centre`: frame differences per column, so a still HUD or a
  pillarbox never pulls it; it moves only when that gains 10% of the motion,
  so a pan or a third-person chase stays centred). The editor's explicit
  "full frame" framing (`force_strategy='WIDE'`) still keeps the whole width.
  `WIDE_CONTENT_HEIGHT_RATIO=0.32` restores the old full-width WIDE. If the check cannot answer (no key, API error) the
  scenes the classifier sent to GENERAL are taken as the screen
  (`fallback_ranges`), so an explicit choice never silently renders GENERAL. One
  still per scene means a long scene that mixes screen and camera is judged by
  its middle frame. A `screen` scene with the presenter's camera floating over
  it (a QuickTime/Loom window mid-screen, not only in a corner) becomes INSET
  with a box centred on that face (`camera_inset.detect_in_scene`: small,
  still across 5 samples); the clip-wide corner inset is applied only to the
  scenes where a face actually sits in its box (`present_in_scene`), since a
  tutorial clip mixes shots with and without it. The old whole-video detector (time ranges + width_fraction)
  was never called in production and is gone: it uploaded the source, ~670k
  tokens for a 37-min tutorial.
- **INSET** (`camera_inset.py`): full-width screen on top, the enlarged webcam
  box below, for a single source with the camera composited in a corner (OBS,
  stream VODs). Chained after the `screencast` decision, **not** asked of Gemini
  (offered as a fourth option it answered `screencast` and overall accuracy
  dropped). The geometric detector needs three filters: a **small** subject,
  **horizontally off-centre** (a talking-head face is centred even when high),
  and **still between samples**. A **face** detection is grown to three face
  widths (the webcam's head-and-shoulders framing) and the box never starts
  below the head (`FACE_HEADROOM`): grown like YOLO's torso box and pinned to
  the bottom edge it cut the forehead off.
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
