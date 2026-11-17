# Pro mode: the talking-head re-edit (`pro_mode.py`)

A second pass over a clip that is already framed (the clean 9:16 file). It cuts
the dead air (silences measured in the audio, with air left around the words),
changes the scale a little on every cut, frames tighter on the face and follows
it, grades the picture and dims the background behind the speaker (MediaPipe
person segmentation), and burns three to five keyword titles at the top of the
frame. No music. Captions and the hook stay the product's own layers, on top;
the captions take pro mode's own look.

It is for one person talking to camera. On a screen recording, a split or a
wide shot there is no face to frame, and cutting inside the picture pulls what
is shown away from what is said.

- **Which clips** (`suggested`): vertical, at least 90 % of the clip rendered as
  TRACK (the layout ranges), and exactly one real face on at least 70 % of the
  clip's keyframes, at a plausible size. Only keyframes are decoded
  (`-skip_frame nokey`), so the check costs a fraction of a second per clip.
  The verdict is stored as `pro_suggested` on the clip and the dashboard shows
  it; it never changes the clip by itself.
- **`PRO_MODE`** (host env, read by the job process): `off` skips the check,
  `suggest` (default) only flags, `auto` also renders the flagged clips inside
  the job. `auto` is opt-in because of the cost below.
- **Cost.** Every frame goes through Python (decode, affine, segmentation,
  encode): CPU time of the order of a whole job's on a GPU host, for one clip.
  Hence three brakes: a render is always its own process (`pro_mode.run`,
  never a thread of the API or of a job; it keeps its caller's priority, so
  one asked for in the dashboard runs ahead of the niced jobs), `PRO_SLOTS` (default 1) caps
  the renders of one host with flock files `output/.pro-N.lock`, and a caller
  that cannot get a slot gives up instead of queueing: the API answers 503
  after `PRO_WAIT_SECONDS` (20), a job ships the clip as it is after
  `PRO_JOB_WAIT_SECONDS` (120). `PRO_THREADS` (2) bounds OpenCV inside a render.
- **The switch**: `POST /api/clip/mode {job_id, clip_index, mode}` with `pro` or
  `normal`, on any clip (the suggestion is advice, not a gate). It takes the
  per-job rerender lock, walks the served file back to its clean base
  (`_strip_burned_captions`, `_strip_burned_hook`), converts that file, and
  puts back the layers that were on it: the hook from the clip's `auto_hook`,
  captions through `_reapply_captions` (so in the clip's `caption_style`),
  the free plan's mark through `_deliver`.
- **Titles** (`pick_titles`, `sanitize_titles`, `place_titles`,
  `write_titles`): one model call per conversion proposes the words (Gemini
  with the caller's entitlement, `resolve_gemini`, or the local LLM); without
  a model, or on any failure, the clip simply has none. The code keeps only
  words really spoken at the index the model names (a drifted proposal falls
  back to the spoken word), and places them on the cut timeline: only where
  the word survived, only on a shot framed on the face (the top of the frame
  is headroom there; over a board it would cover content), never during the
  hook's seconds, 2.5 s apart, five at most. They are burned into the pro
  file itself (Archivo Black, `fonts/`), in the render's own encode: no extra
  pass, and going back to normal is what removes them. The words shown are
  kept as `pro_titles`.
- **Captions.** A clip with burned captions and no style of its own gets the
  `cine` preset (`subtitles.CAPTION_PRESETS`) as its `caption_style`; a style
  the user picked is never replaced. Going back to normal removes it only if
  it is still the one pro mode set (`pro_before.caption_style_set`).
- **Which file.** The job writes no `video_url` into the metadata, so the
  switch finds the served file through `_served_clip_file`: what the dashboard
  says it shows (`input_filename`, only a file of that clip), the in-memory
  result, the metadata, the newest derived file on disk. Starting from the
  clean reframe instead silently dropped the captions and the hook.
- **When.** Only on a finished job (409 while it runs: the job rewrites the
  metadata when it ends) and only while the clip's clean file is on disk (409
  once a purge left the served file alone: reframing a picture with captions
  burned in would crop them).
- **The pro file is a recut.** It is named `recut_<ts>_<hex>_<clean>` and the
  clip's `recipe.segments` are the stretches that survived, on the source
  timeline (`source_segments` maps them through whatever recipe the clip
  already had). So captions (`virtual_transcript`), the clip editor's EDL, the
  walk-backs, the `/videos` guard and the archive need nothing new.
- **Going back.** `convert` keeps what it replaced (`start`, `end`, `recipe`,
  `layout_ranges`, `crop_overrides`, `caption_words`, the clean base's name)
  under `pro_before`; `normal` puts it back. The file served before the switch
  is reused when the layers on the clip are still the ones it was burned with
  (`_clip_layers`), otherwise hook and captions are burned again.
- **Other edits.** A rerender or a manual reframe rebuilds the clip from the
  canonical file or the source, which is the normal look: both clear `mode`
  and `pro_before` (`_leaves_pro_mode`).
- **In the job** (`main.pro_clip`, `auto` only): the canonical reframe stays
  on disk as the clip's normal mode; hook and captions are burned on the pro
  file against the remapped transcript. A busy host or a failed render ships
  the clip as it was: pro mode never costs a clip.
