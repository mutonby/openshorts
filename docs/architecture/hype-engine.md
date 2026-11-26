# Hype engine: finding stream highlights from chat and the streamer

Highlight discovery (`--highlight-discovery`, review mode) used to pick moments
from the transcript. A stream's best moments are the ones where chat goes wild
and the streamer does too, and neither is in a transcript: a scream, a laugh,
chat spamming KEKW. Transcribing a whole multi-hour stream to find them was
also most of the job's time. The hype engine reverses the order: measure the
hype over the whole recording cheaply, then spend speech recognition and the
AI only on the few windows that stand out.

```text
audio (+ chat replay, + Twitch viewer clips)
  → stage 1  hype_engine.scan       whole recording, no ASR       ~10 s per hour
  → stage 2  focused ASR            only the top windows          ~15 min of audio
             hype_judge.prefilter   drops sub/gift/donation alerts
             hype_judge.review      Gemini watches each window as video
  → review   candidate_review       ranked, with reasons, for approval
  → stage 3  ranges in full quality only for the approved clips
```

`livestream.hype_review` runs stages 1–2 for both entry points: audio-first
(`livestream.discover`, YouTube and Twitch) and full-source review in
`main.py`. It returns None when nothing stands out; transcript-first discovery
is then the fallback, unchanged. `HYPE_ENGINE=0` switches the engine off.

## Stage 1: `hype_engine.scan`

Everything is relative to the stream's own normal level, never an absolute
threshold, because streams differ by orders of magnitude.

- **Streamer voice** (`voice_features`, `voice_excitement`): one streaming
  FFmpeg decode at 16 kHz in 60 s blocks (memory flat on any length), analysed
  in 0.5 s frames: speech-band (300–3400 Hz) loudness over a 2-minute rolling
  baseline, 1.5–4 kHz energy (yells, screams) and the 3–7 Hz envelope rhythm of
  laughter, the last judged against the recording's own top 3 % of voiced
  frames (ordinary speech already has some of that rhythm). A frame counts only
  when the speech band dominates it, so a game explosion is not a reaction.
- **Chat** (`chat_curve`, `chat_intensity`): distinct chatters per second, plus
  reaction messages (`HYPE_TOKENS`: KEKW, LUL, OMEGALUL, Pog, `?????`, W, 💀…)
  and clip requests (`!clip`, "clip it", ×3). chat_hype's filters keep
  commands, giveaways and one author's repeats out. Raw chat stays on disk.
- **Chat delay** (`chat_lag`): chat trails the event. The lag between 2 and
  20 s at which the chat curve best matches the voice curve, if clearly better
  than none, else `HIGHLIGHT_CHAT_DELAY` (7 s).
- **Viewer clips** (`viewer_curve`): Twitch viewer clips of the VOD (with
  `TWITCH_CLIENT_ID`) add weight where people clipped.
- **Fusion** (`fuse`): chat shifted back by its lag, plus voice, plus a bonus
  where both agree (`sqrt(chat·voice)`), plus viewer clips.
- **Windows** (`windows`): each peak above the recording's 96th percentile
  becomes a window: the stretch above half its height, 18 s of setup before
  it and 4 s of reaction after, fitted to the clip band (`CLIP_MIN_SECONDS`
  / `CLIP_MAX_SECONDS`, default 30–60 s). Overlaps keep the higher peak.

Measured on a 3.5 h 1080p60 recording (CPU, 8 threads): 32–48 s, 200 MiB.

## Stage 2: focused transcription and the judge

- `transcribe_backends.transcribe_ranges` transcribes only the top
  `HYPE_TRANSCRIBE_CANDIDATES` (20) windows ±8 s, merged, on the recording's
  timeline. The transcript carries `coverage`; outside it nothing is known. It
  runs under the same host slot and `release_models()` as `transcribe_video`
  (`main.transcribe_ranges_job`).
- `hype_judge.prefilter` drops windows whose speech is mostly alert reading
  (`ALERT`: gifted subs, bits, donations, raids, "spam slash claim"), the most
  common false alarm, and keeps the best `HYPE_JUDGE_CANDIDATES` (12).
- `hype_judge.judge` sends each to `GEMINI_MODEL_JUDGE` (default
  `gemini-3.7-flash`) as a 360p / 10 fps / mono video with its sound, plus the
  aggregates (chatters vs normal, reaction categories, clip requests, voice
  excitement, viewer clips) and the speech. Never chat lines or authors. The
  answer (`HypeVerdict`): keep / weak / uncertain, a calibrated 0–100 virality,
  ratings, the tightest cut inside the window, the on-screen hook, title and
  description. Verdicts are cached under `.highlight/judge/` per question.
- `hype_judge.to_short` maps to candidate_review clips (`supported` needs keep,
  virality ≥ 50 and no weak payoff or clarity; weak, an alert or virality under
  `HYPE_MIN_VIRALITY` (35) drops it). `rank` puts supported first, then
  0.65 · virality + 0.35 · hype. Without a key every window goes to manual
  review ranked by hype.

The review page shows the hype curve, a "Streamer reaction" lane and, per
card, the viral score and its reasons ("chat 9x its normal pace", "streamer
laughing", "3 viewer clip(s)"). A judged clip can start after its preview
window; `preview_offset` tells the page where to seek.

## Stage 3: full quality only for what is approved

Audio-first jobs never download the video. YouTube already fetched byte
ranges (`range_media.fetch`). Twitch VODs are muxed HLS whose clock starts
near 62 s:

- `livestream._inspect_twitch` picks the best variant up to 1080p (highest fps,
  then bitrate), a 360–540p preview variant and the "Audio Only" track (the
  smallest variant when there is none). Every playlist must be HTTPS on a
  Twitch CDN host (`TWITCH_MEDIA_HOSTS`) and end in `.m3u8`.
- The audio track downloads with 8 segments in parallel.
- `range_media.fetch_hls` copies the segments covering a range with
  `-copyts`, cut on the packets' own clock (`-to base + end`), with
  `HLS_PROTOCOLS` (`https,tls,tcp,crypto,hls`: no plain http, no file), and
  verifies it with `inspect(..., base=playlist_base)` like a byte range.
  Measured: a 36 s 1080p60 range in 16 s (37 MB).

Twitch chat is not fetched: export it with TwitchDownloader and attach the
JSON (chat_hype reads its `comments` / `content_offset_seconds`).

## Settings

| Variable | Default | |
|---|---|---|
| `HYPE_ENGINE` | `1` | `0`: transcript-first discovery only |
| `HYPE_TRANSCRIBE_CANDIDATES` | 20 | windows transcribed (3–30) |
| `HYPE_JUDGE_CANDIDATES` | 12 | windows the judge watches (1–15) |
| `HYPE_MIN_VIRALITY` | 35 | judged virality under this is dropped |
| `GEMINI_MODEL_JUDGE` | `gemini-3.7-flash` | the model that watches the clips |
| `HIGHLIGHT_CHAT_DELAY` | 7 | chat lag when the curves do not agree |
| `CLIP_MIN_SECONDS` / `CLIP_MAX_SECONDS` | 30 / 60 | clip band |

## The viral edit and the streamer stack (game framings)

`viral_edit.render` replaces the plain gameplay render for `gameplay` and
`streamer` framing: a cold open on the peak (only with 6 s+ of setup), a
punch-in on each appearance of the peak, conservative dead-air cuts (speech
gaps of 2 s+, never near the peak, at most 20 %), colour emoji stickers on
reaction words (PIL + a colour emoji font; skipped without one), synthesised
sound effects (`sfx.py`: hit, whoosh, boom) and `STREAM_CAPTION_STYLE`. One
encode; any failure falls back to `reframe_v2.render_gameplay_range`. The edit
is returned as segments: captions follow it (`recut.virtual_transcript`) and
the clip's `recipe` lets the editor open it. Per job: `CLIP_VIRAL_EDIT`,
`CLIP_SFX` (also `CLIP_COLD_OPEN`, `CLIP_PUNCH`, `CLIP_DEAD_AIR`, `CLIP_EMOJI`).

`streamer` framing (`facecam.py`) lifts the webcam overlay into its own band
above the gameplay; `composition.py` sizes it (see
[layouts-and-reframing.md](layouts-and-reframing.md#stream-and-game-composition-compositionpy)).
The webcam is the face YuNet (vendored, SHA-pinned) finds at the same place in
most of 16 samples taken over the clip and 45 s either side; several people on
one camera are framed together. Its rectangle is the still region around the
face (the game changes, the webcam's room does not), else persistent edges,
else proportions; the facecam crop always holds whole heads. The gameplay crop
slides toward the clip's motion and never onto the webcam; when it cannot, or
no confident overlay is found, the clip keeps the full-picture gameplay layout.

## Live chat recorder

`python -m chat_recorder <channel>...` records Twitch chat through the
anonymous read-only IRC login into `CHAT_RECORDINGS_DIR` (default
`output/.chat-recordings`), one JSONL file per channel per UTC day with Twitch's
own send time. A Twitch VOD's review attaches the stretch it covers as the job's
imported replay (`livestream.attach_recorded_chat`), in VOD seconds from the
VOD's start timestamp; an uploaded replay wins.

## Learning loop

`learning.py` (self-host only) stores each review candidate's features with
its outcomes in `output/.learning/learning.sqlite3`: rendered or skipped
(`/render-candidates`), and the views of clips posted through Upload-Post,
picked up whenever `/api/social/analytics/posts` returns them (matched by
request id). After each outcome it refits: ridge regression of log views with
15+ posted clips, else logistic regression of approvals with 20+ decisions of
both kinds. `hype_judge.rank` blends `learning.score` in at half weight.
`GET /api/learning` reports what it has learned. `LEARNING=0` turns it off.

## Channel watcher

`python -m channel_watch --acknowledge twitch:<login> youtube:@handle` lists
each channel's past streams with yt-dlp every `--interval` seconds and queues
every new finished VOD as an audio-first highlight review (`POST /api/process`,
streamer framing, viral edit on). Finished means not live and the same length
across checks `--settle` seconds apart (a Twitch VOD grows while the stream is
live). The first pass records existing VODs as the baseline (`--backfill N`
queues the N newest instead). `GEMINI_API_KEY` in the watcher's environment is
sent as `X-Gemini-Key`, so watched jobs get the clip judge. State:
`output/.channel-watch.sqlite3`.
