# livewatch: live streams to Shorts, hands-off

`livewatch` watches Twitch, Kick and YouTube channels. When one goes live
(optionally only while it plays a given game), it records a rolling buffer and
finds the moments chat and audio explode. Each moment becomes a 1080x1920 Short
with word-by-word captions and a viral hook, plus a title, description and
hashtags written for each network. Then it posts the Short, or keeps it for
review.

It is standalone (no Docker, no OpenShorts server) and sized for a modest PC.
The reference machine is a 2-core i5-7200U laptop with 8 GB of RAM, no usable
NVENC and Intel QuickSync. On it, a 25-30 s Short is ready 15-30 s after
the moment, while two 720p60 streams keep recording. A stream the connection
cannot keep up with steps itself down in quality.

## Install

```bash
pip install streamlink yt-dlp faster-whisper requests
```

`ffmpeg` must be on the PATH. Check everything with:

```bash
python -m livewatch doctor
```

## The dashboard

Double-click `livewatch.bat`, or run:

```bash
python -m livewatch ui
```

This opens http://127.0.0.1:8765 in your browser. On first launch it creates
`livewatch.toml` from the example. The dashboard only answers on this PC.

| Tab | What it does |
|---|---|
| Dashboard | Start/Stop and Check now. Shows each channel's state (offline, waiting for the game, recording), quality, buffer and chat count, plus a live score graph with the clip line. Also shows counters, the clip being worked on, and the latest activity. |
| Replay test | Runs the whole pipeline on an ended YouTube stream (see below), optionally publishing too. |
| Clips | Every Short, with a player, hook, styles, score and post status. Review & post lets you edit each network's title and caption, tick networks and post. You can also open the folder or delete a clip. |
| Music | The background music library: upload by drag and drop, preview, take tracks in and out of rotation, delete. |
| Settings | Every option with an explanation, the channel list, and the API keys (written to `.env`, never to the settings file). Values are checked before saving. |
| Log | The full log, with a filter. |

## Configure by hand

```bash
cp livewatch.example.toml livewatch.toml
```

List your channels as `[[channel]]` blocks. Secrets go in `.env`, next to the
config:

```
GEMINI_API_KEY=...        # optional: AI hooks/titles/hashtags (offline templates without it)
UPLOADPOST_API_KEY=...    # to post: connect your socials at upload-post.com
```

Set `post.enabled = true` and `post.upload_post_user` to post automatically.

## Run without the dashboard

```bash
python -m livewatch run
```

| Command | What it does |
|---|---|
| `ui [--port N]` | the dashboard |
| `run [--minutes N]` | watch, clip, post (Ctrl+C stops after finishing the queue) |
| `replay <youtube link> [--minutes 30] [--start 0] [--max-clips 5] [--post / --no-post]` | test the pipeline on an ended stream |
| `doctor` | check tools, encoder, keys, config, disk |
| `chat twitch <name>` | show a live chat's peak score, to tune `threshold` |
| `snapshot twitch <name>` | save a frame with a 10% grid, to read the `facecam` box |
| `clip <file> --start 60 --end 105 --style brainrot` | same pipeline on a local video |
| `post <clips/.../dir>` | post a clip that was held back |

Everything lands in `livewatch_data/`:

- `clips/<channel>/<time>/short.mp4`, with `meta.json` holding the
  transcript, chat, every caption and the post status.
- `clips.csv`, one line per clip. Use it to submit links to clipping campaigns.
- `livewatch.log`.

## How moments are found

- **Chat speed.** Twitch (anonymous read-only IRC) and YouTube (yt-dlp live
  chat) count messages per second. An 8 s window is compared with the same
  stream's previous five minutes. Two tests must agree: the stream's own
  spread (median/MAD) and a Poisson test, so a slow chat's handful of
  messages is not a burst. Chat must at least double, and one user spamming
  is discounted.
- **Hype words.** KEKW, OMEGALUL, ???, 💀, "clip it", all caps and similar
  boost a real burst. They never create one.
- **Loudness.** The per-second RMS of the recorded audio is compared with its
  own baseline. Kick has no chat reader, because its supported chat API needs
  a public webhook URL, so Kick moments come from loudness alone.
- Chat reacts a few seconds late, so the clip is centred about 4 s before
  chat's crest. It covers `pre_seconds` before and `post_seconds` after.

On simulated chats, 8 to 30 messages/s with 2.5x bursts were caught 40 out of
40 times with no false peaks in 20 hours of normal chat
(`tests/test_livewatch.py` has the reduced version).

## Styles

All text is English. Two settings, chosen separately.

`caption_style` sets the hook burned on the video: `ooc` | `brainrot` |
`clean` | `rotate`.

- **ooc**: out-of-context, lowercase, deadpan, in a white box. For example,
  "he really thought nobody saw that".
- **brainrot**: slang ("this is peak gta 6 cinema fr"), Anton font, green
  highlight.
- **clean**: a descriptive hook.
- **rotate**: cycles through all three.

`description_style` sets the caption under Reels, TikToks, Facebook and
Threads posts: `normal` | `weird` | `rotate`.

- **normal**: describes the moment.
- **weird**: cryptic, unrelated-sounding lines that make people stop and
  comment, such as "the man in the red car knows what he did". Whether this
  helps reach is unproven, so use `rotate` and compare views per style.

YouTube and X always get the plain English title, which is what search and
the feed read.

Gemini sees the transcript, the chat lines from the moment and the stream
title. It also flags story spoilers: Rockstar takes down standalone cutscene
and ending videos, so `skip_spoilers` holds those back. It rates each clip
0-10, and clips under 4 are held.

## Replay test (ended YouTube streams)

Replay checks your outputs and publishing without waiting for anyone to go
live. Give it a link to a YouTube video that was a live stream with chat. It
then:

1. Downloads the chat replay up to the end of the window you chose. YouTube
   serves it at roughly 10-15x real time, so 30 minutes of stream take about
   2-3 minutes.
2. Runs the same peak detector over the chat.
3. Downloads only the seconds around the best peaks, never the whole
   multi-hour video.
4. Makes each one into a Short with your current settings, and posts it if
   you ticked "Publish".

On the reference laptop, 20 minutes of a GTA RP stream gave 3 Shorts in under
7 minutes. Replays find peaks from chat only: the full audio is never
downloaded, so there is no loudness signal.

## Background music

Put royalty-free tracks in the Music tab or in `livewatch_data/music`, then
set the mode under Settings > Background music:

- `always`: every Short gets music.
- `quiet`: only clips with little speech get music, such as chases, crashes
  or nobody talking.

Each Short uses the track used least recently, starting somewhere inside it.
The track fades in and out and sits at `volume_db`, and dips further whenever
the streamer is loud (sidechain ducking). Copyrighted songs get Shorts
claimed and Reels muted, so use the YouTube Audio Library, Pixabay Music or
your own.

## Posting

Posting goes through Upload-Post, one request per clip carrying every
network: YouTube Shorts, TikTok, Instagram Reels, Facebook Reels, X, Threads
and Bluesky. Each network gets its own text:

- YouTube: a title of 100 characters or less with `#shorts`.
- Instagram: 5 hashtags.
- Threads: 1 topic tag.
- X and Bluesky: kept short.

Posts are spaced by `min_gap_minutes` (later ones are scheduled, not dropped)
and capped by `daily_cap`. Bursts of near-identical uploads get an account's
reach throttled.

## Rules that matter for getting paid

- Clip streamers who allow it, best through their clipping campaigns. Those
  pay per view on every network, which reposting alone does not.
- YouTube does not pay for unedited re-uploads. Captions, a hook and a
  title are the minimum. TikTok Creator Rewards only counts originals over
  60 s.
- Keep GTA 6 story spoilers out (`skip_spoilers = true`).
