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

## Configure

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

## Run

```bash
python -m livewatch run
```

| Command | What it does |
|---|---|
| `run [--minutes N]` | watch, clip, post (Ctrl+C stops after finishing the queue) |
| `doctor` | check tools, encoder, keys, config, disk |
| `chat twitch <name>` | show a live chat's peak score, to tune `threshold` |
| `snapshot twitch <name>` | save a frame with a 10% grid, to read the `facecam` box |
| `clip <file> --start 60 --end 105 --style chinese` | same pipeline on a local video |
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

## Caption styles

`caption_style` = `ooc` | `chinese` | `brainrot` | `clean` | `rotate`

- **ooc**: out-of-context English, lowercase, deadpan, in a white box. For
  example, "he really thought nobody saw that".
- **chinese**: the Douyin look. A big yellow Chinese hook, with a Chinese and
  English title and description.
- **brainrot**: slang ("this is peak gta 6 cinema fr"), Anton font, green
  highlight.
- **clean**: a descriptive hook.
- **rotate**: cycles through all four, so you can test which one your
  audience likes.

Gemini sees the transcript, the chat lines from the moment and the stream
title. It also flags story spoilers: Rockstar takes down standalone cutscene
and ending videos, so `skip_spoilers` holds those back. It rates each clip
0-10, and clips under 4 are held.

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
