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
| `replay <youtube or twitch VOD link> [--minutes 30] [--start 0] [--max-clips 5] [--post / --no-post]` | test the pipeline on an ended stream |
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

## The card look (default)

`style.layout = "card"` presents each Short like a post from a big clip page:

- a black header with your **logo**, page name, a blue badge and **@handle**;
- a news-style headline with an emoji, such as "First look at the motorcycle
  crash physics in GTA 6 👀";
- the stream below, at **full width with nothing cropped**, so a facecam in a
  corner is never cut off;
- the word captions under the video.

Set the brand in **Settings > Your page**:

- **Import logo** takes a square PNG or JPG. Without a logo, the header shows
  the @handle alone.
- The preview shows the result before you save.

The header and headline are drawn with Pillow, because libass cannot draw
colour emoji. `layout = "blur"` keeps the older look (the stream over a
blurred copy of itself), now also without the side crop.

## Styles

All text is English. Two settings, chosen separately. `caption_style` defaults
to `news`, the headline style above. Titles and post captions carry 1–3
emojis.

`caption_style` sets the hook burned on the video: `ooc` | `brainrot` |
`clean` | `rotate`.

- **ooc**: out-of-context, lowercase, deadpan, in a white box. For example,
  "he really thought nobody saw that".
- **brainrot**: slang ("this is peak gta 6 cinema fr"), Anton font, green
  highlight.
- **clean**: a descriptive hook.
- **rotate**: cycles through all three.

`description_style` sets the caption under Reels, TikToks, Facebook and
Threads posts: `normal` | `weird` | `japanese` | `rotate`.

- **normal**: describes the moment.
- **weird**: cryptic, unrelated-sounding lines that make people stop and
  comment, such as "the man in the red car knows what he did".
- **japanese**: Japanese advertising-style lines, like those under many viral
  Reels, such as "#GTA6 話題の最新作がついに解禁 思わず二度見してしまう衝撃の瞬間を一挙公開!".
  Gemini writes them in natural Japanese; anything that is not Japanese (no
  kana) is replaced by one of 15 built-in lines. Nothing shows this trend
  boosts reach, and it may steer a Reel toward Japanese viewers. Whether this
  helps reach is unproven, so use `rotate` and compare views per style.

Whether `weird` or `japanese` helps reach is unproven, so use `rotate` and
compare views per style. YouTube and X always get the plain English title,
which is what search and the feed read.

Gemini sees the transcript, the chat lines from the moment and the stream
title. It also flags story spoilers: Rockstar takes down standalone cutscene
and ending videos, so `skip_spoilers` holds those back. It rates each clip
0-10, and clips under 4 are held.

## Replay test (ended YouTube streams and Twitch VODs)

Replay checks your outputs and publishing without waiting for anyone to go
live. Give it a YouTube video link or a Twitch VOD link
(`twitch.tv/videos/<number>`). It finds the moments in the window you chose
from whatever the video offers:

- **YouTube chat replay**, when the stream kept one. It downloads at roughly
  10-15x real time, so 30 minutes of stream take about 2-3 minutes.
- **Loudness**, when there is no chat. Many channels, IShowSpeed's for
  example, turn chat replay off, and Twitch VODs never have one. Only the
  audio of the window is downloaded (about 20 MB per 30 minutes).
- **Twitch viewer clips**, when `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`
  are set: the moments viewers clipped from that VOD, through Twitch's
  official API. Create a free app at dev.twitch.tv/console and paste its ID
  and secret in Settings > API keys.

It then downloads only the seconds around the best moments, never the whole
multi-hour video. Each one becomes a Short with your current settings, and
is posted if you ticked "Publish".

On the reference laptop, 20 minutes of a GTA RP stream with chat gave 3
Shorts in under 7 minutes.

## Hooks that match the clip

With a Gemini key, the writer is sent 4 frames from the clip: the setup, the
middle, the peak and right after it. It also gets the transcript and the
chat. Every concrete claim in the hook and title must be visible in the
frames or said in the transcript. When it cannot tell what happened, it
writes about the reaction instead ("nobody was ready for this"). Offline
hooks, without a key, never describe an event; they are always about the
reaction.

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

Posting is free. Each network is posted with your own free developer access,
set up once in **Settings > Posting accounts**, where each step is explained:

| Network | `post.*_via` | How |
|---|---|---|
| YouTube Shorts | `youtube_via = "api"` | Official YouTube Data API (`videos.insert`) with your own Google Cloud project and a "Desktop app" OAuth client. **Connect YouTube** opens Google's sign-in and keeps a refresh token. |
| Instagram Reels | `instagram_via = "api"` | Official Instagram API with Instagram Login, using a token you generate for your own Creator/Business account in a Meta developer app. The token renews itself daily. |
| TikTok and the rest | `others_via = "phone"` | Your Telegram bot sends the Short and each network's caption to your phone, ready to paste. |

Things to know:

- **YouTube:** publish the Google OAuth app ("In production"), or Google signs
  you out every 7 days. Some reports say Google keeps API uploads from
  unaudited projects **private**. If that happens, the clip shows "youtube:
  private" with a link, so you can make it public in YouTube Studio or apply
  for Google's free audit.
- **Instagram:** Instagram downloads the Reel from a public link. For each
  post, livewatch serves only that one file, at a random address, through a
  temporary Cloudflare quick tunnel (`cloudflared`), then closes it. The
  dashboard is never exposed.
- **TikTok:** its API keeps posts from apps it hasn't audited private, so the
  phone handoff is the free way to post publicly.
- **Upload-Post:** any network can still go through it
  (`*_via = "upload-post"`) if you pay for it. Its free plan allows 10
  uploads a month.

Each network gets its own text:

- YouTube: a title of 100 characters or less with `#shorts`.
- Instagram: 5 hashtags.
- TikTok: the description plus hashtags.

Posts are spaced by `min_gap_minutes`, and capped by `daily_cap`. Later clips
wait in a queue, which survives a restart, instead of being dropped. Bursts
of near-identical uploads get an account's reach throttled.

## Rules that matter for getting paid

- Clip streamers who allow it, best through their clipping campaigns. Those
  pay per view on every network, which reposting alone does not.
- YouTube does not pay for unedited re-uploads. Captions, a hook and a
  title are the minimum. TikTok Creator Rewards only counts originals over
  60 s.
- Keep GTA 6 story spoilers out (`skip_spoilers = true`).
