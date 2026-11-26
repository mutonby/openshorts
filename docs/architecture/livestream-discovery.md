# Livestream highlight discovery

## Original pipeline and bottlenecks

The React dashboard submits source and render options to FastAPI in `app.py`.
The queue writes a resume manifest and starts `main.py` in a subprocess.
Docker runs separate API, Vite dashboard, and Remotion renderer services.
The same API also serves REST, MCP and CLI clients. Remote workers receive
closed-list job options and may reuse a prepared transcript.

URL jobs call `download_youtube_video` before discovering moments. yt-dlp
extracts formats and FFmpeg merges the video and audio. The early-audio branch
overlaps ASR with the download; when enabled it also tries the Gemini pass. The
main path still waits for the full video. Audio can be downloaded twice.
The source cache reuses recent eligible full downloads and matching
transcripts. The per-job ASR checkpoint survives retries. faster-whisper
builds the spectrogram of everything one call transcribes (~0.65 MB per second
of audio, measured), so media longer than `WHISPER_CHUNK_SECONDS` (default
30 min) is transcribed in pieces cut at the quietest point near each boundary,
and the times are shifted back onto the recording (`transcribe_backends.
_transcribe_whisper_chunked`). One call over an 11-hour stream would ask for
~25 GB and end in SIGKILL. Completed Gemini
batches were held in memory, so an exhausted 503 during early analysis could
cause the entire scoring pass to run again on the same transcript.

The transcript picker ranks overlapping windows, details the shortlist and
snaps boundaries to words. Sparse speech or silence goes to vision proxies.
Candidate review validates the shortlist against frames and speech and saves
it to disk. Approval reuses that plan and queues selected indices. The
framing and FFmpeg paths write the clean cut, then layer captions, hook and
optional watermark. Gameplay Priority keeps the original game frame and
embedded camera together. The dashboard's final preview and download use the
same rendition. Existing exports have no independent square-pixel check.

## Implementation plan

- Checkpoint complete AI batches by prompt and schema, keep all transcript
  windows, and bound retry spend without substituting random picks.
- Collect source metadata, aligned replay where available, transcript, and
  audio reaction hints. Keep transcript scoring as its own discovery route.
- Browse candidate ranges before rendering. Retain source-time provenance and
  use high-quality source packets for approved ranges. Keep the full-source
  workflow selectable when a transport cannot seek reliably.
- Validate decoded output and square pixels before announcing a clip. Keep a
  flagged candidate and its source analysis available for correction.

Platform reference material includes the
[YouTube Live Chat Messages API](https://developers.google.com/youtube/v3/live/docs/liveChatMessages/list),
[Twitch EventSub chat](https://dev.twitch.tv/docs/chat/send-receive-messages/),
[Twitch Clips and VOD offsets](https://dev.twitch.tv/docs/api/clips/),
[yt-dlp setup and usage](https://github.com/yt-dlp/yt-dlp),
[yt-dlp JavaScript runtime setup](https://github.com/yt-dlp/yt-dlp/wiki/EJS),
and [Gemini API errors](https://ai.google.dev/gemini-api/docs/troubleshooting).

## Deployment and workflow

Set `HIGHLIGHT_DISCOVERY_ENABLED=1` in the self-hosted backend environment.
For Compose, pass it in the backend service's `environment` through a local
override. Recreate the backend after changing its environment. The feature
keeps the existing editor, layout and output settings and is inactive on
managed Cloud jobs. Users opt in under **Livestream highlights** while
**Review moments before rendering** is selected. Uploads and Twitch VODs can
use the signals with the existing full-source workflow.

For a compatible YouTube archive, select **Find moments before downloading
full-quality footage**. The source metadata establishes the video ID, duration
and available media tracks. Whisper uses a separate native audio download.
A recording with sparse speech still needs the full-source vision workflow.
Use the full-source workflow for live/incomplete recordings, authenticated
or proxy-only sources, manually bounded source ranges or sources that cannot
serve validated HTTPS MP4 byte ranges. The staged path stops clearly at an
unsupported transport. It does not fall back to a multi-gigabyte download.
Where the full-source cache already contains the video, the review reuses it.

The preview is a short, lower-resolution source rendition. For approval, the
worker pins the source formats from review and copies only the needed
full-quality audio and video packets with their original presentation
timestamps. The loopback range adapter uses verified TLS for origin reads,
declines redirects, and caps bytes across retries. Each retained range must
cover the requested time and match source resolution. Clip Editor reframe
and recut can use retained high-quality packets within that range. Use the
existing full-source workflow to edit beyond it.

Import a CSV with `time_seconds,author_id,text`, a JSON/JSONL replay, a
yt-dlp YouTube replay JSON, or a Twitch export with relative VOD offsets.
A declared video ID must match. Use the offset field only when you have
verified alignment. Replay files are limited to 32 MiB and stay under the
job's private working directory. Only aggregate activity, reaction categories
and participant counts appear in the review. yt-dlp can retrieve the replay
for an eligible completed YouTube archive when the extractor advertises it.
Set `HIGHLIGHT_FETCH_CHAT=0` to use imports only. The official YouTube and
Twitch live APIs are for active subscriptions. Helix clips supply VOD offsets
when the existing Twitch client credentials are configured. Completed Twitch
VOD chat needs an already captured, aligned replay. The third-party
[chat-downloader](https://github.com/xenova/chat-downloader) is not a runtime
dependency. Its relative timestamps can be imported.

Chat compares a 10-second bin with the previous 90 seconds using median
volume and median absolute deviation. Same-author repeats, commands,
giveaways and other repetitive traffic receive little or no weight.
A configurable reaction delay shifts the search earlier than the chat burst.
Mixed-audio level changes and chat windows add candidates alongside the
normal full transcript scoring pass. Frame and speech evidence of setup and
payoff drive the editorial order. A chat spike alone cannot mark a clip
strong. The timeline shows candidate, audio, chat and available Twitch viewer
markers. The full-source player can seek those cues. The audio-first player
shows prepared candidates and can open other timestamps in the original
recording.

Completed AI score and detail batches stay in the job directory, keyed by
provider, model, prompt and schema. Gemini 429 and 503 retry with exponential
backoff and jitter. Pause a running audio-first discovery and resume from
saved audio, transcript and successful analysis batches. Existing resume and
delivery manifests remain compatible. Optional request and prompt-byte limits
are per scoring invocation. The displayed model estimate includes completed
score/detail batches. Candidate visual assessment, provider retries and
other processing may add costs.

The FFmpeg gate measures dimensions, square pixels, duration, track presence
and track alignment, then checks three decode samples for unreadable or
blank video. Candidate renders are gated automatically. Set
`HIGHLIGHT_VALIDATE_EXPORTS=1` to gate other generations and original-source
editor renders too. A failed candidate never becomes a normal downloadable
clip. An editor render that fails validation leaves the previous file and
recipe in place. Alignment checks detect stream timing problems. Captions,
actual gameplay relevance and visual storytelling still need editorial review.

The Docker backend already includes a compatible JavaScript runtime and the
yt-dlp default extras. On a host installation, keep the solver package version
required by the installed yt-dlp release and use a supported Node or Deno
runtime. TLS verification stays enabled. A missing platform token can limit
which formats are offered. These changes use the offered formats and do not
supply credentials or override platform restrictions.
