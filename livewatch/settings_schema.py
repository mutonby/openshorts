"""What the settings page shows: every option with its label, help text and limits.

The page builds its form from ``schema()`` and sends back the raw TOML-shaped
dict; ``config.from_dict`` stays the only validator, so the page can never
save a config the watcher would refuse.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .config import CAPTION_STYLES, DESCRIPTION_STYLES, PLATFORMS, POST_PLATFORMS, Settings

QUALITIES = ["720p60,720p,best", "1080p60,1080p,best", "720p,480p,best", "480p,360p,best"]
WHISPER = ["off", "tiny", "base", "small", "medium"]


def _f(key, kind, label, help="", **extra) -> Dict[str, Any]:
    return {"key": key, "type": kind, "label": label, "help": help, **extra}


def schema() -> Dict[str, Any]:
    d = Settings()
    sections = [
        {"id": "general", "title": "Recording", "fields": [
            _f("max_live_channels", "int", "Channels recorded at once", "Each 720p60 stream needs about "
               "3.5 Mbit/s. A stream the connection cannot keep up with steps itself down in quality.",
               min=1, max=10, default=2),
            _f("quality", "choice", "Recording quality", "First choice; the watcher steps down by itself "
               "when it falls behind.", choices=QUALITIES, default=d.quality),
            _f("buffer_seconds", "int", "Rolling buffer (seconds)", "How much of each live stream is kept "
               "on disk. About 260 MB per 10 minutes at 720p60.", min=120, max=3600, default=d.buffer_seconds),
            _f("check_interval", "int", "Live check every (seconds)", "How often each channel is asked "
               "whether it is live.", min=30, max=3600, default=d.check_interval),
            _f("data_dir", "text", "Data folder", "Clips, log and buffer. Relative to the config file.",
               default="livewatch_data"),
            _f("keep_raw_clips", "bool", "Keep the raw cut", "Keep raw.ts next to each Short (for "
               "re-editing).", default=d.keep_raw_clips),
        ]},
        {"id": "clip", "title": "Clip length", "fields": [
            _f("pre_seconds", "float", "Seconds before the moment", "The setup. Out-of-context clips need "
               "some.", min=3, max=120, default=d.pre_seconds),
            _f("post_seconds", "float", "Seconds after the moment", "The reaction.", min=0, max=60,
               default=d.post_seconds),
            _f("min_seconds", "float", "Shortest clip", min=5, max=180, default=d.min_seconds),
            _f("max_seconds", "float", "Longest clip", "TikTok Creator Rewards only pays videos over 60 s.",
               min=10, max=180, default=d.max_seconds),
            _f("cooldown_seconds", "float", "Pause between clips of one channel (s)", min=10, max=3600,
               default=d.cooldown_seconds),
            _f("max_clips_per_hour", "int", "Max clips per hour per channel", min=1, max=60,
               default=d.max_clips_per_hour),
        ]},
        {"id": "detect", "title": "Peak detection", "fields": [
            _f("threshold", "float", "Sensitivity threshold", "Lower = more clips, higher = only the "
               "biggest moments. The live graph on the dashboard shows this line.", min=1, max=20,
               step=0.1, default=d.threshold),
            _f("warmup_seconds", "float", "Warm-up (seconds)", "Time spent learning each stream's normal "
               "chat speed before clipping.", min=0, max=1800, default=d.warmup_seconds),
            _f("min_chat_rate", "float", "Slowest chat that counts (msg/s)", "Chats slower than this "
               "rely on loudness spikes.", min=0, max=100, step=0.1, default=d.min_chat_rate),
            _f("audio_weight", "float", "Loudness weight", "0 ignores loudness. Kick relies on it "
               "(no chat reader).", min=0, max=5, step=0.1, default=d.audio_weight),
        ]},
        {"id": "style", "title": "Look and captions", "fields": [
            _f("caption_style", "choice", "Hook style (on the video)", "ooc = out-of-context, "
               "brainrot = slang, clean = descriptive, rotate = cycle through all three.",
               choices=list(CAPTION_STYLES) + ["rotate"], default=d.caption_style),
            _f("description_style", "choice", "Post description style", "The caption under Reels, "
               "TikToks, Facebook and Threads. normal = describes the moment, weird = cryptic lines "
               "that make people comment (unproven: use rotate and compare). YouTube and X always "
               "get the plain English title.", choices=list(DESCRIPTION_STYLES) + ["rotate"],
               default=d.description_style),
            _f("captions", "bool", "Word-by-word captions", default=d.captions),
            _f("hook", "bool", "Hook text on top", default=d.hook),
            _f("whisper_model", "choice", "Speech model", "tiny = fastest, base = good default, "
               "small = better on non-English, off = no captions.", choices=WHISPER,
               default=d.whisper_model),
            _f("fps", "int", "Frames per second", min=24, max=60, default=d.fps),
            _f("encoder", "choice", "Video encoder", "auto uses Intel QuickSync when it works.",
               choices=["auto", "qsv", "x264"], default=d.encoder),
        ]},
        {"id": "music", "title": "Background music", "fields": [
            _f("mode", "choice", "Add music", "off; always; quiet = only clips with little speech "
               "(chases, crashes, nobody talking). Tracks come from the Music tab. Use royalty-free "
               "music only: copyrighted songs get Shorts claimed and Reels muted.",
               choices=["off", "always", "quiet"], default=d.music_mode),
            _f("volume_db", "float", "Music volume (dB)", "-20 is a soft bed under speech; it dips "
               "further whenever the streamer is loud.", min=-40, max=0, step=1,
               default=d.music_volume_db),
            _f("folder", "text", "Music folder", "Empty = livewatch_data/music.", default=""),
        ]},
        {"id": "ai", "title": "AI writer", "fields": [
            _f("model", "text", "Gemini model", "Needs GEMINI_API_KEY (Keys section). Without a key, "
               "templates and chat phrases are used.", default=d.ai_model),
        ]},
        {"id": "post", "title": "Posting", "fields": [
            _f("enabled", "bool", "Post automatically", "Off: every clip is saved for review and you post "
               "it from the Clips tab.", default=d.post_enabled),
            _f("platforms", "multi", "Networks", choices=list(POST_PLATFORMS), default=d.post_platforms),
            _f("upload_post_user", "text", "Upload-Post profile name", "Not your email: the name of "
               "the profile your socials are connected to. Use Find my profiles below.", default=""),
            _f("min_gap_minutes", "int", "Minutes between posts", "Later clips are scheduled, not "
               "dropped. Bursts get reach throttled.", min=0, max=1440, default=d.min_gap_minutes),
            _f("daily_cap", "int", "Posts per day (max)", min=1, max=200, default=d.daily_cap),
            _f("skip_spoilers", "bool", "Hold story spoilers", "Rockstar takes down standalone "
               "cutscene and ending videos.", default=d.skip_spoilers),
            _f("base_hashtags", "list", "Hashtags on every post", "Comma separated, without #.",
               default=d.base_hashtags),
        ]},
    ]
    channel = [
        _f("platform", "choice", "Platform", choices=list(PLATFORMS), default="twitch"),
        _f("name", "text", "Channel", "Twitch/Kick login, or @handle for YouTube.", default=""),
        _f("label", "text", "Display name", "Optional; used in titles.", default=""),
        _f("game_filter", "list", "Only while playing", "Comma separated; empty = always. Ignored on "
           "YouTube.", default=["Grand Theft Auto"]),
        _f("tags", "list", "Extra hashtags", "Comma separated, without #.", default=[]),
        _f("caption_style", "choice", "Style for this channel", "Empty = the global style.",
           choices=[""] + list(CAPTION_STYLES), default=""),
        _f("facecam", "facecam", "Facecam box", "x, y, width, height as fractions (0-1). Empty = "
           "blurred layout. Find it with 'snapshot'.", default=None),
        _f("language", "text", "Spoken language", "auto, or a code like en, es, fr.", default="auto"),
    ]
    secrets = [
        _f("GEMINI_API_KEY", "secret", "Gemini API key", "Free at Google AI Studio. Better hooks and "
           "titles, spoiler and quality checks."),
        _f("UPLOADPOST_API_KEY", "secret", "Upload-Post API key", "Needed to post. Connect your "
           "socials at upload-post.com."),
    ]
    return {"sections": sections, "channel": channel, "secrets": secrets}


SECRET_KEYS: List[str] = ["GEMINI_API_KEY", "UPLOADPOST_API_KEY"]
