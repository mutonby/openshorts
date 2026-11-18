"""livewatch: watch live streams, catch their peak moments, post them as Shorts.

A standalone, lightweight companion to OpenShorts, built to run on a modest
home PC (2 cores, 8 GB RAM, no usable NVENC):

- ``recorder``  streamlink -> ffmpeg ``-c copy`` into 10 s chunks, a rolling
  buffer per channel (no re-encoding while watching).
- ``chat``      Twitch chat (anonymous IRC) and YouTube live chat (yt-dlp).
- ``detector``  chat-velocity, hype-word and loudness spikes against each
  stream's own baseline.
- ``captions``  word-timed karaoke captions and the hook text as one ASS file.
- ``viral``     out-of-context / brainrot / clean hooks, normal or "weird"
  descriptions and per-platform hashtags (Gemini, with an offline fallback).
- ``render``    one ffmpeg pass to a 1080x1920 Short (QuickSync when present).
- ``poster``    Upload-Post to YouTube Shorts, TikTok, Instagram, Facebook, X..
- ``service``   ties it together; ``python -m livewatch run``.
"""

__version__ = "1.0.0"
