"""Cloud-facing job log view.

Self-hosted instances see the raw pipeline output (useful for debugging).
Cloud/paying users get a curated, whitelist-based view: friendly progress
lines only — no file paths, model names, encoder details or pipeline
internals. Anything not matched by a rule is hidden.

Progress (download %, transcription %) updates ITS OWN line instead of
adding one per step, so the view reads as a list of stages. Each line keeps
the time it first appeared.
"""
import re

# Strips any slashy path token (output/…/clip.mp4, /app/uploads/x.mp4, …).
_PATH_RE = re.compile(r'(?:/?[\w.\-]+/)+[\w.\-]+')


def _strip_paths(line):
    return _PATH_RE.sub('', line).rstrip(' :')


# Ordered rules; first match wins: (pattern, template, progress kind).
# The template uses the match's groups, or None keeps the (path-stripped)
# line verbatim. A progress kind makes the line update in place.
_RULES = [
    # Worker/job lifecycle + errors: keep, minus any paths.
    (re.compile(r'^(Job started|Process finished|Process failed|'
                r'Execution error|No metadata|❌)'), None, None),
    # The free queue's brake capped this video's clips (app._apply_brake_clips).
    (re.compile(r'^⏳ The free queue is very busy'), None, None),
    # The download waits for the YouTube account (app._schedule_auto_retry).
    (re.compile(r'^⏳ YouTube is limiting downloads'), None, None),
    (re.compile(r'Download attempt'), '📥 Downloading video…', 'download'),
    (re.compile(r'\[download\]\s+(\d+)(?:\.\d+)?% of'), '📥 Downloading video… {0}%', 'download'),
    (re.compile(r'downloading only the first (\d+) min'),
     '✂️ Long video: downloading the first {0} min', None),
    (re.compile(r'Video downloaded in'), '✅ Video downloaded', None),
    (re.compile(r'Audio ready ahead of the video'),
     '🎧 Audio ready — transcribing while the video downloads', None),
    # Live transcription progress emitted by transcribe_backends.
    (re.compile(r'Transcribing… (\d+)%'), '🎙️ Transcribing audio… {0}%', 'asr'),
    (re.compile(r'Transcribing (video|audio)'), '🎙️ Transcribing audio…', 'asr'),
    (re.compile(r'Analyzing with (?:Gemini|local LLM)'), '🤖 Finding the best moments…', None),
    (re.compile(r'Silent video — analyzing'), '🎥 No speech — finding moments in the picture…', None),
    (re.compile(r'Choosing a layout'), '🎛️ Choosing the layout…', None),
    (re.compile(r'Found (\d+) (?:viral )?clips'), '🔥 Found {0} clips!', None),
    (re.compile(r'Processing Clip (\d+)'), '🎬 Creating clip {0}…', None),
    (re.compile(r'Clip (\d+) framed'), '🎞️ Clip {0} framed', None),
    (re.compile(r'Clip (\d+) ready'), '✅ Clip {0} ready', None),
]


def _match(segment):
    """(text, kind) for one raw segment, or None to hide it."""
    stripped = segment.strip()
    if not stripped:
        return None
    for pattern, template, kind in _RULES:
        match = pattern.search(stripped)
        if match:
            if template is None:
                return _strip_paths(stripped), kind
            return template.format(*match.groups()), kind
    return None


def _segments(line):
    """A raw line can hold several outputs: yt-dlp redraws its progress bar
    with \\r and no newline, so a parallel print lands on the same line."""
    parts = []
    for chunk in str(line).split('\r'):
        # A print glued to the end of a progress bar: split before each emoji
        # marker the pipeline starts its own lines with.
        parts.extend(p for p in re.split(r'(?=🎙️|🎧|✅ Video downloaded)', chunk) if p)
    return parts


def friendly_log_line(line):
    """Map one raw log line to its cloud-visible form, or None to hide it."""
    for seg in _segments(line):
        hit = _match(seg)
        if hit:
            return hit[0]
    return None


def friendly_logs_timed(logs, times=None):
    """Curated (text, time) list for cloud users.

    Consecutive duplicates collapse, and a progress line (download %,
    transcription %) is updated in place instead of repeated; a line keeps the
    time it first appeared. ``times`` is a list parallel to ``logs`` (epoch
    seconds) or None.
    """
    out = []           # [text, time]
    live = {}          # progress kind -> index in out still being updated
    for i, line in enumerate(logs):
        t = times[i] if times is not None and i < len(times) else None
        for seg in _segments(line):
            hit = _match(seg)
            if not hit:
                continue
            text, kind = hit
            if kind and kind in live:
                out[live[kind]][0] = text
                continue
            if out and out[-1][0] == text:
                continue
            out.append([text, t])
            # Any other line closes the running progress of other stages only
            # when it is a new stage of the same kind; different kinds (the
            # download and the early transcription) run side by side.
            if kind:
                live[kind] = len(out) - 1
            elif text.startswith(('✅ Video downloaded', '✂️')):
                live.pop('download', None)
            elif text.startswith(('🔥', '🤖', '🎥')):
                live.pop('asr', None)
    return [(text, t) for text, t in out]


def friendly_logs(logs):
    """Curated log list for cloud users (see friendly_logs_timed)."""
    return [text for text, _t in friendly_logs_timed(logs)]
