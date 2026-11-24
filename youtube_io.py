"""Audio-language selection for YouTube extractor results."""

import re

_AUDIO_LANGUAGE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8}){0,4}")


class SourceAudioUnavailable(ValueError):
    def __init__(self, language):
        super().__init__(f"YouTube did not offer the {language} audio track for this video. "
                         "Choose another audio track or use the source default.")


def normalize_audio_language(value):
    """Normalize an audio language reported by yt-dlp, including regional tags."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value.lower() if _AUDIO_LANGUAGE.fullmatch(value) else None


def track_language(media_format):
    if media_format.get("acodec") in (None, "none"):
        return None
    return normalize_audio_language(media_format.get("language"))


def audio_languages(info):
    """Language codes only. Extraction URLs and account data stay server-side."""
    return sorted({code for fmt in info.get("formats") or [] if (code := track_language(fmt))})


def select_audio_language(info, language):
    """Keep the requested audio and silent video formats for yt-dlp's sorter."""
    selected = [fmt for fmt in info.get("formats") or []
                if fmt.get("acodec") == "none" or track_language(fmt) == language]
    if all(fmt.get("acodec") == "none" for fmt in selected):
        raise SourceAudioUnavailable(language)
    return {**info, "formats": selected}
