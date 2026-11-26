"""Shared yt-dlp runtime options and safe download diagnostics."""
import re
import shutil
import subprocess

_URL = re.compile(r"https?://[^\s\]'\"<>]+", re.I)
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


def js_runtime():
    node = shutil.which("node")
    if node:
        try:
            version = subprocess.check_output([node, "--version"], text=True, timeout=3)
            if int(version.lstrip("v").split(".", 1)[0]) >= 22:
                return {"node": {"path": node}}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"deno": {}}


class DownloadLogger:
    def debug(self, message):
        if not message.startswith("[debug]"):
            self.info(message)

    def info(self, message):
        print(_URL.sub("[media URL]", str(message)))

    def warning(self, message):
        if "GVS PO Token" in message:
            self.info("[youtube] Some formats require a platform token and were omitted.")
        else:
            self.info(message)

    def error(self, message):
        self.info(message)


def runtime_options():
    return {"js_runtimes": js_runtime(),
            "nocheckcertificate": False, "verbose": False, "logger": DownloadLogger()}
