"""Optional background-music / sound-effects layer for finished shorts.

This module adds an OPT-IN step that runs after a clip is assembled: it obtains
an audio bed (music and/or sound effects) from a pluggable provider and mixes it
UNDER the clip's existing narration, with the music ducked in the spoken parts.

Design (per the maintainer's conditions on issue #51):

* Provider-agnostic. Everything goes through the small ``MusicProvider``
  interface below. ``LocalAudioProvider`` needs no paid service — the user
  points at an audio file they already have — which keeps the free, no-lock-in
  path first-class. ``SoniloProvider`` is one API implementation among what can
  be many; adding another provider means adding one class here, nothing else.
* Off by default. Nothing in this module runs unless the caller passes a
  provider explicitly (the API endpoint is only reached when the user opts in
  per job from the UI).
* User-supplied API key. ``SoniloProvider`` takes the key as a constructor
  argument; the key is read from the ``X-Sonilo-Key`` request header, which the
  browser fills from its own encrypted local storage. The key is never persisted
  server-side (same pattern as the Gemini / ElevenLabs / fal.ai keys).
* Ducking with a volume control. ``duck_and_mix`` lowers the bed while the
  narration is speaking and lifts it back in the gaps, and exposes a 0..1
  ``music_volume`` knob.

Wording note: Sonilo music is licensed and safe for commercial use (terms
apply); Sonilo sound effects are royalty-free. This module makes no ownership
claim over generated audio.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Optional

import httpx

# Duration caps mirror the Sonilo backend so we fail fast locally instead of
# uploading media the API will reject. Keep in sync with the API.
MUSIC_MAX_DURATION_SECONDS = 360  # 6 minutes — /v1/video-to-music
SFX_MAX_DURATION_SECONDS = 180    # 3 minutes — /v1/video-to-sfx

SONILO_API_BASE = os.getenv("SONILO_API_URL", "https://api.sonilo.com").rstrip("/")

# Audio extensions the local-file provider accepts.
_AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"})


class MusicProviderError(Exception):
    """Raised for any provider-level failure (bad input, API error, timeout)."""


def probe_duration(source: str) -> Optional[float]:
    """Best-effort local duration read via ffprobe.

    Returns the duration in seconds, or ``None`` when ffprobe is missing or the
    source can't be parsed — callers treat ``None`` as "unknown, don't block".
    ``source`` may be a local path or a URL (ffprobe handles both).
    """
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", source,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    if out.returncode != 0:
        return None
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _assert_within_cap(video_path: str, max_seconds: int, label: str) -> None:
    """Fail fast if the clip is known to exceed the provider's duration cap.

    Best-effort: an unknown duration (ffprobe missing/unparseable) is allowed
    through so the backend makes the final call, matching sonilo-mcp's pre-check.
    """
    duration = probe_duration(video_path)
    if duration is not None and duration > max_seconds:
        raise MusicProviderError(
            f"Clip duration {duration:.1f}s exceeds the {label} maximum of "
            f"{max_seconds}s. Trim the clip or pick a shorter range."
        )


class MusicProvider:
    """Interface every music/SFX provider implements.

    A provider returns a path to an audio file on local disk. It does NOT mix
    that audio into the video — mixing/ducking is the app's job (see
    ``duck_and_mix``), so ducking behaviour is identical no matter which
    provider produced the bed.
    """

    #: Human-readable id used in logs and API payloads.
    name = "base"

    #: Whether this provider can produce music beds.
    supports_music = False

    #: Whether this provider can produce sound-effects beds.
    supports_sfx = False

    def generate_music(self, video_path: str, output_path: str,
                       prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate_sfx(self, video_path: str, output_path: str,
                    prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class LocalAudioProvider(MusicProvider):
    """Free, no-lock-in provider: use an audio file the user already has.

    This is the reference implementation that proves the interface is genuinely
    provider-agnostic — it needs no API key and no network. It validates the
    supplied file and copies it into the job directory so the mix step has a
    stable path to read.

    It advertises support for both "music" and "sfx" because either capability,
    from the app's point of view, is just "give me an audio bed to duck under
    the narration"; the distinction only matters for API providers that generate
    the two differently.
    """

    name = "local"
    supports_music = True
    supports_sfx = True

    def __init__(self, audio_path: str):
        self.audio_path = audio_path

    def _resolve(self, output_path: str) -> str:
        path = os.path.expanduser(self.audio_path)
        if not os.path.isabs(path):
            raise MusicProviderError("Local audio path must be absolute")
        if not os.path.exists(path):
            raise MusicProviderError(f"Audio file not found: {path}")
        if not os.path.isfile(path):
            raise MusicProviderError(f"Not a file: {path}")
        if os.path.splitext(path)[1].lower() not in _AUDIO_EXTS:
            raise MusicProviderError(
                "Unsupported audio format. Use one of: "
                + ", ".join(sorted(_AUDIO_EXTS))
            )
        shutil.copyfile(path, output_path)
        return output_path

    def generate_music(self, video_path: str, output_path: str,
                       prompt: Optional[str] = None) -> str:
        # The local file IS the bed; the video/prompt are ignored on purpose.
        return self._resolve(output_path)

    def generate_sfx(self, video_path: str, output_path: str,
                    prompt: Optional[str] = None) -> str:
        return self._resolve(output_path)


class SoniloProvider(MusicProvider):
    """API provider backed by Sonilo's /v1 endpoints.

    * Music: ``POST /v1/video-to-music`` streams NDJSON audio chunks that are
      reassembled to an .m4a bed. Tracks are licensed and safe for commercial
      use (terms apply).
    * SFX: ``POST /v1/video-to-sfx`` returns a task id; we poll
      ``GET /v1/tasks/{id}`` until it finishes, then download the result. SFX
      are royalty-free.

    The key is passed in by the caller (from the request header) and is never
    written to disk here.
    """

    name = "sonilo"
    supports_music = True
    supports_sfx = True

    _POLL_INTERVAL_SECONDS = 5.0

    def __init__(self, api_key: str, timeout: float = 600.0):
        if not api_key:
            raise MusicProviderError("Missing Sonilo API key")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Sonilo-Client": "openshorts",
        }

    @staticmethod
    def _extract_detail(body: str) -> str:
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return body
        if isinstance(parsed, dict):
            for key in ("message", "detail"):
                if key in parsed:
                    return str(parsed[key])
        return body

    def generate_music(self, video_path: str, output_path: str,
                       prompt: Optional[str] = None) -> str:
        _assert_within_cap(video_path, MUSIC_MAX_DURATION_SECONDS, "music")
        url = f"{SONILO_API_BASE}/v1/video-to-music"
        data = {"prompt": prompt} if prompt else None
        chunks = bytearray()
        completed = False
        try:
            with open(video_path, "rb") as fh:
                files = {"video": (os.path.basename(video_path), fh, "video/mp4")}
                with httpx.Client(timeout=self.timeout) as client:
                    with client.stream("POST", url, headers=self._headers(),
                                       data=data, files=files) as resp:
                        if resp.status_code >= 400:
                            body = resp.read().decode("utf-8", "replace")
                            raise MusicProviderError(
                                f"Sonilo music error ({resp.status_code}): "
                                f"{self._extract_detail(body)}"
                            )
                        for line in resp.iter_lines():
                            if not line.strip():
                                continue
                            try:
                                evt = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            etype = evt.get("type")
                            if etype == "audio_chunk":
                                import base64
                                payload = evt.get("data")
                                if isinstance(payload, str):
                                    try:
                                        chunks.extend(base64.b64decode(payload))
                                    except Exception:
                                        continue
                            elif etype == "complete":
                                completed = True
                            elif etype == "error":
                                raise MusicProviderError(
                                    "Sonilo music error: "
                                    + str(evt.get("message") or evt.get("code")
                                          or "stream error")
                                )
        except httpx.HTTPError as e:
            raise MusicProviderError(f"Sonilo request failed: {e}") from e
        if not completed or not chunks:
            raise MusicProviderError(
                "Sonilo music stream ended without a complete track"
            )
        with open(output_path, "wb") as fh:
            fh.write(bytes(chunks))
        return output_path

    def generate_sfx(self, video_path: str, output_path: str,
                    prompt: Optional[str] = None) -> str:
        _assert_within_cap(video_path, SFX_MAX_DURATION_SECONDS, "sfx")
        submit_url = f"{SONILO_API_BASE}/v1/video-to-sfx"
        data = {"prompt": prompt} if prompt else None
        try:
            with open(video_path, "rb") as fh:
                files = {"video": (os.path.basename(video_path), fh, "video/mp4")}
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(submit_url, headers=self._headers(),
                                       data=data, files=files)
            if resp.status_code >= 400:
                raise MusicProviderError(
                    f"Sonilo SFX error ({resp.status_code}): "
                    f"{self._extract_detail(resp.text)}"
                )
            task_id = (resp.json() or {}).get("task_id")
            if not task_id:
                raise MusicProviderError(
                    "Sonilo accepted the request but returned no task_id"
                )
            body = self._poll_task(str(task_id))
            audio = body.get("audio")
            if not (isinstance(audio, dict) and audio.get("url")):
                raise MusicProviderError(
                    "Sonilo SFX task finished without an audio result"
                )
            self._download(audio["url"], output_path)
            return output_path
        except httpx.HTTPError as e:
            raise MusicProviderError(f"Sonilo request failed: {e}") from e

    def _poll_task(self, task_id: str) -> dict:
        url = f"{SONILO_API_BASE}/v1/tasks/{task_id}"
        deadline = time.monotonic() + self.timeout
        while True:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=self._headers())
            if resp.status_code >= 400:
                raise MusicProviderError(
                    f"Sonilo task error ({resp.status_code}): "
                    f"{self._extract_detail(resp.text)}"
                )
            body = resp.json()
            status = body.get("status") if isinstance(body, dict) else None
            if status == "succeeded":
                return body
            if status == "failed":
                err = body.get("error") if isinstance(body, dict) else None
                msg = (err.get("message") if isinstance(err, dict) else err) \
                    or "generation failed"
                raise MusicProviderError(f"Sonilo SFX task failed: {msg}")
            if time.monotonic() >= deadline:
                raise MusicProviderError(
                    f"Sonilo SFX task timed out after {self.timeout:.0f}s "
                    f"(task {task_id})"
                )
            time.sleep(self._POLL_INTERVAL_SECONDS)

    def _download(self, url: str, dest: str) -> None:
        # Presigned result URL — send no auth headers to the storage domain.
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise MusicProviderError(
                        f"Sonilo download failed (status {resp.status_code})"
                    )
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        fh.write(chunk)


# ---------- Provider factory ----------

def build_provider(provider: str, *, api_key: Optional[str] = None,
                  local_audio_path: Optional[str] = None) -> MusicProvider:
    """Return a provider instance for ``provider`` ("local" or "sonilo").

    Adding a provider means adding one branch here and one class above — no
    changes to the endpoint or the mix step. That is the whole point of the
    interface.
    """
    provider = (provider or "").strip().lower()
    if provider == "local":
        if not local_audio_path:
            raise MusicProviderError(
                "The local provider needs a local_audio_path"
            )
        return LocalAudioProvider(local_audio_path)
    if provider == "sonilo":
        return SoniloProvider(api_key or "")
    raise MusicProviderError(f"Unknown music provider: {provider!r}")


# ---------- Ducking / mixing ----------

def duck_and_mix(video_path: str, bed_path: str, output_path: str,
                music_volume: float = 0.35, duck: bool = True,
                duck_amount: float = 0.6) -> str:
    """Mix ``bed_path`` UNDER the narration in ``video_path`` and write a new
    video to ``output_path``.

    * ``music_volume`` (0..1): baseline loudness of the bed relative to source.
    * ``duck``: when True, the bed is sidechain-compressed by the narration so
      it drops while someone is speaking and lifts back in the gaps. When False,
      the bed plays at a flat ``music_volume`` for the whole clip.
    * ``duck_amount`` (0..1): how hard the bed is pushed down while the
      narration is active. Higher = more aggressive ducking.

    Ffmpeg idiom matches the rest of the repo: a single ``subprocess.run`` with
    ``-c:v libx264 -preset medium -crf 18``, re-encoding video losslessly-ish
    and remixing audio. The clip's own audio is preserved as the front voice; if
    the clip has no audio track, the bed is added at ``music_volume`` with no
    ducking (nothing to duck under).
    """
    music_volume = max(0.0, min(1.0, float(music_volume)))
    duck_amount = max(0.0, min(1.0, float(duck_amount)))

    has_voice = _has_audio_stream(video_path)

    if not has_voice:
        # No narration to duck under: lay the bed over the silent clip.
        filter_complex = (
            f"[1:a]volume={music_volume:.3f}[bed]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bed_path,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[bed]",
            "-shortest",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    elif duck:
        # Sidechain the bed by the voice so it ducks under speech. The voice is
        # duplicated: one copy drives the compressor's sidechain, the other is
        # mixed back in at full level as the front layer.
        # ratio/threshold/release tuned so the bed dips clearly under speech and
        # recovers in the gaps; the bed's resting level is `music_volume`.
        threshold = max(0.01, 0.05 + (1.0 - duck_amount) * 0.2)
        ratio = 2 + duck_amount * 18  # 2..20
        filter_complex = (
            f"[0:a]asplit=2[voice_main][voice_sc];"
            f"[1:a]volume={music_volume:.3f}[bed];"
            f"[bed][voice_sc]sidechaincompress="
            f"threshold={threshold:.3f}:ratio={ratio:.1f}:attack=20:release=400"
            f"[bed_ducked];"
            f"[voice_main][bed_ducked]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bed_path,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
    else:
        # No ducking: flat bed under the voice.
        filter_complex = (
            f"[1:a]volume={music_volume:.3f}[bed];"
            f"[0:a][bed]amix=inputs=2:duration=first:"
            f"dropout_transition=0:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", bed_path,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

    print(f"🎵 Mixing audio bed (volume={music_volume}, duck={duck}): "
          f"{' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace")
        print(f"❌ FFmpeg mix error: {err}")
        raise MusicProviderError(f"FFmpeg mix failed: {err}")
    return output_path


def _has_audio_stream(video_path: str) -> bool:
    """Whether ``video_path`` has at least one audio stream.

    Best-effort: on any ffprobe problem we assume there IS audio, so a clip with
    narration is never silently treated as silent (the worse failure).
    """
    if shutil.which("ffprobe") is None:
        return True
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "json", video_path,
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return True
    if out.returncode != 0:
        return True
    try:
        return bool(json.loads(out.stdout).get("streams"))
    except (json.JSONDecodeError, TypeError):
        return True
