"""Background music library: royalty-free tracks mixed quietly under a Short.

Drop audio files in the music folder (``<data_dir>/music`` by default) or
upload them from the web UI. Each Short that gets music:

- takes the track used least recently (ties broken by name), so a run of
  clips cycles through the library instead of repeating one song;
- starts somewhere inside the track (not always the intro), fades in over
  1.5 s and out over the last 2 s;
- sits at ``music_volume_db`` and dips further whenever the clip's own audio
  is loud (sidechain compression), so speech stays clear.

Modes (``music``): ``off``; ``always``; ``quiet`` adds music only to clips
with little speech (a chase, a crash, gameplay with nobody talking), where
silence or game noise alone loses viewers.

Only use music you have the rights to (YouTube Audio Library, Pixabay Music,
your own): copyrighted songs get Shorts claimed and Reels muted.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import procs

EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac")
MAX_BYTES = 50 * 1024 * 1024
MODES = ("off", "always", "quiet")
QUIET_WORDS_PER_SECOND = 1.0   # below this the clip counts as "little speech"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9 ._()\-]")


@dataclass(frozen=True)
class Mix:
    path: Path
    offset: float        # seconds into the track where the Short's music starts
    volume_db: float


def safe_name(name: str) -> str:
    """A plain file name for an uploaded track ("" when unusable)."""
    base = Path(str(name or "")).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = _SAFE_NAME.sub("", stem).strip(" .")[:80]
    if not stem or ext not in EXTENSIONS:
        return ""
    return stem + ext


class Library:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self._lock = threading.Lock()
        self._state_path = self.folder / ".library.json"

    # ------------------------------------------------------------ listing
    def tracks(self) -> List[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(p for p in self.folder.iterdir()
                      if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith("."))

    def _state(self) -> dict:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, state: dict):
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self._state_path)

    def describe(self) -> List[dict]:
        st = self._state()
        disabled = set(st.get("disabled", []))
        used = st.get("last_used", {})
        out = []
        for p in self.tracks():
            out.append({"name": p.name, "size": p.stat().st_size,
                        "duration": duration_of(p), "enabled": p.name not in disabled,
                        "last_used": used.get(p.name)})
        return out

    # ---------------------------------------------------------- editing
    def set_enabled(self, name: str, enabled: bool) -> None:
        with self._lock:
            if not (self.folder / name).is_file():
                raise FileNotFoundError(name)
            st = self._state()
            dis = set(st.get("disabled", []))
            (dis.discard if enabled else dis.add)(name)
            st["disabled"] = sorted(dis)
            self._save(st)

    def add(self, name: str, data: bytes) -> str:
        clean = safe_name(name)
        if not clean:
            raise ValueError(f"use an audio file ({', '.join(EXTENSIONS)})")
        if not data or len(data) > MAX_BYTES:
            raise ValueError("the file is empty or larger than 50 MB")
        self.folder.mkdir(parents=True, exist_ok=True)
        dest = self.folder / clean
        n = 2
        while dest.exists():
            dest = self.folder / f"{Path(clean).stem} ({n}){Path(clean).suffix}"
            n += 1
        tmp = dest.with_name("." + dest.name + ".part")
        tmp.write_bytes(data)
        if duration_of(tmp, cache=False) <= 0:
            tmp.unlink(missing_ok=True)
            raise ValueError("ffmpeg cannot read that file as audio")
        tmp.replace(dest)
        return dest.name

    def remove(self, name: str) -> None:
        p = self.folder / name
        if p.parent != self.folder or not p.is_file():
            raise FileNotFoundError(name)
        p.unlink()

    # ------------------------------------------------------------ picking
    def pick(self, clip_seconds: float, volume_db: float, seed: str = "") -> Optional[Mix]:
        """The next track for a clip, or None when the library is empty."""
        with self._lock:
            st = self._state()
            disabled = set(st.get("disabled", []))
            used: Dict[str, float] = st.get("last_used", {})
            cands = [p for p in self.tracks() if p.name not in disabled and duration_of(p) > 0]
            if not cands:
                return None
            track = min(cands, key=lambda p: (used.get(p.name, 0.0), p.name))
            used[track.name] = time.time()
            st["last_used"] = used
            self._save(st)
        dur = duration_of(track)
        room = dur - clip_seconds - 3
        h = int(hashlib.sha1((seed or track.name).encode()).hexdigest(), 16)
        offset = (h % 1000) / 1000 * room if room > 5 else 0.0
        return Mix(track, round(offset, 2), volume_db)


_durations: Dict[str, float] = {}


def duration_of(path: Path, cache: bool = True) -> float:
    key = f"{path}:{path.stat().st_size if path.exists() else 0}"
    if cache and key in _durations:
        return _durations[key]
    try:
        r = procs.run([procs.ffprobe(), "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(path)], timeout=30, text=True)
        d = float((r.stdout or "0").strip() or 0)
    except (ValueError, OSError):
        d = 0.0
    if cache:
        _durations[key] = d
    return d


def wants_music(mode: str, words: int, seconds: float) -> bool:
    if mode == "always":
        return True
    if mode == "quiet":
        return seconds > 0 and words / seconds < QUIET_WORDS_PER_SECOND
    return False


def audio_graph(has_voice: bool, duration: float, mix: Optional[Mix]) -> str:
    """The audio part of the render filtergraph, ending in [aout]."""
    norm = "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
    if not mix:
        return f"[0:a]{norm}[aout]" if has_voice else ""
    fade_out = max(0.0, duration - 2.0)
    music = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={mix.volume_db:.1f}dB,"
             f"afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out:.2f}:d=2")
    if not has_voice:
        return f"{music},{norm}[aout]"
    return (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit=2[voice][sc];"
            f"{music}[mus];"
            f"[mus][sc]sidechaincompress=threshold=0.04:ratio=8:attack=20:release=400[duck];"
            f"[voice][duck]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,{norm}[aout]")
