"""Peak detection: chat velocity, hype words and loudness against the stream's own baseline.

Every stream has its own normal: 2 messages a second is a riot for a small
streamer and a lull for a big one. So each signal is a robust z-score
(median / MAD) of the last few seconds against the previous five minutes of
the same stream, never an absolute number.

- **chat**: messages per second over an 8 s window. Discounted when a few
  users spam (unique-chatter ratio) and boosted when the window is full of
  hype tokens (KEKW, OMEGALUL, ???, 💀, "clip it"...) above the usual share.
- **audio**: loudest 3 s RMS of the window against the per-second baseline.
  Kick streams, and quiet chats, rely on this alone.
- Chat reacts a few seconds after the moment it reacts to (``reaction_lag``),
  so the moment is placed that far before the chat's crest.

A crossing of ``threshold`` arms the detector; it fires once the burst stops
growing, so the clip is centred on the crest and not the first spike.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, Dict, List, Optional, Tuple

from .chat import ChatMessage

HYPE_TOKENS = {
    "kekw", "lul", "lulw", "omegalul", "icant", "kekl", "lmao", "lmfao", "lol", "lool",
    "w", "ww", "www", "pog", "poggers", "pogchamp", "pogu", "holy", "wtf", "clip",
    "clipped", "clipit", "insane", "omg", "bro", "bruh", "gg", "monkas", "monkaw",
    "aware", "pausechamp", "nah", "noway", "deadass", "jebaited", "ez", "sheesh",
    "hahaha", "hahahaha", "xd", "xdd", "sadge", "pepelaugh", "kappa", "crazy",
}
HYPE_PHRASES = ("no way", "clip it", "clip that", "oh my god", "what the", "holy shit",
                "he did it", "lets go", "let's go", "nah bro")
HYPE_EMOJI = set("💀😭😂🤣🔥😳😱🤯‼❗")
_WORD = re.compile(r"[a-z']+")
_QSPAM = re.compile(r"\?{3,}|!{3,}")


def is_hype(text: str) -> bool:
    t = text.lower()
    if _QSPAM.search(t) or any(ch in HYPE_EMOJI for ch in text):
        return True
    if any(p in t for p in HYPE_PHRASES):
        return True
    words = _WORD.findall(t)
    if any(w in HYPE_TOKENS for w in words):
        return True
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def normalise_message(text: str) -> str:
    """For counting repeats: lower case, collapsed spaces, long runs cut to 3."""
    t = " ".join(text.lower().split())
    return re.sub(r"(.)\1{3,}", r"\1\1\1", t)[:80]


def robust_z(value: float, samples: List[float], floor: float) -> float:
    if not samples:
        return 0.0
    m = median(samples)
    mad = median([abs(s - m) for s in samples])
    sigma = max(1.4826 * mad, floor)
    return (value - m) / sigma


@dataclass
class Peak:
    ts: float                  # wall-clock time of the moment itself
    score: float
    chat_z: float
    audio_z: float
    hype_share: float
    messages: List[str] = field(default_factory=list)  # most repeated chat lines
    reason: str = ""


class PeakDetector:
    WINDOW = 8          # seconds scored at a time
    BASELINE = 300      # seconds of history the window is compared with
    GAP = 20            # recent seconds left out of the baseline
    SETTLE = 2          # seconds of lateness allowed for chat to arrive
    MIN_BLOCKS = 12     # baseline windows needed (about a minute) before chat counts
    MIN_RATIO = 2.0     # chat must reach this multiple of its usual speed
    MIN_MESSAGES = 10   # in the window, for chat to count at all

    def __init__(self, threshold: float = 3.5, warmup: float = 120.0,
                 cooldown: float = 150.0, min_chat_rate: float = 0.3,
                 audio_weight: float = 1.0, reaction_lag: float = 4.0,
                 max_per_hour: int = 6, arm_timeout: float = 14.0):
        self.threshold = threshold
        self.warmup = warmup
        self.cooldown = cooldown
        self.min_chat_rate = min_chat_rate
        self.audio_weight = audio_weight
        self.reaction_lag = reaction_lag
        self.max_per_hour = max_per_hour
        self.arm_timeout = arm_timeout
        self._lock = threading.Lock()
        self._count: Dict[int, int] = defaultdict(int)
        self._hype: Dict[int, int] = defaultdict(int)
        self._users: Dict[int, set] = defaultdict(set)
        self._loud: Dict[int, float] = {}
        self._msgs: Deque[ChatMessage] = deque(maxlen=5000)
        self._first: Optional[float] = None
        self._last_fire = -1e18
        self._fired: Deque[float] = deque()
        self._armed: Optional[dict] = None
        self.last_score: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ input
    def add_message(self, msg: ChatMessage) -> None:
        s = int(msg.ts)
        with self._lock:
            self._count[s] += 1
            self._users[s].add(msg.user)
            if is_hype(msg.text):
                self._hype[s] += 1
            self._msgs.append(msg)
            if self._first is None or msg.ts < self._first:
                self._first = msg.ts

    def add_loudness(self, ts: float, rms_db: float) -> None:
        if rms_db is None or not math.isfinite(rms_db):
            return
        with self._lock:
            self._loud[int(ts)] = max(-90.0, float(rms_db))
            if self._first is None or ts < self._first:
                self._first = ts

    # ---------------------------------------------------------------- scoring
    def _rate_blocks(self, lo: int, hi: int) -> List[float]:
        w = self.WINDOW
        return [sum(self._count.get(s, 0) for s in range(t, t + w)) / w
                for t in range(lo, hi - w + 1, w // 2)]

    def score_at(self, end: int) -> Tuple[float, float, float, float]:
        """(score, chat_z, audio_z, hype_share) of the window ending at ``end``."""
        w = self.WINDOW
        start = end - w
        n = sum(self._count.get(s, 0) for s in range(start, end))
        rate = n / w
        # seconds before the first data are not "zero chat": they are unknown
        first = int(self._first) + 1 if self._first is not None else end
        base_lo, base_hi = max(end - self.BASELINE, first), end - self.GAP
        blocks = self._rate_blocks(base_lo, base_hi)
        if len(blocks) < self.MIN_BLOCKS:
            blocks = []
        base_rate = median(blocks) if blocks else 0.0
        chat_z = 0.0
        hype_share = 0.0
        if n >= self.MIN_MESSAGES and blocks and max(base_rate, rate / 3) >= self.min_chat_rate:
            # Two views of "unusual", the stricter wins: the stream's own spread
            # (real chat is burstier than random) and a Poisson test (on slow
            # chats a handful of messages is not a burst). Anscombe's transform
            # makes Poisson counts roughly standard-normal at any rate.
            lam = max(base_rate, 0.05) * w
            z_pois = 2.0 * (math.sqrt(n + 0.375) - math.sqrt(lam + 0.375))
            z_mad = robust_z(rate, blocks, floor=0.05)
            chat_z = min(z_pois, z_mad)
            if rate < self.MIN_RATIO * max(base_rate, self.min_chat_rate):
                chat_z = min(chat_z, 0.0)  # a real moment at least doubles chat
            users = set().union(*(self._users.get(s, set()) for s in range(start, end)))
            uniq = len(users) / n
            chat_z *= min(1.0, uniq / 0.5)
            hype_n = sum(self._hype.get(s, 0) for s in range(start, end))
            hype_share = hype_n / n
            base_msgs = sum(self._count.get(s, 0) for s in range(base_lo, base_hi))
            base_hype = (sum(self._hype.get(s, 0) for s in range(base_lo, base_hi)) / base_msgs
                         if base_msgs else 0.0)
            # hype words make a real burst count more; they never make one out of noise
            # (a share over a dozen messages is mostly luck, so it is scaled by volume)
            if chat_z >= 2.0 and hype_n >= 8:
                bonus = max(0.0, min(2.0, (hype_share - base_hype) * 5))
                chat_z += bonus * min(1.0, n / 40)
        audio_z = 0.0
        base_loud = [v for s, v in self._loud.items() if base_lo <= s < base_hi]
        if len(base_loud) >= 30:
            vals = [self._loud[s] for s in range(start, end) if s in self._loud]
            if len(vals) >= 3:
                peak3 = max(sum(vals[i:i + 3]) / 3 for i in range(len(vals) - 2))
                audio_z = max(0.0, robust_z(peak3, base_loud, floor=1.5))
                audio_z = min(audio_z, 8.0)
        a = self.audio_weight * audio_z
        c = max(0.0, chat_z)
        score = max(c, a) + 0.5 * min(c, a)
        return score, chat_z, audio_z, hype_share

    def _crest(self, lo: int, hi: int) -> float:
        """Wall time of the moment behind the burst in [lo, hi)."""
        counts = [(self._count.get(s, 0), s) for s in range(lo, hi)]
        best_n, best_s = max(counts) if counts else (0, hi)
        if best_n > 0:
            return best_s - self.reaction_lag
        louds = [(self._loud.get(s, -999), s) for s in range(lo, hi)]
        return float(max(louds)[1]) if louds else float(hi)

    def top_messages(self, lo: float, hi: float, n: int = 8) -> List[str]:
        with self._lock:
            msgs = [m.text for m in self._msgs if lo <= m.ts < hi]
        counts = Counter(normalise_message(t) for t in msgs if t)
        firsts = {}
        for t in msgs:
            firsts.setdefault(normalise_message(t), t)
        return [firsts[k] for k, _ in counts.most_common(n)]

    def evaluate(self, now: Optional[float] = None) -> Optional[Peak]:
        now = time.time() if now is None else now
        end = int(now) - self.SETTLE
        with self._lock:
            self._prune(end)
            if self._first is None or now - self._first < self.warmup:
                return None
            score, chat_z, audio_z, hype = self.score_at(end)
            self.last_score = (score, chat_z, audio_z)
            armed = self._armed
            if armed is None:
                if score < self.threshold or now - self._last_fire < self.cooldown:
                    return None
                while self._fired and now - self._fired[0] > 3600:
                    self._fired.popleft()
                if len(self._fired) >= self.max_per_hour:
                    return None
                self._armed = dict(start=end - self.WINDOW, since=now, best=score,
                                   best_end=end, chat_z=chat_z, audio_z=audio_z, hype=hype)
                return None
            if score > armed["best"]:
                armed.update(best=score, best_end=end, chat_z=chat_z, audio_z=audio_z, hype=hype)
            if score > 0.7 * armed["best"] and now - armed["since"] < self.arm_timeout:
                return None
            self._armed = None
            self._last_fire = now
            self._fired.append(now)
            moment = self._crest(armed["start"], armed["best_end"])
        reason = []
        if armed["chat_z"] > 0:
            reason.append(f"chat x{armed['chat_z']:.1f}")
        if armed["audio_z"] > 0:
            reason.append(f"audio x{armed['audio_z']:.1f}")
        if armed["hype"] > 0:
            reason.append(f"hype {armed['hype']:.0%}")
        return Peak(ts=moment, score=round(armed["best"], 2), chat_z=round(armed["chat_z"], 2),
                    audio_z=round(armed["audio_z"], 2), hype_share=round(armed["hype"], 2),
                    messages=self.top_messages(moment - 5, moment + 20),
                    reason=", ".join(reason))

    def _prune(self, end: int) -> None:
        cutoff = end - self.BASELINE - 60
        for d in (self._count, self._hype, self._users, self._loud):
            for s in [s for s in d if s < cutoff]:
                del d[s]
