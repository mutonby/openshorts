"""Timestamped replay input and baseline-relative, privacy-preserving chat evidence."""
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import itertools
import json
import math
import re
from statistics import median

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_SPIKES = 60
REACTIONS = {
    "laughter": re.compile(r"\b(?:lo+l|lma+o|haha(?:ha)*|rofl)\b|[😂🤣💀]", re.I),
    "surprise": re.compile(r"\b(?:omg|wtf|no way|what just happened)\b|[😱🤯]", re.I),
    "cheering": re.compile(r"^\s*[wW]+\s*$|\b(?:gg|let.s go)\b|[🔥🙌]", re.I),
}
NON_REACTION = re.compile(
    r"^\s*(?:[!/]|(?:hi|hey|hello|good morning)[!. ]*$)|"
    r"\b(?:giveaway|type .* to enter|enter to win|gifted \d+ (?:memberships|subs)|"
    r"follow.*for a chance|subscribers only)\b", re.I)


@dataclass(frozen=True)
class Message:
    seconds: float
    author: str
    text: str
    viewers: int | None = None


def _seconds(value):
    if type(value) is bool:
        return None
    try:
        if isinstance(value, str) and ":" in value:
            parts = [float(part) for part in value.split(":")]
            if len(parts) not in (2, 3) or any(part < 0 for part in parts):
                return None
            result = sum(part * 60 ** i for i, part in enumerate(reversed(parts)))
        else:
            result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _run_text(run):
    if isinstance(run.get("text"), str) and run["text"]:
        return run["text"]
    emoji = run.get("emoji")
    shortcuts = emoji.get("shortcuts") if isinstance(emoji, dict) else None
    return "/".join(s for s in shortcuts or [] if isinstance(s, str))


def _json_lines(first, handle):
    """JSON Lines, one object per line. A line that does not parse is skipped:
    an interrupted replay fetch ends in half a line, and the messages before it
    are still evidence. A file where no line parses at all is not a replay."""
    parsed = skipped = 0
    for line in itertools.chain([first], handle):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        parsed += 1
        yield item
    if skipped and not parsed:
        raise ValueError("The chat replay could not be read.")


def _row(item):
    if not isinstance(item, dict):
        return
    replay = item.get("replayChatItemAction")
    if isinstance(replay, dict):
        offset = _seconds(replay.get("videoOffsetTimeMsec"))
        if offset is None:
            return
        for action in replay.get("actions") or []:
            # One odd entry (a provider format change, a hand-edited file) is
            # skipped; it never makes the rest of the replay unreadable.
            added = action.get("addChatItemAction") if isinstance(action, dict) else None
            container = added.get("item") if isinstance(added, dict) else None
            message = container.get("liveChatTextMessageRenderer") if isinstance(container, dict) else None
            if not isinstance(message, dict):
                continue
            body = message.get("message")
            runs = body.get("runs") if isinstance(body, dict) else None
            text = " ".join(_run_text(run) for run in runs or [] if isinstance(run, dict))
            yield Message(offset / 1000, str(message.get("authorExternalChannelId") or ""), text[:500])
        return
    author = item.get("author_id") or item.get("author") or item.get("commenter") or ""
    if isinstance(author, dict):
        role = str(author.get("type") or "").lower()
        if role == "bot" or author.get("is_bot"):
            return
        author = author.get("id") or author.get("_id") or ""
    kind = str(item.get("type") or item.get("message_type") or "text").lower()
    if item.get("is_bot") in (True, "true", "1") or kind not in (
        "text", "text_message", "textmessageevent", "chat",
    ):
        return
    timestamp = next((item[name] for name in ("time_seconds", "time_in_seconds", "content_offset_seconds",
                                             "seconds", "timestamp") if item.get(name) is not None), None)
    seconds = _seconds(timestamp)
    text = item.get("text") or item.get("message") or ""
    if isinstance(text, dict):
        text = text.get("body") or ""
    if seconds is not None and isinstance(text, str) and text.strip():
        viewers = _seconds(item.get("viewers"))
        yield Message(seconds, str(author)[:150], text[:500], int(viewers) if viewers and viewers >= 1 else None)


def read_messages(path, *, source_id=None):
    """CSV/JSON/JSONL offsets in source seconds. Reject mismatched declared IDs."""
    def rows(data):
        if isinstance(data, dict):
            video = data.get("video") or data.get("Video") or {}
            declared = data.get("video_id") or data.get("source_id")
            if not declared and isinstance(video, dict):
                declared = video.get("id") or video.get("_id")
            if declared is not None and source_id and str(declared) != source_id:
                raise ValueError("This replay identifies a different source.")
            items = data.get("messages") if "messages" in data else data.get("comments") or data.get("Comments")
            if isinstance(items, list):
                yield from items
                return
        if isinstance(data, list):
            yield from data
        else:
            yield data

    with open(path, encoding="utf-8-sig") as handle:
        if str(path).endswith(".csv"):
            stream = csv.DictReader(handle)
        else:
            first = handle.readline()
            if not first:
                return
            if first.lstrip().startswith("["):
                handle.seek(0)
                stream = rows(json.load(handle))
            else:
                try:
                    json.loads(first)
                except json.JSONDecodeError:
                    # One pretty-printed document, or not JSON at all.
                    handle.seek(0)
                    try:
                        stream = rows(json.load(handle))
                    except json.JSONDecodeError:
                        handle.seek(0)
                        stream = (row for item in _json_lines(handle.readline(), handle)
                                  for row in rows(item))
                else:
                    stream = (row for item in _json_lines(first, handle) for row in rows(item))
        for item in stream:
            for message in _row(item):
                yield message


def spikes(messages, duration, *, offset=0.0, window=10, baseline_seconds=90,
           reaction_delay=7.0, emotes=()):
    """Compare volume with nearby bins. Isolated authors and commands cannot drive a spike."""
    bins = defaultdict(list)
    raw = Counter()
    authors_present = False
    accepted = 0
    for m in messages:
        time = m.seconds + offset
        if not 0 <= time < duration:
            continue
        idx = int(time // window)
        raw[idx] += 1
        message = " ".join(m.text.casefold().split())
        if NON_REACTION.search(message):
            continue
        reaction = next((name for name, pattern in REACTIONS.items() if pattern.search(message)), None)
        if reaction is None and any(emote in message for emote in emotes):
            reaction = "community reaction"
        bins[idx].append((m.author, message, reaction, m.viewers))
        authors_present |= bool(m.author)
        accepted += 1

    volume = {}
    participants = {}
    reactions = {}
    views = {}
    for idx, entries in bins.items():
        kept_authors = Counter()
        unique_lines = set()
        public = set()
        categories = Counter()
        viewer_counts = []
        for author, line, reaction, viewers in entries:
            # Many people can share the same emote. Repeated generic messages,
            # commands and copy-pastes carry very little independent evidence.
            key = (author, line) if reaction else ("", line)
            if key in unique_lines or (author and kept_authors[author] >= 3):
                continue
            unique_lines.add(key)
            if author:
                kept_authors[author] += 1
                public.add(author)
            if reaction:
                categories[reaction] += 1
            if viewers:
                viewer_counts.append(viewers)
        volume[idx] = len(unique_lines)
        participants[idx] = len(public) if authors_present and public else None
        reactions[idx] = [name for name, _ in categories.most_common(3)]
        if viewer_counts:
            views[idx] = int(median(viewer_counts))

    result = []
    baseline_bins = max(3, math.ceil(baseline_seconds / window))
    first_bin = min(raw, default=0)
    last_bin = max(raw, default=-1)
    for idx, count in volume.items():
        n = participants[idx]
        if count < 5 or (n is not None and n < 3):
            continue
        prior = [volume.get(j, 0) for j in range(max(first_bin, idx - baseline_bins), idx)]
        if len(prior) < 3:
            prior.extend(volume.get(j, 0) for j in range(idx + 1, min(idx + baseline_bins, last_bin + 1)))
        if len(prior) < 3:
            continue
        base = median(prior)
        dispersion = median(abs(p - base) for p in prior)
        excess = count - base
        ratio = count / max(1.0, base)
        if ratio < 2.0 or excess < max(4.0, 3.0 * dispersion):
            continue
        time = idx * window
        marker = {
            "kind": "chat", "time": time, "start": max(0.0, time - reaction_delay),
            "end": min(duration, time + window), "messages": raw[idx], "activity": count,
            "baseline": base, "ratio": round(ratio, 2), "participants": n,
            "reactions": reactions[idx], "rising": count > volume.get(idx - 1, 0) * 1.5,
            "sustained": volume.get(idx + 1, 0) >= max(4.0, 1.5 * base),
            "confidence": "higher" if len(prior) >= baseline_bins and n is not None else "limited",
        }
        if idx in views:
            marker["viewers"] = views[idx]
            marker["messages_per_100_viewers"] = round(100 * count / views[idx], 2)
        result.append(marker)
    ranked = sorted(result, key=lambda p: (p["activity"] - p["baseline"], p["ratio"]), reverse=True)
    merged = []
    for marker in ranked:
        if all(abs(other["time"] - marker["time"]) > window for other in merged):
            merged.append(marker)
            if len(merged) == MAX_SPIKES:
                break
    return sorted(merged, key=lambda item: item["time"]), accepted
