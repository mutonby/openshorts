"""Brutal Truth clipping engine helpers — score filter, silence removal,
cold-open hook, FCP XML export. Hoisted out of main.py's __main__ block
so they're importable for tests."""
import math
import os
import subprocess


def normalize_intervals(intervals, source_duration=None, min_duration=15.0,
                        pre_roll=0.2, post_roll=0.3):
    """Return sorted, non-overlapping, chronologically valid clip intervals.

    Gemini/transcript boundaries are treated as untrusted input. Invalid rows
    are dropped; valid rows are padded, clamped, sorted, and merged when they
    overlap. The output satisfies ``next.start >= previous.end``.

    ponytail: overlapping candidates are merged rather than rendered twice;
    preserving both would recreate the same footage and cause the exact looping
    symptom this guard prevents.
    """
    cleaned = []
    limit = float(source_duration) if source_duration is not None else None
    for item in intervals or []:
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start -= max(0.0, float(pre_roll))
        end += max(0.0, float(post_roll))
        start = max(0.0, start)
        if limit is not None:
            start = min(start, limit)
            end = min(end, limit)
        if end <= start:
            continue
        if end - start < min_duration:
            end = start + min_duration
            if limit is not None:
                end = min(end, limit)
                start = max(0.0, end - min_duration)
        if end > start:
            cleaned.append((start, end, item))

    cleaned.sort(key=lambda row: (row[0], row[1]))
    merged = []
    for start, end, item in cleaned:
        if not merged or start >= merged[-1][1]:
            merged.append([start, end, item])
            continue
        # Keep the first metadata object, but extend its interval so the
        # overlapping footage is rendered exactly once.
        merged[-1][1] = max(merged[-1][1], end)

    result = []
    for start, end, item in merged:
        copy = dict(item)
        copy["start"] = round(start, 3)
        copy["end"] = round(end, 3)
        result.append(copy)
    return result


def filter_by_score(shorts, threshold=85, top_k=None, transcript=None,
                    top_terms=None):
    """Keep only clips with composite_score ≥ threshold. If nothing survives,
    fall back to the top-K strongest (any K ≤ len). Empty in → empty out.

    ponytail: auto-fallback guards against "0 clips returned" — far better UX
    failure mode than silently producing nothing after a 20-min Gemini run.
    """
    if not shorts:
        return []
    # Re-score before filtering so composite_score is current (Gemini's
    # predicted_score alone is too noisy to gate on).
    enrich_metadata(shorts, transcript=transcript, top_terms=top_terms)
    ranked = sorted(shorts, key=lambda s: s.get("composite_score", 0), reverse=True)
    filtered = [s for s in ranked if s.get("composite_score", 0) >= threshold]
    if filtered:
        return filtered
    if top_k:
        return ranked[:top_k]
    # ponytail: no top_k supplied → keep top 5 (sensible default; matches the
    # 3-15 range Gemini was asked for). Adjust via env MANUAL_CLIP_COUNT if needed.
    return ranked[:5]


# --- Composite scoring v2 — signal blend, not copy ----------------------
# ponytail: weighted blend with env-overridable weights. Each sub-score is
# 0-100; composite is the weighted sum. Calibration against real view counts
# is future work — without published-clip telemetry this is heuristic, not a
# measured model. Realistic ceiling ~7/10 until you wire a feedback loop.

def _clamp01(x):
    return max(0.0, min(1.0, x))


def _duration_score(start, end):
    """Sweet spot 25-45s. Below 15s or above 70s = penalty. Full 0-1."""
    dur = max(0, end - start)
    if 25 <= dur <= 45:
        return 1.0
    if 15 <= dur < 25:
        return 0.7 + 0.3 * (dur - 15) / 10
    if 45 < dur <= 70:
        return 1.0 - 0.6 * (dur - 45) / 25
    if dur < 15:
        return 0.4
    return 0.2  # >70s — too long for a short


def _hook_score(words, clip_start, clip_end):
    """Speech density in the first 3 seconds — high word-rate signals a
    punchy open. Returns 0-1. Reuses climax-window logic but starts at t0."""
    if not words:
        return 0.5  # neutral — silent/unknown
    window = 3.0
    open_words = [w for w in words
                  if clip_start <= w.get("start", 1e9) < clip_start + window]
    # ~3-5 words in 3s = energetic; >7 exceptional; 0-1 weak
    rate = len(open_words) / window  # words/sec
    return _clamp01(rate / 2.2)  # 2.2 wps ≈ 132 wpm → maps to 1.0


def _pacing_score(words, clip_start, clip_end):
    """Words-per-minute normalized to a 0-1 curve. 200+ wpm = energetic,
    <80 = slow. ponytail: simple wpm, no silence-detection pass here."""
    dur = max(1.0, clip_end - clip_start)
    if not words:
        return 0.4
    wpm = (len(words) / dur) * 60
    return _clamp01((wpm - 60) / 180)  # 60-240 wpm → 0-1


def _keyword_score(words, clip_start, clip_end, top_terms=None):
    """Overlap between clip's word-set and the video's top topic terms.
    top_terms is a set of keywords computed once per video (passed in from
    main.py after transcript is finalized). Empty/null → neutral 0.5."""
    if not top_terms:
        return 0.5
    clip_words = {w.get("word", "").lower().strip(" .,!?") for w in (words or [])
                  if clip_start <= w.get("start", 1e9) < clip_end}
    if not clip_words:
        return 0.0
    overlap = len(clip_words & top_terms)
    # Saturating curve: 3+ shared terms → full
    return _clamp01(overlap / 3.0)


def compute_composite(clip, transcript=None, top_terms=None,
                      weights=None):
    """Blend known signals into a 0-100 composite score. Weights default to:
    gemini 0.50, hook 0.20, duration 0.15, pacing 0.10, topic 0.05.
    Overridable via env W_GEMINI / W_HOOK / W_DUR / W_PACE / W_TOPIC (each 0-1,
    rescaled to sum 1 internally).

    ponytail: weights live in env so you can A/B-tune without code edits;
    a feedback loop (real view counts → re-label → re-weight) is the upgrade
    path, not a fancier model with no data to fit it to.
    """
    import os as _os
    if weights is None:
        weights = {
            "gemini": float(_os.environ.get("W_GEMINI", "0.50")),
            "hook":   float(_os.environ.get("W_HOOK", "0.20")),
            "dur":    float(_os.environ.get("W_DUR", "0.15")),
            "pace":   float(_os.environ.get("W_PACE", "0.10")),
            "topic":  float(_os.environ.get("W_TOPIC", "0.05")),
        }
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    gemini = float(clip.get("predicted_score", 0) or 0)

    # Extract word stream for this clip
    words = []
    if transcript:
        for seg in (transcript or {}).get("segments", []):
            words.extend(seg.get("words", []) or [])

    cs, ce = float(clip.get("start", 0)), float(clip.get("end", 0))
    h = _hook_score(words, cs, ce)
    d = _duration_score(cs, ce)
    p = _pacing_score(words, cs, ce)
    t = _keyword_score(words, cs, ce, top_terms)

    composite = (
        weights["gemini"] * gemini +
        weights["hook"]   * h * 100 +
        weights["dur"]     * d * 100 +
        weights["pace"]    * p * 100 +
        weights["topic"]   * t * 100
    )
    clip["composite_score"] = round(composite, 1)
    clip["virality_breakdown"] = {
        "gemini_score": round(gemini, 1),
        "hook_strength": round(h, 2),
        "duration_fit": round(d, 2),
        "pacing": round(p, 2),
        "topic_relevance": round(t, 2),
    }
    return clip["composite_score"]


def top_terms_from_transcript(transcript, top_n=20):
    """Cheap-and-cheerful topic-term extraction: word frequency minus a tiny
    stopword set. ponytail: a real TF-IDF or a keyphrase model is the upgrade
    path when transcripts routinely exceed ~10 min; for short-form source
    material this free heuristic is plenty and avoids a sklearn dep."""
    import re as _re
    STOP = set("the a an and or but of to in on at by for with from is are was "
               "were be been being this that these those it its as so not no "
               "you i we they he she them us my your his her our their "
               "do does did doing have has had having if then than just "
               "really like gonna want got yeah oh um uh like".split())
    counts = {}
    for seg in (transcript or {}).get("segments", []):
        for w in seg.get("words", []) or []:
            word = w.get("word", "").lower().strip(" .,!?\"'")
            if not word or word in STOP or len(word) < 3:
                continue
            counts[word] = counts.get(word, 0) + 1
    # Top-N by count
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {w for w, _ in ranked[:top_n]}


def cap_count(shorts, max_count):
    return shorts[:max_count]


def enrich_metadata(shorts, transcript=None, top_terms=None):
    """Back-compat wrapper around compute_composite: stamps composite_score +
    virality_breakdown on each clip. Existing callers (e.g. main.py) that
    pass only `shorts` get the default heuristic breakdown.

    ponytail: kept as a separate function so the old test (which only passes
    shorts and asserts `hashtags` + `virality_breakdown` keys exist) still
    passes without rewriting its assertions.
    """
    for s in shorts:
        s.setdefault("hashtags", [])
        compute_composite(s, transcript=transcript, top_terms=top_terms)
    return shorts


def remove_silence(clip_path, noise_db="-30dB", min_dur=0.4):
    """Strip dead air while keeping video/audio on one fresh timeline."""
    if not os.path.exists(clip_path):
        return False
    tmp_out = clip_path + ".nonsilence.mp4"
    audio_filter = (
        f"silenceremove=start_periods=1:start_silence=0:"
        f"start_threshold={noise_db}:stop_periods=-1:stop_silence=0:"
        f"stop_threshold={noise_db}:stop_duration={min_dur},"
        "asetpts=PTS-STARTPTS"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", clip_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", "setpts=PTS-STARTPTS", "-af", audio_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", tmp_out,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        r = None
    if r and r.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
        os.replace(tmp_out, clip_path)
        return True
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False


EMOTION_BOOST = {"!", "?", "really", "insane", "crazy", "amazing", "wow",
                 "never", "always", "absolutely", "literally", "actually",
                 "shocking", "unbelievable", "exclusive", "first", "only"}


def find_climax_window(seg_words, clip_start, clip_end):
    """Locate the most intense 0-3s window in the clip.

    Heuristic v2 (retention-tuned):
      - baseline: word count in the window (talking density)
      - weight: avg word length (longer words = more substantive)
      - bonus: emotional/surprise markers (! ? REALLY INSANE...) — these are
        well-documented "scroll-stop" cues in short-form retention research.
    ponytail: no audio loudness feature here; that would need a peek into
    the waveform (peak/RMS) — agenda for v3 if a real retention telemetry
    loop proves the word-signal is too weak.
    """
    words_in = [w for w in seg_words
                if w.get('end', 0) > clip_start and w.get('start', 0) < clip_end]
    if len(words_in) < 5:
        return None
    window = 3.0
    best, best_score = None, -1
    step = 0.5
    t = clip_start
    while t + window <= clip_end:
        ws = [w for w in words_in if t <= w.get('start', 0) < t + window]
        if ws:
            wc = len(ws)
            avg_len = sum(len(w.get('word', '').strip()) for w in ws) / wc
            emote = sum(1 for w in ws
                        if any(m in w.get('word', '').lower() for m in EMOTION_BOOST))
            # weighted: density (3×) + length (1×) + emote bonus (5× each)
            score = wc * 3 + avg_len + emote * 5
            if score > best_score:
                best_score = score
                best = (t, t + window)
        t += step
    return best


def find_weak_tail(seg_words, clip_start, clip_end, tail_window=2.0):
    """Identify a 'weak tail' — the last N seconds where speech density drops
    below 40% of the clip's average. Used to trim the tail for retention: a
    rushed ending tanks completion rate. Returns the new clip_end (or None if
    no trim warranted).

    ponytail: heuristic only; real completion-rate optimization needs an
    actual retention curve from published clips. Upgrade path is to plug in
    aggregated watch-time data once published clips flow back telemetry.
    """
    words_in = [w for w in seg_words
                if w.get('end', 0) > clip_start and w.get('start', 0) < clip_end]
    if len(words_in) < 8:  # too few to judge pacing
        return None
    dur = max(0.1, clip_end - clip_start)
    avg_rate = len(words_in) / dur
    tail_start = clip_end - tail_window
    if tail_start <= clip_start:
        return None
    tail_words = [w for w in words_in if w.get('start', 0) >= tail_start]
    tail_rate = len(tail_words) / tail_window
    if tail_rate >= 0.4 * avg_rate:
        return None  # tail is still energetic
    # Find trim point: end of the last word before the dead zone.
    # If there are words inside the tail, use the latest one's end;
    # otherwise fall back to the last word whose end < tail_start (the
    # last "energetic" word, with a small breath margin).
    candidates = [w for w in tail_words if w.get('end', 0) > 0]
    if candidates:
        candidates.sort(key=lambda w: w.get('end', 0))
        new_end = candidates[-1].get('end', 0) + 0.2
    else:
        # No speech in tail → trim to the last word end before the tail, +breath
        before_tail = [w for w in words_in if w.get('end', 0) <= tail_start]
        if not before_tail:
            return None
        before_tail.sort(key=lambda w: w.get('end', 0))
        new_end = before_tail[-1].get('end', 0) + 0.2
    if clip_start + 5.0 < new_end < clip_end:  # min 5s clip
        return new_end
    return None


def extract_clean_segment(source_video, start, end, output_path):
    """Accurately cut one A/V segment with fresh, monotonic timestamps.

    Stream-copy is deliberately not used here: keyframe-copy cuts preserve
    source PTS/DTS and can produce non-monotonic timestamps at later joins.
    """
    try:
        start = max(0.0, float(start))
        end = float(end)
    except (TypeError, ValueError):
        return False
    if not os.path.exists(source_video) or end <= start:
        return False
    duration = end - start
    tmp_out = output_path + ".cut.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", source_video,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", "setpts=PTS-STARTPTS",
        "-af", "asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-fps_mode", "cfr", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart", tmp_out,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result and result.returncode == 0 and os.path.exists(tmp_out):
        if os.path.getsize(tmp_out) > 0:
            os.replace(tmp_out, output_path)
            return True
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False


def concat_clean_segments(segment_paths, output_path):
    """Join already-cut segments with normalized PTS/DTS and unified A/V.

    Every input is decoded, reset to time zero, then passed through concat.
    This prevents stream-copy concat from carrying incompatible timestamps or
    accidentally repeating a segment at a boundary.
    """
    paths = [p for p in segment_paths if os.path.exists(p)]
    if not paths:
        return False
    inputs = []
    filters = []
    for i, path in enumerate(paths):
        inputs.extend(["-i", path])
        filters.append(
            f"[{i}:v:0]setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a:0]asetpts=PTS-STARTPTS[a{i}]"
        )
    labels = "".join(f"[v{i}][a{i}]" for i in range(len(paths)))
    filters.append(f"{labels}concat=n={len(paths)}:v=1:a=1[v][a]")
    tmp_out = output_path + ".concat.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", tmp_out,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result and result.returncode == 0 and os.path.exists(tmp_out):
        if os.path.getsize(tmp_out) > 0:
            os.replace(tmp_out, output_path)
            return True
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False


def add_cold_open(clip_path, clip_start, clip_end, transcript, source_video, output_dir):
    """No-op by default: a hook selected from inside the same clip is duplicate
    footage, not a separate segment. The old implementation prepended that
    overlapping window with ``-c copy`` and caused the reported looping bug.

    Keep the API for compatibility; callers can build a deliberate editorial
    cold open externally using extract_clean_segment + concat_clean_segments.
    """
    return False


def trim_weak_tail(clip_path, clip_start, clip_end, transcript):
    """Re-cut the clip with a tighter end if pacing analysis shows a dead tail.
    ponytail: re-encodes audio (aac) but stream-copies video for speed. Skipped
    entirely for silent videos — pacing signal needs words.
    """
    if not transcript:
        return False
    seg_words = []
    for seg in transcript.get('segments', []):
        seg_words.extend(seg.get('words', []) or [])
    new_end = find_weak_tail(seg_words, clip_start, clip_end)
    if not new_end:
        return False
    tmp_out = clip_path + ".trimmed.mp4"
    # Note: -to is relative to the input position; using absolute timestamps
    # against the full clip works because we re-input the already-cut clip.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", clip_path,
        "-t", f"{new_end - clip_start:.3f}",
        "-vf", "setpts=PTS-STARTPTS", "-af", "asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-fps_mode", "cfr",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", tmp_out,
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        r = None
    if r and r.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
        os.replace(tmp_out, clip_path)
        return True
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False


def export_xml_timeline(shorts, video_title, output_dir, video_duration):
    """Minimal FCP 7 XML — one sequence per clip, no transitions.
    Enough for Premiere/Final Cut import without clip nesting complexity."""
    xml_path = os.path.join(output_dir, f"{video_title}_timeline.xml")
    tc = lambda s: f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{(s%60):06.3f}"

    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<xmeml version="5">']
    out.append('  <sequence id="truelifeclipper">')
    out.append(f'    <name>{video_title}</name>')
    out.append(f'    <duration>{int(video_duration)}</duration>')
    out.append('    <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>')
    out.append('    <media><video>')
    out.append('      <track>')
    out.append(f'        <enabled>TRUE</enabled>')
    for i, s in enumerate(shorts, 1):
        clip_path = os.path.join(output_dir, f"{video_title}_clip_{i}.mp4")
        out.append(f'        <clipitem id="clip{i}">')
        out.append(f'          <name>Clip {i}: {(s.get("video_title_for_youtube_short") or "")[:50]}</name>')
        out.append(f'          <enabled>TRUE</enabled>')
        out.append(f'          <duration>{int(s.get("end", 0) - s.get("start", 0))}</duration>')
        out.append(f'          <start>{tc(s.get("start", 0))}</start>')
        out.append(f'          <end>{tc(s.get("end", 0))}</end>')
        out.append(f'          <in>0</in>')
        out.append(f'          <out>{int((s.get("end", 0) - s.get("start", 0)))}</out>')
        media_url = Path_win_fix(clip_path)
        out.append('          <file id="file' + str(i) + '">')
        out.append(f'            <pathurl>{media_url}</pathurl>')
        out.append('            <mediaType>Video</mediaType>')
        out.append('            <media><video/></media>')
        out.append('          </file>')
        out.append('        </clipitem>')
    out.append('      </track>')
    out.append('    </video></media>')
    out.append('  </sequence>')
    out.append('</xmeml>')
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    return xml_path


def Path_win_fix(p):
    """Convert Windows path to file:// URL with forward slashes."""
    s = p.replace('\\', '/')
    if not s.startswith('/'):
        s = '/' + s
    return f"file://localhost{s}"



def render_preset_to_ffmpeg_args(preset):
    """Map a render quality preset to renderer kwargs. The renderer (v2 or
    v1) picks the encoder. ponytail: only width + crf change here — codec
    args come from ffmpeg_utils.QUALITY."""
    presets = {
        "draft":   {"width": 540,  "crf": 28, "preset": "ultrafast"},
        "review":  {"width": 720,  "crf": 23, "preset": "veryfast"},
        "final":   {"width": 1080, "crf": 19, "preset": "slow"},
    }
    return presets.get(preset, presets["review"])


def naming_template(template, client, title, fmt="mp4", date=None):
    """Build clip filenames from a templating pattern.

    Supports tokens: {client} {title} {n} {fmt} {date}.

    ponytail: just str.format — pattern strings are user-supplied, no need
    to invent a regex mini-language.
    """
    from datetime import datetime
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    safe_title = "".join(c if (c.isalnum() or c in "-_") else "_" for c in title)[:60]
    safe_client = "".join(c if (c.isalnum() or c in "-_") else "_" for c in client)[:30]
    return template.format(client=safe_client, title=safe_title,
                          n="{n}", fmt=fmt, date=date)


def whitelabel_clip_metadata(clip, client, brand_hashtags=None, brand_caption_prefix=""):
    """Stamp per-client branding on a clip's metadata: hashtags get joined
    with brand hashtags, caption gets prefixed, and a client tag added.

    ponytail: mutates the dict in place; cheap and the consumer (XML export,
    metadata.json) reads from the same object.
    """
    if brand_hashtags is None:
        brand_hashtags = []
    clip['client'] = client
    existing = clip.get('hashtags', []) or []
    seen = set()
    merged = []
    for tag in list(existing) + list(brand_hashtags):
        if tag and tag not in seen:
            seen.add(tag)
            merged.append(tag)
        if len(merged) >= 12:
            break
    clip['hashtags'] = merged
    desc = clip.get('video_description_for_tiktok') or ""
    if brand_caption_prefix and not desc.startswith(brand_caption_prefix):
        clip['video_description_for_tiktok'] = f"{brand_caption_prefix} {desc}".strip()
    return clip


def write_qc_state(output_dir, clip_idx, state, reviewer="", note=""):
    """Append/update a JSON file describing a clip's QC state machine:
    pending → submitted → approved | rejected. Enables an ed assistant
    workflow: render → submit → human approves → publish.
    """
    import json as _json
    from datetime import datetime
    valid = ("pending", "submitted", "approved", "rejected")
    if state not in valid:
        raise ValueError(f"qc state must be one of {valid}")
    qc_path = os.path.join(output_dir, "qc_state.json")
    db = {}
    if os.path.exists(qc_path):
        try:
            db = _json.loads(open(qc_path, encoding="utf-8").read())
        except Exception:
            db = {}
    db.setdefault("clips", {})
    db["clips"][str(clip_idx)] = {
        "state": state, "reviewer": reviewer, "note": note,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(qc_path, "w", encoding="utf-8") as f:
        _json.dump(db, f, indent=2)
    return qc_path


def find_static_broll_insertions(scene_ranges, transcript, min_static_seconds=4.0,
                                 insert_duration=2.0, max_insertions=3):
    """Choose short pattern-interrupt windows from long static scenes.

    ``scene_ranges`` is an iterable of (start_seconds, end_seconds). Keyword
    extraction is intentionally local and deterministic; the asset matcher
    later uses those keywords against local filenames.
    """
    words = []
    for segment in (transcript or {}).get("segments", []):
        words.extend(segment.get("words", []) or [])
    candidates = []
    for scene_start, scene_end in scene_ranges or []:
        start, end = float(scene_start), float(scene_end)
        if end - start < min_static_seconds:
            continue
        insert_start = start + min(2.0, (end - start - insert_duration) / 2)
        insert_end = min(end, insert_start + insert_duration)
        scene_words = [
            str(w.get("word", "")).strip(" .,!?\"'").lower()
            for w in words
            if w.get("start", 0) < end and w.get("end", 0) > start
        ]
        keywords = [w for w in scene_words if len(w) >= 3]
        candidates.append({
            "start": round(insert_start, 3),
            "end": round(insert_end, 3),
            "keywords": list(dict.fromkeys(keywords))[:12],
        })
    return candidates[:max_insertions]


def select_broll_asset(asset_dir, keywords):
    """Select a local video asset whose filename overlaps transcript keywords."""
    if not asset_dir or not os.path.isdir(asset_dir):
        return None
    files = []
    for name in os.listdir(asset_dir):
        if os.path.splitext(name)[1].lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            files.append(name)
    if not files:
        return None
    terms = {str(k).lower() for k in keywords or []}
    ranked = sorted(
        files,
        key=lambda name: sum(
            1 for term in terms if term and term in os.path.splitext(name.lower())[0]
        ),
        reverse=True,
    )
    return os.path.join(asset_dir, ranked[0])


def overlay_broll(video_path, output_path, insertions, asset_dir):
    """Overlay local 2-second pattern interrupts with fresh A/V timestamps."""
    selected = []
    for insertion in insertions or []:
        asset = select_broll_asset(asset_dir, insertion.get("keywords"))
        if asset:
            selected.append((asset, insertion))
    if not selected:
        return False

    inputs = ["-i", video_path]
    filters = ["[0:v]setpts=PTS-STARTPTS[base0]"]
    current = "base0"
    for idx, (asset, insertion) in enumerate(selected, 1):
        input_idx = idx
        inputs.extend(["-stream_loop", "-1", "-i", asset])
        start = float(insertion["start"])
        end = float(insertion["end"])
        filters.append(
            f"[{input_idx}:v]trim=duration={end-start:.3f},setpts=PTS-STARTPTS,"
            f"scale=900:-2[br{idx}]"
        )
        next_label = f"base{idx}"
        filters.append(
            f"[{current}][br{idx}]overlay=(W-w)/2:(H-h)/2:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{next_label}]"
        )
        current = next_label

    tmp_out = output_path + ".broll.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
        "-filter_complex", ";".join(filters),
        "-map", f"[{current}]", "-map", "0:a:0?",
        "-af", "asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-fps_mode", "cfr",
        "-shortest", "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart", tmp_out,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=1800)
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result and result.returncode == 0 and os.path.exists(tmp_out):
        if os.path.getsize(tmp_out) > 0:
            os.replace(tmp_out, output_path)
            return True
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    return False


def expand_batch(sources):
    """Normalize a batch spec into a flat list of source references.
    Each source can be a path/URL string, or a dict with metadata.

    ponytail: just a flatting pass + dedup; full job tracking happens in
    app.py.
    """
    flat = []
    seen = set()
    for src in sources:
        if isinstance(src, str):
            ref = src
            meta = {}
        elif isinstance(src, dict):
            ref = src.get("url") or src.get("path") or src.get("source") or ""
            meta = {k: v for k, v in src.items()
                    if k not in ("url", "path", "source")}
        else:
            continue
        if ref and ref not in seen:
            seen.add(ref)
            flat.append({"url": ref, **meta})
    return flat
