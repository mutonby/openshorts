import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
import math
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp

# import whisper (replaced by faster_whisper inside function)
from openai import OpenAI
from dotenv import load_dotenv
import json

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16
MIN_CLIP_DURATION = 15
MAX_CLIP_DURATION = 60
MIN_HIGHLIGHT_SCORE = 0.85
PODCAST_COMEDY_LANES = ("COMEDY", "HOT_TAKE", "PERSONAL_STORY")
PODCAST_COMEDY_MAX_CANDIDATES = 24
PODCAST_COMEDY_CONTEXT_BEFORE = 8.0
PODCAST_COMEDY_CONTEXT_AFTER = 5.0
PODCAST_COMEDY_HARD_MIN_DURATION = 15.0
PODCAST_COMEDY_HARD_MAX_DURATION = 70.0
PODCAST_COMEDY_DURATION_TOLERANCE = 5.0
PODCAST_COMEDY_BALANCE_GAP = 0.12
PODCAST_COMEDY_JUDGE_MIN_SCORE = 0.7

GEMINI_BASE_PROMPT = """
You are a world-class short-form video editor for TikTok, Instagram Reels, and YouTube Shorts. Identify the 3-10 most viral-worthy moments from the transcript. Each clip MUST be self-contained — a viewer watching only that clip must understand the point without prior context.

VIDEO_DURATION: {video_duration}s | LANGUAGE: {language}
SCENE_BOUNDARIES (align clip edges here when possible): {scene_boundaries}

--- VIRAL PATTERNS (pick the best one per clip) ---
1. CURIOSITY_GAP: Opens a compelling question the viewer MUST know the answer to
2. EMOTIONAL_PEAK: High emotion — anger, excitement, laughter, tears, intense passion
3. VALUE_DROP: Actionable tip, life hack, insight, or framework usable immediately
4. CONTROVERSY: Hot take, debate trigger, strong opinion that sparks comments
5. CLIFFHANGER: Unfinished story, looming reveal
6. STORY_BEAT: Clear setup, conflict, or resolution
7. PATTERN_INTERRUPT: Unexpected twist, surprise reveal, "wait, what?!" moment
8. RELATABLE_MOMENT: "That's so me" — universal experience with high shareability

--- HOOK SIGNALS (check the FIRST sentence of the clip) ---
ANGKA_BESAR: "3 cara...", "5 hal...", "Rp50 juta..."
KLAIM_BERANI: "Ini yang terbaik...", "Gak ada yang berani ngomong ini..."
PERTANYAAN: Opens with "Kenapa...", "Gimana cara...", "Lo tau gak..."
BONGKAR_RAHASIA: "Akhirnya gue buka...", "Rahasianya adalah..."
DRAMA_EMOSI: Immediate laughter, shouting, crying, or intense reaction
Clips without any hook signal → max PERUNGGU.

{category_instructions}

--- QUALITY GATES ---
- START_QUALITY_CHECK: The FIRST 2 SECONDS must contain an understandable hook, reaction, or strong claim.
- COMPLETE_THOUGHT: Include the setup needed to understand the payoff and end after the thought resolves.
- QUOTE_EVIDENCE: Reasoning must cite words actually present inside the selected timestamps.
- SELF-CONTAINED CHECK: Reject clips that require earlier conversation to make sense.

--- AVOID (jangan pilih ini) ---
- Pembukaan generic: "Halo semua, kembali lagi di channel gue..."
- CTA kosong: "Jangan lupa like, comment, subscribe"
- Recap yang sudah dibahas, perkenalan tamu panjang
- Kalimat yang bergantung ke visual: "Lihat nih grafiknya"
- Jeda panjang atau "anu... eee..." berulang
- Segmen sponsor murni tanpa hook

--- CUTTING RULES ---
- Start 0.2-0.4s BEFORE the hook, end 0.2-0.4s AFTER the payoff
- NEVER cut mid-word or mid-sentence — align to topic boundaries (period, question, natural pause)
- Align with scene boundaries when available
- Avoid low-confidence words (p < 0.3 in WORDS_JSON)
- Timestamps in absolute seconds, 3 decimal places (e.g., 12.340)
- 0 ≤ start < end ≤ VIDEO_DURATION

--- CONFIDENCE TIERS ---
EMAS: Hook kuat <2 detik + payoff jelas + high shareability — langsung stop scroll
PERAK: Momen bagus tapi hook butuh 2-4 detik atau payoff agak lemah
PERUNGGU: Hook lambat atau konten generik, works as filler
TOLAK: Intro, outro, sponsor, atau filler tanpa hook (JANGAN output)

TRANSCRIPT: {transcript_text}
WORDS_JSON (array of {{"w", "s", "e", "p"}} where s/e are seconds, p is 0-1 confidence): {words_json}

--- OUTPUT ---
RETURN ONLY VALID JSON. No markdown fences, no extra text, no apologies. Order EMAS first.

{{
  "content_type": "<TUTORIAL|STORYTELLING|INTERVIEW|REACTION|VLOG|REVIEW|DEBATE|OTHER>",
  "shorts": [{{
    "start": <float>,
    "end": <float>,
    "confidence": "<EMAS|PERAK|PERUNGGU>",
    "reasoning": "<2-3 sentences: viral pattern, why it works, timing rationale>",
    "viral_pattern_type": "<one of the 8 patterns above>",
    "viral_hook_text": "<2-5 words, punchy hook overlay text, EXACT SAME LANGUAGE AS VIDEO>",
    "social_caption": "<engaging caption + 3-5 relevant hashtags, EXACT SAME LANGUAGE AS VIDEO>"
  }}]
}}
"""

CATEGORY_INSTRUCTIONS = {
    "podcast": """--- CATEGORY: PODCAST/DISKUSI ---
- DURASI IDEAL: 40-80 detik (diskusi serius) / 15-40 detik (komedi/roasting)
- Best clips: hot takes, bongkar rahasia, cliffhanger di akhir cerita, ketawa spontan
- Hook signals khusus: "Lo tau gak...", "Gue dulu mikir...", "Ini yang orang gak sadar..."
- Prioritas patterns: CURIOSITY_GAP, CONTROVERSY, CLIFFHANGER, STORY_BEAT, EMOTIONAL_PEAK
- Hindari: basa-basi pembuka, perkenalan tamu >5 detik, topik yang baru mulai dibangun""",
    "tutorial": """--- CATEGORY: TUTORIAL/EDUKASI ---
- DURASI IDEAL: 15-45 detik
- Best clips: actionable tip + hasil instan, before/after reveal, "cuma butuh X menit"
- Prioritas patterns: VALUE_DROP, CURIOSITY_GAP, PATTERN_INTERRUPT
- Hindari: penjelasan teknis panjang, setup tanpa payoff""",
    "gaming": """--- CATEGORY: GAMING ---
- DURASI IDEAL: 8-25 detik
- Best clips: clutch win, funny glitch, rage moment, unexpected kill
- Hook signals: jeritan, ketawa spontan, "Gak mungkin!", "WTF!"
- Prioritas patterns: EMOTIONAL_PEAK, PATTERN_INTERRUPT
- Hindari: gameplay biasa tanpa reaksi, loading screen, menu navigation""",
    "reaction": """--- CATEGORY: REAKSI ---
- DURASI IDEAL: 15-40 detik
- Best clips: emotional spike — kaget, nangis, ngakak, marah intens
- Hook signals: reaksi vokal keras, jeda dramatis, ekspresi kaget
- Prioritas patterns: EMOTIONAL_PEAK, PATTERN_INTERRUPT, RELATABLE_MOMENT
- Hindari: reaksi datar, komentar panjang tanpa emosi""",
    "interview": """--- CATEGORY: WAWANCARA ---
- DURASI IDEAL: 30-60 detik
- Best clips: reveal mengejutkan, soundbite kuat, cerita personal yang emosional
- Prioritas patterns: STORY_BEAT, EMOTIONAL_PEAK, CURIOSITY_GAP
- Hindari: perkenalan tamu, pertanyaan basa-basi interviewer, jawaban "ya/tidak" singkat""",
    "news": """--- CATEGORY: BERITA ---
- DURASI IDEAL: 20-50 detik
- Best clips: breaking news moment, kontroversi, angka mengejutkan
- Prioritas patterns: CONTROVERSY, CURIOSITY_GAP, PATTERN_INTERRUPT
- Hindari: pembacaan berita monoton, konteks background panjang""",
    "general": """--- AUTO-DETECT ---
Pertama, klasifikasikan jenis konten video ini, lalu terapkan aturan yang sesuai:
- PODCAST/DISKUSI → 40-80s, fokus hot takes, bongkar rahasia, cliffhanger
- TUTORIAL/EDUKASI → 15-45s, fokus actionable tips, before/after
- GAMING → 8-25s, fokus clutch/funny/rage
- REAKSI → 15-40s, fokus emotional spikes
- WAWANCARA → 30-60s, fokus reveals & soundbites
- BERITA → 20-50s, fokus breaking points & kontroversi
- Tidak jelas → default 15-60s, apply all patterns equally""",
}


def _normalize_whitespace(value):
    """Collapse repeated whitespace and return a safe string."""
    return " ".join(str(value or "").split())


def _normalize_hook_text(value):
    """Normalize hook overlays to uppercase and a maximum of six words."""
    words = _normalize_whitespace(value or "Viral Short").split()
    return " ".join(words[:6]).upper() or "VIRAL SHORT"


def _safe_float(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _split_segment_for_ai(segment, max_duration=30.0):
    """Split long timestamped segments at word boundaries."""
    start = _safe_float(segment.get("start"))
    end = _safe_float(segment.get("end"))
    text = _normalize_whitespace(segment.get("text"))
    words = sorted(
        segment.get("words", []),
        key=lambda word: _safe_float(word.get("start")),
    )
    if end - start <= max_duration or not words:
        return [(start, end, text, words)]
    first_word_start = _safe_float(words[0].get("start"), start)
    last_word_end = _safe_float(words[-1].get("end"), end)
    if first_word_start > start + 1.0 or last_word_end < end - 1.0:
        return [(start, end, text, words)]

    chunks = []
    current_words = []
    current_start = None
    for word in words:
        word_start = _safe_float(word.get("start"), start)
        word_end = _safe_float(word.get("end"), word_start)
        if current_words and word_end - current_start > max_duration:
            chunk_text = _normalize_whitespace(
                " ".join(
                    _normalize_whitespace(item.get("word"))
                    for item in current_words
                )
            )
            chunks.append(
                (
                    current_start,
                    _safe_float(current_words[-1].get("end"), current_start),
                    chunk_text,
                    current_words,
                )
            )
            current_words = []
            current_start = None
        if current_start is None:
            current_start = word_start
        current_words.append(word)

    if current_words:
        chunk_text = _normalize_whitespace(
            " ".join(
                _normalize_whitespace(item.get("word")) for item in current_words
            )
        )
        chunks.append(
            (
                current_start,
                _safe_float(current_words[-1].get("end"), current_start),
                chunk_text,
                current_words,
            )
        )
    return chunks


def _build_timestamped_transcript(transcript_result):
    """Build one compact segment transcript for AI analysis."""
    chunks = []
    for segment in sorted(
        transcript_result.get("segments", []),
        key=lambda item: _safe_float(item.get("start")),
    ):
        chunks.extend(_split_segment_for_ai(segment))

    timestamped = []
    for start, end, text, words in chunks:
        if not text or end <= start:
            continue
        probabilities = [
            max(0.0, min(_safe_float(word.get("probability"), 1.0), 1.0))
            for word in words
            if word.get("probability") is not None
        ]
        confidence = (
            sum(probabilities) / len(probabilities) if probabilities else 1.0
        )
        timestamped.append(
            {
                "id": len(timestamped),
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "confidence": round(confidence, 3),
            }
        )
    return timestamped


def _segments_for_window(segments, window_start, window_end):
    """Return only transcript segments overlapping a bounded time window."""
    return [
        segment
        for segment in segments
        if _safe_float(segment.get("end")) >= window_start
        and _safe_float(segment.get("start")) <= window_end
    ]


def _build_podcast_comedy_scout_prompt(
    timestamped_segments,
    video_duration,
    max_candidates=PODCAST_COMEDY_MAX_CANDIDATES,
    window_num=1,
    total_windows=1,
):
    """Build the Indonesian candidate-discovery prompt using one transcript."""
    transcript_json = json.dumps(timestamped_segments, ensure_ascii=False)
    return f"""
# ROLE: INDONESIAN_PODCAST_CANDIDATE_SCOUT
Cari kandidat clip terbaik dari podcast Indonesia dengan gaya obrolan natural,
roasting, slang, sarkasme, hot take, dan cerita personal.

# VIDEO
- Duration: {video_duration:.3f}s
- Window: {window_num}/{total_windows}
- Maximum candidates across this response: {max_candidates}

# CONTENT LANES
- COMEDY: setup -> build -> punchline -> reaction bernilai
- HOT_TAKE: klaim kuat + alasan/konsekuensi, bukan rage bait kosong
- PERSONAL_STORY: cerita dengan perubahan, reveal, pelajaran, atau payoff emosi

# QUALITY GATES
- START_QUALITY_CHECK: hook/reaction/klaim harus dipahami dalam 1-3 detik.
- COMPLETE_THOUGHT: sertakan konteks minimum dan akhiri setelah payoff selesai.
- QUOTE_EVIDENCE: hook_evidence dan payoff_evidence harus berasal dari timestamp.
- SELF-CONTAINED CHECK: tolak inside joke yang tidak bisa dipahami penonton baru.
- Hindari salam, sponsor, filler, pengulangan, dan acknowledgement kosong.
- SENSOR kata kasar Indonesia pada suggested_hook dengan tanda * di tengah kata.
- COMEDY target 20-40 detik; HOT_TAKE/PERSONAL_STORY target 35-60 detik.
- Semua timestamp absolut dan harus berada dalam video.

# OUTPUT
Return ONLY valid JSON:
{{
  "candidates": [{{
    "candidate_id": "w{window_num}-c1",
    "content_lane": "COMEDY|HOT_TAKE|PERSONAL_STORY",
    "start": 0.0,
    "end": 0.0,
    "setup_start": 0.0,
    "payoff_start": 0.0,
    "payoff_end": 0.0,
    "reaction_end": 0.0,
    "hook_evidence": "kutipan atau referensi langsung",
    "payoff_evidence": "kutipan atau referensi langsung",
    "standalone_summary": "apa yang dipahami penonton baru",
    "suggested_hook": "maksimal enam kata",
    "scout_score": 0.0
  }}]
}}

# TIMESTAMPED_TRANSCRIPT
{transcript_json}
""".strip()


def _normalize_content_lane(value):
    lane = _normalize_whitespace(value).upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ROASTING": "COMEDY",
        "FUNNY": "COMEDY",
        "CONTROVERSY": "HOT_TAKE",
        "HOTTAKE": "HOT_TAKE",
        "STORY": "PERSONAL_STORY",
        "PERSONAL": "PERSONAL_STORY",
    }
    lane = aliases.get(lane, lane)
    return lane if lane in PODCAST_COMEDY_LANES else ""


def _candidate_overlap_ratio(first, second):
    overlap = max(
        0.0,
        min(first["end"], second["end"]) - max(first["start"], second["start"]),
    )
    min_duration = min(
        first["end"] - first["start"], second["end"] - second["start"]
    )
    return overlap / min_duration if min_duration > 0 else 0.0


def _consolidate_scout_candidates(
    candidates,
    video_duration,
    max_candidates=PODCAST_COMEDY_MAX_CANDIDATES,
    max_overlap=0.7,
):
    """Validate, normalize, and deduplicate candidates from scout windows."""
    normalized = []
    for index, raw in enumerate(candidates or []):
        start = _safe_float(raw.get("start"), -1.0)
        end = _safe_float(raw.get("end"), -1.0)
        duration = end - start
        lane = _normalize_content_lane(raw.get("content_lane"))
        hook_evidence = _normalize_whitespace(raw.get("hook_evidence"))
        payoff_evidence = _normalize_whitespace(raw.get("payoff_evidence"))
        if (
            not lane
            or not hook_evidence
            or not payoff_evidence
            or start < 0
            or end > video_duration
            or duration < PODCAST_COMEDY_HARD_MIN_DURATION
            or duration > PODCAST_COMEDY_HARD_MAX_DURATION
        ):
            continue

        score = _safe_float(
            raw.get("scout_score", raw.get("score", raw.get("confidence"))), 0.0
        )
        setup_start = _safe_float(raw.get("setup_start"), start)
        payoff_start = _safe_float(raw.get("payoff_start"), end)
        payoff_end = _safe_float(raw.get("payoff_end"), end)
        reaction_end = _safe_float(raw.get("reaction_end"), end)
        if not (
            start <= setup_start <= end
            and start <= payoff_start <= payoff_end <= reaction_end <= end
        ):
            continue

        candidate = dict(raw)
        candidate.update(
            {
                "candidate_id": _normalize_whitespace(raw.get("candidate_id"))
                or f"candidate-{index + 1}",
                "content_lane": lane,
                "start": round(start, 3),
                "end": round(end, 3),
                "setup_start": round(setup_start, 3),
                "payoff_start": round(payoff_start, 3),
                "payoff_end": round(payoff_end, 3),
                "reaction_end": round(reaction_end, 3),
                "hook_evidence": hook_evidence,
                "payoff_evidence": payoff_evidence,
                "standalone_summary": _normalize_whitespace(
                    raw.get("standalone_summary")
                ),
                "suggested_hook": _normalize_hook_text(raw.get("suggested_hook")),
                "scout_score": max(0.0, min(score, 1.0)),
            }
        )
        normalized.append(candidate)

    deduped = []
    for candidate in sorted(
        normalized, key=lambda item: item["scout_score"], reverse=True
    ):
        if any(
            _candidate_overlap_ratio(candidate, kept) > max_overlap
            for kept in deduped
        ):
            continue
        deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _attach_candidate_context(
    candidates,
    timestamped_segments,
    context_before=PODCAST_COMEDY_CONTEXT_BEFORE,
    context_after=PODCAST_COMEDY_CONTEXT_AFTER,
    video_duration=None,
):
    """Attach bounded transcript context for global judging."""
    contextualized = []
    for candidate in candidates:
        item = dict(candidate)
        context_start = max(0.0, candidate["start"] - context_before)
        context_end = candidate["end"] + context_after
        if video_duration is not None:
            context_end = min(video_duration, context_end)
        item["context_start"] = round(context_start, 3)
        item["context_end"] = round(context_end, 3)
        item["transcript_context"] = _segments_for_window(
            timestamped_segments, context_start, context_end
        )
        contextualized.append(item)
    return contextualized


def _build_podcast_comedy_judge_prompt(candidates, max_clips=10):
    """Build the global Indonesian editor/judge prompt."""
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    return f"""
# ROLE: INDONESIAN_EDITOR_JUDGE
Bandingkan seluruh kandidat podcast Indonesia secara global. Pahami slang,
sarkasme, roasting, callback, banter antarpembicara, dan konteks budaya.

# PRIORITAS PENILAIAN
1. Hook dipahami dalam 1-3 detik.
2. Setup efisien dan payoff kuat.
3. Clip berdiri sendiri tanpa percakapan sebelumnya.
4. Ending selesai; reaction komedi hanya dipertahankan jika menambah nilai.
5. Pilih campuran COMEDY, HOT_TAKE, PERSONAL_STORY jika kualitasnya sebanding.
6. Jangan promosikan lane lemah hanya demi variasi.
7. Jangan membuat candidate_id baru atau klaim yang tidak ada di context.
8. SENSOR kata kasar Indonesia pada hook/caption dengan tanda * di tengah kata.

# DURASI
- COMEDY ideal 20-40 detik, toleransi 5 detik.
- HOT_TAKE/PERSONAL_STORY ideal 35-60 detik, toleransi 5 detik.
- Maksimum output: {max_clips}

# OUTPUT
Return ONLY valid JSON:
{{
  "shorts": [{{
    "candidate_id": "existing-id",
    "start": 0.0,
    "end": 0.0,
    "judge_score": 0.0,
    "hook_score": 0.0,
    "payoff_score": 0.0,
    "standalone_score": 0.0,
    "ending_score": 0.0,
    "reasoning": "alasan dalam Bahasa Indonesia",
    "viral_pattern_type": "EMOTIONAL_PEAK",
    "viral_hook_text": "MAKSIMAL ENAM KATA",
    "social_caption": "caption akurat + hashtag relevan"
  }}]
}}

# CANDIDATES_WITH_LOCAL_CONTEXT
{candidates_json}
""".strip()


def _duration_allowed_for_lane(lane, duration):
    tolerance = PODCAST_COMEDY_DURATION_TOLERANCE
    if lane == "COMEDY":
        return 20.0 - tolerance <= duration <= 40.0 + tolerance
    return 35.0 - tolerance <= duration <= 60.0 + tolerance


def _rank_balanced_clips(
    clips, max_clips=10, balance_gap=PODCAST_COMEDY_BALANCE_GAP
):
    """Prefer strong lane diversity without promoting weak candidates."""
    ordered = sorted(
        clips, key=lambda item: _safe_float(item.get("confidence")), reverse=True
    )
    if not ordered:
        return []

    selected = [ordered[0]]
    selected_ids = {id(ordered[0])}
    best_score = _safe_float(ordered[0].get("confidence"))
    for lane in PODCAST_COMEDY_LANES:
        if lane == ordered[0].get("content_lane"):
            continue
        lane_candidate = next(
            (item for item in ordered if item.get("content_lane") == lane), None
        )
        if (
            lane_candidate
            and _safe_float(lane_candidate.get("confidence"))
            >= best_score - balance_gap
            and len(selected) < max_clips
        ):
            selected.append(lane_candidate)
            selected_ids.add(id(lane_candidate))

    for item in ordered:
        if id(item) not in selected_ids and len(selected) < max_clips:
            selected.append(item)
            selected_ids.add(id(item))
    return selected


def _validate_judge_output(
    judge_payload,
    candidates,
    max_clips=10,
    min_score=PODCAST_COMEDY_JUDGE_MIN_SCORE,
):
    """Validate judge selections against the immutable scout candidate set."""
    candidate_map = {item["candidate_id"]: item for item in candidates}
    validated = []
    seen_ids = set()
    for selected in (judge_payload or {}).get("shorts", []):
        candidate_id = _normalize_whitespace(selected.get("candidate_id"))
        candidate = candidate_map.get(candidate_id)
        if not candidate or candidate_id in seen_ids:
            continue

        start = _safe_float(selected.get("start"), candidate["start"])
        end = _safe_float(selected.get("end"), candidate["end"])
        duration = end - start
        context_start = candidate.get("context_start", candidate["start"])
        context_end = candidate.get("context_end", candidate["end"])
        if (
            start < context_start
            or end > context_end
            or start > candidate["payoff_start"]
            or end < candidate["payoff_end"]
            or duration < PODCAST_COMEDY_HARD_MIN_DURATION
            or duration > PODCAST_COMEDY_HARD_MAX_DURATION
            or not _duration_allowed_for_lane(candidate["content_lane"], duration)
        ):
            continue

        score = max(
            0.0,
            min(
                _safe_float(
                    selected.get("judge_score"), candidate.get("scout_score", 0.0)
                ),
                1.0,
            ),
        )
        if score < min_score:
            continue
        clip = dict(candidate)
        clip.update(selected)
        clip.update(
            {
                "candidate_id": candidate_id,
                "content_lane": candidate["content_lane"],
                "hook_evidence": candidate["hook_evidence"],
                "payoff_evidence": candidate["payoff_evidence"],
                "standalone_summary": candidate["standalone_summary"],
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": score,
                "score": score,
                "viral_hook_text": _normalize_hook_text(
                    selected.get("viral_hook_text") or candidate.get("suggested_hook")
                ),
                "social_caption": _normalize_whitespace(
                    selected.get("social_caption")
                    or candidate.get("standalone_summary")
                ),
                "reasoning": _normalize_whitespace(
                    selected.get("reasoning")
                    or (
                        f"Hook: {candidate['hook_evidence']}. "
                        f"Payoff: {candidate['payoff_evidence']}."
                    )
                ),
            }
        )
        validated.append(clip)
        seen_ids.add(candidate_id)
    return _rank_balanced_clips(validated, max_clips=max_clips)


def _get_podcast_prompt(
    transcript_text,
    video_duration,
    words_json="[]",
    scene_boundaries="No scene data",
    max_clips=10,
    min_duration=None,
    max_duration=None,
    min_score=None,
):
    """Full podcast prompt adapted from ai-repurposer, outputting standard Virlo JSON."""
    min_dur = min_duration or MIN_CLIP_DURATION
    max_dur = max_duration or MAX_CLIP_DURATION
    m_score = min_score or MIN_HIGHLIGHT_SCORE
    return f"""
# ROLE
You are a Viral Growth Hacker and Expert Podcast Editor. Your goal is to find "Gold Nuggets" in a transcript that will stop the scroll on TikTok, Reels, and Shorts.

# VIDEO METADATA
- Total Duration: {video_duration} seconds
- Maximum Clips: {max_clips} (return fewer if not enough high-quality moments exist)
- Clip Duration Range: {min_dur}-{max_dur} seconds (let the natural content dictate exact duration)
- Minimum Quality Score: {m_score} (only include clips scoring >= {m_score})

# SCENE_BOUNDARIES (align clip edges here when possible):
{scene_boundaries}

# INPUT TRANSCRIPT
{transcript_text}

# WORDS_JSON (word-level timestamps for precise timing):
{words_json}

# TASK
Analyze the transcript and identify ALL highly engaging highlights. You must look for segments where the retention will be highest. Return up to {max_clips} clips, ordered by score (highest first). Only include clips with score >= {m_score}.

# SELECTION CRITERIA (The "Viral" Framework)
1. **The "Wait, What?" Moment**: Something so unexpected or controversial it forces a replay.
2. **High-Value Insight**: A "lightbulb moment" where the listener learns something new.
3. **Emotional Peak**: Intense laughter, anger, sadness, or deep vulnerability.
4. **Standalone Value**: The clip must make sense and be impactful even if the viewer hasn't seen the whole podcast.

# QUALITY GATES
- START_QUALITY_CHECK: The first sentence must hook the viewer quickly.
- COMPLETE_THOUGHT: Include enough setup and finish after the payoff resolves.
- QUOTE_EVIDENCE: Cite transcript words inside the selected timestamp range.
- SELF-CONTAINED CHECK: Reject moments that require earlier conversation.

# VIRAL PATTERNS (pick the best one per clip)
1. CURIOSITY_GAP: Opens a compelling question the viewer MUST know the answer to
2. EMOTIONAL_PEAK: High emotion — anger, excitement, laughter, tears, intense passion
3. VALUE_DROP: Actionable tip, life hack, insight, or framework usable immediately
4. CONTROVERSY: Hot take, debate trigger, strong opinion that sparks comments
5. CLIFFHANGER: Unfinished story, looming reveal
6. STORY_BEAT: Clear setup, conflict, or resolution
7. PATTERN_INTERRUPT: Unexpected twist, surprise reveal, "wait, what?!" moment
8. RELATABLE_MOMENT: "That's so me" — universal experience with high shareability

# CUTTING RULES
- Start 0.2-0.4s BEFORE the hook, end 0.2-0.4s AFTER the payoff
- NEVER cut mid-word or mid-sentence — align to topic boundaries (period, question, natural pause)
- Align with scene boundaries when available
- Timestamps in absolute seconds, 3 decimal places (e.g., 12.340)
- 0 ≤ start < end ≤ VIDEO_DURATION

# ANTI-HALLUCINATION RULES (CRITICAL)
- NEVER fabricate timestamps not present in the transcript
- NEVER write a viral_hook_text that misrepresents what is actually spoken
- NEVER write a social_caption about a topic not discussed in the clip's timestamp range
- NEVER invent quotes or paraphrase content that doesn't exist in the transcript
- EVERY clip's viral_hook_text and social_caption MUST be directly derivable from the transcript text at that clip's timestamps
- If you cannot find enough genuinely engaging moments, return FEWER clips — do NOT pad with mediocre or fabricated content

# CONFIDENCE ASSESSMENT
After generating clips, assess the quality of your analysis:
1. **analysis_confidence** (0.0-1.0): How confident are you in identifying the best moments?
2. **needs_deep_analysis** (true/false): Flag true if content has nuanced insights or layered storytelling.
3. **complexity_reason** (string): Brief explanation if needs_deep_analysis is true.

# OUTPUT REQUIREMENTS (IN INDONESIAN)
For each clip, generate:
1. **Timestamps**: Precise start and end (decimal) matching actual transcript timestamps.
2. **Viral Hook**: MUST be provocative, grounded in what is actually said. MAX 6 WORDS, ALL CAPS.
3. **Score**: Float from {m_score} to 1.0. Only include truly engaging moments.
4. **Reasoning**: 2-3 sentences. MUST quote or reference the specific transcript content that makes this clip engaging.
5. **Social Caption**: Engaging caption + 3-5 hashtags. MUST accurately describe the actual clip content.

# STRICT JSON FORMAT
Return ONLY a valid JSON object. No conversational filler.

{{
  "content_type": "PODCAST",
  "analysis_confidence": float,
  "needs_deep_analysis": false,
  "complexity_reason": "",
  "shorts": [
    {{
      "start": float,
      "end": float,
      "score": float,
      "reasoning": "2-3 sentences quoting specific transcript content",
      "viral_pattern_type": "<one of the 8 patterns above>",
      "viral_hook_text": "MAX 6 WORDS ALL CAPS",
      "social_caption": "engaging caption + 3-5 hashtags"
    }}
  ]
}}

# CRITICAL RULES
- Each clip MUST be between {min_dur} and {max_dur} seconds.
- The "viral_hook_text" field MUST NOT exceed 6 words.
- NO mid-sentence cuts. Start and end on natural pauses.
- All text (except hashtags) MUST be in natural, engaging INDONESIAN.
- Ensure timestamps match the transcript markers exactly.
- **Grounding First**: Title and caption MUST reflect the actual spoken content at the clip's timestamps.
- **Quality Over Quantity**: Only include clips scoring >= {m_score}. Return fewer clips if not enough quality moments exist.
"""


def _get_podcast_comedy_prompt(
    transcript_text,
    video_duration,
    words_json="[]",
    scene_boundaries="No scene data",
    max_clips=10,
    min_duration=None,
    max_duration=None,
    min_score=None,
):
    """Full comedy podcast prompt adapted from ai-repurposer, outputting standard Virlo JSON."""
    min_dur = min_duration or MIN_CLIP_DURATION
    max_dur = max_duration or MAX_CLIP_DURATION
    m_score = min_score or MIN_HIGHLIGHT_SCORE
    return f"""
# ROLE
You are a Viral Comedy Scout. Your job is to find the absolute funniest "Comedic Beats" from this transcript that will go viral on TikTok and Reels.

# VIDEO METADATA
- Content Type: COMEDY PODCAST
- Total Duration: {video_duration} seconds
- Maximum Clips: {max_clips} (return fewer if not enough high-quality moments exist)
- Clip Duration Range: {min_dur}-{max_dur} seconds (let the natural content dictate exact duration)
- Minimum Quality Score: {m_score} (only include clips scoring >= {m_score})

# SCENE_BOUNDARIES (align clip edges here when possible):
{scene_boundaries}

# INPUT TRANSCRIPT
{transcript_text}

# WORDS_JSON (word-level timestamps for precise timing):
{words_json}

# TASK
Identify ALL hilarious moments. Return up to {max_clips} clips, ordered by score (highest first). Only include clips with score >= {m_score}.

# THE COMEDY BEAT FORMULA (CRITICAL)
For each clip, you MUST capture the full cycle:
1. **The Setup (Start 5-8s early)**: Give context so the joke makes sense.
2. **The Build**: The rising tension or banter.
3. **The Punchline**: The hilarious payoff.
4. **The Reaction (End 2-3s late)**: Include the laughter or shocked silence.

# QUALITY GATES
- START_QUALITY_CHECK: The first sentence or reaction must create curiosity quickly.
- COMPLETE_THOUGHT: Preserve the minimum setup, punchline, and valuable reaction.
- QUOTE_EVIDENCE: Cite transcript words for both setup/hook and payoff.
- SELF-CONTAINED CHECK: Reject inside jokes that need earlier conversation.

# VIRAL PATTERNS (pick the best one per clip)
1. CURIOSITY_GAP: Opens a compelling question the viewer MUST know the answer to
2. EMOTIONAL_PEAK: High emotion — anger, excitement, laughter, tears, intense passion
3. VALUE_DROP: Actionable tip, life hack, insight, or framework usable immediately
4. CONTROVERSY: Hot take, debate trigger, strong opinion that sparks comments
5. CLIFFHANGER: Unfinished story, looming reveal
6. STORY_BEAT: Clear setup, conflict, or resolution
7. PATTERN_INTERRUPT: Unexpected twist, surprise reveal, "wait, what?!" moment
8. RELATABLE_MOMENT: "That's so me" — universal experience with high shareability

# CUTTING RULES
- Start 0.2-0.4s BEFORE the hook, end 0.2-0.4s AFTER the payoff
- NEVER cut mid-word or mid-sentence — align to topic boundaries (period, question, natural pause)
- Align with scene boundaries when available
- Timestamps in absolute seconds, 3 decimal places (e.g., 12.340)
- 0 ≤ start < end ≤ VIDEO_DURATION

# PROFANITY FILTER (MANDATORY)
You MUST identify Indonesian curse words or sensitive slang (e.g., "anjing", "bangsat", "tolol", "goblok", "kontol", etc.) in the output (viral_hook_text, reasoning).
- SENSOR them using an asterisk (*) in the middle.
- Example: "ANJING" -> "ANJ*NG", "TOLOL" -> "T*LOL".

# ANTI-HALLUCINATION RULES (CRITICAL)
- NEVER fabricate timestamps not present in the transcript
- NEVER write a viral_hook_text that misrepresents what is actually spoken
- NEVER write a social_caption about a topic not discussed in the clip's timestamp range
- NEVER invent quotes or paraphrase content that doesn't exist in the transcript
- EVERY clip's viral_hook_text and social_caption MUST be directly derivable from the transcript text at that clip's timestamps
- If you cannot find enough genuinely funny moments, return FEWER clips — do NOT pad with mediocre or fabricated content

# CONFIDENCE ASSESSMENT
After generating clips, assess the quality of your analysis:
1. **analysis_confidence** (0.0-1.0): How confident are you in identifying the funniest moments?
2. **needs_deep_analysis** (true/false): Flag true if comedy relies on subtle timing, cultural references, or inside jokes.
3. **complexity_reason** (string): Brief explanation if needs_deep_analysis is true.

# OUTPUT REQUIREMENTS (IN INDONESIAN)
For each clip, generate:
1. **Timestamps**: Precise start and end (decimal) matching actual transcript timestamps.
2. **Viral Hook**: MUST be ALL CAPS, provocative, and MAX 6 WORDS. Do not spoil the joke. Grounded in what is actually said.
3. **Score**: Float from {m_score} to 1.0. Only include truly hilarious moments.
4. **Reasoning**: 2-3 sentences explaining the comedic timing. MUST quote or reference the specific transcript content.
5. **Social Caption**: Catchy caption + 5-8 comedy-focused hashtags. MUST accurately describe the actual clip content.

# STRICT JSON FORMAT
Return ONLY a valid JSON object. No conversational filler.

{{
  "content_type": "PODCAST_COMEDY",
  "analysis_confidence": float,
  "needs_deep_analysis": false,
  "complexity_reason": "",
  "shorts": [
    {{
      "start": float,
      "end": float,
      "score": float,
      "reasoning": "2-3 sentences quoting specific transcript content",
      "viral_pattern_type": "<one of the 8 patterns above>",
      "viral_hook_text": "MAX 6 WORDS ALL CAPS CENSORED",
      "social_caption": "catchy caption + 5-8 hashtags"
    }}
  ]
}}

# CRITICAL RULES
- **HOOK LIMIT**: Strictly MAX 6 WORDS for the "viral_hook_text" field.
- **ALL CAPS HOOK**: Use ALL CAPS for the "viral_hook_text" field.
- **CENSORSHIP**: If a funny moment involves a curse word, sensor it in the JSON output.
- **DURATION**: Each clip MUST be between {min_dur} and {max_dur} seconds.
- **COMEDY MARKERS**: Prioritize segments with "(laughs)", "hahaha", or rapid banter.
- **NO MID-WORD CUTS**: Always start and end on natural pauses or laughter aftermath.
- **Grounding First**: Title and caption MUST reflect the actual spoken content at the clip's timestamps.
- **Quality Over Quantity**: Only include clips scoring >= {m_score}. Return fewer clips if not enough quality moments exist.
"""


def _build_prompt(
    video_duration,
    language,
    scene_boundaries,
    transcript_text,
    words_json,
    category="general",
):
    """Build the full Gemini prompt by injecting category-specific instructions."""
    if category == "podcast":
        return _get_podcast_prompt(
            transcript_text, video_duration, words_json, scene_boundaries
        )
    elif category == "podcast_comedy":
        return _get_podcast_comedy_prompt(
            transcript_text, video_duration, words_json, scene_boundaries
        )

    cat_instructions = CATEGORY_INSTRUCTIONS.get(
        category, CATEGORY_INSTRUCTIONS["general"]
    )
    return GEMINI_BASE_PROMPT.format(
        video_duration=video_duration,
        language=language,
        scene_boundaries=scene_boundaries,
        category_instructions=cat_instructions,
        transcript_text=transcript_text,
        words_json=words_json,
    )


GEMINI_PROMPT_TEMPLATE = GEMINI_BASE_PROMPT

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO("yolov8n.pt")

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)


class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """

    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height

        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
            self.crop_width = video_width
            self.crop_height = int(self.crop_width / ASPECT_RATIO)

        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2

    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x

            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1

                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0  # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan

                self.current_center_x += direction * speed

                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (
                    direction == -1 and new_diff > 0
                ):
                    self.current_center_x = self.target_center_x

            # If inside safe zone, DO NOTHING (Stationary Camera)

        # Clamp center
        half_crop = self.crop_width / 2

        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop

        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)

        x1 = max(0, x1)
        x2 = min(self.video_width, x2)

        y1 = 0
        y2 = self.video_height

        return x1, y1, x2, y2


class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """

    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}  # {id: frame_number}
        self.locked_counter = 0  # How long we've been locked on current speaker

        # Hyperparameters
        self.stabilization_threshold = (
            stabilization_frames  # Frames needed to confirm a new speaker
        )
        self.switch_cooldown = cooldown_frames  # Minimum frames before switching again
        self.last_switch_frame = -1000

        # ID tracking
        self.next_id = 0
        self.known_faces = []  # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []

        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face["box"]
            center_x = x + w / 2

            best_match_id = -1
            min_dist = (
                width * 0.15
            )  # Reduced matching radius to avoid jumping in groups

            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if (
                    frame_number - kf["last_frame"] > 30
                ):  # Forgot faces older than 1s (was 2s)
                    continue

                dist = abs(center_x - kf["center"])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf["id"]

            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1

            # Update known face
            self.known_faces = [
                kf for kf in self.known_faces if kf["id"] != best_match_id
            ]
            self.known_faces.append(
                {"id": best_match_id, "center": center_x, "last_frame": frame_number}
            )

            current_candidates.append(
                {"id": best_match_id, "box": face["box"], "score": face["score"]}
            )

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85  # Faster decay (was 0.9)
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand["id"]
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand["score"] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None

        best_candidate = None
        max_score = -1

        for cand in current_candidates:
            pid = cand["id"]
            total_score = self.speaker_scores.get(pid, 0)

            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0  # Sticky factor

            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate["id"]

            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate["box"]

            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next(
                    (
                        c
                        for c in current_candidates
                        if c["id"] == self.active_speaker_id
                    ),
                    None,
                )
                if old_cand:
                    return old_cand["box"]

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate["box"]

        return None


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    candidates = []

    if not results.detections:
        return []

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)

        candidates.append(
            {
                "box": [x, y, w, h],
                "score": w * h,  # Area as score
            }
        )

    return candidates


def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0])  # class 0 is person

    if not results:
        return None

    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]

    return best_box


def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame:
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]

    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(
        frame, (bg_w, output_height), interpolation=cv2.INTER_LANCZOS4
    )

    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0:
        start_x = 0
    background = bg_resized[:, start_x : start_x + output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(
            background, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4
        )

    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)

    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(
        frame, (output_width, fg_h), interpolation=cv2.INTER_LANCZOS4
    )

    # 3. Overlay
    y_offset = (output_height - fg_h) // 2

    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset : y_offset + fg_h, :] = foreground

    return final_frame


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ["TRACK"] * len(scenes)

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5,
        ]

        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))

        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)

        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append("GENERAL")
        else:
            strategies.append("TRACK")

    cap.release()
    return strategies


def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[<>:"/\\|?*#]', "", filename)
    filename = filename.replace(" ", "_")
    return filename[:100]


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = "/app/cookies.txt"
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print(
            "🍪 Found YOUTUBE_COOKIES env var, creating cookies file inside container..."
        )
        try:
            with open(cookies_path, "w") as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                print(
                    f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes"
                )
                with open(cookies_path, "r") as f:
                    content = f.read(100)
                    print(f"   Debug: First 100 chars of cookie file: {content}")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")

    # Common yt-dlp options to work around YouTube bot detection.
    # extractor_args tries multiple player clients in order; tv_embed / android
    # avoid the OAuth/PO-token checks that block server IPs.
    _COMMON_YDL_OPTS = {
        "quiet": False,
        "verbose": True,
        "no_warnings": False,
        "cookiefile": cookies_path if cookies_path else None,
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "nocheckcertificate": True,
        "cachedir": False,
        "extractor_args": {
            "youtube": {
                "player_client": ["tv_embed", "android", "mweb", "web"],
                "player_skip": ["webpage", "configs"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    }

    with yt_dlp.YoutubeDL(_COMMON_YDL_OPTS) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", "youtube_video")
            sanitized_title = sanitize_filename(video_title)
        except Exception as e:
            # Force print to stderr/stdout immediately so it's captured before crash
            import sys
            import traceback

            # Print minimal error first to ensure something gets out
            print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)

            error_msg = f"""
            
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌
            
REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(e)}
            """
            # Print to both streams to ensure capture
            print(error_msg, file=sys.stdout)
            print(error_msg, file=sys.stderr)

            # Force flush
            sys.stdout.flush()
            sys.stderr.flush()

            # Wait a split second to allow buffer to drain before raising
            time.sleep(0.5)

            raise e

    output_template = os.path.join(output_dir, f"{sanitized_title}.%(ext)s")
    expected_file = os.path.join(output_dir, f"{sanitized_title}.mp4")
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print(f"🗑️  Removed existing file to re-download with H.264 codec")

    ydl_opts = {
        **_COMMON_YDL_OPTS,
        "format": "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "overwrites": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    downloaded_file = os.path.join(output_dir, f"{sanitized_title}.mp4")

    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith(".mp4"):
                downloaded_file = os.path.join(output_dir, f)
                break

    step_end_time = time.time()
    print(
        f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}"
    )

    return downloaded_file, sanitized_title


def _get_video_dims_ffprobe(video_path):
    """Get video width, height, and fps using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(result.stdout)
        streams = probe.get("streams", [])
        if streams:
            s = streams[0]
            w = s.get("width", 1920)
            h = s.get("height", 1080)
            fps_str = s.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            return w, h, fps
    except Exception:
        pass
    return 1920, 1080, 30.0


def process_video_blur_bars(
    input_video, output_video, blur_strength=15, brightness=0.7, saturation=0.8
):
    """
    Convert any video to 9:16 using blur bars.
    Preserves full content — no cropping. Original is scaled to fit and
    centered on a dimmed, desaturated blurred background.
    Single FFmpeg pass. Much faster and sharper than TRACK upscaling.
    """
    print(f"\n🎬 Blur Bars: {input_video}")

    w, h, fps = _get_video_dims_ffprobe(input_video)
    canvas_w, canvas_h = 1080, 1920
    input_aspect = w / h if h else 1.0
    canvas_aspect = canvas_w / canvas_h

    # Already 9:16? Stream-copy or light re-encode.
    if abs(input_aspect - canvas_aspect) < 0.02:
        print("   📐 Video already 9:16 — encoding pass only.")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_video,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_video,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"   ✅ Blur Bars complete: {output_video}")
        return True

    # Calculate foreground dimensions
    if input_aspect > canvas_aspect:
        # Wider than 9:16 (horizontal) — fit to width
        video_w = canvas_w
        video_h = int(canvas_w / input_aspect)
        print(f"   📐 Horizontal ({w}x{h}) → fit to width: {video_w}x{video_h}")
    else:
        # Taller than 9:16 — fit to height
        video_w = int(canvas_h * input_aspect)
        video_h = canvas_h
        print(f"   📐 Vertical/tall ({w}x{h}) → fit to height: {video_w}x{video_h}")

    # Build filter_complex
    blur_radius = blur_strength // 2
    blur_power = 2
    eq_brightness = brightness - 1.0  # 0.7 → -0.3 (darken background)

    filter_complex = (
        f"[0:v]split=2[original][forblur];"
        f"[forblur]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
        f"crop={canvas_w}:{canvas_h},"
        f"boxblur=luma_radius={blur_radius}:luma_power={blur_power}:chroma_radius={blur_radius}:chroma_power={blur_power},"
        f"eq=brightness={eq_brightness}:saturation={saturation}[blurred];"
        f"[original]scale={video_w}:{video_h}[scaled];"
        f"[blurred][scaled]overlay=(W-w)/2:(H-h)/2[out]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        output_video,
    ]

    print("   🎨 Applying blur bars...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ FFmpeg failed:\n{result.stderr}")
        return False

    print(f"   ✅ Blur Bars complete: {output_video}")
    return True


def process_video_to_vertical(input_video, final_output_video, crop_style="blur_bars"):
    """
    Core logic to convert horizontal video to vertical.

    crop_style:
      - "blur_bars" (default): Full content preserved on blurred background. Sharp output.
      - "auto": Scene-based AI analysis → TRACK (face-crop) or GENERAL (OpenCV blur bars).
    """
    if crop_style == "blur_bars":
        return process_video_blur_bars(input_video, final_output_video)

    # ---- "auto" mode: existing TRACK / GENERAL pipeline ----
    script_start_time = time.time()

    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output):
        os.remove(temp_video_output)
    if os.path.exists(temp_audio_output):
        os.remove(temp_audio_output)
    if os.path.exists(final_output_video):
        os.remove(final_output_video)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)

    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode

        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)

    # Fixed HD 9:16 output resolution
    OUTPUT_HEIGHT = 1920
    OUTPUT_WIDTH = 1080

    # Initialize Cameraman
    cameraman = SmoothedCameraman(
        OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height
    )

    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes

    print("\n   ✂️ Step 4: Processing video frames...")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "15",
        "-pix_fmt",
        "yuv420p",
        "-an",
        temp_video_output,
    ]

    ffmpeg_process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    current_scene_index = 0

    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if (
                    frame_number >= end_f
                    and current_scene_index < len(scene_boundaries) - 1
                ):
                    current_scene_index += 1

            # Determine Strategy for current frame based on scene
            current_strategy = (
                scene_strategies[current_scene_index]
                if current_scene_index < len(scene_strategies)
                else "TRACK"
            )

            # Apply Strategy
            if current_strategy == "GENERAL":
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)

                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2

            else:
                # "Single Speaker" -> Track & Crop

                # Detect every 2nd frame for performance
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(
                        candidates, frame_number, original_width
                    )
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (
                    frame_number == scene_boundaries[current_scene_index][0]
                )

                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(
                        cropped,
                        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                else:
                    output_frame = cv2.resize(
                        frame,
                        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)

    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    audio_extract_command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vn",
        "-acodec",
        "copy",
        temp_audio_output,
    ]
    try:
        subprocess.run(
            audio_extract_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print(
            "\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio."
        )
        pass

    print("\n   ✨ Step 6: Merging...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_output,
            "-i",
            temp_audio_output,
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            final_output_video,
        ]
    else:
        merge_command = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_output,
            "-c:v",
            "copy",
            final_output_video,
        ]

    try:
        subprocess.run(
            merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output):
        os.remove(temp_video_output)
    if os.path.exists(temp_audio_output):
        os.remove(temp_audio_output)

    return True


def transcribe_video(video_path, method="faster-whisper"):
    """
    Transcribe video audio to text.

    Args:
        video_path: Path to the video file
        method: Transcription method - "faster-whisper" or "groq"
    """
    if method == "groq":
        return transcribe_with_groq(video_path)
    else:
        return transcribe_with_faster_whisper(video_path)


def split_audio_into_chunks(audio_path, chunk_duration_sec=600):
    """
    Splits an audio file into smaller chunks using FFmpeg.
    Returns a list of paths to the created chunks.
    """
    print(f"   ✂️  Splitting audio into {chunk_duration_sec}s chunks...")
    chunks = []
    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    # Use ffmpeg segment muxer for efficient splitting without re-encoding
    # we use -f segment to split into equal durations
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_duration_sec),
        "-c",
        "copy",
        os.path.join(base_dir, f"{base_name}_chunk_%03d.wav"),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Find all created chunks
    for f in sorted(os.listdir(base_dir)):
        if f.startswith(f"{base_name}_chunk_") and f.endswith(".wav"):
            chunks.append(os.path.join(base_dir, f))

    return chunks


def transcribe_with_groq(video_path):
    """Transcribe video using Groq API (faster, cloud-based)."""
    print("🎙️  Transcribing with Groq Whisper...")
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables.")
        return None

    client = Groq(api_key=api_key)

    # Extract audio from video first
    temp_audio = "/tmp/temp_audio.wav"
    extract_audio_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_audio,
    ]
    subprocess.run(extract_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    try:
        # Handle file size limit (25MB for free tier)
        file_size_mb = os.path.getsize(temp_audio) / (1024 * 1024)

        if file_size_mb > 20:  # Use 20MB as a safe threshold
            print(
                f"   ⚠️  Audio file size ({file_size_mb:.2f}MB) exceeds Groq free tier limit. Chunking..."
            )
            chunks = split_audio_into_chunks(temp_audio)

            all_segments = []
            all_words = []
            full_text_parts = []
            cumulative_offset = 0.0
            detected_language = "unknown"

            for i, chunk_path in enumerate(chunks):
                print(
                    f"   Processing chunk {i + 1}/{len(chunks)} ({os.path.basename(chunk_path)})..."
                )
                with open(chunk_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(chunk_path), f.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )

                if isinstance(transcription, dict):
                    text = transcription.get("text", "")
                    language = transcription.get("language", "unknown")
                    segments_data = transcription.get("segments", [])
                    words_data = transcription.get("words", [])
                else:
                    text = transcription.text
                    language = transcription.language
                    segments_data = transcription.segments or []
                    words_data = getattr(transcription, "words", []) or []

                if i == 0:
                    detected_language = language

                full_text_parts.append(text)

                # Offset timestamps for segments
                for seg in segments_data:
                    if isinstance(seg, dict):
                        s_start, s_end, s_text = (
                            seg.get("start", 0),
                            seg.get("end", 0),
                            seg.get("text", ""),
                        )
                    else:
                        s_start, s_end, s_text = seg.start, seg.end, seg.text

                    all_segments.append(
                        {
                            "text": s_text,
                            "start": s_start + cumulative_offset,
                            "end": s_end + cumulative_offset,
                            "words": [],
                        }
                    )

                # Offset timestamps for words
                for word in words_data:
                    if isinstance(word, dict):
                        w_text, w_start, w_end, w_prob = (
                            word.get("word", ""),
                            word.get("start", 0),
                            word.get("end", 0),
                            word.get("probability", 1.0),
                        )
                    else:
                        w_text, w_start, w_end, w_prob = (
                            word.word,
                            word.start,
                            word.end,
                            getattr(word, "probability", 1.0),
                        )

                    all_words.append(
                        {
                            "word": w_text,
                            "start": w_start + cumulative_offset,
                            "end": w_end + cumulative_offset,
                            "probability": w_prob,
                        }
                    )

                # Accurate chunk duration for offset
                if segments_data:
                    last_seg = segments_data[-1]
                    chunk_duration = (
                        last_seg.get("end", 600)
                        if isinstance(last_seg, dict)
                        else last_seg.end
                    )
                else:
                    chunk_duration = 600

                cumulative_offset += chunk_duration
                os.remove(chunk_path)

            # Map offset words to offset segments
            for seg in all_segments:
                for word in all_words:
                    if seg["start"] <= word["start"] < seg["end"]:
                        seg["words"].append(word)

            full_text = " ".join(full_text_parts)
            transcript_segments = all_segments
            language = detected_language

        else:
            # Single-request logic
            with open(temp_audio, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(temp_audio), file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                )

            if isinstance(transcription, dict):
                text = transcription.get("text", "")
                language = transcription.get("language", "unknown")
                segments_data = transcription.get("segments", [])
                words_data = transcription.get("words", [])
            else:
                text = transcription.text
                language = transcription.language
                segments_data = transcription.segments or []
                words_data = getattr(transcription, "words", []) or []

            full_text = text
            transcript_segments = []
            for segment in segments_data:
                if isinstance(segment, dict):
                    s_text, s_start, s_end = (
                        segment.get("text", ""),
                        segment.get("start", 0),
                        segment.get("end", 0),
                    )
                else:
                    s_text, s_start, s_end = segment.text, segment.start, segment.end

                seg_dict = {"text": s_text, "start": s_start, "end": s_end, "words": []}
                if words_data:
                    for word in words_data:
                        if isinstance(word, dict):
                            w_text, w_start, w_end, w_prob = (
                                word.get("word", ""),
                                word.get("start", 0),
                                word.get("end", 0),
                                word.get("probability", 1.0),
                            )
                        else:
                            w_text, w_start, w_end, w_prob = (
                                word.word,
                                word.start,
                                word.end,
                                getattr(word, "probability", 1.0),
                            )
                        if s_start <= w_start < s_end:
                            seg_dict["words"].append(
                                {
                                    "word": w_text,
                                    "start": w_start,
                                    "end": w_end,
                                    "probability": w_prob,
                                }
                            )
                transcript_segments.append(seg_dict)

        print(f"   Detected language: {language}")
        for seg in transcript_segments:
            print(f"   [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        return {
            "text": full_text.strip(),
            "segments": transcript_segments,
            "language": language,
        }

    except Exception as e:
        print(f"   ❌ Groq transcription failed: {e}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return None


def transcribe_with_faster_whisper(video_path):
    """Transcribe video using Faster-Whisper (local, CPU-optimized)."""
    print("🎙️  Transcribing video with Faster-Whisper...")
    from faster_whisper import WhisperModel
    import torch

    # Auto-detect CUDA for faster transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"   Using device: {device} ({compute_type})")

    model = WhisperModel("large-v3", device=device, compute_type=compute_type)

    segments, info = model.transcribe(video_path, word_timestamps=True)

    print(
        f"   Detected language '{info.language}' with probability {info.language_probability:.2f}"
    )

    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""

    for segment in segments:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        seg_dict = {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": [],
        }

        if segment.words:
            for word in segment.words:
                seg_dict["words"].append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )

        transcript_segments.append(seg_dict)
        full_text += segment.text + " "

    return {
        "text": full_text.strip(),
        "segments": transcript_segments,
        "language": info.language,
    }


CONFIDENCE_TIER_MAP = {
    "EMAS": 0.9,
    "PERAK": 0.75,
    "PERUNGGU": 0.5,
    "GOLD": 0.9,
    "SILVER": 0.75,
    "BRONZE": 0.5,
}


def _normalize_confidence(value):
    """Convert tier label (EMAS/PERAK/PERUNGGU) or float to numeric score."""
    if isinstance(value, str):
        return CONFIDENCE_TIER_MAP.get(value.strip().upper(), 0.5)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.5


def _min_confidence_score(min_confidence):
    """Resolve a min_confidence arg that may be a float or a tier label."""
    if isinstance(min_confidence, str):
        return CONFIDENCE_TIER_MAP.get(min_confidence.strip().upper(), 0.5)
    return float(min_confidence)


def post_process_clips(result_json, min_confidence=0.5, max_overlap=0.7, max_clips=10):
    """Filter, deduplicate, and rank clips by quality, duration, and diversity."""
    shorts = result_json.get("shorts") or result_json.get("highlights", [])
    if not shorts:
        return result_json

    # Normalize score → confidence for downstream consumers
    for s in shorts:
        if "score" in s and "confidence" not in s:
            s["confidence"] = s["score"]

    # Normalize all confidence values to numeric for consistent filtering/sorting
    for s in shorts:
        s["confidence"] = _normalize_confidence(s.get("confidence"))

    threshold = _min_confidence_score(min_confidence)

    # 0. Filter by duration — enforce absolute floor and ceiling
    MIN_DURATION = 8
    MAX_DURATION = 90
    duration_filtered = []
    for s in shorts:
        dur = s["end"] - s["start"]
        if dur < MIN_DURATION:
            print(
                f"   ⚠️ Dropping clip at {s['start']:.1f}s: {dur:.1f}s too short (min {MIN_DURATION}s)"
            )
            continue
        if dur > MAX_DURATION:
            print(
                f"   ⚠️ Dropping clip at {s['start']:.1f}s: {dur:.1f}s too long (max {MAX_DURATION}s)"
            )
            continue
        duration_filtered.append(s)
    if not duration_filtered:
        print(f"   ⚠️ All clips outside duration bounds, keeping originals")
        duration_filtered = shorts

    # 1. Filter by confidence (keep low-confidence as fallback)
    has_confidence = any("confidence" in s for s in duration_filtered)
    if has_confidence:
        filtered = [s for s in duration_filtered if s.get("confidence", 0) >= threshold]
        if not filtered:
            print(f"   ⚠️ All clips below confidence {min_confidence}, keeping top 3")
            filtered = sorted(
                duration_filtered, key=lambda s: s.get("confidence", 0), reverse=True
            )[:3]
    else:
        filtered = duration_filtered

    # 2. Deduplicate overlapping clips (>max_overlap overlap)
    deduped = []
    filtered.sort(key=lambda s: s["start"])
    for clip in filtered:
        is_duplicate = False
        for i, kept in enumerate(deduped):
            overlap_start = max(clip["start"], kept["start"])
            overlap_end = min(clip["end"], kept["end"])
            overlap_dur = max(0, overlap_end - overlap_start)
            clip_dur = clip["end"] - clip["start"]
            kept_dur = kept["end"] - kept["start"]
            min_dur = min(clip_dur, kept_dur)
            if min_dur > 0 and overlap_dur / min_dur > max_overlap:
                is_duplicate = True
                if clip.get("confidence", 0) > kept.get("confidence", 0):
                    deduped[i] = clip
                break
        if not is_duplicate:
            deduped.append(clip)

    # 3. Enforce diversity — keep first occurrence of each pattern type, then fill with high-confidence remainder
    seen_types = set()
    diversified = []
    for clip in deduped:
        ptype = clip.get("viral_pattern_type", "")
        if ptype and ptype not in seen_types:
            seen_types.add(ptype)
            diversified.append(clip)
    for clip in deduped:
        if clip not in diversified:
            diversified.append(clip)

    # 4. Sort by confidence desc, limit to max_clips
    diversified.sort(key=lambda s: s.get("confidence", 0), reverse=True)
    result_json["shorts"] = diversified[:max_clips]

    # 5. Backward-compat shim: synthesize legacy field names from new tier-based schema
    # so downstream consumers (ResultCard, ScheduleWeekModal, S3, app.py) keep working.
    for clip in result_json["shorts"]:
        hook = _normalize_hook_text(clip.get("viral_hook_text"))
        caption = _normalize_whitespace(clip.get("social_caption"))
        clip["viral_hook_text"] = hook
        clip["social_caption"] = caption
        if not _normalize_whitespace(clip.get("video_title_for_youtube_short")):
            clip["video_title_for_youtube_short"] = hook[:100]
        if not _normalize_whitespace(clip.get("video_description_for_tiktok")):
            clip["video_description_for_tiktok"] = caption
        if not _normalize_whitespace(clip.get("video_description_for_instagram")):
            clip["video_description_for_instagram"] = caption
        # Convert numeric confidence back to tier label for any consumer that reads it as string
        if not isinstance(clip.get("confidence_label"), str):
            score = clip.get("confidence", 0)
            if score >= 0.85:
                clip["confidence_label"] = "EMAS"
            elif score >= 0.7:
                clip["confidence_label"] = "PERAK"
            else:
                clip["confidence_label"] = "PERUNGGU"

    filtered_out = len(shorts) - len(result_json["shorts"])
    if filtered_out > 0:
        print(
            f"   🔎 Post-process filtered {filtered_out} clip(s) (duration/confidence/dedup/diversity)"
        )

    return result_json


def snap_clip_to_boundaries(
    clip, words, scene_boundaries, snap_padding=0.3, video_duration=None
):
    """Snap clip start/end to nearest clean word boundary or scene boundary."""
    if not words:
        return

    start = clip["start"]
    end = clip["end"]

    # Build flat list of scene boundary points (just the seconds values)
    scene_points = []
    for s_start, s_end in scene_boundaries:
        scene_points.append(s_start)
        scene_points.append(s_end)
    scene_points.sort()

    def _prefer_scene_boundary(value):
        nearby = [
            point for point in scene_points if abs(point - value) <= snap_padding
        ]
        if nearby:
            return min(nearby, key=lambda point: abs(point - value))
        return value

    def _snap_start(t):
        # If the proposed cut lands inside speech, expand before the whole word.
        for w in words:
            if w["s"] <= t <= w["e"]:
                return _prefer_scene_boundary(w["s"] - snap_padding)

        # Find nearest word boundary: previous word end or next word start
        prev_end = None
        next_start = None
        for w in words:
            if w["e"] <= t:
                prev_end = w["e"]
            if w["s"] >= t and next_start is None:
                next_start = w["s"]
                break

        candidates = []
        if prev_end is not None:
            candidates.append(prev_end + snap_padding)
        if next_start is not None:
            candidates.append(next_start - snap_padding)

        if not candidates:
            return t

        snapped = min(candidates, key=lambda x: abs(x - t))

        return _prefer_scene_boundary(snapped)

    def _snap_end(t):
        # If the proposed cut lands inside speech, retain the whole final word.
        for w in words:
            if w["s"] <= t <= w["e"]:
                return w["e"]

        # Find nearest word boundary: previous word end or next word start
        prev_end = None
        next_start = None
        for w in words:
            if w["e"] <= t:
                prev_end = w["e"]
            if w["s"] >= t and next_start is None:
                next_start = w["s"]
                break

        candidates = []
        if prev_end is not None:
            candidates.append(prev_end + snap_padding)
        if next_start is not None:
            candidates.append(next_start - snap_padding)

        if not candidates:
            return t

        snapped = min(candidates, key=lambda x: abs(x - t))

        return _prefer_scene_boundary(snapped)

    new_start = _snap_start(start)
    new_end = _snap_end(end)

    # Clamp to valid range
    new_start = max(0.0, new_start)
    if video_duration:
        new_end = min(video_duration, new_end)

    # Ensure valid range: start < end and minimum duration
    MIN_DURATION = 8
    if new_start >= new_end:
        return  # Keep original if snapping produces invalid range
    if new_end - new_start < MIN_DURATION:
        return  # Keep original if too short after snapping

    clip["start"] = round(new_start, 3)
    clip["end"] = round(new_end, 3)


def _build_window_prompt(
    words,
    transcript_segments,
    video_duration,
    language,
    scene_boundaries,
    window_num,
    total_windows,
    category="general",
):
    """Build a reduced prompt for a window of words."""
    if not words:
        return None, None

    window_start = words[0]["s"]
    window_end = words[-1]["e"]

    # Include only segments that overlap this bounded word window.
    window_segments = [
        seg
        for seg in transcript_segments
        if seg["end"] >= window_start and seg["start"] <= window_end
    ]
    if window_segments:
        window_text = " ".join(seg["text"] for seg in window_segments)
    else:
        window_text = "(no transcript in this window)"

    # Filter scene boundaries in this window
    if scene_boundaries:
        window_scene_parts = []
        for s_start, s_end in scene_boundaries:
            if s_end >= window_start and s_start <= window_end:
                window_scene_parts.append(f"[{s_start:.1f}s - {s_end:.1f}s]")
        window_scene_text = (
            ", ".join(window_scene_parts)
            if window_scene_parts
            else "No scene data in this window"
        )
    else:
        window_scene_text = "No scene data available"

    prompt = _build_prompt(
        video_duration=video_duration,
        language=language,
        scene_boundaries=f"WINDOW {window_num + 1}/{total_windows} [{window_start:.1f}s - {window_end:.1f}s]: {window_scene_text}",
        transcript_text=json.dumps(window_text),
        words_json=json.dumps(words),
        category=category,
    )

    return prompt, window_text


MAX_PROMPT_CHARS = 380_0000  # ~100K tokens, safe budget per chunk


def _chunk_timestamped_segments(timestamped_segments, max_chars=120_000):
    """Split timestamped transcript into bounded prompt chunks."""
    if not timestamped_segments:
        return []

    chunks = []
    current = []
    current_chars = 0
    for segment in timestamped_segments:
        segment_chars = len(json.dumps(segment, ensure_ascii=False))
        if current and current_chars + segment_chars > max_chars:
            chunks.append(current)
            current = current[-2:]
            current_chars = sum(
                len(json.dumps(item, ensure_ascii=False)) for item in current
            )
        current.append(segment)
        current_chars += segment_chars
    if current:
        chunks.append(current)
    return chunks


def _scout_candidates_to_clips(candidates, max_clips=10):
    """Create grounded fallback clips when the judge pass is unavailable."""
    fallback_payload = {
        "shorts": [
            {
                "candidate_id": candidate["candidate_id"],
                "start": candidate["start"],
                "end": candidate["end"],
                "judge_score": candidate["scout_score"],
                "reasoning": (
                    f"Hook: {candidate['hook_evidence']}. "
                    f"Payoff: {candidate['payoff_evidence']}."
                ),
                "viral_pattern_type": (
                    "EMOTIONAL_PEAK"
                    if candidate["content_lane"] == "COMEDY"
                    else "CONTROVERSY"
                    if candidate["content_lane"] == "HOT_TAKE"
                    else "STORY_BEAT"
                ),
                "viral_hook_text": candidate.get("suggested_hook"),
                "social_caption": candidate.get("standalone_summary"),
            }
            for candidate in candidates
        ]
    }
    return _validate_judge_output(
        fallback_payload, candidates, max_clips=max_clips, min_score=0.0
    )


def _format_lane_counts(clips):
    counts = {
        lane: sum(1 for clip in clips if clip.get("content_lane") == lane)
        for lane in PODCAST_COMEDY_LANES
    }
    return ", ".join(f"{lane}={count}" for lane, count in counts.items())


def _run_podcast_comedy_multipass(
    timestamped_segments,
    video_duration,
    analyze_prompt,
    max_clips=10,
    max_candidates=PODCAST_COMEDY_MAX_CANDIDATES,
):
    """Run scout, global judge, and deterministic fallback for podcast comedy."""
    max_scout_chars = _safe_int(
        os.getenv("PODCAST_COMEDY_SCOUT_MAX_CHARS"), 120_000
    )
    scout_temperature = _safe_float(
        os.getenv("PODCAST_COMEDY_SCOUT_TEMPERATURE"), 0.8
    )
    judge_temperature = _safe_float(
        os.getenv("PODCAST_COMEDY_JUDGE_TEMPERATURE"), 0.2
    )
    judge_min_score = _safe_float(
        os.getenv("PODCAST_COMEDY_JUDGE_MIN_SCORE"),
        PODCAST_COMEDY_JUDGE_MIN_SCORE,
    )
    windows = _chunk_timestamped_segments(
        timestamped_segments, max_chars=max_scout_chars
    )
    window_label = "window" if len(windows) == 1 else "windows"
    print(
        f"   🎭 Multi-pass started: {len(timestamped_segments)} transcript segments, "
        f"{len(windows)} scout {window_label}",
        flush=True,
    )
    raw_candidates = []
    used_candidate_ids = set()
    for window_index, window_segments in enumerate(windows, start=1):
        window_start = window_segments[0]["start"]
        window_end = window_segments[-1]["end"]
        print(
            f"   🔎 Scout {window_index}/{len(windows)} started: "
            f"{len(window_segments)} segments ({window_start:.1f}s-{window_end:.1f}s)",
            flush=True,
        )
        prompt = _build_podcast_comedy_scout_prompt(
            window_segments,
            video_duration=video_duration,
            max_candidates=max_candidates,
            window_num=window_index,
            total_windows=len(windows),
        )
        payload = analyze_prompt(prompt, scout_temperature) or {}
        window_candidates = payload.get("candidates") or payload.get("shorts") or []
        print(
            f"   ✅ Scout {window_index}/{len(windows)} completed: "
            f"{len(window_candidates)} raw candidates",
            flush=True,
        )
        for candidate_index, candidate in enumerate(window_candidates, start=1):
            candidate = dict(candidate)
            candidate_id = _normalize_whitespace(candidate.get("candidate_id"))
            if not candidate_id:
                candidate_id = f"w{window_index}-c{candidate_index}"
            elif candidate_id in used_candidate_ids:
                candidate_id = f"w{window_index}-{candidate_id}"
            suffix = 2
            unique_id = candidate_id
            while unique_id in used_candidate_ids:
                unique_id = f"{candidate_id}-{suffix}"
                suffix += 1
            candidate["candidate_id"] = unique_id
            used_candidate_ids.add(unique_id)
            raw_candidates.append(candidate)

    candidates = _consolidate_scout_candidates(
        raw_candidates,
        video_duration=video_duration,
        max_candidates=max_candidates,
    )
    print(
        f"   🧹 Candidate validation completed: {len(raw_candidates)} raw -> "
        f"{len(candidates)} valid",
        flush=True,
    )
    contextualized = _attach_candidate_context(
        candidates, timestamped_segments, video_duration=video_duration
    )
    contextualized = [
        candidate
        for candidate in contextualized
        if candidate.get("transcript_context")
    ]
    if not contextualized:
        print("   ⚠️ Multi-pass stopped: no valid candidates with transcript context")
        return {"content_type": "PODCAST_COMEDY", "shorts": []}

    print(
        f"   🧑‍⚖️ Judge started: reviewing {len(contextualized)} candidates",
        flush=True,
    )
    judge_prompt = _build_podcast_comedy_judge_prompt(
        contextualized, max_clips=max_clips
    )
    try:
        judge_payload = analyze_prompt(judge_prompt, judge_temperature)
        judged = _validate_judge_output(
            judge_payload,
            contextualized,
            max_clips=max_clips,
            min_score=judge_min_score,
        )
        if judged:
            print(
                f"   ✅ Judge completed: {len(judged)} clips selected "
                f"({_format_lane_counts(judged)})",
                flush=True,
            )
            return {"content_type": "PODCAST_COMEDY", "shorts": judged}
        print("   ⚠️ Judge returned no valid clips; using scout fallback")
    except Exception as judge_error:
        print(f"   ⚠️ Judge pass failed; using scout fallback: {judge_error}")

    fallback = _scout_candidates_to_clips(
        contextualized, max_clips=max_clips
    )
    print(
        f"   ✅ Scout fallback completed: {len(fallback)} clips selected "
        f"({_format_lane_counts(fallback)})",
        flush=True,
    )
    return {"content_type": "PODCAST_COMEDY", "shorts": fallback}


def get_viral_clips(
    transcript_result, video_duration, scene_boundaries=None, category="general"
):
    print(f"🤖  Analyzing with AI (category={category})...")

    if not transcript_result:
        print("❌ Error: No transcript available. Skipping viral clip analysis.")
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
        ),
    )
    model_name = os.getenv("AI_MODEL") or os.getenv(
        "GEMINI_MODEL", "gemini-3.5-flash"
    )

    print(f"🤖  Initializing model: {model_name}")

    words = []
    for segment in transcript_result["segments"]:
        for word in segment.get("words", []):
            words.append(
                {
                    "w": word["word"],
                    "s": word["start"],
                    "e": word["end"],
                    "p": round(word.get("probability", 1.0), 3),
                }
            )

    language = transcript_result.get("language", "unknown")

    # Format scene boundaries for full-video prompt
    if scene_boundaries:
        scene_parts = []
        for s_start, s_end in scene_boundaries:
            scene_parts.append(f"[{s_start:.1f}s - {s_end:.1f}s]")
        scene_text = ", ".join(scene_parts)
    else:
        scene_text = "No scene data available — use transcript context to find natural cut points"

    from utils import extract_json

    def _call_llm(prompt_content, extra_system_prompt=None, temperature=1.0):
        messages = []
        if extra_system_prompt:
            messages.append({"role": "system", "content": extra_system_prompt})
        messages.append({"role": "user", "content": prompt_content})
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )

    def _analyze_single(prompt_content, temperature=1.0):
        response = _call_llm(prompt_content, temperature=temperature)
        text = response.choices[0].message.content
        result = extract_json(text)
        if result is None:
            print("   ⚠️ First attempt returned non-JSON output, retrying...")
            try:
                response = _call_llm(
                    prompt_content,
                    "CRITICAL: Return ONLY a single valid JSON object. Start your response with '{' and end with '}'. No markdown, no code fences, no explanatory text.",
                    temperature=temperature,
                )
                text = response.choices[0].message.content
                result = extract_json(text)
            except Exception as retry_err:
                print(f"   ❌ Retry also failed: {retry_err}")
        return result, response

    try:
        multipass_enabled = os.getenv(
            "PODCAST_COMEDY_MULTIPASS", "true"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if category == "podcast_comedy" and multipass_enabled:
            timestamped_segments = _build_timestamped_transcript(transcript_result)

            def _analyze_payload(prompt_content, temperature):
                payload, _ = _analyze_single(
                    prompt_content, temperature=temperature
                )
                return payload

            result_json = _run_podcast_comedy_multipass(
                timestamped_segments,
                video_duration=video_duration,
                analyze_prompt=_analyze_payload,
            )
            if not result_json.get("shorts"):
                raise ValueError("No valid podcast comedy candidates found")
            result_json = post_process_clips(result_json)
            return result_json, words

        # --- Estimate prompt size and decide single-call vs chunked ---
        transcript_json = json.dumps(transcript_result["text"])
        words_json = json.dumps(words)
        full_prompt = _build_prompt(
            video_duration=video_duration,
            language=language,
            scene_boundaries=scene_text,
            transcript_text=transcript_json,
            words_json=words_json,
            category=category,
        )

        if len(full_prompt) <= MAX_PROMPT_CHARS:
            # Single call
            result_json, response = _analyze_single(full_prompt)

            if result_json is None:
                raise ValueError("Failed to parse JSON from model response after retry")

            # Token usage
            try:
                usage = response.usage
                if usage:
                    print(f"💰 Token Usage ({model_name}):")
                    print(f"   - Input Tokens: {usage.prompt_tokens}")
                    print(f"   - Output Tokens: {usage.completion_tokens}")
            except Exception:
                pass

            result_json = post_process_clips(result_json)
            return result_json, words

        # --- Chunked: sliding windows ---
        print(
            f"   📦 Prompt too large ({len(full_prompt)} chars > {MAX_PROMPT_CHARS}), chunking video into windows..."
        )

        # Calculate window size: fit words + transcript within budget
        template_overhead = _build_prompt(
            video_duration=video_duration,
            language=language,
            scene_boundaries="",
            transcript_text="",
            words_json="",
            category=category,
        )
        # Account for both words_json AND transcript_json in per-word estimate
        total_data_chars = len(transcript_json) + len(words_json)
        chars_per_word = total_data_chars / max(len(words), 1)
        available_per_window = (
            MAX_PROMPT_CHARS - len(template_overhead) - 500
        )  # window label
        words_per_window = max(int(available_per_window / chars_per_word), 200)
        overlap_words = words_per_window // 5  # 20% overlap

        total_windows = max(
            1,
            (len(words) + words_per_window - overlap_words - 1)
            // (words_per_window - overlap_words),
        )
        print(
            f"   🪟 Splitting {len(words)} words into ~{total_windows} windows ({words_per_window} words each, {overlap_words} overlap)"
        )

        all_shorts = []
        content_types = []

        for win_idx in range(0, len(words), words_per_window - overlap_words):
            window_words = words[win_idx : win_idx + words_per_window]
            if len(window_words) < 200:
                continue

            window_prompt, _ = _build_window_prompt(
                window_words,
                transcript_result["segments"],
                video_duration,
                language,
                scene_boundaries,
                win_idx // max(words_per_window - overlap_words, 1),
                total_windows,
                category,
            )
            if window_prompt is None:
                continue

            # Safety: if window prompt still too large, trim words from both ends
            while len(window_prompt) > MAX_PROMPT_CHARS and len(window_words) > 200:
                trim_count = int(len(window_words) * 0.15)
                trim_start = trim_count // 2
                trim_end = trim_count - trim_start
                window_words = (
                    window_words[trim_start:-trim_end]
                    if trim_end > 0
                    else window_words[trim_start:]
                )
                window_prompt, _ = _build_window_prompt(
                    window_words,
                    transcript_result["segments"],
                    video_duration,
                    language,
                    scene_boundaries,
                    win_idx // max(words_per_window - overlap_words, 1),
                    total_windows,
                    category,
                )

            if len(window_prompt) > MAX_PROMPT_CHARS:
                print(
                    f"   ⚠️ Window still too large ({len(window_prompt)} chars), skipping"
                )
                continue

            print(
                f"   🔍 Analyzing window {win_idx // max(words_per_window - overlap_words, 1) + 1}/{total_windows} ({len(window_words)} words)..."
            )
            result_json, _ = _analyze_single(window_prompt)

            if result_json:
                if "content_type" in result_json:
                    content_types.append(result_json["content_type"])
                all_shorts.extend(result_json.get("shorts", []))

            # Add 1-minute delay between chunk analyses to avoid API rate limiting
            next_win_idx = win_idx + words_per_window - overlap_words
            if next_win_idx < len(words):
                print(f"   ⏳ Waiting 60 seconds before analyzing next window...")
                time.sleep(60)

        if not all_shorts:
            raise ValueError("No clips found in any window")

        # Merge: dedup highly overlapping clips from adjacent windows, keep best
        merged = []
        all_shorts.sort(key=lambda s: s["start"])
        for clip in all_shorts:
            is_dup = False
            for i, kept in enumerate(merged):
                overlap_start = max(clip["start"], kept["start"])
                overlap_end = min(clip["end"], kept["end"])
                overlap_dur = max(0, overlap_end - overlap_start)
                clip_dur = clip["end"] - clip["start"]
                kept_dur = kept["end"] - kept["start"]
                min_dur = min(clip_dur, kept_dur)
                if (
                    min_dur > 0 and overlap_dur / min_dur > 0.7
                ):  # 70% overlap = same clip
                    is_dup = True
                    if clip.get("confidence", 0) > kept.get("confidence", 0):
                        merged[i] = clip
                    break
            if not is_dup:
                merged.append(clip)

        # Merge cross-window split clips: if two clips are close together
        # and no scene boundary separates them, combine into one complete clip
        MAX_GAP = 10  # max seconds between clips to consider merging
        merged.sort(key=lambda s: s["start"])
        fused = []
        for clip in merged:
            if fused and (clip["start"] - fused[-1]["end"] <= MAX_GAP):
                prev = fused[-1]
                # Check if a scene boundary separates them
                gap_start = prev["end"]
                gap_end = clip["start"]
                has_scene_between = False
                for s_start, s_end in scene_boundaries:
                    if gap_start <= s_start <= gap_end or gap_start <= s_end <= gap_end:
                        has_scene_between = True
                        break
                if not has_scene_between:
                    # Merge: extend prev to cover this clip
                    combined_dur = clip["end"] - prev["start"]
                    if combined_dur <= 90:  # respect MAX_DURATION
                        prev["end"] = clip["end"]
                        prev["confidence"] = max(
                            prev.get("confidence", 0), clip.get("confidence", 0)
                        )
                        if clip.get("reasoning"):
                            prev["reasoning"] = (
                                prev.get("reasoning", "") + " " + clip["reasoning"]
                            )
                        continue
            fused.append(clip)
        merged = fused

        result_json = {"shorts": merged}
        if content_types:
            result_json["content_type"] = max(
                set(content_types), key=content_types.count
            )

        result_json = post_process_clips(result_json)
        print(
            f"   ✅ Chunked analysis complete: {len(all_shorts)} raw → {len(result_json['shorts'])} after merge+filter"
        )
        return result_json, words

    except Exception as e:
        print(f"❌ AI Analysis Error: {e}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoCrop-Vertical with Viral Clip Detection."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i", "--input", type=str, help="Path to the input video file."
    )
    input_group.add_argument(
        "-u", "--url", type=str, help="YouTube URL to download and process."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory or file (if processing whole video).",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the downloaded YouTube video.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip AI analysis and convert the whole video.",
    )
    parser.add_argument(
        "--transcription-method",
        type=str,
        choices=["faster-whisper", "groq"],
        default=None,
        help="Transcription method: faster-whisper (local CPU) or groq (cloud API). Defaults to GROQ if GROQ_API_KEY env var is set, otherwise faster-whisper",
    )
    parser.add_argument(
        "--crop-style",
        type=str,
        choices=["auto", "blur_bars"],
        default="blur_bars",
        help="Crop style: blur_bars (full content on blurred background, sharp output) or auto (AI scene detection with TRACK/GENERAL). Default: blur_bars.",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "podcast",
            "podcast_comedy",
            "tutorial",
            "gaming",
            "reaction",
            "interview",
            "news",
            "general",
        ],
        default="general",
        help="Content category for tailored clip detection. 'general' auto-detects. Default: general.",
    )

    args = parser.parse_args()

    # Auto-detect transcription method if not specified
    transcription_method = args.transcription_method
    if not transcription_method:
        if os.getenv("GROQ_API_KEY"):
            transcription_method = "groq"
            print("🚀 GROQ_API_KEY detected - using Groq for transcription")
        else:
            transcription_method = "faster-whisper"
            print("💻 No GROQ_API_KEY - using Faster-Whisper (local CPU)")
    else:
        print(f"📌 Transcription method specified via CLI: {transcription_method}")

    print(f"🔍 GROQ_API_KEY in env: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
    print(f"🎯 Selected transcription method: {transcription_method}")

    script_start_time = time.time()

    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."

        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]

        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(
                    input_video
                )
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = (
            args.output
            if args.output
            else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        )
        process_video_to_vertical(input_video, output_file, crop_style=args.crop_style)
    else:
        # 3. Transcribe
        transcript = transcribe_video(input_video, method=transcription_method)

        # Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 3.5. Detect scenes for better boundary alignment
        scene_boundaries = None
        try:
            scenes, scene_fps = detect_scenes(input_video)
            scene_boundaries = []
            for s_start, s_end in scenes:
                scene_boundaries.append((s_start.get_seconds(), s_end.get_seconds()))
            print(
                f"   🎞️  Detected {len(scene_boundaries)} scenes for boundary alignment"
            )
        except Exception as e:
            print(f"   ⚠️ Scene detection skipped: {e}")

        # 4. AI Analysis
        clips_data, transcript_words = get_viral_clips(
            transcript, duration, scene_boundaries, category=args.category
        )

        if not clips_data or "shorts" not in clips_data:
            print("❌ Failed to identify clips. Converting whole video as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
            process_video_to_vertical(
                input_video, output_file, crop_style=args.crop_style
            )
        else:
            # Snap clip boundaries to clean word/scene cut points
            for clip in clips_data["shorts"]:
                snap_clip_to_boundaries(
                    clip,
                    transcript_words or [],
                    scene_boundaries,
                    video_duration=duration,
                )

            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")

            # Save metadata
            clips_data["transcript"] = transcript  # Save full transcript for subtitles
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            # 5. Process each clip
            for i, clip in enumerate(clips_data["shorts"]):
                start = clip["start"]
                end = clip["end"]
                print(f"\n🎬 Processing Clip {i + 1}: {start}s - {end}s")
                print(
                    f"   Title: {clip.get('video_title_for_youtube_short') or clip.get('viral_hook_text', 'No Title')}"
                )

                # Cut clip
                clip_filename = f"{video_title}_clip_{i + 1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)

                # ffmpeg cut
                # Using re-encoding for precision as requested by strict seconds
                cut_command = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-i",
                    input_video,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "15",
                    "-preset",
                    "slow",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    clip_temp_path,
                ]
                subprocess.run(
                    cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )

                # Process vertical
                success = process_video_to_vertical(
                    clip_temp_path, clip_final_path, crop_style=args.crop_style
                )

                if success:
                    print(f"   ✅ Clip {i + 1} ready: {clip_final_path}")

                # Clean up temp cut
                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
