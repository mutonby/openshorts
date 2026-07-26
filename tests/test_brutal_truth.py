"""Tests for Brutal Truth-style features in brutal_truth module."""
import json
import os
import sys
from pathlib import Path

import pytest
import brutal_truth


def test_score_threshold_filters_low_scores():
    """Auto mode keeps only clips whose composite_score ≥ threshold; with no
    transcript only the Gemini signal contributes, so composite ≈ gemini_score.
    Threshold 75 keeps the two clips with predicted_score 90/95 and drops the 80
    one (whose composite ≈ 0.5*80 + heuristic ≈ 68).
    """
    shorts = [
        {"start": 0, "end": 30, "predicted_score": 90},
        {"start": 30, "end": 60, "predicted_score": 80},
        {"start": 60, "end": 90, "predicted_score": 95},
    ]
    filtered = brutal_truth.filter_by_score(shorts, 75)
    assert len(filtered) == 2
    assert all(s["composite_score"] >= 75 for s in filtered)


def test_score_threshold_falls_back_to_top_k_when_empty():
    """Old behaviour silently returned [] for low-quality inputs, wasting a
    20-min Gemini run. New behaviour: empty survivors → top-K fallback so the
    user always gets *something* rather than a hard zero."""
    shorts = [
        {"start": 0, "end": 30, "predicted_score": 50},
        {"start": 30, "end": 60, "predicted_score": 40},
    ]
    out = brutal_truth.filter_by_score(shorts, 85, top_k=1)
    assert len(out) == 1
    # Fallback returns strongest by composite_score, not threshold-passing
    assert out[0]["predicted_score"] == 50


def test_manual_mode_caps_clip_count():
    shorts = list(range(10))
    out = brutal_truth.cap_count(shorts, 3)
    assert out == [0, 1, 2]


def test_find_static_broll_insertions_flags_long_scenes():
    transcript = {"segments": [{"words": [
        {"word": "bitcoin", "start": 7, "end": 7.3},
        {"word": "price", "start": 8, "end": 8.3},
    ]}]}
    out = brutal_truth.find_static_broll_insertions(
        [(0, 3), (5, 15)], transcript)
    assert len(out) == 1
    assert 5 <= out[0]["start"] < out[0]["end"] <= 15
    assert "bitcoin" in out[0]["keywords"] or "price" in out[0]["keywords"]


def test_select_broll_asset_matches_filename(tmp_path):
    (tmp_path / "bitcoin_chart.mp4").write_bytes(b"x")
    (tmp_path / "generic.mp4").write_bytes(b"x")
    selected = brutal_truth.select_broll_asset(str(tmp_path), ["bitcoin"])
    assert selected.endswith("bitcoin_chart.mp4")


def test_broll_without_assets_is_noop(tmp_path):
    assert brutal_truth.overlay_broll(
        str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4"),
        [{"start": 1, "end": 3, "keywords": ["x"]}],
        str(tmp_path / "assets")) is False

def test_normalize_intervals_sorts_merges_and_pads():
    intervals = [
        {"start": 10, "end": 20, "id": "b"},
        {"start": 4, "end": 12, "id": "a"},
        {"start": 50, "end": 52, "id": "short"},
        {"start": "bad", "end": 99, "id": "invalid"},
    ]
    out = brutal_truth.normalize_intervals(
        intervals, source_duration=60, min_duration=15,
        pre_roll=0.2, post_roll=0.3)
    assert len(out) == 2
    assert out[0]["start"] == 3.8
    assert out[0]["end"] == 24.8
    assert out[1]["start"] == 45.0
    assert out[1]["end"] == 60.0
    assert all(out[i]["start"] >= out[i - 1]["end"]
               for i in range(1, len(out)))


def test_normalize_intervals_clamps_to_source():
    out = brutal_truth.normalize_intervals(
        [{"start": -5, "end": 200}], source_duration=30,
        min_duration=15)
    assert out == [{"start": 0.0, "end": 30.0}]


def test_extract_clean_segment_missing_file(tmp_path):
    assert brutal_truth.extract_clean_segment(
        str(tmp_path / "missing.mp4"), 0, 10,
        str(tmp_path / "out.mp4")) is False


def test_concat_clean_segments_missing_files(tmp_path):
    assert brutal_truth.concat_clean_segments(
        [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")],
        str(tmp_path / "out.mp4")) is False



def test_enrich_metadata_adds_score_breakdown():
    """enrich_metadata now stamps composite_score + a real virality_breakdown
    (gemini_score + hook_strength + duration_fit + pacing + topic_relevance)
    instead of three copies of predicted_score."""
    short = {"start": 0, "end": 30, "predicted_score": 88}
    brutal_truth.enrich_metadata([short])
    assert short["hashtags"] == []
    assert "composite_score" in short
    assert "virality_breakdown" in short
    vd = short["virality_breakdown"]
    assert vd["gemini_score"] == 88.0
    # No transcript → hook defaults to neutral 0.5, not a copy of 88
    assert vd["hook_strength"] == 0.5
    # 30s clip is in the 25-45 sweet spot → duration_fit 1.0
    assert vd["duration_fit"] == 1.0
    assert 0 <= vd["pacing"] <= 1
    assert 0 <= vd["topic_relevance"] <= 1


def test_xml_export_creates_minimal_xmeml(tmp_path):
    shorts = [{"start": 5.0, "end": 25.0, "predicted_score": 90,
               "video_title_for_youtube_short": "Test Clip"}]
    xml_path = brutal_truth.export_xml_timeline(shorts, "testvideo", str(tmp_path), 60.0)
    assert Path(xml_path).exists()
    content = Path(xml_path).read_text(encoding="utf-8")
    assert '<?xml version="1.0"' in content
    assert '<xmeml version="5">' in content
    assert 'testvideo' in content


def test_find_climax_window_returns_3s_inside_clip():
    # Emotional marker words boost the climax detection heuristic v2.
    words = [
        {"word": "absolutely", "start": 50.0, "end": 50.5},
        {"word": "crazy!", "start": 50.5, "end": 51.0},
        {"word": "insane", "start": 51.0, "end": 51.5},
    ] + [{"word": "ok", "start": t, "end": t + 0.1} for t in range(10, 50)]
    climax = brutal_truth.find_climax_window(words, clip_start=10, clip_end=60)
    assert climax is not None
    cs, ce = climax
    assert 48 <= cs <= 52
    assert ce - cs <= 3.0


def test_find_climax_window_returns_none_when_too_few_words():
    words = [{"word": "hi", "start": 5.0, "end": 5.1}]
    assert brutal_truth.find_climax_window(words, 0, 10) is None


def test_find_weak_tail_returns_none_when_no_words():
    """No transcript words → no pacing signal → return None (don't trim)."""
    assert brutal_truth.find_weak_tail([], 0, 30) is None


def test_find_weak_tail_trims_dead_end():
    """When last 2s has far fewer words than the clip average, return a
    trimmed end time around the last energetic word."""
    # Strong speech 0-25s, silence 25-30s
    words = []
    for i in range(50):
        words.append({"word": "talk", "start": 0.5 * i, "end": 0.5 * i + 0.3})
    new_end = brutal_truth.find_weak_tail(words, clip_start=0, clip_end=30)
    assert new_end is not None
    # Trim should land inside the energetic half, not extend to 30
    assert new_end < 25.5


def test_find_weak_tail_no_trim_when_consistent_pacing():
    """Uniform word density → no weak tail found → return None."""
    words = [{"word": "w", "start": t * 0.5, "end": t * 0.5 + 0.4}
             for t in range(60)]  # uniform across 30s
    new_end = brutal_truth.find_weak_tail(words, 0, 30)
    assert new_end is None


def test_compute_composite_blends_signals():
    """Composite score should grow as Gemini score grows, all else equal."""
    base = {"start": 0, "end": 30, "predicted_score": 50}
    high = {"start": 0, "end": 30, "predicted_score": 90}
    brutal_truth.compute_composite(base)
    brutal_truth.compute_composite(high)
    assert high["composite_score"] > base["composite_score"]
    # 30s clip is in the sweet spot → duration_fit should be 1.0
    assert high["virality_breakdown"]["duration_fit"] == 1.0


def test_compute_composite_respects_env_weights():
    """W_GEMINI=1 (everything else 0) → composite equals predicted_score."""
    import os as _os
    _os.environ["W_GEMINI"] = "1"
    _os.environ["W_HOOK"] = "0"
    _os.environ["W_DUR"] = "0"
    _os.environ["W_PACE"] = "0"
    _os.environ["W_TOPIC"] = "0"
    try:
        clip = {"start": 0, "end": 30, "predicted_score": 88}
        brutal_truth.compute_composite(clip)
        assert clip["composite_score"] == 88.0
    finally:
        for k in ("W_GEMINI", "W_HOOK", "W_DUR", "W_PACE", "W_TOPIC"):
            _os.environ.pop(k, None)


def test_top_terms_from_transcript_extracts_keywords():
    transcript = {
        "segments": [
            {"words": [
                {"word": "Bitcoin", "start": 0, "end": 1},
                {"word": "Bitcoin", "start": 5, "end": 6},
                {"word": "Bitcoin", "start": 10, "end": 11},
                {"word": "price", "start": 2, "end": 3},
                {"word": "the", "start": 1, "end": 1.4},  # stopword
                {"word": "moon", "start": 12, "end": 13},
            ]}
        ]
    }
    terms = brutal_truth.top_terms_from_transcript(transcript, top_n=5)
    assert "bitcoin" in terms
    assert "price" in terms
    assert "the" not in terms  # stopword removed


def test_remove_silence_on_missing_file(tmp_path):
    fake = tmp_path / "nope.mp4"
    assert brutal_truth.remove_silence(str(fake)) is False


def test_trim_weak_tail_no_op_on_missing_file(tmp_path):
    """trim_weak_tail returns False (never raises) on missing input."""
    assert brutal_truth.trim_weak_tail(
        str(tmp_path / "nope.mp4"), 0, 30, {"segments": []}) is False


def test_indent_fix_clip_workers_executes():
    """Smoke test for the P0 main.py indentation bug: import + AST parse + check
    that clip_workers line sits at the same indent as the surrounding else
    block (not nested inside _process_one_clip). ponytail: a real render test
    needs a video fixture; AST check is the one thing that fails if anyone
    re-introduces the dead-block bug.
    """
    import ast, os as _os
    main_path = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "main.py")
    tree = ast.parse(open(main_path, encoding="utf-8").read())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        # Each item is an ast.withitem whose .context_expr is the call.
        for it in node.items:
            call = getattr(it, "context_expr", None)
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "ThreadPoolExecutor":
                # The 'with' must be at module-body indent (col_offset == 8),
                # proving it's back inside the else branch (NOT nested inside
                # _process_one_clip, which would be col_offset >= 12).
                assert node.col_offset == 8, (
                    f"ThreadPoolExecutor 'with' must sit at 8-space indent (the "
                    f"else branch body), got col_offset={node.col_offset}")
                found = True
    assert found, "ThreadPoolExecutor block not found in main.py"
