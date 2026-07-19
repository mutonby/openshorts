import copy
import json
import sys
import types


class _DummyYOLO:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return []


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
    return module


_install_stub("cv2")
_install_stub("scenedetect", open_video=lambda *args, **kwargs: None)
_install_stub("scenedetect.detectors", ContentDetector=object)
sys.modules["scenedetect"].SceneManager = object
_install_stub("ultralytics", YOLO=_DummyYOLO)
_install_stub("torch")
_install_stub("tqdm", tqdm=lambda iterable=None, *args, **kwargs: iterable or [])
_install_stub("yt_dlp")
_install_stub(
    "mediapipe",
    solutions=types.SimpleNamespace(
        face_detection=types.SimpleNamespace(
            FaceDetection=lambda *args, **kwargs: types.SimpleNamespace()
        )
    ),
)
_install_stub("dotenv", load_dotenv=lambda *args, **kwargs: None)
_install_stub("openai", OpenAI=object)

import main

from main import (
    _attach_candidate_context,
    _build_global_video_context,
    _build_multipass_rank_prompt,
    _build_podcast_comedy_judge_prompt,
    _build_podcast_comedy_scout_prompt,
    _build_prompt,
    _build_timestamped_transcript,
    _build_window_prompt,
    _consolidate_scout_candidates,
    _rank_balanced_clips,
    _run_podcast_comedy_multipass,
    _segments_for_window,
    _validate_rank_output,
    _validate_judge_output,
    post_process_clips,
    snap_clip_to_boundaries,
)


def _candidate(
    candidate_id,
    lane,
    start,
    end,
    score,
    hook="hook evidence",
    payoff="payoff evidence",
):
    return {
        "candidate_id": candidate_id,
        "content_lane": lane,
        "start": start,
        "end": end,
        "setup_start": start,
        "payoff_start": max(start, end - 5),
        "payoff_end": end - 1,
        "reaction_end": end,
        "hook_evidence": hook,
        "payoff_evidence": payoff,
        "standalone_summary": "Penonton memahami momen ini tanpa konteks lain.",
        "suggested_hook": "hook kandidat yang sangat panjang sekali",
        "scout_score": score,
    }


def test_build_timestamped_transcript_uses_segment_text_and_average_confidence():
    transcript_result = {
        "segments": [
            {
                "start": 10.0,
                "end": 13.0,
                "text": " Gue takut banget. ",
                "words": [
                    {"word": "Gue", "start": 10.0, "end": 10.4, "probability": 0.8},
                    {
                        "word": "takut",
                        "start": 10.5,
                        "end": 11.0,
                        "probability": 0.6,
                    },
                ],
            }
        ]
    }

    assert _build_timestamped_transcript(transcript_result) == [
        {
            "id": 0,
            "start": 10.0,
            "end": 13.0,
            "text": "Gue takut banget.",
            "confidence": 0.7,
        }
    ]


def test_build_timestamped_transcript_sorts_and_splits_unusually_long_segments():
    transcript = _build_timestamped_transcript(
        {
            "segments": [
                {"start": 50.0, "end": 55.0, "text": "Terakhir.", "words": []},
                {
                    "start": 0.0,
                    "end": 40.0,
                    "text": "Satu dua tiga empat.",
                    "words": [
                        {"word": "Satu", "start": 0.0, "end": 1.0, "probability": 0.9},
                        {"word": "dua", "start": 10.0, "end": 11.0, "probability": 0.9},
                        {"word": "tiga", "start": 20.0, "end": 21.0, "probability": 0.9},
                        {
                            "word": "empat",
                            "start": 39.0,
                            "end": 40.0,
                            "probability": 0.9,
                        },
                    ],
                },
            ]
        }
    )

    assert [segment["id"] for segment in transcript] == [0, 1, 2]
    assert [(segment["start"], segment["end"]) for segment in transcript] == [
        (0.0, 21.0),
        (39.0, 40.0),
        (50.0, 55.0),
    ]
    assert transcript[0]["text"] == "Satu dua tiga"
    assert transcript[1]["text"] == "empat"


def test_segments_for_window_excludes_segments_outside_both_bounds():
    segments = [
        {"id": 0, "start": 0.0, "end": 5.0, "text": "before", "confidence": 1.0},
        {"id": 1, "start": 9.0, "end": 12.0, "text": "overlap", "confidence": 1.0},
        {"id": 2, "start": 15.0, "end": 20.0, "text": "inside", "confidence": 1.0},
        {"id": 3, "start": 25.0, "end": 30.0, "text": "after", "confidence": 1.0},
    ]

    selected = _segments_for_window(segments, 10.0, 22.0)

    assert [segment["id"] for segment in selected] == [1, 2]


def test_legacy_window_prompt_does_not_include_transcript_after_window_end():
    _, window_text = _build_window_prompt(
        words=[
            {"w": "inside", "s": 10.0, "e": 10.5, "p": 1.0},
            {"w": "inside", "s": 19.0, "e": 19.5, "p": 1.0},
        ],
        transcript_segments=[
            {"start": 0.0, "end": 5.0, "text": "before"},
            {"start": 9.0, "end": 12.0, "text": "overlapping"},
            {"start": 15.0, "end": 20.0, "text": "inside"},
            {"start": 30.0, "end": 35.0, "text": "after"},
        ],
        video_duration=60.0,
        language="id",
        scene_boundaries=[],
        window_num=0,
        total_windows=1,
        category="general",
    )

    assert window_text == "overlapping inside"


def test_scout_prompt_contains_one_timestamped_transcript_and_no_words_json():
    segments = [
        {
            "id": 0,
            "start": 1.0,
            "end": 4.0,
            "text": "Ini cuma dikirim sekali.",
            "confidence": 0.9,
        }
    ]

    prompt = _build_podcast_comedy_scout_prompt(segments, video_duration=120)

    assert prompt.count("TIMESTAMPED_TRANSCRIPT") == 1
    assert prompt.count("Ini cuma dikirim sekali.") == 1
    assert "WORDS_JSON" not in prompt
    assert "SENSOR" in prompt
    assert "SCORING RUBRIC" in prompt
    assert "<0.70: jangan output" in prompt
    assert "setup lambat tanpa payoff" in prompt


def test_global_context_summarizes_full_video_for_multipass_prompts():
    segments = [
        {"id": 0, "start": 0.0, "end": 10.0, "text": "opening setup", "confidence": 1.0},
        {"id": 1, "start": 60.0, "end": 70.0, "text": "middle conflict", "confidence": 1.0},
        {"id": 2, "start": 120.0, "end": 130.0, "text": "ending payoff", "confidence": 1.0},
    ]

    context = _build_global_video_context(segments, video_duration=130.0)
    scout_prompt = _build_podcast_comedy_scout_prompt(
        [segments[1]], video_duration=130.0, global_context=context
    )

    assert "# GLOBAL_VIDEO_CONTEXT" in scout_prompt
    assert "opening setup" in scout_prompt
    assert "ending payoff" in scout_prompt
    assert "Window transcript is local; use global context only to judge standalone context" in scout_prompt


def test_judge_prompt_prefers_quality_over_filling_slots():
    prompt = _build_podcast_comedy_judge_prompt(
        [_candidate("c1", "COMEDY", 15.0, 40.0, 0.9)], max_clips=5
    )

    assert "Return fewer than 5 clips" in prompt
    assert "kualitas mengalahkan kuantitas" in prompt
    assert "Tie-breaker" in prompt


def test_judge_prompt_requires_evidence_first_dimensions_and_no_score_anchoring():
    prompt = _build_podcast_comedy_judge_prompt(
        [_candidate("c1", "COMEDY", 15.0, 40.0, 0.9)],
        max_clips=5,
        global_context="early: setup\nlate: payoff",
    )

    assert "EVIDENCE-FIRST" in prompt
    assert "transcript evidence -> detected signals -> dimension scores" in prompt
    assert "Do not anchor on scout_score" in prompt
    for dimension in (
        "hook_strength",
        "flow_retention",
        "comment_potential",
        "quotability",
        "boundary_quality",
        "context_dependency",
    ):
        assert dimension in prompt


def test_consolidate_scout_candidates_rejects_invalid_and_keeps_strong_duplicate():
    candidates = [
        _candidate("weak", "COMEDY", 10.0, 35.0, 0.72),
        _candidate("strong", "comedy", 11.0, 36.0, 0.94),
        _candidate("missing-evidence", "HOT_TAKE", 50.0, 90.0, 0.99, hook=""),
        _candidate("too-short", "PERSONAL_STORY", 100.0, 108.0, 0.99),
        _candidate("outside", "COMEDY", 190.0, 220.0, 0.99),
        _candidate("not-finite", "COMEDY", float("nan"), 150.0, 0.99),
    ]

    result = _consolidate_scout_candidates(candidates, video_duration=200)

    assert [candidate["candidate_id"] for candidate in result] == ["strong"]
    assert result[0]["content_lane"] == "COMEDY"


def test_consolidate_scout_candidates_filters_low_score_and_missing_summary():
    missing_summary = _candidate("missing-summary", "COMEDY", 10.0, 35.0, 0.95)
    missing_summary["standalone_summary"] = ""

    result = _consolidate_scout_candidates(
        [
            _candidate("low", "COMEDY", 40.0, 65.0, 0.69),
            missing_summary,
            _candidate("valid", "COMEDY", 70.0, 95.0, 0.7),
        ],
        video_duration=120.0,
    )

    assert [candidate["candidate_id"] for candidate in result] == ["valid"]


def test_consolidate_scout_candidates_dedupes_semantic_overlap_without_temporal_overlap():
    first = _candidate("first", "HOT_TAKE", 10.0, 45.0, 0.86)
    first["standalone_summary"] = "Host explains creators fake urgency to farm comments"
    first["hook_evidence"] = "creators fake urgency"
    first["payoff_evidence"] = "farm comments"
    second = _candidate("second", "HOT_TAKE", 52.0, 87.0, 0.93)
    second["standalone_summary"] = "Host explains creators fake urgency to farm comments"
    second["hook_evidence"] = "fake urgency"
    second["payoff_evidence"] = "farm comments"

    result = _consolidate_scout_candidates(
        [first, second], video_duration=120.0
    )

    assert [candidate["candidate_id"] for candidate in result] == ["second"]


def test_general_multipass_preserves_controversy_lane():
    config = main._get_multipass_config("general")
    config["_category"] = "general"

    result = _consolidate_scout_candidates(
        [_candidate("c1", "CONTROVERSY", 10.0, 35.0, 0.9)],
        video_duration=60.0,
        config=config,
    )

    assert result[0]["content_lane"] == "CONTROVERSY"


def test_candidate_context_is_bounded_and_judge_prompt_uses_compact_context():
    segments = [
        {"id": 0, "start": 0.0, "end": 4.0, "text": "too early", "confidence": 1.0},
        {"id": 1, "start": 8.0, "end": 12.0, "text": "lead in", "confidence": 1.0},
        {"id": 2, "start": 15.0, "end": 30.0, "text": "candidate", "confidence": 1.0},
        {"id": 3, "start": 32.0, "end": 35.0, "text": "follow up", "confidence": 1.0},
        {"id": 4, "start": 40.0, "end": 45.0, "text": "too late", "confidence": 1.0},
    ]
    candidates = [_candidate("c1", "COMEDY", 15.0, 30.0, 0.9)]

    contextualized = _attach_candidate_context(
        candidates, segments, video_duration=33.0
    )
    prompt = _build_podcast_comedy_judge_prompt(contextualized, max_clips=5)

    assert [item["id"] for item in contextualized[0]["transcript_context"]] == [
        1,
        2,
        3,
    ]
    assert contextualized[0]["context_start"] == 7.0
    assert contextualized[0]["context_end"] == 33.0
    assert "too early" not in prompt
    assert "too late" not in prompt


def test_validate_judge_output_rejects_unknown_ids_boundaries_and_bad_lane_duration():
    candidates = _attach_candidate_context(
        [
            _candidate("comedy", "COMEDY", 20.0, 50.0, 0.9),
            _candidate("story", "PERSONAL_STORY", 80.0, 125.0, 0.88),
        ],
        [
            {"id": 0, "start": 10.0, "end": 60.0, "text": "comedy", "confidence": 1.0},
            {"id": 1, "start": 70.0, "end": 135.0, "text": "story", "confidence": 1.0},
        ],
    )
    judge_payload = {
        "shorts": [
            {
                "candidate_id": "unknown",
                "start": 20.0,
                "end": 45.0,
                "judge_score": 0.99,
            },
            {
                "candidate_id": "comedy",
                "start": 5.0,
                "end": 35.0,
                "judge_score": 0.98,
            },
            {
                "candidate_id": "comedy",
                "start": 20.0,
                "end": 68.0,
                "judge_score": 0.97,
            },
            {
                "candidate_id": "story",
                "start": 80.0,
                "end": 125.0,
                "judge_score": 0.93,
                "viral_hook_text": "cerita ini ternyata sangat berbeda dari dugaan semua orang",
                "social_caption": "Cerita yang berubah total di akhir.",
            },
            {
                "candidate_id": "comedy",
                "start": 20.0,
                "end": 44.0,
                "judge_score": 0.96,
            },
        ]
    }

    valid = _validate_judge_output(judge_payload, candidates)

    assert len(valid) == 1
    assert valid[0]["candidate_id"] == "story"
    assert valid[0]["viral_hook_text"] == "CERITA INI TERNYATA SANGAT BERBEDA DARI"


def test_validate_judge_output_rejects_low_score_and_cut_before_payoff():
    candidates = _attach_candidate_context(
        [_candidate("c1", "COMEDY", 20.0, 50.0, 0.9)],
        [{"id": 0, "start": 10.0, "end": 55.0, "text": "context", "confidence": 1.0}],
        video_duration=60.0,
    )

    valid = _validate_judge_output(
        {
            "shorts": [
                {
                    "candidate_id": "c1",
                    "start": 20.0,
                    "end": 44.0,
                    "judge_score": 0.95,
                },
                {
                    "candidate_id": "c1",
                    "start": 20.0,
                    "end": 50.0,
                    "judge_score": 0.4,
                },
            ]
        },
        candidates,
        min_score=0.7,
    )

    assert valid == []


def test_rank_balanced_clips_keeps_strong_lanes_but_not_weak_token_candidate():
    clips = [
        {"candidate_id": "c1", "content_lane": "COMEDY", "confidence": 0.95},
        {"candidate_id": "c2", "content_lane": "COMEDY", "confidence": 0.93},
        {"candidate_id": "h1", "content_lane": "HOT_TAKE", "confidence": 0.89},
        {"candidate_id": "s1", "content_lane": "PERSONAL_STORY", "confidence": 0.60},
    ]

    ranked = _rank_balanced_clips(clips, max_clips=3)

    assert [clip["candidate_id"] for clip in ranked] == ["c1", "h1", "c2"]


def test_validate_rank_output_orders_by_tournament_rank_and_calibrates_score():
    clips = [
        {
            **_candidate("a", "COMEDY", 10.0, 35.0, 0.95),
            "confidence": 0.95,
            "judge_score": 0.95,
            "dimension_scores": {"hook_strength": 4, "standalone_comprehensibility": 4},
        },
        {
            **_candidate("b", "HOT_TAKE", 50.0, 90.0, 0.8),
            "confidence": 0.8,
            "judge_score": 0.8,
            "dimension_scores": {"hook_strength": 3, "standalone_comprehensibility": 3},
        },
    ]
    rank_payload = {
        "ranked_clips": [
            {"candidate_id": "b", "rank": 1, "final_score": 91, "ranking_evidence": "Better comment trigger."},
            {"candidate_id": "a", "rank": 2, "final_score": 84, "ranking_evidence": "Funny but less shareable."},
        ]
    }

    ranked = _validate_rank_output(rank_payload, clips, max_clips=2)

    assert [clip["candidate_id"] for clip in ranked] == ["b", "a"]
    assert ranked[0]["confidence"] == 0.91
    assert ranked[0]["final_score"] == 91
    assert ranked[0]["ranking_evidence"] == "Better comment trigger."


def test_rank_prompt_compares_shortlist_instead_of_rescoring_independently():
    prompt = _build_multipass_rank_prompt(
        [_candidate("a", "COMEDY", 10.0, 35.0, 0.9), _candidate("b", "HOT_TAKE", 50.0, 90.0, 0.88)],
        category="podcast_comedy",
        max_clips=2,
    )

    assert "PAIRWISE" in prompt
    assert "tournament" in prompt.lower()
    assert "final_score" in prompt
    assert "Do not re-read the full transcript" in prompt


def test_multipass_falls_back_to_scout_candidates_when_judge_fails(monkeypatch):
    calls = []

    def analyze(prompt, temperature):
        calls.append(prompt)
        if "INDONESIAN_EDITOR_JUDGE" in prompt:
            raise RuntimeError("judge unavailable")
        return {"candidates": [_candidate("c1", "COMEDY", 10.0, 35.0, 0.91)]}

    monkeypatch.setenv("PODCAST_COMEDY_SCOUT_MAX_CHARS", "invalid")
    result = _run_podcast_comedy_multipass(
        [
            {
                "id": 0,
                "start": 0.0,
                "end": 40.0,
                "text": "Setup, punchline, lalu reaksi.",
                "confidence": 0.95,
            }
        ],
        video_duration=40.0,
        analyze_prompt=analyze,
        max_clips=5,
    )

    assert len(calls) == 2
    assert result["shorts"][0]["candidate_id"] == "c1"
    assert (
        result["shorts"][0]["viral_hook_text"]
        == "HOOK KANDIDAT YANG SANGAT PANJANG SEKALI"
    )


def test_multipass_makes_duplicate_candidate_ids_unique_across_windows(monkeypatch):
    scout_calls = 0

    def analyze(prompt, temperature):
        nonlocal scout_calls
        if "INDONESIAN_EDITOR_JUDGE" in prompt:
            raise RuntimeError("judge unavailable")
        scout_calls += 1
        start = 0.0 if scout_calls == 1 else 30.0
        return {
            "candidates": [
                _candidate("c1", "COMEDY", start, start + 20.0, 0.9)
            ]
        }

    monkeypatch.setenv("PODCAST_COMEDY_SCOUT_MAX_CHARS", "1")
    result = _run_podcast_comedy_multipass(
        [
            {"id": 0, "start": 0.0, "end": 20.0, "text": "first", "confidence": 1.0},
            {
                "id": 1,
                "start": 30.0,
                "end": 50.0,
                "text": "second",
                "confidence": 1.0,
            },
        ],
        video_duration=60.0,
        analyze_prompt=analyze,
    )

    assert scout_calls == 2
    assert [clip["candidate_id"] for clip in result["shorts"]] == ["c1", "w2-c1"]


def test_multipass_logs_each_processing_stage(capsys):
    responses = [
        {"candidates": [_candidate("c1", "COMEDY", 10.0, 35.0, 0.91)]},
        {
            "shorts": [
                {
                    "candidate_id": "c1",
                    "start": 10.0,
                    "end": 35.0,
                    "judge_score": 0.94,
                    "viral_hook_text": "punchline bikin kaget",
                    "social_caption": "Punchline tak terduga.",
                }
            ]
        },
        {
            "ranked_clips": [
                {
                    "candidate_id": "c1",
                    "rank": 1,
                    "final_score": 94,
                    "ranking_evidence": "Paling kuat.",
                }
            ]
        },
    ]

    def analyze(prompt, temperature):
        return responses.pop(0)

    _run_podcast_comedy_multipass(
        [
            {
                "id": 0,
                "start": 0.0,
                "end": 40.0,
                "text": "Setup, punchline, lalu reaksi.",
                "confidence": 0.95,
            }
        ],
        video_duration=40.0,
        analyze_prompt=analyze,
        max_clips=5,
    )

    output = capsys.readouterr().out
    assert "Multi-pass (podcast_comedy) started: 1 transcript segments, 1 scout window" in output
    assert "Scout 1/1 started:" in output
    assert "Scout 1/1 completed: 1 raw candidates" in output
    assert "Candidate validation completed: 1 raw -> 1 valid" in output
    assert "Judge started: reviewing 1 candidates" in output
    assert "Judge completed: 1 clips selected" in output
    assert "Final ranker started: comparing 1 analyzed clips" in output
    assert "Final ranker completed: 1 clips ranked" in output
    assert "COMEDY=1" in output


def test_get_viral_clips_routes_podcast_comedy_through_multipass_pipeline(monkeypatch):
    calls = []
    responses = [
        {"candidates": [_candidate("c1", "COMEDY", 10.0, 35.0, 0.91)]},
        {
            "shorts": [
                {
                    "candidate_id": "c1",
                    "start": 10.0,
                    "end": 35.0,
                    "judge_score": 0.94,
                    "viral_hook_text": "punchline ini bikin semua kaget",
                    "social_caption": "Punchline tak terduga. #komedi",
                }
            ]
        },
        {
            "ranked_clips": [
                {
                    "candidate_id": "c1",
                    "rank": 1,
                    "final_score": 94,
                    "ranking_evidence": "Best evidence-backed clip.",
                }
            ]
        },
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            payload = responses.pop(0)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(payload))
                    )
                ],
                usage=None,
            )

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PODCAST_COMEDY_MULTIPASS", "true")
    monkeypatch.setattr(main, "OpenAI", lambda **kwargs: fake_client)

    result, words = main.get_viral_clips(
        {
            "text": "Setup, punchline, lalu reaksi.",
            "language": "id",
            "segments": [
                {
                    "start": 0.0,
                    "end": 40.0,
                    "text": "Setup, punchline, lalu reaksi.",
                    "words": [
                        {
                            "word": "Setup",
                            "start": 0.0,
                            "end": 0.5,
                            "probability": 0.95,
                        }
                    ],
                }
            ],
        },
        video_duration=40.0,
        scene_boundaries=[],
        category="podcast_comedy",
    )

    assert len(calls) == 3
    assert "INDONESIAN_PODCAST_CANDIDATE_SCOUT" in calls[0]["messages"][0]["content"]
    assert "INDONESIAN_EDITOR_JUDGE" in calls[1]["messages"][0]["content"]
    assert "INDONESIAN_FINAL_RANKER" in calls[2]["messages"][0]["content"]
    assert "WORDS_JSON" not in calls[0]["messages"][0]["content"]
    assert result["shorts"][0]["candidate_id"] == "c1"
    assert result["shorts"][0]["confidence"] == 0.94
    assert result["shorts"][0]["final_score"] == 94
    assert words[0]["w"] == "Setup"


def test_general_prompt_requires_evidence_and_rejects_weak_starts():
    prompt = _build_prompt(
        video_duration=120,
        language="id",
        scene_boundaries="[10.0s - 30.0s]",
        transcript_text="sample transcript",
        words_json="[]",
        category="general",
    )

    assert "QUOTE_EVIDENCE" in prompt
    assert "START_QUALITY_CHECK" in prompt
    assert "FIRST 2 SECONDS" in prompt
    assert "SELF-CONTAINED CHECK" in prompt


def test_specialized_prompts_require_complete_thought_and_grounded_evidence():
    for category in ("podcast", "podcast_comedy"):
        prompt = _build_prompt(
            video_duration=120,
            language="id",
            scene_boundaries="[10.0s - 30.0s]",
            transcript_text="sample transcript",
            words_json="[]",
            category=category,
        )

        assert "QUOTE_EVIDENCE" in prompt
        assert "COMPLETE_THOUGHT" in prompt
        assert "SELF-CONTAINED CHECK" in prompt


def test_post_process_keeps_strongest_near_duplicate_and_normalizes_metadata():
    result = {
        "shorts": [
            {
                "start": 10.0,
                "end": 34.0,
                "score": 0.74,
                "viral_pattern_type": "HOOK",
                "viral_hook_text": "  this    changes everything immediately now  ",
                "social_caption": "  useful point   #clip  ",
            },
            {
                "start": 11.0,
                "end": 35.0,
                "score": 0.91,
                "viral_pattern_type": "HOOK",
                "viral_hook_text": "better    hook text",
                "social_caption": "better caption",
            },
            {
                "start": 80.0,
                "end": 120.0,
                "confidence": "PERAK",
                "viral_pattern_type": "INSIGHT",
                "video_title_for_youtube_short": "   ",
            },
            {
                "start": 130.0,
                "end": 134.0,
                "score": 0.98,
                "viral_pattern_type": "TOO_SHORT",
            },
        ]
    }

    processed = post_process_clips(copy.deepcopy(result), min_confidence=0.7)
    shorts = processed["shorts"]

    assert len(shorts) == 2
    assert shorts[0]["start"] == 11.0
    assert shorts[0]["confidence"] == 0.91
    assert shorts[0]["viral_hook_text"] == "BETTER HOOK TEXT"
    assert shorts[0]["video_title_for_youtube_short"] == "BETTER HOOK TEXT"
    assert shorts[0]["video_description_for_tiktok"] == "better caption"
    assert shorts[1]["viral_hook_text"] == "VIRAL SHORT"
    assert shorts[1]["confidence_label"] == "PERAK"


def test_post_process_returns_best_fallback_when_all_candidates_are_weak():
    result = {
        "shorts": [
            {"start": 0.0, "end": 20.0, "score": 0.2, "viral_hook_text": "weak one"},
            {"start": 40.0, "end": 65.0, "score": 0.4, "viral_hook_text": "less weak"},
        ]
    }

    processed = post_process_clips(copy.deepcopy(result), min_confidence=0.85)

    assert len(processed["shorts"]) == 2
    assert processed["shorts"][0]["viral_hook_text"] == "LESS WEAK"
    assert processed["shorts"][1]["viral_hook_text"] == "WEAK ONE"


def test_snap_clip_expands_inside_word_boundaries_without_trimming_speech():
    clip = {"start": 10.2, "end": 20.7}
    words = [
        {"w": "first", "s": 10.0, "e": 10.5},
        {"w": "middle", "s": 12.0, "e": 12.5},
        {"w": "last", "s": 20.4, "e": 21.0},
    ]
    scene_boundaries = [(0.0, 9.9), (9.9, 21.2), (21.2, 30.0)]

    snap_clip_to_boundaries(
        clip, words, scene_boundaries, snap_padding=0.3, video_duration=30.0
    )

    assert clip["start"] == 9.9
    assert clip["end"] == 21.0


def test_snap_clip_accepts_missing_scene_boundaries():
    clip = {"start": 10.2, "end": 20.7}
    words = [
        {"w": "first", "s": 10.0, "e": 10.5},
        {"w": "last", "s": 20.4, "e": 21.0},
    ]

    snap_clip_to_boundaries(clip, words, None, snap_padding=0.3, video_duration=30.0)

    assert clip == {"start": 9.7, "end": 21.0}


def test_snap_clip_keeps_original_when_snap_would_make_clip_too_short():
    clip = {"start": 10.2, "end": 15.0}
    original = clip.copy()
    words = [
        {"w": "first", "s": 10.0, "e": 10.5},
        {"w": "last", "s": 14.5, "e": 15.2},
    ]

    snap_clip_to_boundaries(clip, words, [], snap_padding=0.3, video_duration=30.0)

    assert clip == original
