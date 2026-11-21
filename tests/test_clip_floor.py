"""The clip floor must survive a model that ignores it.

`min_clips` lived only in the detail prompt, so a run that came back with
half the requested clips shipped half the clips: the measured 9:10 source
asked for 6-12 and delivered 3-4. The windows the detail pass skipped get
one more call before the job settles for less.
"""
import json

import pytest

main = pytest.importorskip("main")


def _transcript(n_segments=40):
    segments = []
    for i in range(n_segments):
        start = i * 15.0
        segments.append({
            "start": start, "end": start + 15.0,
            "text": f"sentence number {i} with enough words to score",
            "words": [{"word": f"w{i}", "start": start, "end": start + 0.4}],
        })
    return {"language": "es", "segments": segments}


def _clip(start, window_id, score=80):
    return {
        "start": start, "end": start + 30.0, "source_window_id": window_id,
        "predicted_score": score, "video_description_for_tiktok": "d",
        "video_description_for_instagram": "d",
        "video_title_for_youtube_short": "t", "viral_hook_text": "h",
        "why": "because",
    }


def _run(monkeypatch, detail_responses):
    """Drive get_viral_clips with canned stage results; returns (clips, calls)."""
    calls = []

    def fake_stage(client, model, prompt, schema):
        payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0]
                             if "WINDOWS_JSON:" in prompt else "[]")
        if schema is main.gemini_worker.ScoreResponse:
            ids = [w["id"] for w in payload]
            calls.append(("score", ids))
            # The prompt asks for every window to be scored.
            return {"windows": [{"id": i, "start": 0, "end": 90, "score": 90 - n,
                                 "reason": "r"}
                                for n, i in enumerate(ids)]}, None
        windows = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1]
                             .split("Return only:")[0])
        ids = [w["id"] for w in windows]
        calls.append(("detail", ids))
        return {"shorts": detail_responses.pop(0)}, None

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda api_key=None: object())
    out = main.get_viral_clips(_transcript(), 600.0)
    return (out or {}).get("shorts", []), calls


def test_a_short_detail_result_asks_the_unused_windows(monkeypatch):
    # Two clips back on a shortlist that asked for six: the four windows that
    # produced nothing get a second call, and its clips are kept.
    first = [_clip(0, "window_001"), _clip(120, "window_003")]
    second = [_clip(240, "window_005"), _clip(300, "window_006"),
              _clip(360, "window_007"), _clip(420, "window_008")]
    clips, calls = _run(monkeypatch, [first, second])

    detail_calls = [c for c in calls if c[0] == "detail"]
    assert len(detail_calls) == 2, "the floor never triggered a second call"
    # The retry only revisits windows the first pass returned nothing for.
    assert "window_001" not in detail_calls[1][1]
    assert "window_003" not in detail_calls[1][1]
    assert len(clips) >= 6
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)


def test_no_second_call_when_the_floor_is_met(monkeypatch):
    enough = [_clip(i * 60, f"window_{i + 1:03d}") for i in range(6)]
    clips, calls = _run(monkeypatch, [enough])
    assert len([c for c in calls if c[0] == "detail"]) == 1
    assert len(clips) == 6


def test_an_empty_retry_is_accepted_rather_than_padded(monkeypatch):
    # Material that genuinely holds two clips still returns two, not filler.
    clips, calls = _run(monkeypatch, [[_clip(0, "window_001")], []])
    assert len([c for c in calls if c[0] == "detail"]) == 2
    assert len(clips) == 1


def test_scoring_fills_the_shortlist_it_targets(monkeypatch):
    # 600s -> target 8. Every window is scored, so the shortlist is the top 8
    # globally instead of the 3-per-batch a constant cap allowed.
    clips, calls = _run(monkeypatch, [[_clip(i * 60, f"window_{i + 1:03d}")
                                       for i in range(6)]])
    detail_windows = [c[1] for c in calls if c[0] == "detail"][0]
    assert len(detail_windows) == 8


# --- a long source: one detail call per stretch, clips from all of it ---------

def _run_long(monkeypatch, minutes, per_window=1):
    """get_viral_clips on a source ``minutes`` long whose every window holds a
    clip; returns (clips, detail calls)."""
    for name in ("CLIP_TARGET_MIN", "CLIP_TARGET_MAX", "CLIP_LONG_SECONDS_PER_CLIP",
                 "CLIP_MAX_PER_JOB"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def fake_stage(client, model, prompt, schema):
        if schema is main.gemini_worker.ScoreResponse:
            payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0])
            return {"windows": [{"id": w["id"], "start": w["start"], "end": w["end"],
                                 "score": 50 + (hash(w["id"]) % 40), "reason": "r"}
                                for w in payload]}, None
        windows = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1]
                             .split("Return only:")[0])
        calls.append(windows)
        return {"shorts": [_clip(float(w["start"]) + 5, w["id"])
                           for w in windows[:len(windows) * per_window]]}, None

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.genai, "Client", lambda api_key=None: object())
    seconds = minutes * 60.0
    out = main.get_viral_clips(_transcript(int(seconds // 15)), seconds)
    return (out or {}).get("shorts", []), calls


def test_a_two_hour_source_gives_the_clips_of_two_hours(monkeypatch):
    clips, calls = _run_long(monkeypatch, 120)
    # It was 6-12 clips whatever the length. Now: four stretches of the video,
    # one detail call each, and at least a clip per five minutes.
    assert len(calls) == 4 and all(len(c) == 10 for c in calls)
    starts = [float(w["start"]) for c in calls for w in c]
    assert starts == sorted(starts)                         # each call its own stretch
    assert 24 <= len(clips) <= 48
    assert max(c["start"] for c in clips) > 3600            # from the second hour too
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)


def test_half_an_hour_is_still_one_call_and_six_to_twelve_clips(monkeypatch):
    clips, calls = _run_long(monkeypatch, 30)
    assert len(calls) == 1 and len(calls[0]) == 10
    assert 6 <= len(clips) <= 12


def test_a_job_never_delivers_more_than_its_cap(monkeypatch):
    clips, calls = _run_long(monkeypatch, 480)
    assert len(calls) == 10
    assert len(clips) == 60


def test_a_capped_plan_gets_its_cap_however_long_the_source(monkeypatch):
    clips, calls = _run_long(monkeypatch, 120)
    assert len(clips) >= 24
    monkeypatch.setenv("CLIP_TARGET_MIN", "2")
    monkeypatch.setenv("CLIP_TARGET_MAX", "2")
    calls.clear()

    def fake_stage(client, model, prompt, schema):
        if schema is main.gemini_worker.ScoreResponse:
            payload = json.loads(prompt.split("WINDOWS_JSON:")[-1].split("Return only:")[0])
            return {"windows": [{"id": w["id"], "start": w["start"], "end": w["end"],
                                 "score": 60, "reason": "r"} for w in payload]}, None
        windows = json.loads(prompt.split("CANDIDATE_WINDOWS_JSON:")[-1]
                             .split("Return only:")[0])
        calls.append(windows)
        return {"shorts": [_clip(float(w["start"]) + 5, w["id"]) for w in windows]}, None

    monkeypatch.setattr(main, "_run_gemini_stage", fake_stage)
    out = main.get_viral_clips(_transcript(480), 7200.0)
    assert len(calls) == 1 and len(calls[0]) == 10          # the old shortlist, one call
    assert len(out["shorts"]) == 2
