"""Stage 2 of highlight discovery: focused transcription, the alert filter,
the video judge and the ranking (no network: the Gemini call is stubbed)."""
from types import SimpleNamespace

import pytest

import candidate_review
import hype_judge as hj
import transcribe_backends as tb


def _transcript(*spans):
    segments = []
    for start, end, text in spans:
        words = text.split()
        step = (end - start) / max(1, len(words))
        segments.append({"start": start, "end": end, "text": text, "words": [
            {"word": f" {w}", "start": start + i * step, "end": start + (i + 1) * step}
            for i, w in enumerate(words)]})
    return {"language": "en", "segments": segments}


def _cand(start, end, hype=0.8, **signals):
    return {"start": start, "end": end, "peak": int(start + 20), "hype": hype, "area": 10.0,
            "signals": {"hype": hype, "chat_lag": 7.0, **signals}}


def _verdict(**over):
    base = {"decision": "keep", "virality": 82, "event": "He clutches a 1v4", "streamer_reaction": "screams",
            "outcome": "Wins the round", "title": "Impossible 1v4 clutch", "opening_title": "HE ACTUALLY DID IT",
            "description": "No way. #clutch #gaming #fyp", "hook": "strong", "visual_action": "strong",
            "reaction": "strong", "clarity": "clear", "payoff": "strong", "best_start": 4.0, "best_end": 40.0,
            "is_alert_or_ad": False, "source_captions": False}
    return {**base, **over}


# --- focused transcription --------------------------------------------------

def test_merge_ranges_pads_and_joins():
    assert tb.merge_ranges([(100, 130), (10, 40), (135, 160)], pad=5, total=162) == [(5.0, 45.0), (95.0, 162.0)]
    assert tb.merge_ranges([(0, 5)], pad=10) == [(0.0, 15.0)]


def test_transcribe_ranges_puts_words_on_the_recording_timeline(monkeypatch):
    decoded = []
    monkeypatch.setattr(tb, "_decode_whisper_audio",
                        lambda path, start=None, duration=None: decoded.append((start, duration)) or "pcm")
    calls = []

    def fake_run(audio, *, offset, progress, **params):
        calls.append(params.get("language"))
        seg = SimpleNamespace(start=1.0, end=2.0, text="no way",
                              words=[SimpleNamespace(word=" no", start=1.0, end=1.4),
                                     SimpleNamespace(word=" way", start=1.4, end=2.0)])
        return [seg], SimpleNamespace(language="en")

    monkeypatch.setattr(tb, "run_whisper_transcription", fake_run)
    out = tb.transcribe_ranges("stream.m4a", [(1000, 1030), (5000, 5045)], pad=8)
    assert decoded == [(992.0, 46.0), (4992.0, 61.0)]
    assert [s["start"] for s in out["segments"]] == [993.0, 4993.0]
    assert out["coverage"] == [[992.0, 1038.0], [4992.0, 5053.0]]
    assert calls == [None, "en"]


# --- alert filter -------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Yo 50 gifted members, thank you so much spam slash claim in the chat",
    "thank you for the 500 bits bro, thanks for the sub",
    "it was a tiktok gift it was a tiktok donation",
])
def test_alert_reading_is_recognised(text):
    assert hj.alert_share(text) >= 0.75


@pytest.mark.parametrize("text", [
    "EW! Bro, this is sick. This game's so peak.",
    "oh shoot oh shoot we out we out",
    "",
])
def test_gameplay_talk_is_not_an_alert(text):
    assert hj.alert_share(text) < 0.3


def test_prefilter_drops_alert_windows_and_keeps_the_order():
    transcript = _transcript((100, 130, "thank you for the 10 gifted subs thank you so much for the bits"),
                             (500, 530, "no no no no he is right behind me"),
                             (900, 930, "okay let's go"))
    cands = [_cand(100, 130, 0.9), _cand(500, 530, 0.7), _cand(900, 930, 0.8, chat={"clip_requests": 3})]
    kept = hj.prefilter(cands, transcript, limit=5)
    assert [c["start"] for c in kept] == [900, 500]
    assert kept[0]["speech"] == "okay let's go"


# --- verdicts -----------------------------------------------------------------

def test_supported_verdict_trims_and_writes_the_copy():
    transcript = _transcript((1000, 1045, " ".join(["word"] * 60)))
    short = hj.to_short(_cand(1000, 1045), _verdict(), transcript, 7200,
                        gameplay=True, min_secs=30, max_secs=60)
    a = short["assessment"]
    assert a["verdict"] == "supported" and a["virality"] == 82
    assert 1003.5 <= short["start"] <= 1004.5 and 1039.5 <= short["end"] <= 1040.5
    assert short["viral_hook_text"] == "HE ACTUALLY DID IT"
    assert short["video_title_for_youtube_short"] == "Impossible 1v4 clutch"
    assert a["ratings"]["reaction"] == "strong"


@pytest.mark.parametrize("over,expected", [
    ({"decision": "weak"}, "weak"),
    ({"is_alert_or_ad": True}, "weak"),
    ({"virality": 20}, "weak"),
    ({"decision": "uncertain"}, "review"),
    ({"payoff": "weak"}, "review"),
])
def test_verdict_mapping(over, expected):
    short = hj.to_short(_cand(10, 50), _verdict(**over), _transcript(), 600,
                        gameplay=False, min_secs=30, max_secs=60)
    assert short["assessment"]["verdict"] == expected
    if expected != "supported":
        assert short["viral_hook_text"] == "" and short["video_title_for_youtube_short"] == "Moment from the stream"


def test_an_absurd_trim_is_ignored():
    short = hj.to_short(_cand(100, 140), _verdict(best_start=-5, best_end=900), _transcript(), 600,
                        gameplay=False, min_secs=30, max_secs=60)
    assert (short["start"], short["end"]) == (100, 140)


def test_without_a_judge_the_window_goes_to_review():
    short = hj.to_short(_cand(100, 140, chat={"peak_chatters_per_second": 40, "normal_chatters_per_second": 4,
                                              "reactions": ["laughter"], "clip_requests": 2}),
                        None, _transcript(), 600, gameplay=True, min_secs=30, max_secs=60)
    a = short["assessment"]
    assert a["verdict"] == "review" and a["virality"] is None
    assert "chat 10x its normal pace" in a["reasons"] and "2 clip request(s)" in a["reasons"]


def test_rank_puts_supported_and_high_scores_first():
    def clip(verdict, virality, hype):
        return {"assessment": {"verdict": verdict, "virality": virality}, "hype": {"score": hype}}
    ranked = hj.rank([clip("review", None, 1.2), clip("supported", 60, 0.4),
                      clip("supported", 90, 0.5), clip("review", 70, 0.9)])
    assert [(c["assessment"]["verdict"], c["assessment"]["virality"]) for c in ranked] == [
        ("supported", 90), ("supported", 60), ("review", 70), ("review", None)]


# --- the judge call -------------------------------------------------------------

def test_judge_answers_once_per_question(tmp_path, monkeypatch):
    clip = tmp_path / "judge.mp4"
    clip.write_bytes(b"video")
    calls = []
    monkeypatch.setattr(hj, "ask_gemini", lambda video, prompt, key, model: calls.append(prompt) or (
        _verdict(), {"total_cost": 0.01}))
    cand = {**_cand(0, 40), "speech": "no way"}
    for _ in range(2):
        verdict, cost = hj.judge(clip, cand, language="en", api_key="k", cache_dir=str(tmp_path / "cache"),
                                 min_secs=30, max_secs=60)
        assert verdict["virality"] == 82 and cost == {"total_cost": 0.01}
    assert len(calls) == 1
    # Aggregates only: the prompt never carries chat lines or authors.
    assert "no way" in calls[0] and "combined hype" in calls[0]


def test_judge_failure_is_not_fatal(tmp_path, monkeypatch):
    clip = tmp_path / "judge.mp4"
    clip.write_bytes(b"video")

    def boom(*a):
        raise RuntimeError("secret provider detail")

    monkeypatch.setattr(hj, "ask_gemini", boom)
    assert hj.judge(clip, _cand(0, 40), language="en", api_key="k", cache_dir=str(tmp_path),
                    min_secs=30, max_secs=60) == (None, None)


def test_review_drops_weak_and_calls_the_proxy_only_when_needed(tmp_path, monkeypatch):
    proxies = []

    def proxy(index, cand):
        path = tmp_path / f"p{index}.mp4"
        path.write_bytes(b"v")
        proxies.append(index)
        return path

    answers = iter([_verdict(virality=90), _verdict(decision="weak")])
    monkeypatch.setattr(hj, "ask_gemini", lambda *a: (next(answers), None))
    cands = [_cand(10, 50), _cand(100, 140)]
    shorts, _ = hj.review(cands, _transcript(), 600, proxy=proxy, cache_dir=str(tmp_path / "c"),
                          gameplay=False, min_secs=30, max_secs=60, api_key="k", workers=1)
    assert [s["hype_slot"] for s in shorts] == [1] and proxies == [1, 2]

    proxies.clear()
    shorts, _ = hj.review(cands, _transcript(), 600, proxy=proxy, cache_dir=str(tmp_path / "c2"),
                          gameplay=False, min_secs=30, max_secs=60, api_key=None)
    assert proxies == [] and len(shorts) == 2
    shorts, _ = hj.review(cands, _transcript(), 600, proxy=proxy, cache_dir=str(tmp_path / "c3"),
                          gameplay=False, min_secs=30, max_secs=60, api_key=None, need_proxy=True)
    assert sorted(proxies) == [1, 2]


# --- the review the page reads ---------------------------------------------------

def test_public_result_reports_the_engine_and_preview_offset():
    plan = {"source_video": ".audio-first", "source_duration": 600.0, "deferred": {"id": "x"}, "hype": True,
            "shorts": [{"start": 104.0, "end": 140.0, "preview_slot": 1, "preview_origin": 100.0,
                        "hype": {"score": 0.9}, "assessment": {"verdict": "supported"}}]}
    out = candidate_review.public_result(plan)
    assert out["engine"] == "hype"
    assert out["candidates"][0]["preview_offset"] == 4.0
    assert out["candidates"][0]["hype"] == {"score": 0.9}


def test_a_refused_key_stops_the_judge_after_the_first_refusal(tmp_path, monkeypatch, capsys):
    calls = []

    def refused(*a):
        calls.append(1)
        raise RuntimeError("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.")

    monkeypatch.setattr(hj, "ask_gemini", refused)

    def proxy(index, cand):
        path = tmp_path / f"p{index}.mp4"
        path.write_bytes(bytes([index]))
        return path

    cands = [_cand(i * 100, i * 100 + 40) for i in range(1, 6)]
    shorts, _ = hj.review(cands, _transcript(), 1000, proxy=proxy, cache_dir=str(tmp_path / "c"),
                          gameplay=False, min_secs=30, max_secs=60, api_key="bad", workers=1)
    assert len(calls) == 1 and len(shorts) == 5
    assert all(s["assessment"]["verdict"] == "review" for s in shorts)
    assert capsys.readouterr().out.count("refused the key") == 1


def test_other_provider_errors_only_cost_their_own_candidate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hj, "ask_gemini", lambda *a: calls.append(1) or (_ for _ in ()).throw(RuntimeError("500 INTERNAL")))

    def proxy(index, cand):
        path = tmp_path / f"q{index}.mp4"
        path.write_bytes(bytes([index]))
        return path

    hj.review([_cand(100, 140), _cand(300, 340)], _transcript(), 1000, proxy=proxy,
              cache_dir=str(tmp_path / "c"), gameplay=False, min_secs=30, max_secs=60, api_key="k", workers=1)
    assert len(calls) == 2


class _Quota(Exception):
    code = 429

    def __init__(self, text):
        super().__init__(text)


def test_a_daily_quota_switches_to_the_fallback_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL_JUDGE", "judge-model")
    monkeypatch.setenv("GEMINI_MODEL", "fallback-model")
    asked = []

    def gemini(video, prompt, key, model):
        asked.append(model)
        if model == "judge-model":
            raise hj.QuotaExhausted()
        return _verdict(virality=88), None

    monkeypatch.setattr(hj, "ask_gemini", gemini)

    def proxy(index, cand):
        path = tmp_path / f"z{index}.mp4"
        path.write_bytes(bytes([index]))
        return path

    shorts, _ = hj.review([_cand(100, 140), _cand(300, 340)], _transcript(), 1000, proxy=proxy,
                          cache_dir=str(tmp_path / "c"), gameplay=False, min_secs=30, max_secs=60,
                          api_key="k", workers=1, context="BEATING GTA 6 — Some Streamer")
    assert asked == ["judge-model", "fallback-model", "fallback-model"]
    assert [s["assessment"]["judge"] for s in shorts] == ["fallback-model", "fallback-model"]
    assert all(s["assessment"]["virality"] == 88 for s in shorts)


def test_quota_errors_are_classified():
    daily = _Quota("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier "
                   "{'retryDelay': '82785s'}")
    with pytest.raises(hj.QuotaExhausted):
        hj._quota(daily)
    assert hj._quota(_Quota("429 RESOURCE_EXHAUSTED {'retryDelay': '12s'}")) == 12.0
    assert hj._quota(RuntimeError("500 INTERNAL")) is None


def test_the_stream_title_reaches_the_judge(tmp_path, monkeypatch):
    prompts = []
    monkeypatch.setattr(hj, "ask_gemini", lambda v, prompt, k, m: prompts.append(prompt) or (_verdict(), None))
    clip = tmp_path / "j.mp4"
    clip.write_bytes(b"v")
    hj.judge(clip, _cand(0, 40), language="en", api_key="k", cache_dir=str(tmp_path / "c"),
             min_secs=30, max_secs=60, context="BEATING GTA 6 (ALL ENDINGS) — Kreek")
    assert "BEATING GTA 6 (ALL ENDINGS) — Kreek" in prompts[0]
