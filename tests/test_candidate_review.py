import asyncio
import json
import uuid

import pytest

import candidate_review
from gemini_worker import MomentReview


def proposed(start, end):
    return {"start": start, "end": end, "viral_hook_text": "Unverified event",
            "video_description_for_tiktok": "Completely different incident"}


def test_review_rejects_filler_and_keeps_coherent_copy(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def answer(_video, clip, _words, _key):
        return MomentReview(
            decision="weak" if clip["start"] == 30 else "keep",
            evidence="visible", event="A phone call arrives while both players are together.",
            outcome="The players react to the incoming call.", title="Calling from the same seat",
            opening_title="A call from the seat next to you", hook="clear",
            visual_action="clear", reaction="strong", clarity="strong", payoff="clear",
            source_captions=True)

    monkeypatch.setattr(candidate_review, "_assessment", answer)
    result = candidate_review.assess("source.mp4", [proposed(30, 51), proposed(61, 87)],
                                     {}, 100, gameplay=True, manual_review=True)
    assert [item["start"] for item in result] == [61]
    clip = result[0]
    assert clip["video_description_for_tiktok"] == clip["video_description_for_instagram"]
    assert clip["assessment"]["source_captions"]
    assert clip["assessment"]["verdict"] == "supported"


def test_uncertain_moments_require_review_and_get_neutral_copy(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    clips = [proposed(1, 25)]
    kept = candidate_review.assess("source.mp4", clips, {}, 45,
                                  gameplay=True, manual_review=True)
    assert kept[0]["video_title_for_youtube_short"] == "Gameplay moment"
    assert kept[0]["viral_hook_text"] == ""
    assert kept[0]["assessment"]["verdict"] == "review"
    assert candidate_review.assess("source.mp4", clips, {}, 45,
                                   gameplay=True, manual_review=False) == []


def test_review_takes_the_best_suggestions_across_the_source(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    proposals = [proposed(i * 30, i * 30 + 20) for i in range(candidate_review.MAX_CANDIDATES + 1)]
    proposals[-1]["predicted_score"] = 95
    kept = candidate_review.assess("source.mp4", proposals, {}, 500,
                                  gameplay=True, manual_review=True)
    assert len(kept) == candidate_review.MAX_CANDIDATES
    assert any(clip["start"] == proposals[-1]["start"] for clip in kept)


def test_approval_restores_a_review_from_disk_and_queues_only_selected_ids(tmp_path, monkeypatch):
    app = pytest.importorskip("app")
    from httpx import ASGITransport, AsyncClient

    job_id = str(uuid.uuid4())
    directory = tmp_path / job_id
    directory.mkdir()
    source = directory / "source.mp4"
    source.touch()
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", False)
    enqueued = []
    monkeypatch.setattr(app, "_enqueue_job", lambda *args: enqueued.append(args))
    plan = {"source_video": source.name, "source_duration": 100,
            "output_format": "vertical", "framing_method": "gameplay",
            "shorts": [proposed(20, 45), proposed(55, 80)],
            "transcript": {"segments": [{"text": "Private test transcript"}]}}
    candidate_review.write(directory, plan)
    app._write_resume_manifest(job_id, ["main.py", "--review-candidates"], 2, None, None, False,
                               generation_options={"AUTO_HOOK": "0"})
    app.jobs[job_id] = {"cmd": ["main.py", "--review-candidates"], "logs": app._TimedLog()}
    app._finish_job(job_id, 0, str(directory))
    assert app.jobs.pop(job_id)["status"] == "review"
    assert app._resume_candidates(set(), 10**10)[1] == []
    assert not list(directory.glob("*_metadata.json"))

    async def call():
        async with AsyncClient(transport=ASGITransport(app=app.app), base_url="http://test") as client:
            status = (await client.get(f"/api/status/{job_id}")).json()
            assert status["status"] == "review"
            assert [row["id"] for row in status["result"]["candidates"]] == [1, 2]
            assert "Private test transcript" not in json.dumps(status)
            path = f"/api/jobs/{job_id}/render-candidates"
            for ids in ([], [3], [1, 1], [True]):
                result = await client.post(path, json={"candidate_ids": ids})
                assert result.status_code in (400, 422)
            assert (await client.post(path, json={"candidate_ids": [2]})).status_code == 202
            assert (await client.post(path, json={"candidate_ids": [2]})).status_code == 409

    try:
        asyncio.run(call())
        assert enqueued == [(job_id, 2)]
        job = app.jobs[job_id]
        assert job["cmd"][-2:] == ["--render-candidates", "2"]
        assert job["cmd"][job["cmd"].index("-i") + 1] == str(source)
        assert job["env"]["AUTO_HOOK"] == "0"
        assert "stage" not in app._read_manifest(job_id)
    finally:
        app.jobs.pop(job_id, None)


@pytest.mark.parametrize("slot", ["../../escape", 0, 16, 2.0, True, None])
def test_a_deferred_review_with_an_unsafe_preview_slot_is_not_read(tmp_path, slot):
    """preview_slot names a file (source-<slot>.mkv): a review whose slot is
    anything but a small integer is refused, not turned into a path."""
    clip = {"start": 1.0, "end": 20.0, "preview_slot": slot}
    plan = {"source_video": ".audio-first", "source_duration": 100.0, "shorts": [clip],
            "deferred": {"id": "abcdefghijk"}}
    candidate_review.write(tmp_path, plan)
    assert candidate_review.read(tmp_path) is None


def test_a_deferred_review_with_valid_slots_is_read(tmp_path):
    plan = {"source_video": ".audio-first", "source_duration": 100.0,
            "shorts": [{"start": 1.0, "end": 20.0, "preview_slot": 3}],
            "deferred": {"id": "abcdefghijk"}}
    candidate_review.write(tmp_path, plan)
    assert candidate_review.read(tmp_path)["shorts"][0]["preview_slot"] == 3
