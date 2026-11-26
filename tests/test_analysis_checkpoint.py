import pytest
from pydantic import BaseModel

from analysis_checkpoint import AnalysisBudgetExceeded, AnalysisCheckpoint


class Scores(BaseModel):
    windows: list[dict]


def test_successful_batches_survive_503_and_identical_restart(tmp_path):
    def first_stage(prompt):
        if prompt == "last":
            raise RuntimeError("HTTP 503")
        return {"windows": [{"score": 83}]}, {"input_tokens": 150}

    original = AnalysisCheckpoint(tmp_path, "gemini")
    assert original.batch("model", Scores, "first", lambda: first_stage("first"))[0]
    with pytest.raises(RuntimeError, match="503"):
        original.batch("model", Scores, "last", lambda: first_stage("last"))

    called = []
    resumed = AnalysisCheckpoint(tmp_path, "gemini")
    for prompt in ("first", "last"):
        out, _cost = resumed.batch("model", Scores, prompt, lambda: (
            called.append(prompt) or {"windows": [{"score": 91}]}, None))
        assert out["windows"]
    assert resumed.reused == 1
    assert called == ["last"]


def test_request_budget_charges_attempts_not_reused_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HIGHLIGHT_MAX_AI_REQUESTS", "1")
    run = AnalysisCheckpoint(tmp_path, "gemini")
    def submit():
        run.charge("prompt")
        return {"windows": []}, None

    run.batch("model", Scores, "a", submit)
    run.batch("model", Scores, "a", submit)
    with pytest.raises(AnalysisBudgetExceeded):
        run.batch("model", Scores, "b", submit)
    assert run.report() == {"requests": 1, "reused_batches": 1, "prompt_bytes": 6}


def test_checkpoint_invalidation_includes_schema_model_and_provider(tmp_path):
    class Different(BaseModel):
        windows: list[dict]
        verified: bool

    run = AnalysisCheckpoint(tmp_path, "gemini")
    run.batch("old", Scores, "same prompt", lambda: ({"windows": []}, None))
    assert run.batch("new", Scores, "same prompt", lambda: ({"windows": [{"score": 20}]}, None))[0]["windows"]
    assert run.batch("old", Different, "same prompt", lambda: (
        {"windows": [], "verified": True}, None))[0]["verified"]
    other = AnalysisCheckpoint(tmp_path, "local")
    assert other.batch("old", Scores, "same prompt", lambda: ({"windows": [{"score": 30}]}, None))[0]["windows"]
