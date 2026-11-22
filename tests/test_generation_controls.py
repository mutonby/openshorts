"""Unit tests for the manual generation controls (discussion #65): the
clip-duration band helper and the prompt templates it feeds.

Pure logic only — no API calls, no heavy imports (gemini_worker's templates are
read via ast so the ML stack never loads in CI).
"""
import ast
import os

import pytest

import clip_selection


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLIP_MIN_SECONDS", raising=False)
    monkeypatch.delenv("CLIP_MAX_SECONDS", raising=False)


class TestClipDurationBounds:
    def test_defaults_are_the_classic_band(self):
        assert clip_selection.clip_duration_bounds() == (15.0, 60.0)

    def test_env_overrides_apply(self, monkeypatch):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "20")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "30")
        assert clip_selection.clip_duration_bounds() == (20.0, 30.0)

    def test_clamped_to_platform_limits(self, monkeypatch):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "1")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "9999")
        assert clip_selection.clip_duration_bounds() == (5.0, 180.0)

    def test_degenerate_band_keeps_a_real_range(self, monkeypatch):
        # max below min must widen, not invert or collapse.
        monkeypatch.setenv("CLIP_MIN_SECONDS", "60")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "20")
        lo, hi = clip_selection.clip_duration_bounds()
        assert lo == 60.0 and hi >= lo + 5.0

    def test_garbage_falls_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "garbage")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "")
        assert clip_selection.clip_duration_bounds() == (15.0, 60.0)


def _templates():
    """The prompt template constants, read without importing gemini_worker
    (which pulls google-genai — absent in the thin CI env)."""
    mod = ast.parse(open(os.path.join(os.path.dirname(__file__), "..",
                                      "gemini_worker.py"), encoding="utf-8").read())
    return {
        node.targets[0].id: node.value.value
        for node in mod.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.endswith("_PROMPT_TEMPLATE")
    }


class TestPromptTemplates:
    def test_detail_template_carries_the_band(self):
        text = _templates()["DETAIL_PROMPT_TEMPLATE"].format(
            video_duration=100, language="es", min_clips=2, max_clips=4,
            min_secs=10.0, max_secs=20.0, windows_json="[]")
        assert "10 to 20 seconds" in text
        assert "return 2 to 4 clips" in text

    def test_visual_template_carries_band_and_count(self):
        text = _templates()["VISUAL_PROMPT_TEMPLATE"].format(
            video_duration=100, language="es", min_clips=2, max_clips=4,
            min_secs=10.0, max_secs=20.0)
        assert "2–4 MOST engaging" in text
        assert "10 to 20 seconds" in text

    def test_score_template_ranks_every_window(self):
        # The pass used to SELECT ("choose up to 3 windows from this batch"),
        # which capped the shortlist at 3 * n_batches and threw away the
        # ranking it was computing. It scores everything now.
        text = _templates()["SCORE_PROMPT_TEMPLATE"].format(
            video_duration=100, language="es", windows_json="[]")
        assert "Score EVERY window" in text
        assert "up to 3 windows" not in text
