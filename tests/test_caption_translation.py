"""Translate a clip's caption TEXT to another language (no re-dubbing).

The pure helpers keep the line count/order and degrade to the original line
when the model drops one; the endpoint returns the translated captions as SRT.
"""
import json

import pytest

translate = pytest.importorskip("translate")


def test_prompt_names_the_language_and_every_line():
    p = translate.build_caption_translation_prompt(["a", "b"], "German")
    assert "German" in p and "1. a" in p and "2. b" in p


def test_parse_numbered_translations_recovers_lines():
    assert translate.parse_numbered_translations("1. a\n2. b\n3. c", 3) == ["a", "b", "c"]
    assert translate.parse_numbered_translations("2) b\n1) a", 2) == ["a", "b"]   # out of order
    assert translate.parse_numbered_translations("1. a", 2) == ["a", ""]          # missing -> ''


def test_translate_caption_lines_keeps_count_and_degrades():
    lines = ["one", "two", "three"]
    # the model drops line 2 -> that line keeps its original text
    out = translate.translate_caption_lines(lines, "Spanish", lambda p: "1. uno\n3. tres")
    assert out == ["uno", "two", "tres"]
    assert translate.translate_caption_lines([], "Spanish", lambda p: "") == []


# --- endpoint -------------------------------------------------------------- #

app_module = pytest.importorskip("app")
from starlette.testclient import TestClient


def _meta():
    return {
        "transcript": {"language": "en", "segments": [{"words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": " world", "start": 0.5, "end": 1.0},
        ]}]},
        "shorts": [{"start": 0.0, "end": 5.0}],
    }


def test_translate_captions_endpoint_returns_srt(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "job1").mkdir(parents=True)
    (out / "job1" / "clip_metadata.json").write_text(json.dumps(_meta()))
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "jobs", {"job1": {"status": "completed"}})
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    # Stand in for the Gemini call so the test does no network I/O.
    monkeypatch.setattr(translate, "translate_caption_lines",
                        lambda lines, lang, call: [f"ES:{line}" for line in lines])

    r = TestClient(app_module.app).post(
        "/api/clip/job1/0/translate-captions", json={"target_lang": "es"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert "ES:Hello world" in r.text
    assert "00:00:00,000 --> 00:00:01,000" in r.text


def test_unsupported_language_is_rejected():
    r = TestClient(app_module.app).post(
        "/api/clip/whatever/0/translate-captions", json={"target_lang": "zz"})
    assert r.status_code == 400
