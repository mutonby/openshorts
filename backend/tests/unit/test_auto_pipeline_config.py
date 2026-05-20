"""Tests for the short-form auto-pipeline config parsers in app.main.

These pin SubtitleStyle bounds, _parse_subtitle_style error paths, and
_normalize_bool_form coercion. The pipeline itself (apply_ai_edit /
apply_subtitles) is exercised by Phase 5 e2e tests.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import (
    ColorGradeRequest,
    SilenceCutRequest,
    SubtitleStyle,
    _AUTO_ALLOWED_CATEGORIES,
    _normalize_bool_form,
    _parse_subtitle_style,
)


# --- SubtitleStyle ----------------------------------------------------------

def test_subtitle_style_defaults_are_valid():
    s = SubtitleStyle.model_validate({})
    assert s.position == "bottom"
    assert s.font_size == 16
    assert s.font_name == "Verdana"
    assert s.font_color == "#FFFFFF"
    assert s.bg_opacity == 0.0
    assert s.words_per_line is None
    assert s.text_case is None


@pytest.mark.parametrize(
    "grid, expected",
    [
        ("bottom-center", "bottom"),
        ("TOP-LEFT", "top"),
        ("middle-right", "middle"),
        ("bottom", "bottom"),
    ],
)
def test_subtitle_style_aliases_brand_kit_grid(grid, expected):
    s = SubtitleStyle.model_validate({"position": grid})
    assert s.position == expected


def test_subtitle_style_rejects_unknown_position():
    with pytest.raises(ValidationError):
        SubtitleStyle.model_validate({"position": "centre-of-mass"})


@pytest.mark.parametrize(
    "case, expected",
    [
        ("upper", "upper"),
        ("LOWER", "lower"),
        ("original", "original"),
        ("", None),
        (None, None),
    ],
)
def test_subtitle_style_text_case_normalization(case, expected):
    s = SubtitleStyle.model_validate({"text_case": case})
    assert s.text_case == expected


def test_subtitle_style_rejects_invalid_text_case():
    with pytest.raises(ValidationError):
        SubtitleStyle.model_validate({"text_case": "title"})


@pytest.mark.parametrize(
    "field, bad",
    [
        ("font_size", 7),    # below ge=8
        ("font_size", 121),  # above le=120
        ("border_width", -1),
        ("border_width", 21),
        ("bg_opacity", -0.1),
        ("bg_opacity", 1.01),
        ("words_per_line", -1),
        ("words_per_line", 21),
    ],
)
def test_subtitle_style_numeric_bounds(field, bad):
    with pytest.raises(ValidationError):
        SubtitleStyle.model_validate({field: bad})


@pytest.mark.parametrize(
    "hex_value",
    ["#fff", "FFFFFF", "#1234567", "rgb(0,0,0)", "#ZZZZZZ", "", None, 123],
)
def test_subtitle_style_rejects_bad_hex_color(hex_value):
    with pytest.raises(ValidationError):
        SubtitleStyle.model_validate({"font_color": hex_value})


def test_subtitle_style_hex_color_uppercased():
    s = SubtitleStyle.model_validate({"font_color": "#abcdef"})
    assert s.font_color == "#ABCDEF"


# --- _normalize_bool_form ---------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        (None, False),
        ("anything-else", False),
    ],
)
def test_normalize_bool_form(raw, expected):
    assert _normalize_bool_form(raw) is expected


# --- _parse_subtitle_style --------------------------------------------------

def test_parse_subtitle_style_empty_returns_empty_dict():
    assert _parse_subtitle_style(None) == {}
    assert _parse_subtitle_style("") == {}


def test_parse_subtitle_style_round_trip_defaults():
    out = _parse_subtitle_style("{}")
    assert out["position"] == "bottom"
    assert out["font_size"] == 16


def test_parse_subtitle_style_applies_validation_and_aliasing():
    payload = json.dumps({
        "position": "bottom-center",
        "font_size": 72,
        "font_color": "#ffffff",
        "border_color": "#000000",
        "border_width": 6,
        "words_per_line": 2,
        "text_case": "upper",
    })
    out = _parse_subtitle_style(payload)
    assert out["position"] == "bottom"
    assert out["font_size"] == 72
    assert out["font_color"] == "#FFFFFF"
    assert out["text_case"] == "upper"


def test_parse_subtitle_style_rejects_non_json():
    with pytest.raises(HTTPException) as exc:
        _parse_subtitle_style("not-json-at-all")
    assert exc.value.status_code == 400


def test_parse_subtitle_style_rejects_non_object():
    with pytest.raises(HTTPException) as exc:
        _parse_subtitle_style("[1, 2, 3]")
    assert exc.value.status_code == 400


def test_parse_subtitle_style_rejects_out_of_bounds():
    payload = json.dumps({"font_size": 500})
    with pytest.raises(HTTPException) as exc:
        _parse_subtitle_style(payload)
    assert exc.value.status_code == 400


# --- _AUTO_ALLOWED_CATEGORIES sanity check ---------------------------------

def test_categories_allowlist_matches_frontend_contract():
    # Matches the four cards in frontend/src/pages/ShortForm/steps/Categorize.jsx.
    assert _AUTO_ALLOWED_CATEGORIES == {"educational", "yap", "live", "viral"}


# --- Phase 2: ColorGradeRequest --------------------------------------------

def test_color_grade_request_normalizes_lut_case():
    r = ColorGradeRequest(job_id="j", clip_index=0, lut_name=" Teal_Orange ")
    assert r.lut_name == "teal_orange"


@pytest.mark.parametrize("name", ["cool", "noir", "teal_orange", "vivid", "warm"])
def test_color_grade_request_accepts_every_preset(name):
    r = ColorGradeRequest(job_id="j", clip_index=0, lut_name=name)
    assert r.lut_name == name


def test_color_grade_request_rejects_unknown_lut():
    with pytest.raises(ValidationError):
        ColorGradeRequest(job_id="j", clip_index=0, lut_name="neon_dreams")


def test_color_grade_request_rejects_empty_lut():
    with pytest.raises(ValidationError):
        ColorGradeRequest(job_id="j", clip_index=0, lut_name="")


# --- Phase 2: SilenceCutRequest --------------------------------------------

def test_silence_cut_request_defaults():
    r = SilenceCutRequest(job_id="j", clip_index=0)
    assert r.noise_db == -30.0
    assert r.min_silence_sec == 0.5
    assert r.input_filename is None


def test_silence_cut_request_rejects_positive_noise_db():
    with pytest.raises(ValidationError):
        SilenceCutRequest(job_id="j", clip_index=0, noise_db=5.0)


def test_silence_cut_request_rejects_overlong_silence_threshold():
    with pytest.raises(ValidationError):
        SilenceCutRequest(job_id="j", clip_index=0, min_silence_sec=999.0)


def test_silence_cut_request_rejects_too_short_silence_threshold():
    with pytest.raises(ValidationError):
        SilenceCutRequest(job_id="j", clip_index=0, min_silence_sec=0.001)
