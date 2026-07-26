"""Tests for agency-tier helpers in brutal_truth module."""
import json
import os
from pathlib import Path

import pytest
import brutal_truth


# ---- render presets ----

def test_render_preset_draft():
    p = brutal_truth.render_preset_to_ffmpeg_args("draft")
    assert p["width"] == 540
    assert p["crf"] == 28
    assert p["preset"] == "ultrafast"


def test_render_preset_review():
    p = brutal_truth.render_preset_to_ffmpeg_args("review")
    assert p["width"] == 720
    assert p["crf"] == 23


def test_render_preset_final():
    p = brutal_truth.render_preset_to_ffmpeg_args("final")
    assert p["width"] == 1080
    assert p["crf"] == 19
    assert p["preset"] == "slow"


def test_render_preset_unknown_falls_back_to_review():
    p = brutal_truth.render_preset_to_ffmpeg_args("garbage")
    assert p["width"] == 720
    assert p["crf"] == 23


# ---- naming templates ----

def test_naming_template_basic():
    name = brutal_truth.naming_template(
        "{client}_{title}_clip_{n}_{date}.{fmt}",
        client="Acme Corp",
        title="Mr Beast vs Cake!",
        fmt="mp4",
        date="20260723",
    )
    assert "Acme_Corp" in name
    assert "Mr_Beast_vs_Cake" in name
    assert "{n}" in name  # left for caller to fill per-clip
    assert "_20260723.mp4" in name


def test_naming_template_default_date_today():
    import re
    name = brutal_truth.naming_template(
        "{client}_{title}_clip_{n}_{date}.{fmt}", "Acme", "My Clip")
    base, ext = os.path.splitext(name)
    # default date is today's YYYYMMDD (8 digits) embedded in name
    assert re.search(r"\d{8}", base), f"no date in {name!r}"
    assert "{n}" in name  # caller fills per-clip
    assert "(date)" not in name  # date was auto-filled


def test_naming_template_sanitizes_special_chars():
    name = brutal_truth.naming_template(
        "{client}_{title}_clip_{n}.{fmt}",
        "Acme!?/Co",
        "Fire! & Ice!",
        "mp4",
        "20260101")
    assert "Acme" in name
    assert "Co" in name
    assert "Fire" in name
    assert "Ice" in name
    # special chars replaced with underscores
    assert "!" not in name
    assert "/" not in name


# ---- white-label metadata ----

def test_whitelabel_adds_client_tag():
    clip = {"hashtags": ["#viralk"], "video_description_for_tiktok": "Hello"}
    brutal_truth.whitelabel_clip_metadata(clip, "acme")
    assert clip["client"] == "acme"


def test_whitelabel_merges_brand_hashtags_dedup():
    clip = {"hashtags": ["#a", "#b"], "video_description_for_tiktok": "x"}
    brutal_truth.whitelabel_clip_metadata(
        clip, "acme", brand_hashtags=["#b", "#c", "#d"])
    assert clip["hashtags"] == ["#a", "#b", "#c", "#d"]


def test_whitelabel_caps_hashtags_at_12():
    clip = {"hashtags": ["#t%d" % i for i in range(10)]}
    brutal_truth.whitelabel_clip_metadata(
        clip, "acme", brand_hashtags=["#b1", "#b2", "#b3", "#b4"])
    assert len(clip["hashtags"]) == 12


def test_whitelabel_prefixes_caption_once():
    clip = {"video_description_for_tiktok": "Original caption"}
    brutal_truth.whitelabel_clip_metadata(
        clip, "acme", brand_caption_prefix="Sponsored by Acme:")
    # Should only be prefixed once
    count = clip["video_description_for_tiktok"].count("Sponsored by Acme:")
    assert count == 1


# ---- QC state ----

def test_qc_state_creates_file(tmp_path):
    path = brutal_truth.write_qc_state(str(tmp_path), 1, "submitted",
                                       reviewer="alice", note="CG ok?")
    assert Path(path).exists()
    db = json.loads(Path(path).read_text())
    assert db["clips"]["1"]["state"] == "submitted"
    assert db["clips"]["1"]["reviewer"] == "alice"
    assert db["clips"]["1"]["note"] == "CG ok?"
    assert "timestamp" in db["clips"]["1"]


def test_qc_state_rejects_invalid_state(tmp_path):
    with pytest.raises(ValueError):
        brutal_truth.write_qc_state(str(tmp_path), 1, "approved_by_ceo")


def test_qc_state_merges_clips(tmp_path):
    brutal_truth.write_qc_state(str(tmp_path), 1, "pending")
    brutal_truth.write_qc_state(str(tmp_path), 2, "approved")
    brutal_truth.write_qc_state(str(tmp_path), 1, "submitted")
    db = json.loads(Path(tmp_path, "qc_state.json").read_text())
    assert db["clips"]["1"]["state"] == "submitted"
    assert db["clips"]["2"]["state"] == "approved"


# ---- batch expansion ----

def test_expand_batch_with_strings():
    out = brutal_truth.expand_batch(["a.mp4", "b.mp4"])
    assert out == [{"url": "a.mp4"}, {"url": "b.mp4"}]


def test_expand_batch_with_dicts_and_dedup():
    sources = [
        {"url": "a.mp4", "client": "acme"},
        "b.mp4",
        {"path": "a.mp4", "client": "other"},
    ]
    out = brutal_truth.expand_batch(sources)
    # 'a.mp4' appears twice under different keys, but dedup should drop the third
    assert any(o["url"] == "a.mp4" for o in out)
    urls = [o["url"] for o in out]
    assert urls.count("a.mp4") == 1


def test_expand_batch_keeps_metadata():
    sources = [{"url": "a.mp4", "client": "acme", "format": "vertical"}]
    out = brutal_truth.expand_batch(sources)
    assert out[0]["client"] == "acme"
    assert out[0]["format"] == "vertical"
