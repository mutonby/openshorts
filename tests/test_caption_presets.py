"""Caption presets: the API, the MCP tool and the dashboard agree on them."""
import asyncio
import re

from subtitles import AUTO_CAPTION_STYLE, CAPTION_PRESETS, line_budget


def test_dashboard_offers_every_server_preset():
    src = open("dashboard/src/components/SubtitleModal.jsx").read()
    ids = set(re.findall(r"\{ id: '([a-z]+)'", src))
    missing = set(CAPTION_PRESETS) - ids
    assert not missing, f"presets the dashboard does not offer: {missing}"


def test_default_preset_is_what_clips_ship_with():
    d = CAPTION_PRESETS["default"]
    for k in ("font_name", "font_size", "highlight_color", "border_width",
              "effect", "uppercase", "max_chars", "max_duration"):
        assert d[k] == AUTO_CAPTION_STYLE[k], k


def test_line_budget_shrinks_with_size_and_wide_fonts():
    assert line_budget("Anton", 44) == 16
    assert line_budget("Anton", 70) < line_budget("Anton", 44) < line_budget("Anton", 34)
    assert line_budget("Montserrat ExtraBold", 44) < line_budget("Anton", 44)


class _FakeResp:
    status_code = 200

    def json(self):
        return {"ok": True}


class _FakeClient:
    def __init__(self):
        self.body = None

    async def post(self, path, json):
        assert path == "/api/subtitle"
        self.body = json
        return _FakeResp()


def _call(**args):
    from mcp_server import _tool_add_subtitles
    client = _FakeClient()
    asyncio.run(_tool_add_subtitles(client, {"job_id": "j", "clip_index": 0, **args}))
    return client.body


def test_mcp_forwards_preset_and_new_options():
    body = _call(preset="pill", reveal=True, shadow=2, effect="highlight")
    assert body["preset"] == "pill"
    assert (body["reveal"], body["shadow"], body["effect"]) == (True, 2, "highlight")
    assert "max_chars" not in body  # the preset's own budget applies


def test_mcp_one_word_and_resize_set_the_line_budget():
    assert _call(one_word=True)["max_chars"] == 1
    assert _call(font_size=70)["max_chars"] == line_budget("Anton", 70)
    # Resizing the one-word preset keeps it one word.
    assert "max_chars" not in _call(preset="oneword", font_size=56)


def test_unknown_preset_is_a_400():
    import pytest
    from fastapi import HTTPException
    from app import SubtitleRequest, add_subtitles
    with pytest.raises(HTTPException) as exc:
        asyncio.run(add_subtitles(SubtitleRequest(job_id="j", clip_index=0, preset="nope"), None))
    assert exc.value.status_code == 400


def test_preset_fills_only_fields_not_sent():
    from app import SubtitleRequest
    req = SubtitleRequest(job_id="j", clip_index=0, preset="hormozi", highlight_color="#00FF00")
    merged = req.model_copy(update={k: v for k, v in CAPTION_PRESETS["hormozi"].items()
                                    if k not in req.model_fields_set})
    assert merged.highlight_color == "#00FF00"
    assert merged.reveal is True and merged.font_name == "Montserrat ExtraBold"


# --- one line per block ----------------------------------------------------

def _words(text, step=0.3):
    return {"segments": [{"words": [
        {"word": " " + w, "start": i * step, "end": i * step + step - 0.02}
        for i, w in enumerate(text.split())]}]}


def _default_ass(tmp_path, text, **over):
    from subtitles import generate_ass
    st = {**AUTO_CAPTION_STYLE, **over}
    out = tmp_path / "c.ass"
    assert generate_ass(_words(text), 0, 60, str(out), max_chars=st["max_chars"],
                        max_duration=st["max_duration"], fontsize=st["font_size"],
                        font_name=st["font_name"], border_width=st["border_width"],
                        effect=st["effect"], uppercase=st["uppercase"])
    return out.read_text(encoding="utf-8-sig")


def _lines(ass):
    """The text of every event, override tags removed."""
    return [re.sub(r"\{[^}]*\}", "", ln.split(",", 9)[9])
            for ln in ass.splitlines() if ln.startswith("Dialogue:")]


def test_a_block_never_needs_a_second_line(tmp_path):
    from subtitles import line_fitter, line_width, _LINE_UNITS
    # Inside the character budget yet wider than the frame in Anton
    # capitals: this is the block that used to jump to two lines.
    text = "immortality, which is a mmmm wwww mmmm wwww il it is at if in"
    ass = _default_ass(tmp_path, text)
    assert "WrapStyle: 2" in ass
    size = int(AUTO_CAPTION_STYLE["font_size"] * 0.85)
    shown = set(_lines(ass))
    assert "IMMORTALITY, WHICH IS A" not in shown
    for line in shown:
        assert line_width("Anton", size, line) <= _LINE_UNITS, line
    assert line_fitter("Anton", size, uppercase=True)(["il", "it", "is", "at"])


def test_a_lone_word_wider_than_the_frame_is_narrowed(tmp_path):
    ass = _default_ass(tmp_path, "ok " + "w" * 30 + " ok")
    event = next(ln for ln in ass.splitlines() if "W" * 30 in ln)
    assert re.search(r"\\fscx(\d+)", event) and int(re.search(r"\\fscx(\d+)", event).group(1)) < 100


def test_an_unmeasured_font_keeps_the_old_wrapping(tmp_path):
    from subtitles import line_fitter
    assert line_fitter("Some Font Nobody Ships", 27) is None
    assert "WrapStyle: 0" in _default_ass(tmp_path, "one two three four", font_name="Verdana")
