import text_watermark as tw


def test_clean_clamps_and_rejects_unknown_values():
    s = tw.clean({"enabled": True, "text": "  @my\nchannel  ", "opacity": 7, "size": -3,
                  "position": "nowhere"})
    assert s == {"enabled": True, "text": "@my channel", "opacity": 1.0, "size": 1.0,
                 "position": "bottom-right"}


def test_empty_text_is_never_enabled():
    assert tw.clean({"enabled": True, "text": "   "})["enabled"] is False


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / ".wm.json")
    tw.save({"enabled": True, "text": "मेरा चैनल", "opacity": 0.4, "size": 6,
             "position": "top-left"}, path)
    assert tw.load(path) == {"enabled": True, "text": "मेरा चैनल", "opacity": 0.4,
                             "size": 6.0, "position": "top-left"}


def test_load_missing_file_is_off(tmp_path):
    assert tw.load(str(tmp_path / "missing.json"))["enabled"] is False


def test_ass_document_places_and_fades_the_text():
    doc = tw.ass_document({"enabled": True, "text": "{\\b1}hi", "opacity": 0.5, "size": 5,
                           "position": "top-left"}, 1080, 1920)
    assert "PlayResX: 1080" in doc and "PlayResY: 1920" in doc
    style = next(line for line in doc.splitlines() if line.startswith("Style: Mark"))
    fields = style.split(",")
    assert fields[2] == "96"
    assert fields[3] == "&H80FFFFFF"
    assert fields[18] == "7"
    assert doc.rstrip().endswith("(/b1)hi")


def test_apply_does_nothing_when_disabled(tmp_path):
    assert tw.apply(str(tmp_path / "clip.mp4"), {"enabled": False, "text": "x"}) is None
