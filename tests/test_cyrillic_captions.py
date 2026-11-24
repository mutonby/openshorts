import re

import subtitles


def test_cyrillic_captions_keep_lines_inside_the_frame(tmp_path):
    transcript = {
        "language": "ru",
        "segments": [{
            "start": 0.0, "end": 1.0, "text": "И ВОТ В ЧЁМ,",
            "words": [
                {"word": " И", "start": 0.0, "end": 0.15},
                {"word": " ВОТ", "start": 0.15, "end": 0.35},
                {"word": " В", "start": 0.35, "end": 0.5},
                {"word": " ЧЁМ,", "start": 0.5, "end": 0.9},
            ],
        }],
    }
    out = tmp_path / "captions.ass"

    assert subtitles.generate_ass(
        transcript, 0.0, 1.0, str(out), max_chars=18, fontsize=40,
        font_name="Anton", uppercase=True, highlight_color="#FFE500")

    dialogue = [line for line in out.read_text(encoding="utf-8-sig").splitlines()
                if line.startswith("Dialogue:")]
    texts = [re.sub(r"\{[^}]*\}", "", line.split(",", 9)[-1]).strip()
             for line in dialogue]
    assert texts
    assert all(len(text) <= subtitles._CYRILLIC_MAX_CHARS for text in texts)
    assert "И ВОТ В ЧЁМ," not in texts
