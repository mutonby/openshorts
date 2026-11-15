"""Find subtitles already burned into the source, so ours do not stack on them.

A TV show re-upload came with its own subtitles burned into the bottom of the
picture. The pipeline then burned its own captions as well, and every clip
carried two layers of text: ours just above the source's, the source's cut off
at both sides wherever the 9:16 crop was narrower than the line.

The fix keeps one layer, ours, by trimming the source's subtitle band off the
bottom of every cut clip before it is reframed (``crop_filter``). Cropping from
the bottom keeps the top-left origin, so every face box and crop position the
reframer computes is unchanged; only the height shrinks.

The question goes to Gemini on the same 12 stills at 1024px the layout picker
uses, asked once per source video. It is a separate call on purpose: the layout
prompt's accuracy was measured as is, and a second question in its schema
would change what was measured. Per frame, the model gives the top of the
subtitle text or null; the decision (present in at least half the frames) and
the band (the highest line seen, plus a margin) are made here. The band is a
position, not a routing decision: off by a few percent it trims a sliver more or
less, it never changes a layout.

Off with ``SUBTITLE_BAND_CHECK=0``. Any failure means "no subtitles": the clip
renders exactly as it did before this module existed.
"""
import json
import os

ENABLED = os.environ.get("SUBTITLE_BAND_CHECK", "1").strip() != "0"

# Subtitles must be in at least this share of the readable frames. Lines come
# and go with speech, so a real subtitled video still has frames without text:
# a subtitled TV show showed them in 6 of 12 stills, twice; an unsubtitled
# podcast and a news interview with a ticker checked alongside, in 0 of 12.
MIN_PRESENCE = 0.4

# Never trim more than the bottom quarter. A band reported higher than this is
# not a subtitle track (a lower-third, a caption card, a misread) and trimming
# that much would cost a real part of the picture.
HIGHEST_BAND = 0.75

# Below this the band is a sliver not worth a re-encode.
LOWEST_BAND = 0.97

# Head-room above the highest line seen: descenders of the line above, a box
# behind the text, a frame the model read a little low.
MARGIN = 0.02

PROMPT = """
Each image is a frame sampled from the same landscape video, in order.

For EVERY image, look only for SUBTITLES burned into the picture: lines of
dialogue text, usually centred near the bottom, that transcribe or translate
what is being said. Return its number and subtitle_top: the vertical position
of the TOP of the highest subtitle line, as a fraction of the image height
(0 = top edge, 1 = bottom edge), or null when the frame has no subtitle text.

NOT subtitles (answer null): channel logos, watermarks, news tickers and
crawls, name lower-thirds, score bugs, titles, on-screen graphics, text that is
part of the scene (signs, slides, screens, product labels).
"""


def band_top(answers):
    """Where to cut, as a fraction of the height, from per-frame answers.

    ``answers`` is a list of ``subtitle_top`` values (float or None), one per
    readable frame. Returns None when there is no subtitle track to remove.
    """
    if not answers:
        return None
    tops = []
    for value in answers:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < v < 1.0:
            tops.append(v)
    if len(tops) < MIN_PRESENCE * len(answers):
        return None
    tops.sort()
    # The highest line seen, ignoring one stray reading: two-line subtitles
    # set the band, a single frame misread as half-way up does not.
    top = tops[1] if len(tops) >= 4 else tops[0]
    cut = top - MARGIN
    if cut < HIGHEST_BAND or cut > LOWEST_BAND:
        return None
    return round(cut, 3)


def crop_filter(band):
    """ffmpeg filter keeping the picture above ``band``, with an even height."""
    return f"crop=iw:trunc(ih*{float(band):.4f}/2)*2:0:0"


def detect(video_path, frames=None):
    """Top of the source's burned-in subtitle band (0..1), or None.

    Never raises: like the layout picker, a missing answer degrades to today's
    render.
    """
    if not ENABLED:
        return None
    import llm_backend

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not llm_backend.active():
        return None
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    try:
        from google import genai
        from google.genai import types as genai_types
        from pydantic import BaseModel

        import gemini_worker
        import layout_picker

        class FrameSubtitle(BaseModel):
            frame: int
            subtitle_top: float | None

        class SubtitleAnswer(BaseModel):
            frames: list[FrameSubtitle]

        frames = frames if frames is not None else layout_picker.sample_frames(video_path)
        if not frames:
            return None
        if llm_backend.active():
            labelled = []
            for n, jpg in enumerate(frames):
                labelled.append(f"Frame {n}:")
                labelled.append(jpg)
            answer, _ = llm_backend.generate_json(
                PROMPT + f"\nThere are exactly {len(frames)} frames, numbered 0 to "
                f"{len(frames) - 1}. Return exactly {len(frames)} entries, one per frame, "
                "then stop.", SubtitleAnswer, images=labelled)
        else:
            client = genai.Client(api_key=api_key)
            parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                     for b in frames]
            response = client.models.generate_content(
                model=model_name,
                contents=parts + [PROMPT],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SubtitleAnswer,
                ))
            gemini_worker.raise_if_blocked(response)
            answer = json.loads(response.text) or {}
    except Exception as e:
        print(f"   ⚠️ Burned-in subtitle check failed ({e}) — captions as usual.")
        return None

    per_frame = [f.get("subtitle_top") for f in answer.get("frames", [])
                 if isinstance(f, dict)][:len(frames)]
    band = band_top(per_frame)
    seen = sum(1 for v in per_frame if v is not None)
    if band is None:
        print(f"   💬 No burned-in subtitle track ({seen}/{len(per_frame)} frames with text).")
    else:
        print(f"   💬 Source has burned-in subtitles ({seen}/{len(per_frame)} frames) — "
              f"trimming below {band:.0%} so only our captions show.")
    return band
