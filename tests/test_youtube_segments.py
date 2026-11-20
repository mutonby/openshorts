import youtube_segments as ys


def test_plan_ranges_merges_overlapping_padded_clips():
    clips = [
        {"start": 10, "end": 20},
        {"start": 19, "end": 30},
        {"start": 100, "end": 110},
    ]
    ranges = ys.plan_ranges(clips, 200, padding=2)
    assert len(ranges) == 2
    assert ranges[0]["start"] == 8
    assert ranges[0]["end"] == 32
    assert ranges[0]["clip_indexes"] == [0, 1]
    assert ranges[1]["start"] == 98
    assert ranges[1]["end"] == 112


def test_remap_clips_preserves_original_timestamps_and_compacts_ranges():
    clips = [
        {"start": 10, "end": 20, "video_title_for_youtube_short": "A"},
        {"start": 100, "end": 110, "video_title_for_youtube_short": "B"},
    ]
    ranges = ys.plan_ranges(clips, 200, padding=2)
    mapped, duration = ys.remap_clips(clips, ranges)
    assert mapped[0]["start"] == 2
    assert mapped[0]["end"] == 12
    assert mapped[0]["source_start"] == 10
    assert mapped[1]["start"] == 16
    assert mapped[1]["end"] == 26
    assert mapped[1]["source_start"] == 100
    assert duration == 28
    assert clips[0]["start"] == 10  # input objects are not mutated


def test_remap_transcript_shifts_segments_and_word_timestamps():
    transcript = {
        "text": "hello world again",
        "language": "en",
        "segments": [
            {"start": 0, "end": 5, "text": "before", "words": []},
            {"start": 10, "end": 20, "text": "hello world",
             "words": [
                 {"start": 10, "end": 12, "text": "hello"},
                 {"start": 18, "end": 20, "text": "world"},
             ]},
            {"start": 100, "end": 110, "text": "again",
             "words": [{"start": 100, "end": 102, "text": "again"}]},
        ],
    }
    ranges = ys.plan_ranges(
        [{"start": 10, "end": 20}, {"start": 100, "end": 110}],
        200, padding=2)
    _clips, _duration = ys.remap_clips(
        [{"start": 10, "end": 20}, {"start": 100, "end": 110}], ranges)
    out = ys.remap_transcript(transcript, ranges)
    assert len(out["segments"]) == 2
    assert out["segments"][0]["start"] == 2
    assert out["segments"][0]["end"] == 12
    assert out["segments"][0]["words"][0]["start"] == 2
    assert out["segments"][1]["start"] == 16
    assert out["segments"][1]["words"][0]["start"] == 16
    assert "before" not in out["text"]
