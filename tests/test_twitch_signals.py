"""Viewer clips from a Twitch VOD as a hint for the moment picker."""
import twitch_signals as ts


def test_parse_vod_id_accepts_vod_links_only():
    assert ts.parse_vod_id("https://www.twitch.tv/videos/2290012345") == "2290012345"
    assert ts.parse_vod_id("https://m.twitch.tv/videos/2290012345?t=1h2m") == "2290012345"
    assert ts.parse_vod_id("https://www.twitch.tv/somechannel/video/2290012345") == "2290012345"
    assert ts.parse_vod_id("https://www.twitch.tv/somechannel") is None
    assert ts.parse_vod_id("https://clips.twitch.tv/FunnyClipSlug") is None
    assert ts.parse_vod_id("https://www.twitch.tv/somechannel/clip/FunnyClipSlug") is None
    assert ts.parse_vod_id("https://evil.example/videos/123") is None
    assert ts.parse_vod_id(None) is None


def test_parse_helix_duration():
    assert ts.parse_helix_duration("3h2m1s") == 3 * 3600 + 2 * 60 + 1
    assert ts.parse_helix_duration("45s") == 45
    assert ts.parse_helix_duration("") == 0


def test_shift_clips_moves_into_the_clipped_stretch():
    clips = [{"start": 100, "duration": 30, "views": 5},     # before the stretch
             {"start": 3700, "duration": 30, "views": 9},    # inside
             {"start": 3590, "duration": 30, "views": 2}]    # middle at 3605: inside
    out = ts.shift_clips(clips, 3600, 1800)
    assert out == [{"start": 100.0, "end": 130.0, "views": 9},
                   {"start": 0.0, "end": 20.0, "views": 2}]


def test_shift_clips_without_a_known_duration_keeps_everything_after_start():
    out = ts.shift_clips([{"start": 50, "duration": 20, "views": 1}], 0, float("inf"))
    assert out == [{"start": 50.0, "end": 70.0, "views": 1}]


def _windows(n, size=90):
    return [{"id": f"w{i}", "start": i * size, "end": (i + 1) * size, "text": ""}
            for i in range(n)]


def test_window_heat_counts_each_clip_once_by_midpoint():
    ws = _windows(3)
    clips = [{"start": 80, "end": 110, "views": 10},   # mid 95 -> w1, not w0
             {"start": 100, "end": 120, "views": 3},   # w1
             {"start": 200, "end": 230, "views": 7}]   # w2
    heat = ts.window_heat(ws, clips)
    assert heat["w1"]["clips"] == 2 and heat["w1"]["views"] == 13
    assert heat["w1"]["spans"] == [[80, 110], [100, 120]]
    assert heat["w2"]["clips"] == 1
    assert "w0" not in heat


def test_promote_hot_windows_reserves_a_third_and_keeps_size():
    ws = _windows(12)
    ranked = list(ws)                       # model order: w0 best ... w11 worst
    shortlist = ranked[:6]
    heat = {"w10": {"clips": 3, "views": 900, "spans": []},
            "w9": {"clips": 1, "views": 50, "spans": []},
            "w11": {"clips": 1, "views": 10, "spans": []}}
    out = ts.promote_hot_windows(shortlist, ranked, heat, 6)
    ids = [w["id"] for w in out]
    assert len(out) == 6
    assert "w10" in ids and "w9" in ids          # reserve = 6 // 3 = 2
    assert "w11" not in ids
    assert "w4" not in ids and "w5" not in ids    # the weakest unclipped go
    assert ids[:4] == ["w0", "w1", "w2", "w3"]


def test_promote_hot_windows_is_a_noop_without_heat_or_when_already_in():
    ws = _windows(4)
    assert ts.promote_hot_windows(ws[:3], ws, {}, 3) == ws[:3]
    heat = {"w0": {"clips": 1, "views": 1, "spans": []}}
    assert ts.promote_hot_windows(ws[:3], ws, heat, 3) == ws[:3]


def test_fetch_viewer_clips_without_credentials_is_empty(monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    assert ts.fetch_viewer_clips("123") == []
    assert ts.viewer_clips_for_job("https://www.twitch.tv/videos/123", 0) == []


def test_best_stretch_finds_the_clipped_hour_not_the_start():
    clips = [{"start": 3 * 3600 + 120, "duration": 30, "views": 400},
             {"start": 3 * 3600 + 900, "duration": 25, "views": 150},
             {"start": 600, "duration": 30, "views": 20}]
    s, e = ts.best_stretch(clips, 6 * 3600, 30 * 60)
    assert s == 3 * 3600 + 60 and e == s + 1800      # opens a minute before


def test_best_stretch_stays_inside_the_vod_and_needs_a_signal():
    clips = [{"start": 5 * 3600 + 3500, "duration": 30, "views": 10}]
    s, e = ts.best_stretch(clips, 6 * 3600, 1800)
    assert e == 6 * 3600 and s == e - 1800
    assert ts.best_stretch([], 6 * 3600, 1800) is None
    assert ts.best_stretch(clips, 1200, 1800) is None   # VOD fits whole
