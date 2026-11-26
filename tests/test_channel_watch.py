"""The channel watcher: new finished VODs are queued once, live ones wait."""
import pytest

import channel_watch as cw


@pytest.mark.parametrize("value,expected", [
    ("twitch:SomeStreamer", ("twitch", "https://www.twitch.tv/somestreamer/videos?filter=archives&sort=time")),
    ("https://www.twitch.tv/somestreamer", ("twitch", "https://www.twitch.tv/somestreamer/videos?filter=archives&sort=time")),
    ("youtube:@SomeChannel", ("youtube", "https://www.youtube.com/@SomeChannel/streams")),
    ("https://www.youtube.com/@SomeChannel/videos", ("youtube", "https://www.youtube.com/@SomeChannel/streams")),
    ("https://evil.example/x", None), ("twitch:no spaces", None),
])
def test_channel_listing_urls(value, expected):
    assert cw.listing(value) == expected


def _vod(n, live=False):
    url = f"https://www.twitch.tv/videos/{n}"
    return {"id": url, "url": url, "live": live}


class Fakes:
    def __init__(self):
        self.vods = []
        self.durations = {}
        self.sent = []

    def lister(self, channel):
        return list(self.vods)

    def duration_of(self, url):
        return self.durations.get(url)

    def sender(self, api, body):
        self.sent.append(body)
        return 200, {"job_id": f"job{len(self.sent)}"}


def _check(state, fakes, now, **kw):
    return cw.check(state, ["twitch:somestreamer"], api="http://api", now=now, lister=fakes.lister,
                    duration_of=fakes.duration_of, sender=fakes.sender, **kw)


def test_existing_vods_are_the_baseline_and_new_ones_wait_until_finished(tmp_path):
    state, fakes = cw.State(str(tmp_path / "w.sqlite3")), Fakes()
    fakes.vods = [_vod(1), _vod(2)]
    assert _check(state, fakes, 0) == [] and fakes.sent == []           # baseline, nothing queued

    fakes.vods = [_vod(3, live=True), _vod(1), _vod(2)]                  # a stream is live
    assert _check(state, fakes, 600) == []
    fakes.vods = [_vod(3), _vod(1), _vod(2)]                             # ended, VOD still growing?
    fakes.durations[_vod(3)["url"]] = 3600.0
    assert _check(state, fakes, 1200) == []                              # first sight of its length
    assert _check(state, fakes, 1500) == []                              # unchanged, not settled yet
    assert _check(state, fakes, 1900) == ["job1"]                        # unchanged for 10+ minutes
    assert _check(state, fakes, 2600) == []                              # never queued twice
    body = fakes.sent[0]
    assert body["url"] == "https://www.twitch.tv/videos/3"
    assert body["audio_first"] and body["review_candidates"] and body["highlight_discovery"]
    assert body["framing_method"] == "streamer" and body["viral_edit"] == "1"


def test_a_growing_vod_restarts_the_settle_clock(tmp_path):
    state, fakes = cw.State(str(tmp_path / "w.sqlite3")), Fakes()
    _check(state, fakes, 0)
    fakes.vods = [_vod(9)]
    fakes.durations[_vod(9)["url"]] = 1000.0
    _check(state, fakes, 100)
    fakes.durations[_vod(9)["url"]] = 1600.0                              # still being recorded
    assert _check(state, fakes, 800) == []
    assert _check(state, fakes, 1500) == ["job1"]


def test_backfill_queues_the_newest_on_first_sight(tmp_path):
    state, fakes = cw.State(str(tmp_path / "w.sqlite3")), Fakes()
    fakes.vods = [_vod(5), _vod(4), _vod(3)]
    for n in (5, 4, 3):
        fakes.durations[_vod(n)["url"]] = 3000.0
    assert _check(state, fakes, 0, backfill=1) == ["job1"]
    assert fakes.sent[0]["url"].endswith("/5")


def test_a_refused_submission_is_retried_next_pass(tmp_path):
    state, fakes = cw.State(str(tmp_path / "w.sqlite3")), Fakes()
    _check(state, fakes, 0)
    fakes.vods = [_vod(7)]
    fakes.durations[_vod(7)["url"]] = 3000.0
    fakes.sender = lambda api, body: (503, {"detail": "restarting"})
    _check(state, fakes, 100)
    assert _check(state, fakes, 800) == []
    fakes.sender = Fakes().sender
    assert _check(state, fakes, 900) == ["job1"]


def test_acknowledge_is_required():
    with pytest.raises(SystemExit):
        cw.main(["twitch:somestreamer", "--once"])
