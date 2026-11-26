"""Autopilot (cloud/autopilot.py): the decisions that run without a person.

Everything here fires on its own schedule and spends the user's minutes, so
the pure rules are pinned down: which channel videos get picked, which clips
get published and when, and how each /api/process answer is recorded.
"""
from datetime import datetime, timedelta, timezone

from cloud import autopilot as ap

NOW = datetime(2026, 9, 22, 18, 0, tzinfo=timezone.utc)


def _video(vid, hours_ago):
    ts = (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"id": vid, "timestamp": ts, "permalink": f"https://www.youtube.com/watch?v={vid}"}


class TestPickNewVideos:
    def test_only_videos_after_the_baseline(self):
        baseline = NOW - timedelta(hours=5)
        media = [_video("old", 10), _video("new", 2)]
        assert [v["id"] for v in ap.pick_new_videos(media, baseline, set(), NOW)] == ["new"]

    def test_switching_on_never_picks_the_back_catalogue(self):
        media = [_video("a", 1), _video("b", 30)]
        assert ap.pick_new_videos(media, NOW, set(), NOW) == []

    def test_no_baseline_means_nothing(self):
        assert ap.pick_new_videos([_video("a", 1)], None, set(), NOW) == []

    def test_already_seen_videos_are_skipped(self):
        baseline = NOW - timedelta(days=1)
        media = [_video("a", 1), _video("b", 2)]
        assert [v["id"] for v in ap.pick_new_videos(media, baseline, {"a"}, NOW)] == ["b"]

    def test_too_old_even_if_after_baseline(self):
        baseline = NOW - timedelta(days=30)
        media = [_video("stale", 24 * 5), _video("fresh", 3)]
        assert [v["id"] for v in ap.pick_new_videos(media, baseline, set(), NOW)] == ["fresh"]

    def test_oldest_first(self):
        baseline = NOW - timedelta(days=2)
        media = [_video("newest", 1), _video("older", 20), _video("middle", 8)]
        assert [v["id"] for v in ap.pick_new_videos(media, baseline, set(), NOW)] == \
            ["older", "middle", "newest"]

    def test_items_without_id_or_timestamp_are_ignored(self):
        baseline = NOW - timedelta(days=1)
        media = [{"id": "x"}, {"timestamp": "2026-09-22T17:00:00Z"}, {"id": "y", "timestamp": "garbage"}]
        assert ap.pick_new_videos(media, baseline, set(), NOW) == []


class TestRankClips:
    def test_best_scores_first(self):
        clips = [{"predicted_score": 40}, {"predicted_score": 90}, {"predicted_score": 70}]
        assert ap.rank_clips(clips, 2) == [1, 2]

    def test_unscored_clips_keep_order_after_scored(self):
        clips = [{}, {"predicted_score": "55"}, {}, {"predicted_score": None}]
        assert ap.rank_clips(clips, 4) == [1, 0, 2, 3]

    def test_n_larger_than_clips(self):
        assert ap.rank_clips([{"predicted_score": 1}], 3) == [0]

    def test_zero(self):
        assert ap.rank_clips([{"predicted_score": 1}], 0) == []


class TestScheduleSlots:
    def test_one_a_day_at_the_hour(self):
        now = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
        assert ap.schedule_slots(now, 3, hour=17) == [
            "2026-09-22T17:00:00", "2026-09-23T17:00:00", "2026-09-24T17:00:00"]

    def test_starts_tomorrow_when_the_slot_is_under_an_hour_away(self):
        now = datetime(2026, 9, 22, 16, 30, tzinfo=timezone.utc)
        assert ap.schedule_slots(now, 1, hour=17) == ["2026-09-23T17:00:00"]

    def test_local_timezone(self):
        # 08:00 UTC is 10:00 in Madrid (CEST): 17:00 local is still today.
        now = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
        assert ap.schedule_slots(now, 1, hour=17, tz_name="Europe/Madrid") == ["2026-09-22T17:00:00"]

    def test_unknown_timezone_falls_back_to_utc(self):
        now = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
        assert ap.schedule_slots(now, 1, hour=17, tz_name="Not/AZone") == ["2026-09-22T17:00:00"]


class TestOutcomeFromProcess:
    def test_job_started(self):
        o = ap.outcome_from_process(200, {"job_id": "j1", "status": "queued"})
        assert o == {"status": "processing", "reason": None, "job_id": "j1", "retry": False}

    def test_quality_gate_is_a_skip(self):
        assert ap.outcome_from_process(200, {"needs_confirmation": True})["reason"] == "low_quality"

    def test_out_of_minutes_is_a_skip(self):
        o = ap.outcome_from_process(402, {"detail": {"error": "quota_exceeded"}})
        assert (o["status"], o["reason"], o["retry"]) == ("skipped", "out_of_minutes", False)

    def test_job_limit_is_retried_later(self):
        o = ap.outcome_from_process(429, {"detail": "max jobs"})
        assert o["retry"] is True and o["reason"] == "busy"

    def test_short_source(self):
        detail = ("This video is only 40s long — clip generation needs at least 45s "
                  "of material to cut from. It already is short-form content.")
        assert ap.outcome_from_process(400, {"detail": detail})["reason"] == "too_short"

    def test_other_400_is_unavailable(self):
        o = ap.outcome_from_process(400, {"detail": "Could not determine the video duration."})
        assert (o["status"], o["reason"]) == ("skipped", "unavailable")

    def test_server_error_is_retried(self):
        assert ap.outcome_from_process(502, "not json")["retry"] is True


class TestSmallRules:
    def test_platforms_are_filtered_and_ordered(self):
        assert ap.clean_platforms(["YouTube", "facebook", "tiktok", "x"]) == ["tiktok", "youtube", "facebook"]
        assert ap.clean_platforms(None) == []

    def test_only_paid_plans_are_eligible(self):
        class U:
            def __init__(self, plan, entitled=True):
                self.plan, self.entitled = plan, entitled
        assert ap.eligible(U("starter"))
        assert ap.eligible(U("pro"))
        assert not ap.eligible(U("free"))
        assert not ap.eligible(U(None))
        assert not ap.eligible(U("creator", entitled=False))
        assert not ap.eligible(None)

    def test_parse_ts(self):
        assert ap.parse_ts("2026-09-22T10:01:35Z") == datetime(2026, 9, 22, 10, 1, 35, tzinfo=timezone.utc)
        assert ap.parse_ts("nope") is None
        assert ap.parse_ts(None) is None


class TestASkipSaysWhy:
    """A channel that only has Shorts gets every video skipped. "Skipped" with
    a three-word badge read as Autopilot doing nothing (most runs so far ended
    this way), so the API carries a sentence and says it once for the channel."""

    def test_every_reason_the_code_gives_has_a_sentence(self):
        import inspect
        import re
        src = inspect.getsource(ap)
        given = set(re.findall(r'"reason": "([a-z_]+)"', src))
        given |= set(re.findall(r'reason="([a-z_]+)"', src))
        # Not shown to anyone: retried on a later tick, or answered as an error.
        given -= {"busy", "server_error", "already_picked", "no_job_id"}
        assert "youtube_short" in given and "private_video" in given
        assert given <= set(ap.REASON_MESSAGES), given - set(ap.REASON_MESSAGES)

    def test_a_short_is_explained_not_just_named(self):
        text = ap.reason_message("youtube_short")
        assert "already a YouTube Short" in text and "longer videos" in text
        assert ap.reason_message(None) is None
        assert ap.reason_message("http_418") is None     # no sentence: the page falls back

    def test_the_run_a_page_lists_carries_the_sentence(self):
        from types import SimpleNamespace
        run = SimpleNamespace(
            id="r1", video_id="v1", video_url="u", video_title="t", thumbnail_url=None,
            published_at=None, trigger="manual", status="skipped", reason="youtube_short",
            job_id=None, clips_count=None, posted_count=None, created_at=None)
        view = ap._run_view(run)
        assert view["reason"] == "youtube_short"
        assert view["message"] == ap.REASON_MESSAGES["youtube_short"]
        run.status, run.reason = "completed", None
        assert ap._run_view(run)["message"] is None

    def test_a_channel_of_only_shorts_is_told_once(self):
        from types import SimpleNamespace as R
        short = R(status="skipped", reason="youtube_short")
        assert ap.only_shorts([short, short])
        assert not ap.only_shorts([])
        assert not ap.only_shorts([short, R(status="completed", reason=None)])
        assert not ap.only_shorts([short, R(status="skipped", reason="out_of_minutes")])

    def test_clip_it_on_a_short_answers_with_the_reason_in_words(self, monkeypatch):
        import asyncio
        from types import SimpleNamespace
        from cloud import auth

        user = SimpleNamespace(id="u-1", email="a@example.com", plan="creator", entitled=True)
        updates = []

        async def _user(_request):
            return user

        async def _none(*_a, **_k):
            return None

        async def _zero(*_a, **_k):
            return 0

        async def _profile(_uid):
            return "os_profile"

        async def _videos(_profile, limit=12, accounts=None):
            return [{"id": "sh0rt", "source": "youtube",
                     "permalink": "https://www.youtube.com/watch?v=sh0rt"}]

        async def _claim(_uid, _video, _trigger):
            return "run-1"

        async def _false(_vid):
            return False

        async def _true(_vid):
            return True

        async def _update(run_id, **values):
            updates.append((run_id, values))

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def begin(self):
                return self

            async def execute(self, *_a, **_k):
                return None

        monkeypatch.setattr(auth, "get_current_user_required", _user)
        monkeypatch.setattr(ap, "_get_settings", _none)
        monkeypatch.setattr(ap, "_count_recent", _zero)
        monkeypatch.setattr(ap, "_profile_for", _profile)
        monkeypatch.setattr(ap, "fetch_all_videos", _videos)
        monkeypatch.setattr(ap, "_claim", _claim)
        monkeypatch.setattr(ap, "is_youtube_private", _false)
        monkeypatch.setattr(ap, "is_youtube_short", _true)
        monkeypatch.setattr(ap, "_update_run", _update)
        monkeypatch.setattr(ap.database, "session", lambda: _Session())

        answer = asyncio.run(ap.run_now(ap.ManualRun(video_id="sh0rt"), request=None))
        assert answer == {"status": "skipped", "reason": "youtube_short", "job_id": None,
                          "message": ap.REASON_MESSAGES["youtube_short"]}
        assert updates == [("run-1", {"status": "skipped", "reason": "youtube_short"})]


def test_connect_return_targets_are_a_closed_list():
    from cloud import social_profiles
    assert set(social_profiles.RETURN_TARGETS) == {"account", "autopilot"}
    assert all(v.startswith("/#") for v in social_profiles.RETURN_TARGETS.values())


class TestTwitchSource:
    def test_live_broadcasts_wait_for_a_later_tick(self):
        baseline = NOW - timedelta(days=1)
        live = dict(_video("twitch:1", 2), is_live=True)
        done = _video("twitch:2", 3)
        assert [v["id"] for v in ap.pick_new_videos([live, done], baseline, set(), NOW)] == ["twitch:2"]

    def test_normalise_media_prefixes_twitch_ids_and_keeps_youtube(self):
        tw = ap.normalise_media([{"id": "2290012345", "title": "GTA heist night",
                                  "timestamp": "2026-11-20T01:00:00Z"}], "twitch")
        assert tw[0]["id"] == "twitch:2290012345"
        assert tw[0]["source_id"] == "2290012345"
        assert tw[0]["permalink"] == "https://www.twitch.tv/videos/2290012345"
        assert tw[0]["caption"] == "GTA heist night"
        assert tw[0]["source"] == "twitch"
        yt = ap.normalise_media([{"id": "abcDEF12345", "caption": "x"}, {"no": "id"}], "youtube")
        assert [m["id"] for m in yt] == ["abcDEF12345"] and yt[0]["source"] == "youtube"

    def test_video_url(self):
        assert ap.video_url({"id": "twitch:9"}) == "https://www.twitch.tv/videos/9"
        assert ap.video_url({"id": "abc"}) == "https://www.youtube.com/watch?v=abc"

    def test_channel_sources_follow_the_flag(self, monkeypatch):
        monkeypatch.delenv("AUTOPILOT_TWITCH", raising=False)
        assert ap.channel_sources({}) == ["youtube"]
        assert ap.channel_sources({"youtube": {}, "twitch": {}}) == ["youtube"]
        assert ap.channel_sources({"twitch": {}}) == []
        monkeypatch.setenv("AUTOPILOT_TWITCH", "1")
        assert ap.channel_sources({"youtube": {}, "twitch": {}}) == ["youtube", "twitch"]
        assert ap.channel_sources({"twitch": {}, "tiktok": {}}) == ["twitch"]

    def test_connect_page_offers_twitch_only_with_the_flag(self, monkeypatch):
        from cloud import social_profiles
        monkeypatch.delenv("AUTOPILOT_TWITCH", raising=False)
        assert "twitch" not in social_profiles.connect_platforms()
        monkeypatch.setenv("AUTOPILOT_TWITCH", "1")
        assert social_profiles.connect_platforms()[-1] == "twitch"

    def test_connect_page_names_twitch_as_read_only_only_when_offered(self):
        from cloud import social_profiles
        without = social_profiles.connect_description(["tiktok", "instagram", "youtube"])
        assert "Twitch" not in without
        offered = social_profiles.connect_description(["tiktok", "instagram", "youtube", "twitch"])
        assert offered.startswith(without)
        assert "Twitch" in offered and "read-only" in offered
        assert "—" not in offered

    def test_connect_url_sends_the_description_matching_its_platforms(self, monkeypatch):
        import asyncio
        from cloud import social_profiles
        sent = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"access_url": "https://connect.example/?token=t"}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                sent.update(json)
                return _Resp()

        monkeypatch.setattr(social_profiles.httpx, "AsyncClient", _Client)
        monkeypatch.setenv("AUTOPILOT_TWITCH", "1")
        asyncio.run(social_profiles.get_connect_url("os_x", "autopilot"))
        assert sent["platforms"][-1] == "twitch"
        assert "Twitch" in sent["connectDescription"]
        monkeypatch.delenv("AUTOPILOT_TWITCH")
        asyncio.run(social_profiles.get_connect_url("os_x", "autopilot"))
        assert "twitch" not in sent["platforms"]
        assert "Twitch" not in sent["connectDescription"]
