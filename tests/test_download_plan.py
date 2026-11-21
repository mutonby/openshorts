"""Download attempt ordering: cheapest bandwidth first.

Direct (server IP) -> flat-rate static ISP proxies (uncapped 1080p, free
bytes) -> per-GB paid proxy (720p cost cap) -> conservative fallback. The
``capped`` flag marks the attempts whose bytes are billed per GB (drives the
720p format cap and the PROXY_BYTES monthly counter).
"""
import pytest

main = pytest.importorskip("main")

plan = main.plan_download_attempts

STATICS = ["http://s1", "http://s2", "http://s3"]
PAID = "http://paid"


class TestOrdering:
    def test_full_chain(self):
        got = plan(True, STATICS, PAID, True)
        assert got == [
            ('HD-direct', False, None),
            ('HD-static1', False, "http://s1"),
            ('HD-static2', False, "http://s2"),
            ('HD-static3', False, "http://s3"),
            # Free rescue for the fake "Video unavailable" the HD client gets
            # from datacenter-ISP IPs (4-sep-2026): conservative clients
            # through a static BEFORE any per-GB attempt.
            ('fallback-static', False, "http://s1"),
            ('HD', True, PAID),
            ('fallback', True, PAID),
        ]

    def test_statics_only_fallback_uses_a_static(self):
        got = plan(False, STATICS, None, True)
        assert got[-1] == ('fallback', False, "http://s1")
        assert ('HD', False, None) in got  # direct HD still tried, uncapped

    def test_selfhost_no_proxies_matches_legacy_chain(self):
        assert plan(False, [], None, True) == [
            ('HD', False, None),
            ('fallback', False, None),
        ]

    def test_no_hd_args_still_has_fallback(self):
        assert plan(False, [], PAID, False) == [('fallback', True, PAID)]

    def test_paid_attempts_are_the_only_capped_ones(self):
        got = plan(True, STATICS, PAID, True)
        for _label, capped, proxy in got:
            assert capped == (proxy == PAID)


def test_fallback_static_only_exists_when_paid_is_configured():
    # Without a paid proxy the final fallback already runs through a static;
    # a fallback-static before it would be the same attempt twice.
    got = plan(False, STATICS, None, True)
    assert ('fallback-static', False, "http://s1") not in got
    assert got[-1] == ('fallback', False, "http://s1")


def test_direct_file_urls_skip_the_proxy_chain():
    """A catbox/tmpfiles/CDN mp4 has no YouTube ban to dodge: own IP first,
    one static as fallback, never the per-GB proxy (prod 27-aug: 9 KB/s
    through the ISP proxy vs 17 MB/s direct)."""
    got = plan(False, ["s1", "s2"], "paid", True, youtube=False)
    # Every static is listed: the caller moves past one only when it refuses
    # the tunnel (tests/test_refused_exit.py).
    assert got == [("direct", False, None), ("static-fallback", False, "s1"),
                   ("static-fallback2", False, "s2")]
    assert all(proxy != "paid" for _label, _capped, proxy in got)
    assert plan(False, [], "paid", True, youtube=False) == [("direct", False, None)]
    assert main.is_youtube_url("https://www.youtube.com/watch?v=x")
    assert main.is_youtube_url("https://youtu.be/x")
    assert not main.is_youtube_url("https://litter.catbox.moe/u90j4q.mp4")
    assert not main.is_youtube_url("https://tmpfiles.org/dl/1/2/v.mp4")


class TestSkipStatics:
    """The probe already saw every static IP bot-checked for this video
    (30-sep-2026: per video, identical on every IP): the download goes
    straight to the paid attempts instead of repeating four anonymous hits."""

    def test_skip_statics_goes_straight_to_paid(self):
        got = plan(True, STATICS, PAID, True, skip_statics=True)
        assert got == [('HD', True, PAID), ('fallback', True, PAID)]

    def test_skip_statics_without_paid_keeps_the_statics(self):
        assert plan(False, STATICS, None, True, skip_statics=True) == \
            plan(False, STATICS, None, True)

    def test_default_is_unchanged(self):
        assert plan(True, STATICS, PAID, True, skip_statics=False) == plan(True, STATICS, PAID, True)
