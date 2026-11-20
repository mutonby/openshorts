from datetime import datetime, timezone

from social_metrics import summarise, row_metrics

SINCE = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _row(post_id, views, uploaded, captured="2026-09-30T13:00:00", platform="tiktok"):
    return {"platform": platform, "post_id": post_id, "upload_timestamp": uploaded,
            "captured_at": captured, "post_url": f"https://x/{post_id}",
            "metrics": {"views": views}}


def test_native_posts_without_an_upload_timestamp_are_not_ours():
    out = summarise([_row("a", 100, "2026-09-20T10:00:00"), _row("b", 50000, None)], SINCE)
    assert out["total_impressions"] == 100
    assert out["posts_count"] == 1


def test_posts_published_before_the_window_are_left_out():
    # The vendor's `since` is the snapshot date: an old post rides along
    # with its lifetime views.
    out = summarise([_row("old", 9000, "2026-08-27T20:00:00"),
                     _row("new", 10, "2026-09-26T10:00:00")], SINCE)
    assert out["total_impressions"] == 10


def test_each_post_counts_once_from_its_latest_snapshot():
    rows = [_row("a", 100, "2026-09-20T10:00:00", captured="2026-09-25T00:00:00"),
            _row("a", 400, "2026-09-20T10:00:00", captured="2026-09-30T00:00:00")]
    out = summarise(rows, SINCE)
    assert out["total_impressions"] == 400
    assert out["updated_at"].startswith("2026-09-30")


def test_top_posts_and_platform_split():
    rows = [_row("a", 5, "2026-09-20T10:00:00", platform="youtube"),
            _row("b", 50, "2026-09-21T10:00:00"),
            _row("c", 20, "2026-09-22T10:00:00")]
    out = summarise(rows, SINCE, top_n=2)
    assert [p["views"] for p in out["top_posts"]] == [50, 20]
    assert out["per_platform"] == {"youtube": 5, "tiktok": 70}


def test_nothing_published_reads_as_empty():
    out = summarise([], SINCE)
    assert out["posts_count"] == 0 and out["total_impressions"] == 0
    assert out["updated_at"] is None


def test_refresh_due_when_stale_or_missing():
    from social_metrics import refresh_due
    now = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
    assert refresh_due(None, now)
    assert refresh_due("2026-09-30T13:17:31", now)
    assert not refresh_due("2026-10-07T08:00:00", now)


def test_request_ids_newest_first_successful_in_window_and_capped():
    from social_metrics import request_ids_to_refresh
    rows = [
        {"request_id": "old", "success": True, "upload_timestamp": "2026-08-21T19:40:02Z"},
        {"request_id": "a", "success": True, "upload_timestamp": "2026-09-20T10:00:00Z"},
        {"request_id": "a", "success": True, "upload_timestamp": "2026-09-20T10:00:00Z"},
        {"request_id": "failed", "success": False, "upload_timestamp": "2026-10-01T10:00:00Z"},
        {"request_id": "b", "success": True, "upload_timestamp": "2026-10-06T10:02:47Z"},
    ]
    assert request_ids_to_refresh(rows, SINCE) == ["b", "a"]
    assert request_ids_to_refresh(rows, SINCE, limit=1) == ["b"]


def test_row_metrics_reads_engagement_and_degrades_to_zero():
    full = {"metrics": {"views": 100, "likes": 9, "comments": 3, "shares": 2}}
    assert row_metrics(full) == {"views": 100.0, "likes": 9.0, "comments": 3.0, "shares": 2.0}
    # A platform that reports only views still returns the full shape.
    assert row_metrics({"metrics": {"plays": 50}}) == {
        "views": 50.0, "likes": 0.0, "comments": 0.0, "shares": 0.0}


def test_summarise_totals_engagement_and_keeps_the_view_contract():
    row = {"platform": "tiktok", "post_id": "p1", "upload_timestamp": "2026-09-20T10:00:00",
           "captured_at": "2026-09-30T13:00:00", "post_url": "https://x/p1",
           "metrics": {"views": 100, "likes": 9, "comments": 3, "shares": 2}}
    out = summarise([row], SINCE)
    assert out["total_impressions"] == 100            # unchanged contract
    assert out["engagement"] == {"likes": 9, "comments": 3, "shares": 2}
    assert out["total_engagement"] == 14
    assert out["top_posts"][0]["likes"] == 9 and out["top_posts"][0]["shares"] == 2
