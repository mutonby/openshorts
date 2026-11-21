import json
import sqlite3

import pytest

from cli import openshorts_cli as cli


def test_trends_parses_the_regional_rss(monkeypatch, capsys):
    def discover(url, label):
        assert url == "https://trends.google.com/trending/rss?geo=ZA"
        assert label == "Google Trends"
        return b'''<rss xmlns:ht="https://trends.google.com/trending/rss"><channel>
            <item><title>Fuel price</title><ht:approx_traffic>20K+</ht:approx_traffic>
            <pubDate>Fri, 20 Nov 2026 06:00:00 GMT</pubDate></item>
            <item><title>Another topic</title></item>
            </channel></rss>'''

    monkeypatch.setattr(cli, "_discover", discover)
    cli.main(["--json", "trends", "--region", "za", "--limit", "1"])
    assert json.loads(capsys.readouterr().out) == {
        "region": "ZA", "source": "google_trending_searches", "items": [
            {"topic": "Fuel price", "approx_searches": "20K+",
             "published_at": "Fri, 20 Nov 2026 06:00:00 GMT"}
        ]
    }


def test_youtube_references_require_recent_bucket_and_exact_duration(monkeypatch, capsys):
    def youtube_data(resource, params, key):
        assert key == "test-key"
        if resource == "search":
            assert params["q"] == "candy making"
            assert params["regionCode"] == "ZA"
            assert params["videoDuration"] == "short"
            assert params["publishedAfter"]
            return [{"id": {"videoId": char * 11}} for char in "abcd"]
        return [
            {"id": char * 11, "contentDetails": {"duration": duration},
             "snippet": {"title": char}, "statistics": {"viewCount": views}}
            for char, duration, views in (
                ("a", "PT3M", "9"), ("b", "PT15S", "15"), ("c", "PT3M1S", "90"),
                ("d", "PT0S", "100"),
            )
        ]

    monkeypatch.setenv("YOUTUBE_DATA_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_youtube_data", youtube_data)
    cli.main(["--json", "trends", "--region", "ZA", "--topic", "candy making"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "youtube_short_video_references"
    assert [(row["title"], row["seconds"]) for row in payload["items"]] == [("b", 15), ("a", 180)]


def test_only_matching_library_sources_are_previewed_and_queued_once(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps([
        {"url": "https://example.com/candy.mp4", "topics": ["candy"], "profile": "process",
         "source_start": 30, "source_end": 210},
        {"url": "https://example.com/candles.mp4", "topics": ["candle"]},
        {"url": "https://example.com/car.mp4", "topics": ["car"]},
    ]))
    monkeypatch.setattr(cli, "_search_trends", lambda region: [
        {"topic": "cartoons"}, {"topic": "CANDY-MAKING"}, {"topic": "candy recipe"}
    ])
    calls = []

    def request(method, path, body):
        calls.append((method, path, body))
        return 200, {"job_id": "job-1"}

    monkeypatch.setattr(cli, "_request", request)
    args = ["--json", "trend-generate", str(manifest)]
    cli.main(args)
    assert not calls and not manifest.with_suffix(".trends.sqlite3").exists()
    preview = json.loads(capsys.readouterr().out)
    assert preview == {"dry_run": True, "items": [
        {"profile": "process", "topic": "CANDY-MAKING", "url": "https://example.com/candy.mp4",
         "source_start": 30, "source_end": 210}
    ]}

    with pytest.raises(SystemExit) as exc:
        cli.main([*args, "--submit"])
    assert exc.value.code == 1
    assert not calls
    capsys.readouterr()

    authorized = [*args, "--submit", "--acknowledge", "--limit", "2"]
    cli.main(authorized)
    assert calls == [("POST", "/api/process", {
        "url": "https://example.com/candy.mp4", "acknowledged": True, "output_format": "vertical",
        "framing_method": "auto", "target_clips": 3, "clip_min_seconds": 15, "clip_max_seconds": 40,
        "source_start": 30, "source_end": 210, "auto_hook": False,
    })]
    assert json.loads(capsys.readouterr().out)["items"][0]["job_id"] == "job-1"
    cli.main(authorized)
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out) == {"dry_run": False, "items": []}


@pytest.mark.parametrize("status, expected_calls", [(200, 2), (429, 2), (503, 1)])
def test_trend_ledger_retries_rejected_jobs_but_keeps_ambiguous_claims(
    tmp_path, monkeypatch, capsys, status, expected_calls,
):
    db_path = tmp_path / "jobs.sqlite3"
    calls = []

    def request(method, path, body):
        # A second database connection represents another scheduler run.
        with cli.closing(sqlite3.connect(db_path)) as observer:
            assert observer.execute("SELECT job_id FROM jobs").fetchall() == [(None,)]
        calls.append(body)
        if status == 200:
            return 200, {"needs_confirmation": True, "quality_check": {"max_height": 480}}
        return status, {"detail": "try later"}

    monkeypatch.setattr(cli, "_request", request)
    source = ("https://example.com/footage", None, None)
    with cli.closing(sqlite3.connect(db_path)) as db:
        db.execute("CREATE TABLE jobs(source TEXT PRIMARY KEY, job_id TEXT)")
        for _ in range(2):
            try:
                cli._submit_trend_job(db, source, {})
            except SystemExit as exc:
                assert exc.code == 1
    assert len(calls) == expected_calls
    assert ("unconfirmed trend job" in capsys.readouterr().err) == (status == 503)
