"""The download cache: a YouTube video downloaded whole is reused by the next
jobs that ask for it instead of spending a YouTube account request again."""
import os
import threading
import time

import pytest

import source_cache


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("SOURCE_CACHE_DIR", str(d))
    monkeypatch.setenv("SOURCE_CACHE_HOURS", "48")
    monkeypatch.setenv("SOURCE_CACHE_MAX_GB", "1")
    monkeypatch.setenv("SOURCE_CACHE_WAIT_SECONDS", "5")
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    return d


URL = "https://www.youtube.com/watch?v=q6fgVVmdpgY&t=42"
TRANSCRIPT = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}


def _download(tmp_path, name="job1", data=b"x" * 1000):
    job = tmp_path / name
    job.mkdir()
    f = job / "Some Title.mp4"
    f.write_bytes(data)
    return job, str(f)


@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=q6fgVVmdpgY", "q6fgVVmdpgY"),
    ("https://youtube.com/watch?feature=share&v=q6fgVVmdpgY&list=PL1", "q6fgVVmdpgY"),
    ("https://m.youtube.com/watch?v=q6fgVVmdpgY", "q6fgVVmdpgY"),
    ("https://youtu.be/q6fgVVmdpgY?si=abc", "q6fgVVmdpgY"),
    ("https://www.youtube.com/shorts/q6fgVVmdpgY", "q6fgVVmdpgY"),
    ("https://www.youtube.com/live/q6fgVVmdpgY?feature=shared", "q6fgVVmdpgY"),
    ("https://www.youtube.com/embed/q6fgVVmdpgY", "q6fgVVmdpgY"),
    ("https://www.youtube.com/clip/UgkxAbCdEfGhIjKlMnOpQrStUvWxYz012345", None),
    ("https://www.youtube.com/watch?v=short", None),
    ("https://evil.example/watch?v=q6fgVVmdpgY", None),
    ("https://www.twitch.tv/videos/123", None),
    ("not a url", None),
    (None, None),
])
def test_video_id(url, vid):
    assert source_cache.video_id(url) == vid


def test_off_without_a_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("SOURCE_CACHE_DIR")
    job, path = _download(tmp_path)
    assert not source_cache.store(URL, path, "Some Title")
    assert source_cache.fetch(URL, str(tmp_path)) is None


def test_store_then_fetch_shares_the_file(tmp_path):
    job, path = _download(tmp_path)
    assert source_cache.store(URL, path, "Some Title")
    other = tmp_path / "job2"
    other.mkdir()
    got, title = source_cache.fetch("https://youtu.be/q6fgVVmdpgY", str(other))
    assert title == "Some Title"
    assert os.path.dirname(got) == str(other)
    assert open(got, "rb").read() == b"x" * 1000
    # Hard-linked, not copied: same inode as the cached file.
    cached = os.path.join(os.environ["SOURCE_CACHE_DIR"], "q6fgVVmdpgY.mp4")
    assert os.stat(got).st_ino == os.stat(cached).st_ino


def test_a_job_replacing_its_source_leaves_the_cache_whole(tmp_path):
    """cap_source_duration cuts with os.replace: the cache keeps the original."""
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    tmp = path + ".capped"
    with open(tmp, "wb") as f:
        f.write(b"y" * 10)
    os.replace(tmp, path)
    other = tmp_path / "job2"
    other.mkdir()
    got, _ = source_cache.fetch(URL, str(other))
    assert os.path.getsize(got) == 1000


def test_a_deleted_job_dir_keeps_the_cached_file(tmp_path):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    os.remove(path)
    other = tmp_path / "job2"
    other.mkdir()
    assert source_cache.fetch(URL, str(other)) is not None


def test_expired_entries_are_dropped(tmp_path, monkeypatch):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    real = time.time
    monkeypatch.setattr(source_cache.time, "time", lambda: real() + 49 * 3600)
    assert source_cache.fetch(URL, str(job)) is None
    assert not os.path.exists(os.path.join(os.environ["SOURCE_CACHE_DIR"], "q6fgVVmdpgY.mp4"))


def test_a_truncated_cached_file_is_not_served(tmp_path):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    cached = os.path.join(os.environ["SOURCE_CACHE_DIR"], "q6fgVVmdpgY.mp4")
    os.remove(cached)
    with open(cached, "wb") as f:
        f.write(b"x" * 10)
    assert source_cache.fetch(URL, str(job)) is None


def test_the_title_read_back_is_only_a_file_name(tmp_path):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "../../etc/passwd")
    other = tmp_path / "job2"
    other.mkdir()
    got, title = source_cache.fetch(URL, str(other))
    assert os.path.dirname(got) == str(other)
    assert "/" not in title and not title.startswith(".")


def test_least_recently_used_goes_first_past_the_cap(tmp_path, monkeypatch):
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    d = os.environ["SOURCE_CACHE_DIR"]
    for i, vid in enumerate(ids):
        job, path = _download(tmp_path, name=f"j{i}")
        source_cache.store(f"https://youtu.be/{vid}", path, "t")
        os.utime(os.path.join(d, f"{vid}.json"), (time.time() - 100 + i,) * 2)
    # The first one was used again just now: the second is the oldest use.
    os.utime(os.path.join(d, "aaaaaaaaaaa.json"))
    monkeypatch.setenv("SOURCE_CACHE_MAX_GB", str(2500 / 1e9))   # room for two
    source_cache.prune()
    left = sorted(n for n in os.listdir(d) if n.endswith(".mp4"))
    assert left == ["aaaaaaaaaaa.mp4", "ccccccccccc.mp4"]


def test_a_source_too_big_to_be_worth_its_room_is_not_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_CACHE_MAX_ENTRY_GB", "0.000002")       # 2 kB
    _, small = _download(tmp_path, "a", b"x" * 1000)
    _, giant = _download(tmp_path, "b", b"x" * 5000)
    other = "https://youtu.be/bbbbbbbbbbb"
    assert source_cache.store(URL, small, "small")
    assert not source_cache.store(other, giant, "giant")
    assert source_cache.fetch(other, str(tmp_path / "c")) is None
    assert source_cache.fetch(URL, str(tmp_path / "a")) is not None   # the small one stays
    monkeypatch.setenv("SOURCE_CACHE_MAX_ENTRY_GB", "0")              # no limit
    assert source_cache.store(other, giant, "giant")
    # One kept under a larger limit goes at the next clean-up, the others stay.
    monkeypatch.setenv("SOURCE_CACHE_MAX_ENTRY_GB", "0.000002")
    source_cache.prune()
    assert source_cache.fetch(other, str(tmp_path / "b")) is None
    assert source_cache.fetch(URL, str(tmp_path / "a")) is not None


def test_only_whole_free_downloads_are_kept():
    ok = {"winner": "HD-static1", "attempts": [
        {"label": "HD-static1", "ok": False, "paid": False},
        {"label": "HD-static2", "ok": True, "paid": False}], "ranged": False}
    assert source_cache.is_whole_free_download(ok)
    assert not source_cache.is_whole_free_download({**ok, "ranged": True})
    paid = {"winner": "HD", "attempts": [{"label": "HD", "ok": True, "paid": True}]}
    assert not source_cache.is_whole_free_download(paid)
    assert not source_cache.is_whole_free_download({"winner": None, "attempts": []})
    assert not source_cache.is_whole_free_download(None)


def test_transcript_rides_along_for_the_same_source(tmp_path, monkeypatch):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    assert source_cache.store_transcript(URL, 600.0, TRANSCRIPT, path)
    assert source_cache.fetch_transcript(URL, 600.2) == TRANSCRIPT
    # Another length is another file (a job that cut its source).
    assert source_cache.fetch_transcript(URL, 300.0) is None
    # Another ASR backend transcribes differently.
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "whisper")
    assert source_cache.fetch_transcript(URL, 600.0) is None


def test_a_cut_source_does_not_store_its_transcript(tmp_path):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    tmp = path + ".capped"
    with open(tmp, "wb") as f:
        f.write(b"y" * 10)
    os.replace(tmp, path)
    assert not source_cache.store_transcript(URL, 300.0, TRANSCRIPT, path)
    assert source_cache.fetch_transcript(URL, 300.0) is None


def test_storing_a_new_download_drops_the_old_transcript(tmp_path):
    job, path = _download(tmp_path)
    source_cache.store(URL, path, "Some Title")
    source_cache.store_transcript(URL, 600.0, TRANSCRIPT, path)
    job2, path2 = _download(tmp_path, name="job2", data=b"z" * 900)
    source_cache.store(URL, path2, "Some Title")
    assert source_cache.fetch_transcript(URL, 600.0) is None


def test_claim_makes_a_second_job_wait_for_the_first_download(tmp_path):
    order = []
    started = threading.Event()

    def first():
        with source_cache.claim(URL):
            started.set()
            time.sleep(0.5)
            job, path = _download(tmp_path, name="first")
            source_cache.store(URL, path, "Some Title")
            order.append("stored")

    t = threading.Thread(target=first)
    t.start()
    started.wait(2)
    with source_cache.claim(URL):
        order.append("second got the lock")
        other = tmp_path / "second"
        other.mkdir()
        assert source_cache.fetch(URL, str(other)) is not None
    t.join()
    assert order == ["stored", "second got the lock"]


def test_claim_gives_up_after_the_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_CACHE_WAIT_SECONDS", "0")
    held = threading.Event()
    release = threading.Event()

    def holder():
        with source_cache.claim(URL):
            held.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    held.wait(2)
    t0 = time.monotonic()
    with source_cache.claim(URL):
        pass
    assert time.monotonic() - t0 < 3
    release.set()
    t.join()


def test_claim_is_a_no_op_for_other_urls():
    with source_cache.claim("https://www.twitch.tv/videos/123"):
        pass
    assert not os.path.exists(os.environ["SOURCE_CACHE_DIR"])


def test_probe_facts_are_kept_per_video():
    assert source_cache.probe_put(URL, 754.0, 1080)
    got = source_cache.probe_get("https://youtu.be/q6fgVVmdpgY")
    assert got == {"seconds": 754.0, "max_height": 1080}
    assert source_cache.probe_get("https://youtu.be/bbbbbbbbbbb") is None
    # No length, no entry: a live stream never gets one.
    assert not source_cache.probe_put("https://youtu.be/bbbbbbbbbbb", 0)


def test_probe_facts_expire(monkeypatch):
    source_cache.probe_put(URL, 754.0, 1080)
    real = time.time
    monkeypatch.setattr(source_cache.time, "time", lambda: real() + 49 * 3600)
    assert source_cache.probe_get(URL) is None


def test_a_repeated_submission_does_not_probe_youtube_again(monkeypatch):
    metering = pytest.importorskip("cloud.metering")
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    calls = []

    def fake_probe(url, *a, **k):
        calls.append(url)
        metering._probe_facts[url] = {"at": time.time(), "max_height": 720, "duration": 600}
        return 10.0

    monkeypatch.setattr(metering, "_probe_with_proxies", fake_probe)
    assert metering.probe_url_minutes(URL) == 10.0
    other = "https://www.youtube.com/shorts/q6fgVVmdpgY"
    metering._probe_facts.pop(other, None)
    assert metering.probe_url_minutes(other) == 10.0
    assert calls == [URL]                       # the second one never asked
    assert metering.probe_facts(other)["max_height"] == 720


def test_the_probe_still_asks_with_the_cache_off(monkeypatch):
    metering = pytest.importorskip("cloud.metering")
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.delenv("SOURCE_CACHE_DIR")
    calls = []
    monkeypatch.setattr(metering, "_probe_with_proxies",
                        lambda url, *a, **k: calls.append(url) or 10.0)
    metering.probe_url_minutes(URL)
    metering.probe_url_minutes(URL)
    assert len(calls) == 2
