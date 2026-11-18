"""cloud/log_archive.py: chunked spool, gzip upload, retry, cap, and the
guarantee that logging never breaks the process writing."""
import gzip
import io
import os

import pytest

from cloud import log_archive as la


class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class _FakeS3:
    def __init__(self, fail=False):
        self.objects = {}
        self.fail = fail

    def put_object(self, Bucket, Key, Body, **kw):
        if self.fail:
            raise ConnectionError("down")
        self.objects[(Bucket, Key)] = Body


def _sink(tmp_path, clock, max_bytes=10 * 1024 * 1024):
    return la._Sink(str(tmp_path), 600, "b00t", max_bytes, clock=clock)


def test_lines_are_timestamped_and_chunked(tmp_path):
    clock = _Clock(1_700_000_000)          # 2023-11-14 22:13:20 UTC
    sink = _sink(tmp_path, clock)
    sink.write("hello ")
    sink.write("world\nsecond line\n")
    clock.t += 600                          # next chunk
    sink.write("third\n")
    files = sorted(os.listdir(tmp_path))
    assert len(files) == 2
    first = open(tmp_path / files[0]).read().splitlines()
    assert first[0].endswith("Z hello world") and first[0].startswith("2023-11-14T22:13:20")
    assert first[1].endswith(" second line")
    assert open(tmp_path / files[1]).read().strip().endswith(" third")


def test_only_closed_chunks_upload_then_shutdown_takes_the_rest(tmp_path):
    clock = _Clock(1_700_000_000)
    sink = _sink(tmp_path, clock)
    s3 = _FakeS3()
    up = la._Uploader(sink, "logs", "api", "host1", client_factory=lambda: s3)
    sink.write("a\n")
    assert up.upload_ready() == 0           # chunk still open
    clock.t += 600
    sink.write("b\n")
    assert up.upload_ready() == 1
    (bucket, key), body = next(iter(s3.objects.items()))
    assert bucket == "logs"
    assert key == "api/2023-11-14/2210-host1-b00t.log.gz"
    assert gzip.decompress(body).decode().strip().endswith(" a")
    assert up.upload_ready(include_open=True) == 1
    assert os.listdir(tmp_path) == []


def test_quiet_chunk_is_closed_by_the_upload_pass(tmp_path):
    clock = _Clock(1_700_000_000)
    sink = _sink(tmp_path, clock)
    s3 = _FakeS3()
    up = la._Uploader(sink, "logs", "api", "h", client_factory=lambda: s3)
    sink.write("only line\n")
    clock.t += 3600                         # nothing written since
    assert up.upload_ready() == 1


def test_failed_upload_keeps_the_file_for_the_next_pass(tmp_path):
    clock = _Clock(1_700_000_000)
    sink = _sink(tmp_path, clock)
    s3 = _FakeS3(fail=True)
    up = la._Uploader(sink, "logs", "api", "h", client_factory=lambda: s3)
    sink.write("x\n")
    assert up.upload_ready(include_open=True) == 0
    assert len(os.listdir(tmp_path)) == 1
    s3.fail = False
    assert up.upload_ready(include_open=True) == 1


def test_cap_drops_oldest_chunks(tmp_path):
    clock = _Clock(1_700_000_000)
    sink = _sink(tmp_path, clock, max_bytes=200)
    for i in range(5):
        sink.write("y" * 100 + "\n")
        clock.t += 600
    sink.rotate()
    sink.enforce_cap()
    total = sum(os.path.getsize(tmp_path / n) for n in os.listdir(tmp_path))
    assert total <= 200 or len(os.listdir(tmp_path)) == 1


def test_tee_writes_stream_first_and_survives_a_broken_sink(tmp_path):
    out = io.StringIO()
    sink = _sink(tmp_path, _Clock(1_700_000_000))
    sink._open = lambda start: (_ for _ in ()).throw(OSError("disk full"))
    tee = la._Tee(out, sink)
    assert tee.write("still printed\n") == len("still printed\n")
    assert out.getvalue() == "still printed\n"
    assert tee.getvalue() == "still printed\n"   # other attributes delegate


def test_object_key_ignores_foreign_files():
    assert la.object_key("api", "/tmp/x/notes.txt", "h") is None


def test_off_without_bucket(monkeypatch):
    monkeypatch.delenv("LOG_ARCHIVE_BUCKET", raising=False)
    assert la.enabled() is False
    assert la.install() is False


def test_refuses_the_public_media_bucket(monkeypatch):
    monkeypatch.setenv("LOG_ARCHIVE_BUCKET", "media")
    monkeypatch.setattr(la, "_bucket_is_public", lambda b: True)
    assert la.install() is False


@pytest.mark.parametrize("raw,expected", [("api", "api"), ("../etc", "etc"), ("", "unknown")])
def test_prefix_is_sanitised(raw, expected):
    assert la._safe(raw) == expected


def test_processes_sharing_the_spool_leave_each_others_chunks_alone(tmp_path):
    """The engine and the front processes of a split API (serve.py) spool in
    one directory: each uploaded and deleted the chunk another one was still
    appending to (18-nov-2026). Another process's chunk goes up only once
    nothing has written to it for two chunk lengths (that process ended)."""
    clock = _Clock(1_700_000_000)
    mine = la._Sink(str(tmp_path), 600, "engine1", 10 * 1024 * 1024, clock=clock)
    other = la._Sink(str(tmp_path), 600, "front1", 10 * 1024 * 1024, clock=clock)
    s3 = _FakeS3()
    up = la._Uploader(mine, "logs", "api", "h", client_factory=lambda: s3)
    other.write("front line\n")
    clock.t += 600
    other.write("front still writing\n")       # its previous chunk is closed, this one open
    mine.write("engine line\n")
    for path in [os.path.join(tmp_path, n) for n in os.listdir(tmp_path)]:
        os.utime(path, (clock.t, clock.t))
    up.upload_ready()
    assert all("front1" not in key for _, key in s3.objects)
    assert len([n for n in os.listdir(tmp_path) if "front1" in n]) == 2
    # The front process died: an hour later its chunks are taken.
    clock.t += 3600
    mine.write("engine later\n")
    up.upload_ready()
    assert len([key for _, key in s3.objects if "front1" in key]) == 2
    assert not [n for n in os.listdir(tmp_path) if "front1" in n]


def test_a_chunk_another_process_took_is_not_a_failed_upload(tmp_path):
    clock = _Clock(1_700_000_000)
    sink = _sink(tmp_path, clock)
    s3 = _FakeS3()
    up = la._Uploader(sink, "logs", "api", "h", client_factory=lambda: s3)
    sink.write("a\n")
    clock.t += 600
    sink.write("b\n")
    real_ready = sink.ready_files

    def ready_then_gone(include_open=False):
        paths = real_ready(include_open)
        for p in paths:
            os.remove(p)
        return paths
    sink.ready_files = ready_then_gone
    assert up.upload_ready() == 0 and up.failing is False
