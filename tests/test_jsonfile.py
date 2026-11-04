"""write_json_atomic: a reader never sees a half-written *_metadata.json."""
import glob
import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jsonfile  # noqa: E402
from jsonfile import write_json_atomic  # noqa: E402


def _leftovers(directory):
    return [n for n in os.listdir(directory) if n.endswith(".tmp")]


def test_writes_the_document(tmp_path):
    path = tmp_path / "job_metadata.json"
    write_json_atomic(str(path), {"shorts": [1, 2]}, indent=2)
    assert json.loads(path.read_text()) == {"shorts": [1, 2]}
    assert _leftovers(tmp_path) == []


def test_reader_mid_write_sees_the_old_document(tmp_path, monkeypatch):
    # The /transcript 500: the job process rewrote metadata.json in place and a
    # concurrent read hit the truncated file. Read the target while json.dump
    # is still writing: it must still parse, and hold the previous version
    # (with open(path, 'w') it is already truncated at this point).
    path = tmp_path / "job_metadata.json"
    path.write_text(json.dumps({"v": 1}))
    seen = []
    real_dump = json.dump

    def dump_and_peek(obj, fp, **kw):
        seen.append(json.loads(path.read_text()))
        real_dump(obj, fp, **kw)

    monkeypatch.setattr(jsonfile.json, "dump", dump_and_peek)
    write_json_atomic(str(path), {"v": 2})

    assert seen == [{"v": 1}]
    assert json.loads(path.read_text()) == {"v": 2}
    assert _leftovers(tmp_path) == []


def test_failed_write_keeps_the_old_file_and_no_temp(tmp_path):
    path = tmp_path / "job_metadata.json"
    path.write_text(json.dumps({"v": 1}))
    with pytest.raises(TypeError):
        write_json_atomic(str(path), {"v": object()})
    assert json.loads(path.read_text()) == {"v": 1}
    assert _leftovers(tmp_path) == []


def test_temp_file_is_invisible_to_metadata_globs(tmp_path, monkeypatch):
    path = tmp_path / "job_metadata.json"
    matches = []
    real_dump = json.dump

    def dump_and_glob(obj, fp, **kw):
        matches.append(glob.glob(os.path.join(str(tmp_path), "*_metadata.json")))
        real_dump(obj, fp, **kw)

    monkeypatch.setattr(jsonfile.json, "dump", dump_and_glob)
    write_json_atomic(str(path), {"v": 1})
    assert matches == [[]]


def test_keeps_the_existing_file_mode(tmp_path):
    path = tmp_path / "job_metadata.json"
    path.write_text("{}")
    os.chmod(path, 0o640)
    write_json_atomic(str(path), {"v": 1})
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640

    fresh = tmp_path / "new_metadata.json"
    write_json_atomic(str(fresh), {"v": 1})
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o644
