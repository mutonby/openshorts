"""Exercise the real CLI entry point without loading the optional ML stack."""
import argparse
import ast
import atexit
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest


MAIN = Path(__file__).resolve().parents[1] / "main.py"


@pytest.fixture
def run_cli(monkeypatch, tmp_path):
    # Only the render/download boundaries are replaced. Argument parsing,
    # filesystem checks and whole-video dispatch run from main.py itself.
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    entry = next(node for node in tree.body
                 if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name)
                 and node.test.left.id == "__name__")
    code = compile(ast.Module(body=entry.body, type_ignores=[]), str(MAIN), "exec")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(atexit, "register", lambda callback: None)
    for name in ("MAX_SOURCE_MINUTES", "SOURCE_CAP_MINUTES"):
        monkeypatch.delenv(name, raising=False)

    def run(source, output):
        renders, downloads = [], []
        local_input = tmp_path / "source.mp4"
        local_input.touch()

        def download(url, directory, **kwargs):
            downloads.append(directory)
            Path(directory).mkdir(parents=True, exist_ok=True)
            downloaded = os.path.join(directory, "source.mp4")
            Path(downloaded).touch()
            return downloaded, "source"

        argv = [str(MAIN), "--skip-analysis", "--format", "vertical"]
        if source == "url":
            argv += ["--url", "https://www.youtube.com/watch?v=example", "--keep-original"]
        else:
            argv += ["--input", str(local_input)]
        if output is not None:
            argv += ["--output", str(output)]
        monkeypatch.setattr(sys, "argv", argv)
        namespace = {
            "argparse": argparse, "os": os, "time": time, "json": json,
            "threading": threading, "layout_picker": SimpleNamespace(ENABLED=False),
            "download_youtube_video": download,
            "render_clip": lambda video, path, fmt: renders.append((video, path, fmt)),
        }
        exec(code, namespace)
        assert len(renders) == 1
        return renders[0][1], downloads

    return run


@pytest.mark.parametrize("source", ["local", "url"])
def test_creates_missing_output_directory(run_cli, tmp_path, source):
    output = tmp_path / "output" / "missing_dir"
    rendered, downloads = run_cli(source, output)
    assert rendered == str(output / "source_vertical.mp4")
    assert output.is_dir()
    if source == "url":
        assert downloads == [str(output)]


@pytest.mark.parametrize("source", ["local", "url"])
@pytest.mark.parametrize("name", ["existing_dir", "existing.dir"])
def test_existing_directory_still_gets_a_filename(run_cli, tmp_path, source, name):
    output = tmp_path / name
    output.mkdir()
    rendered, _ = run_cli(source, output)
    assert rendered == str(output / "source_vertical.mp4")


@pytest.mark.parametrize("source", ["local", "url"])
def test_creates_directory_with_trailing_separator(run_cli, tmp_path, source):
    output = str(tmp_path / "new.dir") + os.sep
    rendered, _ = run_cli(source, output)
    assert rendered == os.path.join(output, "source_vertical.mp4")
    assert Path(output).is_dir()


@pytest.mark.parametrize("source", ["local", "url"])
def test_explicit_output_file_is_preserved(run_cli, tmp_path, source):
    output = tmp_path / "output" / "clip.mp4"
    rendered, _ = run_cli(source, output)
    assert rendered == str(output)


@pytest.mark.parametrize("source", ["local", "url"])
def test_existing_extensionless_file_is_preserved(run_cli, tmp_path, source):
    output = tmp_path / "existing_file"
    output.touch()
    rendered, _ = run_cli(source, output)
    assert rendered == str(output)


@pytest.mark.parametrize("source", ["local", "url"])
def test_default_output_location_is_preserved(run_cli, tmp_path, source):
    rendered, _ = run_cli(source, None)
    directory = "." if source == "url" else str(tmp_path)
    assert rendered == os.path.join(directory, "source_vertical.mp4")
