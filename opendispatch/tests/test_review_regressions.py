"""Regression tests for defects found in review of the campaign PR."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from opendispatch.api.app import app

client = TestClient(app, raise_server_exceptions=False)


# ── Defect 1: non-numeric Content-Length crashed /dispatch/bulk ────────────
# int("abc") raised ValueError, which surfaced as HTTP 500 on a
# state-changing route driven by unauthenticated input.

def test_bulk_rejects_non_numeric_content_length() -> None:
    r = client.post(
        "/dispatch/bulk",
        content=b'"twitter","twitter_thread","hi",""',
        headers={"content-type": "text/csv", "content-length": "abc"},
    )
    assert r.status_code == 400, r.text


def test_bulk_still_rejects_oversize_content_length() -> None:
    r = client.post(
        "/dispatch/bulk",
        content=b"x",
        headers={"content-type": "text/csv", "content-length": "9999999"},
    )
    assert r.status_code == 413, r.text


def test_bulk_accepts_normal_csv() -> None:
    r = client.post(
        "/dispatch/bulk",
        content=b'"twitter","twitter_thread","hello",""',
        headers={"content-type": "text/csv"},
    )
    assert r.status_code == 202, r.text


# ── Defect 2: transcode_image output location contradicted its docstring ───
# With a media root configured, transcode output is confined there by design.
# The docstring still promised "<src>.<platform>.<ext> next to the source",
# so callers resolving output relative to the input got the wrong file.

def test_transcode_output_is_confined_to_media_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    src_dir = tmp_path / "elsewhere"
    src_dir.mkdir()

    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths
    import opendispatch.media.transcode as media_transcode

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    src = src_dir / "pic.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(src)

    dest = media_transcode.transcode_image(src, "twitter")

    # Confined: the output lives under the media root, never beside an
    # untrusted source path.
    assert media_root.resolve() in dest.parents, dest
    assert dest.name == "pic.png.twitter.jpg"
    assert dest.is_file()

    # The documented contract must match the behavior.
    doc = (media_transcode.transcode_image.__doc__ or "")
    assert "next to the source" not in doc, (
        "docstring still promises output next to the source, but the "
        "destination is confined to the media root"
    )


def test_transcode_accepts_explicit_relative_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths
    import opendispatch.media.transcode as media_transcode

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    src = media_root / "pic.png"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(src)

    dest = media_transcode.transcode_image(src, "twitter", dest_path="out/custom.jpg")

    assert dest == (media_root / "out" / "custom.jpg").resolve()
    assert dest.is_file()


def test_transcode_rejects_destination_escaping_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths
    import opendispatch.media.transcode as media_transcode

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    src = media_root / "pic.png"
    Image.new("RGB", (8, 8), (0, 0, 255)).save(src)

    with pytest.raises(media_paths.MediaPathError):
        media_transcode.transcode_image(src, "twitter", dest_path="../escape.jpg")


# ── Defect 3: the is_symlink() guard was inert after .resolve() ─────────────
# resolve() dereferences the link, so resolved.is_symlink() can never be True.
# The real protection is the relative_to(root) check; assert that instead.

def test_resolve_media_path_blocks_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(outside)
    (media_root / "link.png").symlink_to(outside)

    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    with pytest.raises(media_paths.MediaPathError):
        media_paths.resolve_media_path("link.png", strict_root=True)


def test_resolve_media_path_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "secret.png")
    (media_root / "ok.png").write_bytes(b"x")

    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    with pytest.raises(media_paths.MediaPathError):
        media_paths.resolve_media_path("../secret.png", strict_root=True)


def test_resolve_media_path_allows_regular_file_in_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    (media_root / "ok.png").write_bytes(b"x")

    monkeypatch.setenv("OPEN_DISPATCH_MEDIA_DIR", str(media_root))
    import opendispatch.media.paths as media_paths

    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media_root.resolve())

    resolved = media_paths.resolve_media_path("ok.png", strict_root=True)
    assert resolved == (media_root / "ok.png").resolve()
