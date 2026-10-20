"""Confined local media path resolution for worker-side file access."""
from __future__ import annotations

import os
from pathlib import Path

MEDIA_ROOT = Path(os.getenv("OPEN_DISPATCH_MEDIA_DIR", str(Path(os.getenv("OPEN_DISPATCH_DATA", "data")) / "media"))).expanduser()


class MediaPathError(ValueError):
    """Raised when a media path escapes the configured media root."""


def resolve_media_path(value: str | Path, *, must_exist: bool = True, strict_root: bool = False) -> Path:
    """Resolve `value` to a readable file, refusing anything outside the media root.

    Note on symlinks: every branch calls `.resolve()`, which dereferences
    symlinks before the check. A post-`resolve()` `is_symlink()` test can
    therefore never be true, so the real protection against a symlink escape
    is the `relative_to(root)` containment check below — a link inside the
    root pointing outside resolves to a path outside the root and is rejected.
    """
    candidate = Path(value)
    if candidate.is_absolute() and (not strict_root or not os.getenv("OPEN_DISPATCH_MEDIA_DIR")):
        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise MediaPathError("media path does not exist") from exc
        if must_exist and not resolved.is_file():
            raise MediaPathError("media path must be a regular file")
        return resolved
    if candidate.is_absolute():
        raise MediaPathError("absolute media paths are not allowed")
    root = MEDIA_ROOT.resolve()
    resolved = (root / candidate).resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MediaPathError("media path escapes the configured media root") from exc
    if must_exist and not resolved.is_file():
        raise MediaPathError("media path must be a regular file")
    return resolved


def resolve_media_destination(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and not os.getenv("OPEN_DISPATCH_MEDIA_DIR"):
        resolved = candidate.resolve(strict=False)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    if candidate.is_absolute():
        raise MediaPathError("absolute media paths are not allowed")
    root = MEDIA_ROOT.resolve()
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MediaPathError("media destination escapes the configured media root") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
