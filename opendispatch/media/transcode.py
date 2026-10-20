"""Per-platform image transcoding.

Each platform has its own preferred aspect ratios + size caps. Posting a
4032x3024 phone photo straight to Twitter wastes bandwidth and quality
(Twitter re-encodes it on their side). Pre-sizing on the way out gives
us:

- Predictable file sizes (uploads finish, no timeouts)
- Predictable aspect (no surprise crops)
- Clean re-encode (we control the quality dial)

Pillow is the only dependency. No ffmpeg, no native bins.

Public API
----------
- `transcode_image(src_path, platform, dest_path=None) -> Path`
- `transcode_image_bytes(blob, platform) -> bytes`
- `PLATFORM_IMAGE_SPECS` — the spec table, public so callers can introspect

Adding a platform = adding one row to PLATFORM_IMAGE_SPECS.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from opendispatch.media.paths import resolve_media_destination, resolve_media_path

log = logging.getLogger("open-dispatch.media.transcode")


class TranscodeError(Exception):
    """Raised when transcoding can't produce a valid output."""


@dataclass(frozen=True)
class ImageSpec:
    """Spec for a single platform-flavored image."""
    max_width: int
    max_height: int
    aspect_min: float | None  # null = don't enforce
    aspect_max: float | None
    format: str  # "JPEG" or "PNG"
    quality: int  # for JPEG, 1-100
    description: str

    @property
    def max_size(self) -> tuple[int, int]:
        return self.max_width, self.max_height


# Platform-spec table. Numbers from the each platform's current developer docs.
# When a platform supports multiple aspects, we pick the most common:
#   Instagram → 1080x1080 square (carousel-safe)
#   IG Reels   → 1080x1920 vertical (use platform="instagram_reels")
#   Twitter   → 1200x675 16:9 (best engagement)
#   LinkedIn  → 1200x627 16:9 (best engagement)
#   YouTube Shorts thumbnail → 1080x1920 vertical
PLATFORM_IMAGE_SPECS: dict[str, ImageSpec] = {
    "instagram": ImageSpec(
        max_width=1080, max_height=1080,
        aspect_min=1.0, aspect_max=1.0,  # pure 1:1 square (safest IG default)
        format="JPEG", quality=90,
        description="Square (1080x1080) — safest IG default, works in carousels.",
    ),
    "instagram_portrait": ImageSpec(
        max_width=1080, max_height=1350,
        aspect_min=0.8, aspect_max=0.8,
        format="JPEG", quality=90,
        description="4:5 portrait (1080x1350) — max IG feed real estate.",
    ),
    "instagram_reels": ImageSpec(
        max_width=1080, max_height=1920,
        aspect_min=0.5625, aspect_max=0.5625,  # 9:16
        format="JPEG", quality=90,
        description="9:16 vertical reel cover (1080x1920).",
    ),
    "twitter": ImageSpec(
        max_width=1200, max_height=675,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=85,
        description="16:9 (1200x675) — Twitter cards display best at this size.",
    ),
    "linkedin": ImageSpec(
        max_width=1200, max_height=627,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=85,
        description="LinkedIn share image (1200x627).",
    ),
    "bluesky": ImageSpec(
        max_width=2000, max_height=2000,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=85,
        description="Bluesky max 2000px on the long side, 1MB cap.",
    ),
    "telegram": ImageSpec(
        max_width=1280, max_height=1280,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=85,
        description="Telegram photo — keeps it under their auto-compress threshold.",
    ),
    "threads": ImageSpec(
        max_width=1080, max_height=1350,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=88,
        description="Threads 4:5 portrait (1080x1350).",
    ),
    "youtube_thumbnail": ImageSpec(
        max_width=1280, max_height=720,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=90,
        description="YouTube video thumbnail (1280x720).",
    ),
    "youtube_short_thumbnail": ImageSpec(
        max_width=1080, max_height=1920,
        aspect_min=None, aspect_max=None,
        format="JPEG", quality=90,
        description="YouTube Shorts cover (1080x1920 vertical).",
    ),
}


# Video specs are aspirational here — full transcoding needs ffmpeg, which
# we may add as an optional dep in a later release. Right now we expose
# the spec table so callers can validate dimensions themselves.
PLATFORM_VIDEO_SPECS: dict[str, dict[str, int | str]] = {
    "instagram_reels":      {"max_seconds": 90,  "aspect": "9:16", "max_mb": 100},
    "instagram_post":       {"max_seconds": 60,  "aspect": "1:1",  "max_mb": 100},
    "tiktok":               {"max_seconds": 600, "aspect": "9:16", "max_mb": 287},  # ~300 MB
    "twitter":              {"max_seconds": 140, "aspect": "16:9", "max_mb": 512},
    "threads":              {"max_seconds": 300, "aspect": "9:16", "max_mb": 1000},
    "youtube_short":        {"max_seconds": 60,  "aspect": "9:16", "max_mb": 256},
    "linkedin":             {"max_seconds": 600, "aspect": "16:9", "max_mb": 5000},
}


# ─── Public API ──────────────────────────────────────────────────────────

def transcode_image(src_path: str | Path, platform: str,
                    dest_path: str | Path | None = None) -> Path:
    """Read an image from disk, transcode for `platform`, write back to disk.

    Both the source read and the write are confined to the media root
    (`media.paths.resolve_media_path` / `resolve_media_destination`), so
    output lands under that root — normally `<media_root>/<src name>.<platform>.<ext>`
    — not beside the source file. `dest_path` is therefore interpreted
    relative to the media root and may not escape it.

    Returns the destination Path.
    """
    try:
        src = resolve_media_path(src_path)
    except Exception as e:
        raise TranscodeError(f"source image not found: {src_path}") from e
    spec = PLATFORM_IMAGE_SPECS.get(platform)
    if not spec:
        raise TranscodeError(f"no image spec for platform: {platform!r}")

    out_bytes = transcode_image_bytes(src.read_bytes(), platform)

    if dest_path is None:
        ext = "jpg" if spec.format == "JPEG" else spec.format.lower()
        dest = resolve_media_destination(f"{src.name}.{platform}.{ext}")
    else:
        dest = resolve_media_destination(dest_path)
    dest.write_bytes(out_bytes)
    return dest


def transcode_image_bytes(blob: bytes, platform: str) -> bytes:
    """Pure-bytes transcode — useful when the source isn't on disk yet
    (e.g. user uploaded into a form, or proxying from a remote URL).
    """
    spec = PLATFORM_IMAGE_SPECS.get(platform)
    if not spec:
        raise TranscodeError(f"no image spec for platform: {platform!r}")

    try:
        from PIL import Image, ImageOps
        Image.MAX_IMAGE_PIXELS = 25_000_000
    except ImportError as e:
        raise TranscodeError("Pillow not installed — `pip install Pillow`") from e

    try:
        img = Image.open(io.BytesIO(blob))
    except Exception as e:  # noqa: BLE001
        raise TranscodeError(f"could not open image: {e}") from e

    # Honor EXIF orientation (phone photos are often rotated in metadata)
    img = ImageOps.exif_transpose(img)

    # JPEG can't carry alpha — flatten any RGBA/LA against white before encoding.
    if spec.format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        img = bg
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Aspect crop (if spec demands a specific aspect)
    if spec.aspect_min is not None and spec.aspect_max is not None:
        target_aspect = (spec.aspect_min + spec.aspect_max) / 2
        img = _center_crop_to_aspect(img, target_aspect)

    # Resize to fit within (max_width, max_height) while preserving the
    # (possibly cropped) aspect — never upscale.
    if img.size[0] > spec.max_width or img.size[1] > spec.max_height:
        img.thumbnail(spec.max_size, Image.Resampling.LANCZOS)

    out = io.BytesIO()
    save_kwargs: dict = {"format": spec.format, "optimize": True}
    if spec.format == "JPEG":
        save_kwargs["quality"] = spec.quality
        save_kwargs["progressive"] = True
    img.save(out, **save_kwargs)
    return out.getvalue()


# ─── Helpers ─────────────────────────────────────────────────────────────

def _center_crop_to_aspect(img, target_aspect: float):
    """Crop the image to `target_aspect` (w / h) without distortion."""
    w, h = img.size
    current = w / h
    if abs(current - target_aspect) < 0.01:
        return img  # close enough
    if current > target_aspect:
        # too wide — crop sides
        new_w = int(h * target_aspect)
        x = (w - new_w) // 2
        return img.crop((x, 0, x + new_w, h))
    else:
        # too tall — crop top/bottom
        new_h = int(w / target_aspect)
        y = (h - new_h) // 2
        return img.crop((0, y, w, y + new_h))
