"""Compat shim: re-exports openshorts.overlays.hooks at the original import path.

This module moved to openshorts/overlays/hooks.py as part of the restructure.
New code should import from `openshorts.overlays.hooks` directly; this shim
keeps existing `from hooks import ...` calls working.
"""
from openshorts.overlays.hooks import *  # noqa: F401,F403
from openshorts.overlays.hooks import (  # noqa: F401
    FONT_URL,
    FONT_DIR,
    FONT_PATH,
    download_font_if_needed,
    create_hook_image,
    add_hook_to_video,
)
