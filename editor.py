"""Compat shim: re-exports openshorts.editing.ai_filters.VideoEditor at the original path.

This module was split into three files as part of the restructure:
- openshorts/editing/ai_filters.py  (VideoEditor class)
- openshorts/editing/prompts.py     (Gemini prompt templates)
- openshorts/utils/filters.py       (shared FFmpeg filter helpers)

New code should import from those modules directly. This shim keeps existing
`from editor import VideoEditor` calls working.
"""
from openshorts.editing.ai_filters import VideoEditor  # noqa: F401
