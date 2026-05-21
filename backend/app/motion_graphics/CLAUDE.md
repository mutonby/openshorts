# `openshorts/motion_graphics/`

Animated overlays (lower-thirds, callouts, progress bars, animated emoji,
etc.). Each effect subclasses `MotionGraphicEffect` from `base.py` and is
registered in `library/__init__.py`. The compositor in `compositor.py`
batches a timeline of effects into a single `filter_complex` chain so the
video is encoded only once.

See `ROADMAP.md` (feature C) — this ships first because it's the
prerequisite for the audio mixer's batching pattern.
