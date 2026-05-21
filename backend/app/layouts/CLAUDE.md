# `openshorts/layouts/`

Layout templates control how a vertical clip is composed (single-subject
panorama, two-pane educational, side-by-side, picture-in-picture). Each
layout subclasses `Layout` from `base.py` and exposes a single
`render_frame(frame, detections, frame_number)` method. The pipeline's
hot loop calls that — don't bypass it with raw cv2 in routers or other
high-level callers.

See `ROADMAP.md` (feature B) for the migration plan from today's inline
TRACK/GENERAL branching to polymorphic layouts.
