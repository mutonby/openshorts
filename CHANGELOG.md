# Changelog

All notable changes to OpenShorts are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] — 2026-07-08

Major overhaul release: new output formats, watermarking, a completely reworked AI Auto-Edit,
modern karaoke subtitles, a reliability pass over the whole pipeline, and a test suite with CI.

### Added

**Output & Rendering**
- **Output format selection** (Opus-Clip style): choose **Auto (smart)**, **9:16** (Shorts/Reels with speaker tracking), **16:9** (original, no reframing) or **1:1** (square) when starting a job. Auto detects sources that already match the target aspect (e.g. phone footage) and passes them through untouched instead of re-cropping. The chosen format is persisted per job (`render_config.json`), so resumed jobs keep rendering identically.
- **Watermark**: a subtle, centered watermark (default: OpenShorts logo at ~8% opacity) is blended into every rendered clip directly inside the frame loop — zero extra encode passes. Fully configurable via `.env`: `WATERMARK_TEXT` (e.g. your channel name), `WATERMARK_IMAGE` (custom PNG), `WATERMARK_OPACITY`, `WATERMARK_WIDTH_FRACTION`, or `WATERMARK_ENABLED=0` to disable.

**Auto-Edit v2 (complete rework)**
- Gemini no longer writes raw FFmpeg filter strings. It now returns a structured **edit decision list** (what / when / how strong), and a deterministic builder (`edit_builder.py`) converts it into a guaranteed-valid filter chain.
- 7 effect types: `zoom_in`, `punch_in`, `zoom_pulse`, `color_pop`, `bw_moment`, `flash`, `vignette`.
- Hard safety limits: max zoom 1.15, no overlapping zooms, zoom window anchored with facial headroom, max 12 edits, max 2 flashes.
- **Caption-safe editing**: if the clip already has burned-in subtitles or a hook, all zoom effects are automatically blocked so text always stays visible.
- 2-second FFmpeg dry-run before the full encode, with one Gemini self-repair round-trip on failure.

**Subtitles**
- Modern **karaoke captions** with word-level highlighting (active word in a custom color).
- **11 preset looks**: TikTok, Reels, Shorts Pop, Gold Glow, Neon, Cyber, Karaoke, Minimal, Beast, Boxed, Classic.
- Visual effects per preset: **glow**, **pop** (scale-in per word), **box**; plus dim control for non-active words, UPPERCASE toggle, font size slider (14–40), custom fonts and colors.
- **Bulk subtitles**: configure a style once and apply it to all clips of a job in one click.
- **Download all clips as ZIP.**

**Gemini pipeline**
- Two-stage analysis: transcript windows are scored first, then only the shortlist gets the expensive detail pass — better clips at lower cost.
- Structured output via Pydantic `response_schema` (no more JSON-parse failures), split temperatures per strategy, word-boundary snapping of clip cuts, transcript-language enforcement for all generated text, viral hook playbook, hashtag & diversity rules.
- Models configurable per task via `.env` (`GEMINI_MODEL`, `GEMINI_MODEL_ANALYSIS`, `GEMINI_MODEL_EDITOR`, `GEMINI_MODEL_THUMBNAIL`, `GEMINI_MODEL_IMAGE`, `GEMINI_MODEL_SAAS`), optional thinking control (`GEMINI_THINKING_SCORE`), full cost tracking including thinking tokens.
- Editor video uploads use `media_resolution=low` (~70 tokens/frame) — large video-input cost cut with no quality loss for edit decisions.

**Reliability & job management**
- **Pre-flight quality gate**: before a job starts, a fast probe (`quality_probe.py`) checks which resolution YouTube actually offers. If it is below `QUALITY_GATE_MIN_HEIGHT` (default 720p), the UI shows a popup — process anyway or fix cookies first (with step-by-step incognito export instructions) — instead of silently burning 20+ minutes on a 360p source.
- **Self-learning ETA**: each phase's real duration per video-second is persisted (`.phase_stats.json`, median of last 10 jobs), the current phase is extrapolated from its live measured rate, and a one-time total estimate ("Estimated total processing time: ~X min") is announced right after download and shown in the dashboard.
- **Clean stop/cancel**: cancel endpoint kills worker processes server-side and frees the queue slot; recovery banner offers resume-or-restart choice after crashes.
- **Keepalive heartbeat** thread and **Windows standby prevention** (`SetThreadExecutionState`) — long silent transcriptions no longer trigger false "stalled" states or freeze when the laptop sleeps.
- **Resume hardening**: jobs that died mid-transcription (no checkpoints yet) now resume from the source video instead of crashing; resume resets the elapsed clock.
- One-time notice (instead of a repeating alert) when the Upload-Post key has no profiles yet.

**Project infrastructure**
- Test suite: **63 pytest tests** covering subtitle engine, clip selection/snapping, edit builder, hooks and translation helpers.
- **GitHub Actions CI**: backend tests, frontend lint + build, Docker build.
- `start.bat` for local Windows startup without Docker.
- `.env.example` documenting all new configuration (watermark, Whisper, Gemini models/thinking, quality gate).

### Changed
- **yt-dlp**: removed all hardcoded `player_client` overrides — yt-dlp's curated default clients are the ones that still serve HD without PO tokens. Restores up to 4K downloads (verified 2160p) where only 360p was available before.
- Whisper default model is now `small` (noticeably better German than `base`), with tuned transcription parameters (`beam_size=5`, VAD filter, no cross-segment conditioning against hallucinations); configurable via `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE`.
- Recommended default analysis model: `gemini-3.1-flash-lite` (cheaper **and** stronger than 2.5 Flash).
- `-movflags +faststart` on all rendered outputs — downloaded clips play instantly everywhere.
- Frontend performance: `React.lazy` code splitting (initial JS 372 KB → 245 KB), GZip on API responses (status polling 249 KB → 30 KB), optimized logo (266 KB → 7 KB), reduced polling churn and log payloads.

### Fixed
- Subtitle word-gluing ("ichhabe" instead of "ich habe"): whisper continuation fragments are now merged into the previous word using the leading-space token signal.
- Gray/muddy dimmed subtitle text: semi-transparent fill blended with the libass outline; replaced with fully opaque RGB dimming — colors now match the preview exactly.
- Double subtitles when re-subtitling an already subtitled clip (prefix stripping now applies to both bulk and manual paths).
- Enlarged/fullscreen clip preview looked cropped and wrong: `object-cover` was cutting the video both in the card and in native fullscreen; previews now letterbox (`object-contain`) and a global CSS rule enforces contain in fullscreen. Works for all output formats.
- Resume crash ("Process failed with exit code 1") when a job died before any checkpoint was written.
- Runtime display counting frozen hours (e.g. "ETA 2m at 5h24m runtime") — elapsed clock resets on resume, ETA is now measurement-based.
- Zombie FFmpeg/worker processes after cancel or crash; encoder subprocess cleanup on aborted frame loops.
- Frontend memory leaks, polling races after completion, unbounded job-state growth, `datetime.utcnow` deprecations, missing `raise_for_status` checks, thumbnail upload size limits, title-slug collisions.
- Hook overlays: emoji rendering (color emoji font runs), text measurement, and positioning fixes.

### Security
- Color/font/number inputs sanitized before entering ASS subtitle files and FFmpeg filter strings (injection hardening).
- API keys remain client-side encrypted; YouTube cookies (`*_cookies.txt`), runtime stats and local notes are excluded from the repository via `.gitignore`.

---

## [1.x] — earlier

Initial platform: Clip Generator (Gemini viral-moment detection, dual-mode 9:16 reframing with
MediaPipe/YOLOv8 tracking), AI Shorts UGC pipeline, YouTube Studio (thumbnails/titles/descriptions),
ElevenLabs dubbing, Upload-Post social publishing, S3 backup, Docker setup.
