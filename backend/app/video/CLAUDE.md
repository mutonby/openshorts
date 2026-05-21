# `openshorts/video/`

All video processing. **FFmpeg is invoked only through `ffmpeg.py`.** Never
call `subprocess.run(['ffmpeg', ...])` directly from a module in this folder
or any caller of it — funnel through the wrapper.
