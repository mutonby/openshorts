"""VideoEditor: Gemini-driven FFmpeg filter generation and application.

The shared filter helpers (sanitize_filter_string, enforce_zoompan_output_size)
live in openshorts/utils/filters.py and are exposed as static/classmethods on
VideoEditor for backwards compatibility.
"""
import os
import json
import time

from app.video import ffmpeg as ffmpeg_wrapper

from google import genai
from google.genai import types

from app.editing.prompts import (
    build_ffmpeg_filter_prompt,
    build_effects_config_prompt,
)
from app.utils.filters import (
    split_filter_chain as _split_filter_chain_fn,
    enforce_zoompan_output_size as _enforce_zoompan_output_size_fn,
    sanitize_filter_string as _sanitize_filter_string_fn,
    validate_filter_string as _validate_filter_string_fn,
    UnsafeFilterError,
)


class VideoEditor:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"

    def upload_video(self, video_path):
        """Uploads video to Gemini File API."""
        print(f"📤 Uploading {video_path} to Gemini...")

        # Ensure we are passing a path that exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Using 'file' keyword instead of 'path'
        try:
            file_upload = self.client.files.upload(file=video_path)
        except Exception as e:
            print(f"❌ Gemini Upload Error: {e}")
            raise e

        # Wait for processing
        print("⏳ Waiting for video processing by Gemini...")
        while True:
            file_info = self.client.files.get(name=file_upload.name)
            if file_info.state == "ACTIVE":
                print("✅ Video processed and ready.")
                return file_upload
            elif file_info.state == "FAILED":
                raise Exception("Video processing failed by Gemini.")
            time.sleep(2)

    def get_ffmpeg_filter(self, video_file_obj, duration, fps=30, width=None, height=None, transcript=None):
        """Asks Gemini for a raw FFmpeg filter string."""
        if width is None or height is None:
            # Keep prompt usable even if caller didn't pass dimensions.
            width, height = 1080, 1920

        prompt = build_ffmpeg_filter_prompt(duration, fps, width, height, transcript)

        print("🤖 Asking Gemini for FFmpeg filter...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file_obj, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        print(f"🔍 DEBUG: Gemini Raw Response:\n{response.text}")

        try:
            # Clean response text (remove potential markdown blocks)
            text = response.text
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Additional cleanup for potential trailing characters outside JSON
            # Find the first '{' and last '}'
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]

            print(f"🔍 DEBUG: Cleaned JSON Text:\n{text}")

            return json.loads(text)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse JSON: {response.text}")
            return None

    def get_effects_config(self, video_file_obj, duration, fps=30, width=None, height=None, transcript=None):
        """Asks Gemini for a structured EffectsConfig JSON for Remotion rendering."""
        if width is None or height is None:
            width, height = 1080, 1920

        prompt = build_effects_config_prompt(duration, fps, width, height, transcript)

        print("🤖 Asking Gemini for Remotion effects config...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file_obj, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        print(f"🔍 DEBUG: Gemini Raw Response:\n{response.text}")

        try:
            # Clean response text (remove potential markdown blocks)
            text = response.text
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

            # Find the first '{' and last '}'
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]

            print(f"🔍 DEBUG: Cleaned JSON Text:\n{text}")

            return json.loads(text)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse effects config JSON: {response.text}")
            return None

    @staticmethod
    def _split_filter_chain(filter_string):
        return _split_filter_chain_fn(filter_string)

    @classmethod
    def _enforce_zoompan_output_size(cls, filter_string, width, height):
        return _enforce_zoompan_output_size_fn(filter_string, width, height)

    @staticmethod
    def _sanitize_filter_string(filter_string):
        return _sanitize_filter_string_fn(filter_string)

    def apply_edits(self, input_path, output_path, filter_data):
        """Executes FFmpeg with the generated filter."""

        if not filter_data or "filter_string" not in filter_data:
            print("⚠️ No filter string found. Copying original.")
            ffmpeg_wrapper.run(['-y', '-i', input_path, '-c', 'copy', output_path])
            return

        filter_string = filter_data["filter_string"]

        # Sanitize common expression pitfalls (e.g., t<3 / on>=75) before
        # validating: post-sanitization is the form that actually executes,
        # so that's what must pass the allowlist.
        sanitized = _sanitize_filter_string_fn(filter_string)
        if sanitized != filter_string:
            print("🧼 Sanitized AI Filter (converted comparisons to lt/lte/gt/gte functions)")
            print(f"🧼 Before: {filter_string}")
            print(f"🧼 After:  {sanitized}")
            filter_string = sanitized

        # SAFETY: reject any LLM-produced filter that calls a non-allowlisted
        # FFmpeg filter (movie/amovie/subtitles/concat/ass/...). Raises
        # UnsafeFilterError, which the /api/edit route should surface as 400.
        _validate_filter_string_fn(filter_string)

        # Get input dimensions so we can enforce geometry (avoid broken aspect ratios).
        try:
            w, h = ffmpeg_wrapper.probe_resolution(input_path)
        except Exception as e:
            print(f"⚠️ Could not probe resolution: {e}")
            w, h = None, None

        # Enforce zoompan output size to preserve aspect ratio / resolution.
        if w and h:
            enforced = _enforce_zoompan_output_size_fn(filter_string, w, h)
            if enforced != filter_string:
                print(f"📐 Enforced zoompan output size to {w}x{h}")
                filter_string = enforced

            # Ensure square pixels (avoid weird display stretching in some players).
            if "setsar=" not in filter_string:
                filter_string = f"{filter_string},setsar=1"

        print(f"🎬 Executing AI Filter: {filter_string}")

        # Wrapper sets LANG/LC_ALL=C.UTF-8 so the prior bytes-encoding hack
        # is no longer needed; subprocess on Python 3 with a UTF-8 locale
        # handles unicode args correctly.
        ffmpeg_wrapper.run([
            '-y',
            '-i', input_path,
            '-vf', filter_string,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'copy',
            output_path,
        ])
