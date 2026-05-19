"""VideoEditor: Gemini-driven FFmpeg filter generation and application.

The shared filter helpers (sanitize_filter_string, enforce_zoompan_output_size)
live in openshorts/utils/filters.py and are exposed as static/classmethods on
VideoEditor for backwards compatibility.
"""
import os
import json
import subprocess
import time

from google import genai
from google.genai import types

from openshorts.editing.prompts import (
    build_ffmpeg_filter_prompt,
    build_effects_config_prompt,
)
from openshorts.utils.filters import (
    split_filter_chain as _split_filter_chain_fn,
    enforce_zoompan_output_size as _enforce_zoompan_output_size_fn,
    sanitize_filter_string as _sanitize_filter_string_fn,
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
            subprocess.run(['ffmpeg', '-y', '-i', input_path, '-c', 'copy', output_path])
            return

        filter_string = filter_data["filter_string"]

        # Get input dimensions so we can enforce geometry (avoid broken aspect ratios).
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', input_path]
            res_out = subprocess.check_output(probe_cmd, env={**os.environ, "LANG": "C.UTF-8"}).decode().strip()
            w, h = map(int, res_out.split('x'))
        except Exception as e:
            print(f"⚠️ Could not probe resolution: {e}")
            w, h = None, None

        # Sanitize common expression pitfalls (e.g., t<3 / on>=75) before executing FFmpeg.
        sanitized = _sanitize_filter_string_fn(filter_string)
        if sanitized != filter_string:
            print("🧼 Sanitized AI Filter (converted comparisons to lt/lte/gt/gte functions)")
            print(f"🧼 Before: {filter_string}")
            print(f"🧼 After:  {sanitized}")
            filter_string = sanitized

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

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', filter_string,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'copy',
            output_path
        ]

        # Use explicit environment with UTF-8 to avoid ascii errors in subprocess
        env = os.environ.copy()
        # On some minimal docker images, we need to ensure we use a UTF-8 locale
        # Try C.UTF-8 first, fallback to en_US.UTF-8 if available, but C.UTF-8 is usually safer for minimal
        env["LANG"] = "C.UTF-8"
        env["LC_ALL"] = "C.UTF-8"

        try:
            # We must encode arguments if filesystem is ascii but we have unicode chars
            # But subprocess in Python 3 handles unicode args by encoding them with os.fsencode().
            # If sys.getfilesystemencoding() is ascii, this fails.
            # We can't change fs encoding at runtime easily.
            # Workaround: pass bytes directly? subprocess allows bytes in args.

            # Convert command elements to bytes assuming utf-8 if they are strings
            cmd_bytes = []
            for arg in cmd:
                if isinstance(arg, str):
                    cmd_bytes.append(arg.encode('utf-8'))
                else:
                    cmd_bytes.append(arg)

            subprocess.run(cmd_bytes, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")
            raise e
