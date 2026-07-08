import os
import json
import re
import subprocess
import time
from typing import List
from google import genai
from google.genai import types
from pydantic import BaseModel

from edit_builder import build_filter_string


class EditDecision(BaseModel):
    type: str
    start: float
    end: float
    strength: float = 0.0
    reason: str = ""


class EditPlan(BaseModel):
    edits: List[EditDecision]


class VideoEditor:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_name = (
            os.environ.get("GEMINI_MODEL_EDITOR")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-3-flash-preview"
        )

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
        deadline = time.time() + 120
        while True:
            file_info = self.client.files.get(name=file_upload.name)
            state = getattr(file_info, "state", file_info)
            state_name = str(getattr(state, "name", state)).upper()
            if state_name == "ACTIVE":
                print("✅ Video processed and ready.")
                return file_upload
            if state_name == "FAILED":
                raise Exception("Video processing failed by Gemini.")
            if time.time() > deadline:
                raise TimeoutError("Gemini file processing timed out after 120s.")
            time.sleep(2)

    def get_ffmpeg_filter(self, video_file_obj, duration, fps=30, width=None, height=None, transcript=None, has_captions=False):
        """Asks Gemini for an edit decision list, then builds the FFmpeg filter
        deterministically (edit_builder) so syntax and zoom limits are always safe."""
        if width is None or height is None:
            # Keep prompt usable even if caller didn't pass dimensions.
            width, height = 1080, 1920

        transcript_text = json.dumps(transcript) if transcript else "Not available."

        caption_rule = ""
        if has_captions:
            caption_rule = (
                "\n        CRITICAL: This video already has burned-in captions and/or a hook text. "
                "Any zoom would crop or shift them off screen. Do NOT use zoom_in, punch_in or zoom_pulse — "
                "only color_pop, bw_moment, flash and vignette are allowed.\n"
            )

        prompt = f"""
        You are a viral short-form video editor (TikTok / Reels / Shorts). You receive a video and its transcript.
        Decide WHERE a small set of tasteful, high-impact effects belongs. You do NOT write FFmpeg —
        you return an edit decision list, and a deterministic renderer applies it safely.

        Video duration: {duration:.1f} seconds. Resolution: {width}x{height}.

        Available effect types:
        - "zoom_in": slow push-in across the segment. Builds tension or focus on a key statement. strength 0.06-0.15, segment 1.5-6s.
        - "punch_in": instant tighter framing for the whole segment. Emphasis, punchlines, "listen to this" moments. strength 0.06-0.15, segment 0.8-5s.
        - "zoom_pulse": quick in-and-out pulse peaking mid-segment. Beat drops, single impactful words. strength 0.04-0.10, segment 0.4-1.2s.
        - "color_pop": richer saturation/contrast. Energy, excitement, product/visual highlights. strength 0.2-1.0.
        - "bw_moment": black & white. Drama, serious quotes, flashbacks.
        - "flash": brief bright flash at segment start. Hard-cut energy, reveals. Use at most 2.
        - "vignette": darkened edges. Focus, intimacy, storytelling moments.

        Rules:
        1. Match effects to the CONTENT: place them on the exact words or moments they emphasize, using the transcript timestamps.
        2. Less is more: 2-6 edits per 30 seconds of video. A random effect is worse than no effect.
        3. Never overlap two zoom-type edits (zoom_in/punch_in/zoom_pulse).
        4. Calm, serious delivery => few or no motion effects. High-energy content => more punch.
        {caption_rule}
        TRANSCRIPT (with timestamps, the context of what is being said):
        {transcript_text}

        Return JSON only:
        {{"edits": [{{"type": "punch_in", "start": 3.2, "end": 5.4, "strength": 0.1, "reason": "punchline"}}]}}
        If no effects genuinely improve the video, return {{"edits": []}}.
        """

        print("🤖 Asking Gemini for an edit decision list...")
        # Low media resolution: Gemini samples at 1 FPS and normalizes frames to
        # fixed token sizes, so "low" (~70 tokens/frame) cuts video-input cost
        # without hurting motion/scene understanding. Fall back to defaults if
        # the SDK or model rejects the option.
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EditPlan,
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            )
        except Exception:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EditPlan,
            )
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video_file_obj, prompt],
                config=config,
            )
        except Exception as e:
            if getattr(config, "media_resolution", None) is None:
                raise
            print(f"⚠️ media_resolution=low rejected ({e}); retrying with defaults...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[video_file_obj, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EditPlan,
                ),
            )

        raw_edits = self._extract_edits(response)
        if raw_edits is None:
            return None

        filter_string, applied = build_filter_string(
            raw_edits, duration=duration, fps=fps, width=width, height=height,
            has_captions=has_captions,
        )
        if not filter_string:
            print("ℹ️ Gemini suggested no (valid) edits — keeping the clip untouched.")
            return {"filter_string": None, "edits": []}
        print(f"🎯 Applying {len(applied)} edits: " + ", ".join(f"{e['type']}@{e['start']:.1f}s" for e in applied))
        return {"filter_string": filter_string, "edits": applied}

    @staticmethod
    def _extract_edits(response):
        """Pull the edit list out of a Gemini response (parsed schema preferred,
        raw-JSON fallback). Returns a list of dicts, or None on hard failure."""
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                return [edit.model_dump() for edit in parsed.edits]
            except Exception:
                pass

        text = (getattr(response, "text", None) or "").strip()
        print(f"🔍 DEBUG: Gemini Raw Response:\n{text}")
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"❌ Failed to parse edit plan JSON: {text[:500]}")
            return None
        edits = data.get("edits") if isinstance(data, dict) else None
        return edits if isinstance(edits, list) else []

    @staticmethod
    def _split_filter_chain(filter_string: str) -> list[str]:
        """Split a -vf filter chain on commas, respecting single-quoted substrings."""
        parts: list[str] = []
        start = 0
        in_quote = False
        for i, ch in enumerate(filter_string):
            if ch == "'":
                in_quote = not in_quote
            elif ch == "," and not in_quote:
                parts.append(filter_string[start:i])
                start = i + 1
        parts.append(filter_string[start:])
        return parts

    @classmethod
    def _enforce_zoompan_output_size(cls, filter_string: str, width: int, height: int) -> str:
        """Force any zoompan filter to output the same geometry as the input clip."""
        parts = cls._split_filter_chain(filter_string)
        out_parts: list[str] = []
        for part in parts:
            if "zoompan=" in part:
                # Force s=WxH inside zoompan options (digitsxdigits only).
                if re.search(r":s=\d+x\d+", part):
                    part = re.sub(r":s=\d+x\d+", f":s={width}x{height}", part)
                else:
                    part = f"{part}:s={width}x{height}"
            out_parts.append(part)
        return ",".join(out_parts)

    @staticmethod
    def _sanitize_filter_string(filter_string: str) -> str:
        """
        Best-effort sanitizer for Gemini-generated FFmpeg expressions.
        Converts comparison operators (t<3, on>=75, etc.) into FFmpeg expr functions (lt(), gte(), ...),
        which are far more reliably parsed across FFmpeg builds.
        """
        s = filter_string

        # Order matters: handle >= / <= before > / <
        patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>=\s*(-?\d+(?:\.\d+)?)"), r"gte(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<=\s*(-?\d+(?:\.\d+)?)"), r"lte(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>\s*(-?\d+(?:\.\d+)?)"), r"gt(\1,\2)"),
            (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<\s*(-?\d+(?:\.\d+)?)"), r"lt(\1,\2)"),
        ]
        for pat, repl in patterns:
            s = pat.sub(repl, s)

        return s

    @staticmethod
    def _test_filter(input_path, filter_string, env):
        """Dry-run the filter on the first 2 seconds (no output written).

        Catches broken Gemini-generated filter syntax in seconds instead of
        failing after a full-length encode. Returns (ok, stderr_tail)."""
        cmd = [
            'ffmpeg', '-v', 'error', '-t', '2',
            '-i', input_path,
            '-vf', filter_string,
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(
                [a.encode('utf-8') if isinstance(a, str) else a for a in cmd],
                env=env, capture_output=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return False, "filter dry-run timed out after 120s"
        if result.returncode == 0:
            return True, ""
        stderr_text = (result.stderr or b"").decode(errors="replace")
        return False, stderr_text[-1500:]

    def _repair_filter(self, filter_string, error_text, width, height):
        """One self-repair round-trip: show Gemini the FFmpeg error and ask for
        a corrected filter string. Returns the new string or None."""
        if not self.client:
            return None
        prompt = f"""
        The following FFmpeg -vf filter string you generated fails to run.

        FILTER:
        {filter_string}

        FFMPEG ERROR:
        {error_text}

        Fix the filter. Keep the same creative intent, obey the same rules as before:
        - exact output resolution {width}x{height} (zoompan must set s={width}x{height}),
        - no bare comparison operators (use between/lt/lte/gt/gte),
        - expression values in single quotes.
        Output JSON only: {{"filter_string": "..."}}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = (response.text or "").strip()
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx + 1]
            repaired = json.loads(text).get("filter_string")
            return repaired if isinstance(repaired, str) and repaired.strip() else None
        except Exception as e:
            print(f"⚠️ Filter self-repair failed: {e}")
            return None

    def apply_edits(self, input_path, output_path, filter_data):
        """Executes FFmpeg with the generated filter."""
        
        if not filter_data or not filter_data.get("filter_string"):
            print("⚠️ No filter string found. Copying original.")
            try:
                subprocess.run(['ffmpeg', '-y', '-i', input_path, '-c', 'copy', output_path], timeout=1800)
            except subprocess.TimeoutExpired:
                raise RuntimeError("FFmpeg copy timed out after 1800s.")
            return

        filter_string = filter_data["filter_string"]
        
        # Get input dimensions so we can enforce geometry (avoid broken aspect ratios).
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', input_path]
            res_out = subprocess.check_output(probe_cmd, env={**os.environ, "LANG": "C.UTF-8"}, timeout=60).decode().strip()
            w, h = map(int, res_out.split('x'))
        except Exception as e:
            print(f"⚠️ Could not probe resolution: {e}")
            w, h = None, None

        # Sanitize common expression pitfalls (e.g., t<3 / on>=75) before executing FFmpeg.
        sanitized = self._sanitize_filter_string(filter_string)
        if sanitized != filter_string:
            print("🧼 Sanitized AI Filter (converted comparisons to lt/lte/gt/gte functions)")
            print(f"🧼 Before: {filter_string}")
            print(f"🧼 After:  {sanitized}")
            filter_string = sanitized

        # Enforce zoompan output size to preserve aspect ratio / resolution.
        if w and h:
            enforced = self._enforce_zoompan_output_size(filter_string, w, h)
            if enforced != filter_string:
                print(f"📐 Enforced zoompan output size to {w}x{h}")
                filter_string = enforced

            # Ensure square pixels (avoid weird display stretching in some players).
            if "setsar=" not in filter_string:
                filter_string = f"{filter_string},setsar=1"

        # Use explicit environment with UTF-8 to avoid ascii errors in subprocess
        env = os.environ.copy()
        # On some minimal docker images, we need to ensure we use a UTF-8 locale
        # Try C.UTF-8 first, fallback to en_US.UTF-8 if available, but C.UTF-8 is usually safer for minimal
        env["LANG"] = "C.UTF-8"
        env["LC_ALL"] = "C.UTF-8"

        # Dry-run the filter on 2 seconds before committing to a full encode;
        # on failure, give Gemini one self-repair attempt with the real error.
        ok, error_text = self._test_filter(input_path, filter_string, env)
        if not ok:
            print(f"⚠️ AI filter failed dry-run: {error_text}")
            repaired = self._repair_filter(filter_string, error_text, w or 1080, h or 1920)
            if repaired:
                repaired = self._sanitize_filter_string(repaired)
                if w and h:
                    repaired = self._enforce_zoompan_output_size(repaired, w, h)
                    if "setsar=" not in repaired:
                        repaired = f"{repaired},setsar=1"
                ok, error_text = self._test_filter(input_path, repaired, env)
                if ok:
                    print("🔧 Self-repaired AI filter passed dry-run.")
                    filter_string = repaired
            if not ok:
                raise RuntimeError(f"AI filter failed validation even after self-repair: {error_text}")

        print(f"🎬 Executing AI Filter: {filter_string}")

        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', filter_string,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'copy',
            '-movflags', '+faststart',
            output_path
        ]
        
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
            
            subprocess.run(cmd_bytes, check=True, env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg filter timed out after 1800s.")
            raise RuntimeError("FFmpeg filter timed out after 1800s.")
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg failed: {e}")
            raise e

if __name__ == "__main__":
    pass