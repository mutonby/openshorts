import os
import json
import re
import subprocess
import time
from google import genai
from google.genai import types

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
        
        transcript_text = json.dumps(transcript) if transcript else "Not available."

        prompt = f"""
        You are an expert FFmpeg video editor. Your task is to generate a complex video filter string to make a short video viral, BUT ONLY apply effects where they make sense contextually.

        Video Duration: {duration} seconds.
        Video FPS: {fps}
        Video Resolution (MUST KEEP EXACT): {width}x{height}
        
        TRANSCRIPT (Context of what is being said):
        {transcript_text}

        Goal: Enhance the video with dynamic zooms, cuts (simulated with punch-ins), and visual effects to increase retention, but DO NOT overdo it. Random effects are bad. Contextual effects are good.

        Instructions:
        1. ANALYZE THE VIDEO AND TRANSCRIPT: Understand the mood, the pacing, and the key moments.
        2. APPLY EFFECTS ONLY WHEN RELEVANT:
           - Use "punch-in" zooms (zoompan) to emphasize key points, jokes, or dramatic moments in the speech.
           - slow zooms to face when the speaker is speaking
           - Use visual effects (contrast, saturation, sharpness) to highlight mood changes or specific segments.
           - If nothing significant is happening, keep it simple. It is BETTER to have no effect than a random/distracting one.
           - Avoid constant motion if the speaker is delivering a serious or steady message.
        3. Create a single valid FFmpeg filter complex string (for the -vf flag).
        4. Use filters like `zoompan`, `eq` (contrast), `hue` (saturation/bw), `unsharp`.
        5. Pacing: Align effects with the rhythm of the speech (from transcript) or visual action.
        6. CRITICAL SYNTAX RULES:
           - DO NOT use comparison operators like `<`, `>`, `<=`, `>=` anywhere. They frequently break FFmpeg expression parsing.
           - USE FFmpeg expression FUNCTIONS instead:
             - `between(x,a,b)`
             - `lt(x,y)`, `lte(x,y)`, `gt(x,y)`, `gte(x,y)`
             - `if(cond,then,else)`
           - Always wrap expression values in single quotes: `z='...'`, `x='...'`, `y='...'`, `enable='...'`.
           
           - FOR `zoompan`: 
             - Prefer `on` (output frame index) to avoid time-variable quirks.
             - Convert seconds to frames using FPS={fps}: `frame = seconds * {fps}`.
             - Use `between(on, startFrame, endFrame)` for segmenting and pacing.
             - Example:
              `zoompan=z='1.1*between(on,0,75)+1.3*between(on,76,150)+1.15*between(on,151,300)+1.2*gte(on,301)'`
             - ALWAYS set zoompan output size to EXACT `{width}x{height}` using `s={width}x{height}`.
             - ALWAYS set `fps={fps}` and `d=1`.
             - DO NOT use `scale`, `crop`, `pad` unless you keep EXACT `{width}x{height}` (no aspect ratio changes).
             
           - FOR `eq`, `hue`, `curves`, `unsharp` (Visual Effects): 
             - **DO NOT** use dynamic expressions for parameter values (e.g. `contrast='1+0.5*t'`).
             - **USE TIMELINE EDITING** via the `enable` option.
             - Create MULTIPLE filter instances for different time ranges.
             - **SYNTAX FOR ENABLE:**
              - **USE** `between(t,start,end)` for clarity and robustness.
              - **USE** single quotes around the enable expression.
              - **Example:** `eq=contrast=1.2:enable='between(t,0,3)'`
              - **Example:** `hue=s=0:enable='between(t,10,12)'`
             - This is much safer and robust than boolean multiplication.
        
        Constraints:
        - Output JSON with a single key: "filter_string".
        - The value must be the RAW filter string ready to be passed to `-vf`.
        - OUTPUT MUST KEEP EXACT RESOLUTION AND ASPECT RATIO: {width}x{height}.
        - Do NOT output 1280x720 or 1080x1080 unless the input is exactly that.
        - IMPORTANT: Do NOT include the `-vf` flag itself, just the filter content.
        - IMPORTANT: Ensure syntax is correct for FFmpeg. 
        
        Output JSON:
        {{
            "filter_string": "..."
        }}
        """

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

if __name__ == "__main__":
    pass