"""Gemini prompt templates for AI video-effect generation.

Kept as functions returning rendered strings with width/height/fps/duration/transcript
substituted in. The matching call sites in ai_filters.py each pass these args through.
"""
import json


def build_ffmpeg_filter_prompt(duration, fps, width, height, transcript):
    """Prompt for raw FFmpeg filter_complex string (used by /api/edit)."""
    transcript_text = json.dumps(transcript) if transcript else "Not available."
    return f"""
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


def build_effects_config_prompt(duration, fps, width, height, transcript):
    """Prompt for structured EffectsConfig JSON (used by Remotion renderer)."""
    transcript_text = json.dumps(transcript) if transcript else "Not available."
    return f"""
        You are an expert video editor analyzing a video and its transcript to generate dynamic visual effects for a Remotion-based renderer.

        Video Duration: {duration} seconds.
        Video FPS: {fps}
        Video Resolution: {width}x{height}

        TRANSCRIPT (Context of what is being said):
        {transcript_text}

        Your task is to produce a structured JSON describing time-based effect segments that cover the FULL video duration.

        Each segment has these fields:
        - "startSec" (number): Start time in seconds.
        - "endSec" (number): End time in seconds.
        - "zoom" (number): Zoom level. 1.0 = no zoom, max 1.5. Use subtle values like 1.05-1.2 for most cases.
        - "zoomCenterX" (number): Horizontal focus point for zoom, 0.0 (left) to 1.0 (right). 0.5 = center.
        - "zoomCenterY" (number): Vertical focus point for zoom, 0.0 (top) to 1.0 (bottom). 0.5 = center.
        - "brightness" (number): Brightness multiplier. 1.0 = normal. Range 0.8-1.2.
        - "contrast" (number): Contrast multiplier. 1.0 = normal. Range 0.8-1.3.
        - "saturate" (number): Saturation multiplier. 1.0 = normal. Range 0.8-1.3.

        Instructions:
        1. ANALYZE the video content and transcript to understand mood, pacing, and key moments.
        2. Apply CONTEXTUAL effects aligned with speech and action:
           - Use slow, subtle zooms toward the speaker's face during speaking moments.
           - Emphasize key moments, punchlines, or dramatic beats with slightly stronger zoom or contrast.
           - Keep transitions smooth — avoid jarring jumps between segments.
           - If nothing significant is happening, keep values at defaults (zoom 1.0, all multipliers 1.0).
        3. Segments MUST cover the entire video duration from 0 to {duration} seconds with no gaps.
        4. Prefer fewer, longer segments with gradual changes over many rapid short segments.
        5. Output ONLY valid JSON, no explanations.

        Output format:
        {{
            "segments": [
                {{
                    "startSec": 0,
                    "endSec": 3.5,
                    "zoom": 1.0,
                    "zoomCenterX": 0.5,
                    "zoomCenterY": 0.5,
                    "brightness": 1.0,
                    "contrast": 1.0,
                    "saturate": 1.0
                }}
            ]
        }}
        """
