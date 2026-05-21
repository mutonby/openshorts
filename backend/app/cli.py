"""Compat shim + CLI entrypoint.

The implementation was split across the openshorts package:
- openshorts/video/tracking.py        (SmoothedCameraman, SpeakerTracker)
- openshorts/video/scene_analysis.py  (detect_scenes, analyze_scenes_strategy, get_video_resolution)
- openshorts/video/reframing.py       (create_general_frame)
- openshorts/video/pipeline.py        (process_video_to_vertical)
- openshorts/ml/detection.py          (detect_face_candidates, detect_person_yolo)
- openshorts/ml/transcription.py      (transcribe_video)
- openshorts/ml/viral_extraction.py   (GEMINI_PROMPT_TEMPLATE, get_viral_clips)
- openshorts/ingest/youtube.py        (download_youtube_video, sanitize_filename)

New code should import from those modules directly. This shim re-exports the
public surface so existing `from main import ...` calls keep working, and
preserves the CLI entrypoint (`python main.py -i ... -o ...`).
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

from dotenv import load_dotenv
load_dotenv()

# Re-exports (used by app.py, thumbnail.py, and existing tests)
from app.video.tracking import (  # noqa: F401
    ASPECT_RATIO,
    SmoothedCameraman,
    SpeakerTracker,
)
from app.video.scene_analysis import (  # noqa: F401
    detect_scenes,
    get_video_resolution,
    analyze_scenes_strategy,
)
from app.video.reframing import create_general_frame  # noqa: F401
from app.video.pipeline import process_video_to_vertical  # noqa: F401
from app.ml.detection import detect_face_candidates, detect_person_yolo  # noqa: F401
from app.ml.transcription import transcribe_video  # noqa: F401
from app.ml.viral_extraction import GEMINI_PROMPT_TEMPLATE, get_viral_clips  # noqa: F401
from app.ingest.youtube import download_youtube_video, sanitize_filename  # noqa: F401


def _cli():
    import argparse
    import json
    import os
    import subprocess
    import time

    import cv2

    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")

    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")

    args = parser.parse_args()

    script_start_time = time.time()

    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."

        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]

        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        process_video_to_vertical(input_video, output_file)
    else:
        # 3. Transcribe
        transcript = transcribe_video(input_video)

        # Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 4. Gemini Analysis
        clips_data = get_viral_clips(transcript, duration)

        if not clips_data or 'shorts' not in clips_data:
            print("❌ Failed to identify clips. Converting whole video as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
            process_video_to_vertical(input_video, output_file)
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")

            # Save metadata
            clips_data['transcript'] = transcript # Save full transcript for subtitles
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            # 5. Process each clip
            for i, clip in enumerate(clips_data['shorts']):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

                # Cut clip
                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)

                # ffmpeg cut
                # Using re-encoding for precision as requested by strict seconds
                cut_command = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-to', str(end),
                    '-i', input_video,
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                    '-c:a', 'aac',
                    clip_temp_path
                ]
                subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                # Process vertical
                success = process_video_to_vertical(clip_temp_path, clip_final_path)

                if success:
                    print(f"   ✅ Clip {i+1} ready: {clip_final_path}")

                # Clean up temp cut
                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")


if __name__ == '__main__':
    _cli()
