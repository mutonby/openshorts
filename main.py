import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from openai import OpenAI
from dotenv import load_dotenv
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor for TikTok/IG Reels/YouTube Shorts. Identify the 3-15 most viral-worthy moments from the transcript.

VIDEO_DURATION_SECONDS: {video_duration}
DETECTED_LANGUAGE: {language}

SCENE_BOUNDARIES (natural visual cut points in seconds — prefer aligning clip edges here):
{scene_boundaries}

--- VIRAL CRITERIA ---
Evaluate each candidate moment against these viral pattern types:
1. PATTERN_INTERRUPT: Unexpected twist, surprise reveal, "wait what?!" moment
2. EMOTIONAL_PEAK: High emotion — anger, excitement, laughter, tears, passion
3. VALUE_DROP: Actionable tip, hack, insight, or framework the viewer can use immediately
4. CONTROVERSY: Hot take, debate trigger, strong opinion that sparks comments
5. CLIFFHANGER: Unfinished story, looming reveal, "you won't believe what happened next"
6. CURIOSITY_GAP: Opens a question the viewer MUST know the answer to
7. STORY_BEAT: Key narrative moment — setup, conflict, resolution
8. RELATABLE_MOMENT: "That's so me" — universal experience, high shareability

--- CONTENT TYPE ---
First, classify the dominant content type: TUTORIAL / STORYTELLING / INTERVIEW / REACTION / VLOG / REVIEW / DEBATE / OTHER. This guides clip length and selection logic.

--- CLIP DIVERSITY ---
Select clips covering at least 3 DIFFERENT viral pattern types. Avoid homogeneous picks.

--- DURATION GUIDANCE ---
Fast-paced/reactions/quick tips: 8-25 s
Storytelling/narrative: 40-90 s
Tutorials/interviews: 20-60 s
Default when uncertain: 15-60 s

--- TIMING RULES ---
- Return timestamps in ABSOLUTE SECONDS from video start (usable in: ffmpeg -ss <start> -to <end> -i <input> ...)
- Only NUMBERS with decimal point, up to 3 decimals (e.g., 12.340)
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS
- Prefer cutting 0.2-0.4 s BEFORE the hook and 0.2-0.4 s AFTER the payoff
- Align cuts with scene boundaries when available
- Never cut in the middle of a word or phrase
- Avoid cutting near LOW-PROBABILITY words (p < 0.3 in WORDS_JSON) — they indicate transcription uncertainty
- No generic intros/outros or purely sponsorship segments unless they contain a hook

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e, p}} where s/e are seconds, p is 0-1 transcription confidence):
{words_json}

OUTPUT — RETURN ONLY VALID JSON (no markdown fences, no extra text). Order clips by predicted performance (best first). Include a CTA in descriptions like "Follow me and comment X and I'll send you the workflow":
{{
  "content_type": "<TUTORIAL|STORYTELLING|INTERVIEW|REACTION|VLOG|REVIEW|DEBATE|OTHER>",
  "shorts": [
    {{
      "start": <number>,
      "end": <number>,
      "confidence": <0.0 to 1.0, your confidence this clip will perform well>,
      "reasoning": "<2-3 sentences: which viral pattern this hits, why it works, why the timestamps are optimal>",
      "viral_pattern_type": "<one of the 8 types above>",
      "video_description_for_tiktok": "<description for TikTok>",
      "video_description_for_instagram": "<description for Instagram>",
      "video_title_for_youtube_short": "<title for YouTube Short, 100 chars max>",
      "viral_hook_text": "<SHORT punchy overlay (max 10 words). MUST BE IN THE SAME LANGUAGE AS THE VIDEO>"
    }}
  ]
}}
"""

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO('yolov8n.pt')

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        
        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        
        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / ASPECT_RATIO)
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        y1 = 0
        y2 = self.video_height
        
        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0]) # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height))
    
    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
        
    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h))
    
    # 3. Overlay
    y_offset = (output_height - fg_h) // 2
    
    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame

def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []
    
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)
        
    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]
        
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)
            
        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)
        
        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')
            
    cap.release()
    return strategies

def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:100]


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, creating cookies file inside container...")
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
                 with open(cookies_path, 'r') as f:
                     content = f.read(100)
                     print(f"   Debug: First 100 chars of cookie file: {content}")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")
    
    # Common yt-dlp options to work around YouTube bot detection.
    # extractor_args tries multiple player clients in order; tv_embed / android
    # avoid the OAuth/PO-token checks that block server IPs.
    _COMMON_YDL_OPTS = {
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'nocheckcertificate': True,
        'cachedir': False,
        'extractor_args': {
            'youtube': {
                'player_client': ['tv_embed', 'android', 'mweb', 'web'],
                'player_skip': ['webpage', 'configs'],
            }
        },
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        },
    }

    with yt_dlp.YoutubeDL(_COMMON_YDL_OPTS) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'youtube_video')
            sanitized_title = sanitize_filename(video_title)
        except Exception as e:
            # Force print to stderr/stdout immediately so it's captured before crash
            import sys
            import traceback
            
            # Print minimal error first to ensure something gets out
            print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)
            
            error_msg = f"""
            
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌
            
REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(e)}
            """
            # Print to both streams to ensure capture
            print(error_msg, file=sys.stdout)
            print(error_msg, file=sys.stderr)
            
            # Force flush
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Wait a split second to allow buffer to drain before raising
            time.sleep(0.5)
            
            raise e
    
    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    expected_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print(f"🗑️  Removed existing file to re-download with H.264 codec")
    
    ydl_opts = {
        **_COMMON_YDL_OPTS,
        'format': 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'overwrites': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break
    
    step_end_time = time.time()
    print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
    
    return downloaded_file, sanitized_title

def process_video_to_vertical(input_video, final_output_video):
    """
    Core logic to convert horizontal video to vertical using scene detection and Active Speaker Tracking (MediaPipe).
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)
    
    # Fixed HD 9:16 output resolution
    OUTPUT_HEIGHT = 1920
    OUTPUT_WIDTH = 1080

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop
                
                # Detect every 2nd frame for performance
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                
                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 6: Merging...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-c:v', 'copy', final_output_video
        ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    
    return True

def transcribe_video(video_path, method="faster-whisper"):
    """
    Transcribe video audio to text.
    
    Args:
        video_path: Path to the video file
        method: Transcription method - "faster-whisper" or "groq"
    """
    if method == "groq":
        return transcribe_with_groq(video_path)
    else:
        return transcribe_with_faster_whisper(video_path)

def split_audio_into_chunks(audio_path, chunk_duration_sec=600):
    """
    Splits an audio file into smaller chunks using FFmpeg.
    Returns a list of paths to the created chunks.
    """
    print(f"   ✂️  Splitting audio into {chunk_duration_sec}s chunks...")
    chunks = []
    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Use ffmpeg segment muxer for efficient splitting without re-encoding
    # we use -f segment to split into equal durations
    cmd = [
        'ffmpeg', '-y', '-i', audio_path,
        '-f', 'segment',
        '-segment_time', str(chunk_duration_sec),
        '-c', 'copy',
        os.path.join(base_dir, f"{base_name}_chunk_%03d.wav")
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    # Find all created chunks
    for f in sorted(os.listdir(base_dir)):
        if f.startswith(f"{base_name}_chunk_") and f.endswith(".wav"):
            chunks.append(os.path.join(base_dir, f))
            
    return chunks

def transcribe_with_groq(video_path):
    """Transcribe video using Groq API (faster, cloud-based)."""
    print("🎙️  Transcribing with Groq Whisper...")
    from groq import Groq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables.")
        return None
    
    client = Groq(api_key=api_key)
    
    # Extract audio from video first
    temp_audio = "/tmp/temp_audio.wav"
    extract_audio_cmd = [
        'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1', temp_audio
    ]
    subprocess.run(extract_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    try:
        # Handle file size limit (25MB for free tier)
        file_size_mb = os.path.getsize(temp_audio) / (1024 * 1024)
        
        if file_size_mb > 20: # Use 20MB as a safe threshold
            print(f"   ⚠️  Audio file size ({file_size_mb:.2f}MB) exceeds Groq free tier limit. Chunking...")
            chunks = split_audio_into_chunks(temp_audio)
            
            all_segments = []
            all_words = []
            full_text_parts = []
            cumulative_offset = 0.0
            detected_language = "unknown"
            
            for i, chunk_path in enumerate(chunks):
                print(f"   Processing chunk {i+1}/{len(chunks)} ({os.path.basename(chunk_path)})...")
                with open(chunk_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(chunk_path), f.read()),
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"]
                    )
                
                if isinstance(transcription, dict):
                    text = transcription.get("text", "")
                    language = transcription.get("language", "unknown")
                    segments_data = transcription.get("segments", [])
                    words_data = transcription.get("words", [])
                else:
                    text = transcription.text
                    language = transcription.language
                    segments_data = transcription.segments or []
                    words_data = getattr(transcription, "words", []) or []
                
                if i == 0:
                    detected_language = language
                
                full_text_parts.append(text)
                
                # Offset timestamps for segments
                for seg in segments_data:
                    if isinstance(seg, dict):
                        s_start, s_end, s_text = seg.get("start", 0), seg.get("end", 0), seg.get("text", "")
                    else:
                        s_start, s_end, s_text = seg.start, seg.end, seg.text
                    
                    all_segments.append({
                        'text': s_text,
                        'start': s_start + cumulative_offset,
                        'end': s_end + cumulative_offset,
                        'words': []
                    })
                
                # Offset timestamps for words
                for word in words_data:
                    if isinstance(word, dict):
                        w_text, w_start, w_end, w_prob = word.get("word", ""), word.get("start", 0), word.get("end", 0), word.get("probability", 1.0)
                    else:
                        w_text, w_start, w_end, w_prob = word.word, word.start, word.end, getattr(word, "probability", 1.0)
                    
                    all_words.append({
                        'word': w_text,
                        'start': w_start + cumulative_offset,
                        'end': w_end + cumulative_offset,
                        'probability': w_prob
                    })
                
                # Accurate chunk duration for offset
                if segments_data:
                    last_seg = segments_data[-1]
                    chunk_duration = last_seg.get("end", 600) if isinstance(last_seg, dict) else last_seg.end
                else:
                    chunk_duration = 600
                
                cumulative_offset += chunk_duration
                os.remove(chunk_path)
            
            # Map offset words to offset segments
            for seg in all_segments:
                for word in all_words:
                    if seg['start'] <= word['start'] < seg['end']:
                        seg['words'].append(word)
            
            full_text = " ".join(full_text_parts)
            transcript_segments = all_segments
            language = detected_language
            
        else:
            # Single-request logic
            with open(temp_audio, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(temp_audio), file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"]
                )
            
            if isinstance(transcription, dict):
                text = transcription.get("text", "")
                language = transcription.get("language", "unknown")
                segments_data = transcription.get("segments", [])
                words_data = transcription.get("words", [])
            else:
                text = transcription.text
                language = transcription.language
                segments_data = transcription.segments or []
                words_data = getattr(transcription, "words", []) or []
            
            full_text = text
            transcript_segments = []
            for segment in segments_data:
                if isinstance(segment, dict):
                    s_text, s_start, s_end = segment.get("text", ""), segment.get("start", 0), segment.get("end", 0)
                else:
                    s_text, s_start, s_end = segment.text, segment.start, segment.end
                
                seg_dict = {'text': s_text, 'start': s_start, 'end': s_end, 'words': []}
                if words_data:
                    for word in words_data:
                        if isinstance(word, dict):
                            w_text, w_start, w_end, w_prob = word.get("word", ""), word.get("start", 0), word.get("end", 0), word.get("probability", 1.0)
                        else:
                            w_text, w_start, w_end, w_prob = word.word, word.start, word.end, getattr(word, "probability", 1.0)
                        if s_start <= w_start < s_end:
                            seg_dict['words'].append({'word': w_text, 'start': w_start, 'end': w_end, 'probability': w_prob})
                transcript_segments.append(seg_dict)
            
        print(f"   Detected language: {language}")
        for seg in transcript_segments:
            print(f"   [{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
        
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        return {
            'text': full_text.strip(),
            'segments': transcript_segments,
            'language': language
        }
    
    except Exception as e:
        print(f"   ❌ Groq transcription failed: {e}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return None

def transcribe_with_faster_whisper(video_path):
    """Transcribe video using Faster-Whisper (local, CPU-optimized)."""
    print("🎙️  Transcribing video with Faster-Whisper...")
    from faster_whisper import WhisperModel
    import torch
    
    # Auto-detect CUDA for faster transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"   Using device: {device} ({compute_type})")
    
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    
    segments, info = model.transcribe(video_path, word_timestamps=True)
    
    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""
    
    for segment in segments:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }
        
        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
        
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "
        
    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }

def post_process_clips(result_json, min_confidence=0.6, max_overlap=0.7, max_clips=10):
    """Filter, deduplicate, and rank clips by quality, duration, and diversity."""
    shorts = result_json.get('shorts', [])
    if not shorts:
        return result_json

    # 0. Filter by duration — enforce absolute floor and ceiling
    MIN_DURATION = 8
    MAX_DURATION = 90
    duration_filtered = []
    for s in shorts:
        dur = s['end'] - s['start']
        if dur < MIN_DURATION:
            print(f"   ⚠️ Dropping clip at {s['start']:.1f}s: {dur:.1f}s too short (min {MIN_DURATION}s)")
            continue
        if dur > MAX_DURATION:
            print(f"   ⚠️ Dropping clip at {s['start']:.1f}s: {dur:.1f}s too long (max {MAX_DURATION}s)")
            continue
        duration_filtered.append(s)
    if not duration_filtered:
        print(f"   ⚠️ All clips outside duration bounds, keeping originals")
        duration_filtered = shorts

    # 1. Filter by confidence (keep low-confidence as fallback)
    has_confidence = any('confidence' in s for s in duration_filtered)
    if has_confidence:
        filtered = [s for s in duration_filtered if s.get('confidence', 0) >= min_confidence]
        if not filtered:
            print(f"   ⚠️ All clips below confidence {min_confidence}, keeping top 3")
            filtered = sorted(duration_filtered, key=lambda s: s.get('confidence', 0), reverse=True)[:3]
    else:
        filtered = duration_filtered

    # 2. Deduplicate overlapping clips (>max_overlap overlap)
    deduped = []
    filtered.sort(key=lambda s: s['start'])
    for clip in filtered:
        is_duplicate = False
        for i, kept in enumerate(deduped):
            overlap_start = max(clip['start'], kept['start'])
            overlap_end = min(clip['end'], kept['end'])
            overlap_dur = max(0, overlap_end - overlap_start)
            clip_dur = clip['end'] - clip['start']
            kept_dur = kept['end'] - kept['start']
            min_dur = min(clip_dur, kept_dur)
            if min_dur > 0 and overlap_dur / min_dur > max_overlap:
                is_duplicate = True
                if clip.get('confidence', 0) > kept.get('confidence', 0):
                    deduped[i] = clip
                break
        if not is_duplicate:
            deduped.append(clip)

    # 3. Enforce diversity — keep first occurrence of each pattern type, then fill with high-confidence remainder
    seen_types = set()
    diversified = []
    for clip in deduped:
        ptype = clip.get('viral_pattern_type', '')
        if ptype and ptype not in seen_types:
            seen_types.add(ptype)
            diversified.append(clip)
    for clip in deduped:
        if clip not in diversified:
            diversified.append(clip)

    # 4. Sort by confidence desc, limit to max_clips
    diversified.sort(key=lambda s: s.get('confidence', 0), reverse=True)
    result_json['shorts'] = diversified[:max_clips]

    filtered_out = len(shorts) - len(result_json['shorts'])
    if filtered_out > 0:
        print(f"   🔎 Post-process filtered {filtered_out} clip(s) (duration/confidence/dedup/diversity)")

    return result_json


def _build_window_prompt(words, transcript_segments, video_duration, language, scene_boundaries, window_num, total_windows):
    """Build a reduced prompt for a window of words."""
    if not words:
        return None, None

    window_start = words[0]['s']
    window_end = words[-1]['e']

    # Filter segments that overlap this window
    window_segments = [seg for seg in transcript_segments
                       if seg['end'] >= window_start and seg['start'] <= window_end]
    if window_segments:
        window_text = " ".join(seg['text'] for seg in window_segments)
    else:
        window_text = "(no transcript in this window)"

    # Filter scene boundaries in this window
    if scene_boundaries:
        window_scene_parts = []
        for s_start, s_end in scene_boundaries:
            if s_end >= window_start and s_start <= window_end:
                window_scene_parts.append(f"[{s_start:.1f}s - {s_end:.1f}s]")
        window_scene_text = ", ".join(window_scene_parts) if window_scene_parts else "No scene data in this window"
    else:
        window_scene_text = "No scene data available"

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        language=language,
        scene_boundaries=f"WINDOW {window_num+1}/{total_windows} [{window_start:.1f}s - {window_end:.1f}s]: {window_scene_text}",
        transcript_text=json.dumps(window_text),
        words_json=json.dumps(words)
    )

    return prompt, window_text


MAX_PROMPT_CHARS = 380_000  # ~100K tokens, safe budget per chunk


def get_viral_clips(transcript_result, video_duration, scene_boundaries=None):
    print("🤖  Analyzing with AI...")
    
    if not transcript_result:
        print("❌ Error: No transcript available. Skipping viral clip analysis.")
        return None
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    print(f"🤖  Initializing model: {model_name}")

    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({
                'w': word['word'],
                's': word['start'],
                'e': word['end'],
                'p': round(word.get('probability', 1.0), 3)
            })

    language = transcript_result.get('language', 'unknown')

    # Format scene boundaries for full-video prompt
    if scene_boundaries:
        scene_parts = []
        for s_start, s_end in scene_boundaries:
            scene_parts.append(f"[{s_start:.1f}s - {s_end:.1f}s]")
        scene_text = ", ".join(scene_parts)
    else:
        scene_text = "No scene data available — use transcript context to find natural cut points"

    from utils import extract_json

    def _call_llm(prompt_content, extra_system_prompt=None):
        messages = []
        if extra_system_prompt:
            messages.append({"role": "system", "content": extra_system_prompt})
        messages.append({"role": "user", "content": prompt_content})
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"}
        )

    def _analyze_single(prompt_content):
        response = _call_llm(prompt_content)
        text = response.choices[0].message.content
        result = extract_json(text)
        if result is None:
            print("   ⚠️ First attempt returned non-JSON output, retrying...")
            try:
                response = _call_llm(prompt_content, "CRITICAL: Return ONLY a single valid JSON object. Start your response with '{' and end with '}'. No markdown, no code fences, no explanatory text.")
                text = response.choices[0].message.content
                result = extract_json(text)
            except Exception as retry_err:
                print(f"   ❌ Retry also failed: {retry_err}")
        return result, response

    try:
        # --- Estimate prompt size and decide single-call vs chunked ---
        transcript_json = json.dumps(transcript_result['text'])
        words_json = json.dumps(words)
        full_prompt = GEMINI_PROMPT_TEMPLATE.format(
            video_duration=video_duration,
            language=language,
            scene_boundaries=scene_text,
            transcript_text=transcript_json,
            words_json=words_json
        )

        if len(full_prompt) <= MAX_PROMPT_CHARS:
            # Single call
            result_json, response = _analyze_single(full_prompt)

            if result_json is None:
                raise ValueError("Failed to parse JSON from model response after retry")

            # Token usage
            try:
                usage = response.usage
                if usage:
                    print(f"💰 Token Usage ({model_name}):")
                    print(f"   - Input Tokens: {usage.prompt_tokens}")
                    print(f"   - Output Tokens: {usage.completion_tokens}")
            except Exception:
                pass

            result_json = post_process_clips(result_json)
            return result_json

        # --- Chunked: sliding windows ---
        print(f"   📦 Prompt too large ({len(full_prompt)} chars > {MAX_PROMPT_CHARS}), chunking video into windows...")

        # Calculate window size: fit words + transcript within budget
        template_overhead = GEMINI_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            scene_boundaries="", transcript_text="", words_json=""
        )
        # Account for both words_json AND transcript_json in per-word estimate
        total_data_chars = len(transcript_json) + len(words_json)
        chars_per_word = total_data_chars / max(len(words), 1)
        available_per_window = MAX_PROMPT_CHARS - len(template_overhead) - 500  # window label
        words_per_window = max(int(available_per_window / chars_per_word), 200)
        overlap_words = words_per_window // 5  # 20% overlap

        total_windows = max(1, (len(words) + words_per_window - overlap_words - 1) // (words_per_window - overlap_words))
        print(f"   🪟 Splitting {len(words)} words into ~{total_windows} windows ({words_per_window} words each, {overlap_words} overlap)")

        all_shorts = []
        content_types = []

        for win_idx in range(0, len(words), words_per_window - overlap_words):
            window_words = words[win_idx:win_idx + words_per_window]
            if len(window_words) < 200:
                continue

            window_prompt, _ = _build_window_prompt(
                window_words, transcript_result['segments'],
                video_duration, language, scene_boundaries,
                win_idx // max(words_per_window - overlap_words, 1), total_windows
            )
            if window_prompt is None:
                continue

            # Safety: if window prompt still too large, trim words
            while len(window_prompt) > MAX_PROMPT_CHARS and len(window_words) > 200:
                trim_count = int(len(window_words) * 0.15)
                window_words = window_words[:-trim_count]
                window_prompt, _ = _build_window_prompt(
                    window_words, transcript_result['segments'],
                    video_duration, language, scene_boundaries,
                    win_idx // max(words_per_window - overlap_words, 1), total_windows
                )

            if len(window_prompt) > MAX_PROMPT_CHARS:
                print(f"   ⚠️ Window still too large ({len(window_prompt)} chars), skipping")
                continue

            print(f"   🔍 Analyzing window {win_idx // max(words_per_window - overlap_words, 1) + 1}/{total_windows} ({len(window_words)} words)...")
            result_json, _ = _analyze_single(window_prompt)

            if result_json:
                if 'content_type' in result_json:
                    content_types.append(result_json['content_type'])
                all_shorts.extend(result_json.get('shorts', []))

            # Add 1-minute delay between chunk analyses to avoid API rate limiting
            next_win_idx = win_idx + words_per_window - overlap_words
            if next_win_idx < len(words):
                print(f"   ⏳ Waiting 60 seconds before analyzing next window...")
                time.sleep(60)

        if not all_shorts:
            raise ValueError("No clips found in any window")

        # Merge: dedup highly overlapping clips from adjacent windows, keep best
        merged = []
        all_shorts.sort(key=lambda s: s['start'])
        for clip in all_shorts:
            is_dup = False
            for i, kept in enumerate(merged):
                overlap_start = max(clip['start'], kept['start'])
                overlap_end = min(clip['end'], kept['end'])
                overlap_dur = max(0, overlap_end - overlap_start)
                clip_dur = clip['end'] - clip['start']
                kept_dur = kept['end'] - kept['start']
                min_dur = min(clip_dur, kept_dur)
                if min_dur > 0 and overlap_dur / min_dur > 0.9:  # 90% overlap = same clip
                    is_dup = True
                    if clip.get('confidence', 0) > kept.get('confidence', 0):
                        merged[i] = clip
                    break
            if not is_dup:
                merged.append(clip)

        result_json = {"shorts": merged}
        if content_types:
            result_json['content_type'] = max(set(content_types), key=content_types.count)

        result_json = post_process_clips(result_json)
        print(f"   ✅ Chunked analysis complete: {len(all_shorts)} raw → {len(result_json['shorts'])} after merge+filter")
        return result_json

    except Exception as e:
        print(f"❌ AI Analysis Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--transcription-method', type=str, choices=['faster-whisper', 'groq'], default=None, help="Transcription method: faster-whisper (local CPU) or groq (cloud API). Defaults to GROQ if GROQ_API_KEY env var is set, otherwise faster-whisper")
    
    args = parser.parse_args()

    # Auto-detect transcription method if not specified
    transcription_method = args.transcription_method
    if not transcription_method:
        if os.getenv("GROQ_API_KEY"):
            transcription_method = "groq"
            print("🚀 GROQ_API_KEY detected - using Groq for transcription")
        else:
            transcription_method = "faster-whisper"
            print("💻 No GROQ_API_KEY - using Faster-Whisper (local CPU)")
    else:
        print(f"📌 Transcription method specified via CLI: {transcription_method}")

    print(f"🔍 GROQ_API_KEY in env: {'Yes' if os.getenv('GROQ_API_KEY') else 'No'}")
    print(f"🎯 Selected transcription method: {transcription_method}")

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
        transcript = transcribe_video(input_video, method=transcription_method)
        
        # Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 3.5. Detect scenes for better boundary alignment
        scene_boundaries = None
        try:
            scenes, scene_fps = detect_scenes(input_video)
            scene_boundaries = []
            for s_start, s_end in scenes:
                scene_boundaries.append((s_start.get_seconds(), s_end.get_seconds()))
            print(f"   🎞️  Detected {len(scene_boundaries)} scenes for boundary alignment")
        except Exception as e:
            print(f"   ⚠️ Scene detection skipped: {e}")

        # 4. AI Analysis
        clips_data = get_viral_clips(transcript, duration, scene_boundaries)
        
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
                    '-c:v', 'libx264', '-crf', '15', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '192k',
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
