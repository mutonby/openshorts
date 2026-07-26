import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import sys
import numpy as np

# Strip PYTHONPATH early to avoid protobuf 6.x leaking into .venv subprocesses
os.environ.pop("PYTHONPATH", None)
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types

import gemini_worker
from clip_selection import build_transcript_windows, snap_clip_to_words
from ffmpeg_utils import video_encode_args, QUALITY, QUALITY_FAST, METADATA_SCRUB
from dotenv import load_dotenv
import json
import brutal_truth  # Brutal Truth engine: score filter, cold-open, XML export

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words). MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. Examples: 'POV: You realized...', 'Did you know?', 'Stop doing this!'>"
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
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio

        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25
        self.jump_confirm_frames = 3
        self._pending_jump_center = None
        self._pending_jump_count = 0

    def update_target(self, face_box):
        """Update target center; require confirmation for large detector jumps."""
        if not face_box:
            return
        x, y, w, h = face_box
        proposed = x + w / 2
        jump = abs(proposed - self.target_center_x)
        if jump > self.crop_width * 0.6:
            if self._pending_jump_center == proposed:
                self._pending_jump_count += 1
            else:
                self._pending_jump_center = proposed
                self._pending_jump_count = 1
            if self._pending_jump_count < self.jump_confirm_frames:
                return
            self._pending_jump_center = None
            self._pending_jump_count = 0
        else:
            self._pending_jump_center = None
            self._pending_jump_count = 0
        self.target_center_x = proposed
    
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
                # Active speaker is temporarily missing: hold the camera rather
                # than switching to a different face during the cooldown.
                return None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

# Detectors never need full-resolution frames: MediaPipe returns relative
# coords and YOLO boxes are scaled back up. Running them on a ≤640px copy cuts
# per-frame preprocessing cost hard, which is what dominates CPU-only renders.
DETECT_MAX_WIDTH = 640
# The global MediaPipe graph and YOLO model are NOT thread-safe; clips render
# in parallel, so every inference goes through this lock. Contention is small
# (a few ms per call) — the ffmpeg renders are where the parallel time goes.
DETECT_LOCK = threading.Lock()
# Detect every Nth frame; SmoothedCameraman interpolates between updates.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
# YOLO fallback (no face found) is far heavier than MediaPipe — extra throttle.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2


def _detection_frame(frame):
    """Downscaled copy for detectors. Returns (small_frame, scale) with
    scale mapping small-frame pixel coords back to the original frame."""
    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / DETECT_MAX_WIDTH
    small = cv2.resize(frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Boxes are in ORIGINAL frame coordinates (detection runs downscaled;
    MediaPipe's relative coords make the mapping exact).
    """
    height, width, _ = frame.shape
    small, _scale = _detection_frame(frame)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with DETECT_LOCK:
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
    Returns [x, y, w, h] of the person's 'upper body' approximation, in
    ORIGINAL frame coordinates (inference runs on a downscaled copy).
    """
    small, scale = _detection_frame(frame)
    # Use the globally loaded model
    with DETECT_LOCK:
        results = model(small, verbose=False, classes=[0]) # class 0 is person

    if not results:
        return None

    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
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
    Full-frame 9:16 crop for GENERAL scenes — center-crop to target aspect,
    no blurred background bars.
    """
    orig_h, orig_w = frame.shape[:2]
    # Center-crop to target aspect ratio
    target_ratio = output_width / output_height
    src_ratio = orig_w / orig_h
    if src_ratio > target_ratio:
        # Source is wider — crop width
        crop_w = int(orig_h * target_ratio)
        x_start = (orig_w - crop_w) // 2
        cropped = frame[:, x_start:x_start+crop_w]
    else:
        # Source is taller — crop height
        crop_h = int(orig_w / target_ratio)
        y_start = (orig_h - crop_h) // 2
        cropped = frame[y_start:y_start+crop_h, :]

    final_frame = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    return final_frame

def create_split_frame(frame, output_width, output_height):
    """Top/bottom split fallback for multi-speaker scenes."""
    half_height = max(2, output_height // 2)
    top = cv2.resize(frame[:frame.shape[0] // 2],
                     (output_width, half_height), interpolation=cv2.INTER_LINEAR)
    bottom = cv2.resize(frame[frame.shape[0] // 2:],
                        (output_width, half_height), interpolation=cv2.INTER_LINEAR)
    return np.vstack((top, bottom))


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        s_f, e_f = start.get_frames(), end.get_frames()
        # Sample 5 frames spread across the scene, clamped inside it (the old
        # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
        margin = min(2, max(0, (e_f - s_f - 1) // 2))
        frames_to_check = sorted(set(
            int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)
        ))

        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue

            # Near-black frames (fades, cut-to-black) carry no faces and used
            # to drag single-person scenes into GENERAL. Skip them.
            if frame.mean() < 16:
                continue

            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))

        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> optional top/bottom split; center crop otherwise.
        if avg_faces > 1.2:
            split_enabled = os.environ.get("MULTI_SPEAKER_SPLIT", "1") == "1"
            strategies.append('SPLIT' if split_enabled else 'GENERAL')
        elif avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if (dur < max_flip_frames
                and strategies[i - 1] == strategies[i + 1] != strategies[i]):
            strategies[i] = strategies[i - 1]

    return strategies

def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


MAX_TITLE_BYTES = 180


def truncate_bytes(text, max_bytes):
    """Trim UTF-8 text without splitting a multibyte character."""
    text = str(text or "")
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max(0, int(max_bytes))].decode("utf-8", "ignore")


def sanitize_filename(filename):
    """Remove unsafe characters and enforce a filesystem byte budget."""
    filename = re.sub(r'[<>:"/\\|?*#]', '', str(filename or ''))
    filename = filename.replace(' ', '_')
    return truncate_bytes(filename, MAX_TITLE_BYTES)


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    # SSRF guard: block non-http(s) schemes and private/loopback/metadata hosts
    # before handing the URL to yt-dlp.
    from security_utils import assert_public_url
    assert_public_url(url)

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
                 # Never print file CONTENT here: with a headerless cookies
                 # blob this would leak live YouTube session cookies to logs.
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")
    
    # Optional HTTP proxy. Set PROXY_URL to route downloads through it; unset
    # (self-host) goes direct as before.
    _proxy = os.environ.get("PROXY_URL", "").strip() or None
    if _proxy:
        print("🌐 Using proxy for download.")

    # Two download strategies, tried in order so a break in the HD path degrades
    # gracefully instead of failing the whole job: an HD attempt first, then a
    # conservative fallback (also the only strategy for self-host).
    _bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    _bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    if _bgutil_http:
        hd_args = {'youtubepot-bgutilhttp': {'base_url': [_bgutil_http]}}
    elif _bgutil_script:
        hd_args = {'youtubepot-bgutilscript': {'script_path': [_bgutil_script]}}
    else:
        hd_args = None
    fallback_args = {
        'youtube': {
            'player_client': ['tv_embed', 'android', 'mweb', 'web'],
            'player_skip': ['webpage', 'configs'],
        }
    }

    # Cap at 720p when using a paid proxy (bandwidth cost); direct keeps best.
    if _proxy:
        hd_fmt = ('bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]/'
                  'bestvideo[vcodec^=avc1][height<=720]+bestaudio/best[height<=720][ext=mp4]/best[height<=720]/best')
    else:
        hd_fmt = 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best'
    fallback_fmt = 'best[ext=mp4]/best'

    def _base_opts(extractor_args, proxy):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if cookies_path else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
            'nocheckcertificate': True, 'cachedir': False,
            'extractor_args': extractor_args,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ),
            },
        }

    # Wire bytes actually pulled through the (paid) proxy, summed across
    # fragments/streams. Reported to app.py via the PROXY_BYTES= line below.
    _dl_bytes = {"total": 0}

    def _progress_hook(d):
        if d.get('status') == 'finished':
            _dl_bytes["total"] += int(d.get('total_bytes')
                                      or d.get('total_bytes_estimate')
                                      or d.get('downloaded_bytes') or 0)

    def _attempt(extractor_args, fmt, proxy):
        _dl_bytes["total"] = 0
        with yt_dlp.YoutubeDL(_base_opts(extractor_args, proxy)) as ydl:
            info = ydl.extract_info(url, download=False)
        sanitized = sanitize_filename(info.get('title', 'youtube_video'))
        expected = os.path.join(output_dir, f'{sanitized}.mp4')
        if os.path.exists(expected):
            os.remove(expected)
        dl_opts = {
            **_base_opts(extractor_args, proxy),
            'format': fmt,
            'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
            'merge_output_format': 'mp4', 'overwrites': True,
            'progress_hooks': [_progress_hook],
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        return sanitized

    # DIRECT_FIRST=1: try the server's own IP before spending proxy bandwidth.
    # Needs cookies + a PO-token provider — without both, YouTube flags the
    # datacenter IP after the first request (verified in prod, 21-jul-2026).
    _direct_first = (os.environ.get("DIRECT_FIRST", "").strip() == "1"
                     and _proxy and hd_args and cookies_path)

    attempts = (
        ([('HD-direct', hd_args, hd_fmt, None)] if _direct_first else [])
        + ([('HD', hd_args, hd_fmt, _proxy)] if hd_args else [])
        + [('fallback', fallback_args, fallback_fmt, _proxy)]
    )

    sanitized_title = None
    last_err = None
    used_proxy = False
    for label, ea, fmt, proxy in attempts:
        # A 403 on the media fetch is usually transient: the googlevideo URL is
        # bound to the IP that extracted it, and the residential proxy rotates
        # its exit IP between requests. Retrying re-extracts and usually lands
        # on a consistent IP (3 of 62 downloads hit this on 22-jul-2026).
        for retry in range(2):
            try:
                print(f"📥 Download attempt: {label}" + (f" (retry {retry})" if retry else ""))
                sanitized_title = _attempt(ea, fmt, proxy)
                used_proxy = proxy is not None
                print(f"✅ Download succeeded ({label}).")
                break
            except Exception as e:
                last_err = e
                print(f"⚠️  Download attempt '{label}' failed: {str(e)[:200]}")
                retryable = '403' in str(e) or 'Forbidden' in str(e)
                if not retryable or retry == 1:
                    break
                time.sleep(3)
        if sanitized_title is not None:
            break

    if sanitized_title is None:
        import sys
        error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)
❌ ================================================================= ❌
REASON: YouTube blocked the request or the download tooling is out of date.
👇 SOLUTION FOR USER: download the video manually and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        raise last_err

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break

    if used_proxy and _dl_bytes["total"]:
        # Machine-parseable marker consumed by app.py's log reader for the
        # monthly proxy-bandwidth counter. Not shown to clients (log filter).
        # Only emitted when the winning attempt actually went through the
        # proxy — direct-first successes are free bandwidth.
        print(f"PROXY_BYTES={_dl_bytes['total']}")
    print(f"✅ Video downloaded in {time.time() - step_start_time:.2f}s: {downloaded_file}")
    return downloaded_file, sanitized_title

def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (for horizontal/16:9 output).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    print(f"🎬 Passthrough (native framing): {input_video}")
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    print(f"✅ Clip saved to {final_output_video}")
    return True


def render_clip(input_video, final_output_video, output_format="auto"):
    """Route a cut clip through the right renderer for the chosen output format.
    vertical/auto -> 9:16 reframe, square -> 1:1 reframe, horizontal -> keep."""
    if output_format == "horizontal":
        return finalize_clip_passthrough(input_video, final_output_video)
    aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, aspect_ratio=aspect)


def process_video_to_vertical(input_video, final_output_video, aspect_ratio=ASPECT_RATIO):
    """
    Core logic to reframe a horizontal video to a target aspect ratio using
    scene detection and Active Speaker Tracking (MediaPipe).
    aspect_ratio: width/height of the output (9/16 vertical, 1.0 square).
    """
    script_start_time = time.time()

    # v2 engine: analyze downscaled, render natively in ffmpeg. Any failure
    # falls back to the v1 frame loop below so a v2 edge case can't kill jobs.
    if os.environ.get("REFRAME_ENGINE", "v2").strip().lower() != "v1":
        try:
            import reframe_v2
            t0 = time.time()
            result = reframe_v2.render(input_video, final_output_video, aspect_ratio)
            print(f"   ⏱️ Reframe v2 total: {time.time() - t0:.1f}s")
            return result
        except Exception as e:
            print(f"   ⚠️ Reframe v2 failed ({type(e).__name__}: {e}) — "
                  f"falling back to v1 frame loop")

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
    
    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * aspect_ratio)
    # Never ask for a crop wider than the source; shrink height to fit instead.
    if OUTPUT_WIDTH > original_width:
        OUTPUT_WIDTH = original_width
        OUTPUT_HEIGHT = int(OUTPUT_WIDTH / aspect_ratio)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=aspect_ratio)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        *video_encode_args(QUALITY_FAST), '-an', temp_video_output
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

    # Per-stage wall time (server-side diagnostics; hidden from cloud logs).
    stage_seconds = {'detect': 0.0, 'write': 0.0}
    loop_started = time.time()

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
                # "Plano General" -> full-frame crop
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)

                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                speaker_tracker.known_faces = []
            elif current_strategy == 'SPLIT':
                output_frame = create_split_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                speaker_tracker.known_faces = []
            else:
                # "Single Speaker" -> Track & Crop

                # Detect every Nth frame for performance (cameraman smooths in
                # between); the much heavier YOLO fallback gets its own stride.
                if frame_number % DETECT_STRIDE == 0:
                    t_det = time.time()
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    elif frame_number % YOLO_FALLBACK_STRIDE == 0:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)
                    stage_seconds['detect'] += time.time() - t_det

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])

                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

            t_wr = time.time()
            ffmpeg_process.stdin.write(output_frame.tobytes())
            stage_seconds['write'] += time.time() - t_wr
            frame_number += 1
            pbar.update(1)
    
    loop_total = time.time() - loop_started
    other = loop_total - stage_seconds['detect'] - stage_seconds['write']
    print(f"\n   ⏱️ Frame loop: {loop_total:.1f}s total — "
          f"detect {stage_seconds['detect']:.1f}s, "
          f"encode-wait {stage_seconds['write']:.1f}s, "
          f"decode+render {other:.1f}s ({frame_number} frames)")

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
            '-map', '0:v:0', '-map', '1:a:0',
            '-vf', 'setpts=PTS-STARTPTS', '-af', 'asetpts=PTS-STARTPTS',
            *video_encode_args(QUALITY_FAST),
            '-c:a', 'aac', '-b:a', '192k', '-fps_mode', 'cfr',
            '-shortest', *METADATA_SCRUB, '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-map', '0:v:0', '-vf', 'setpts=PTS-STARTPTS',
            *video_encode_args(QUALITY_FAST),
            '-fps_mode', 'cfr', *METADATA_SCRUB,
            '-avoid_negative_ts', 'make_zero', '-movflags', '+faststart',
            final_output_video
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

def transcribe_video(video_path):
    print("🎙️  Transcribing video...")
    from transcribe_backends import transcribe_media

    transcript = transcribe_media(video_path)

    print(f"   Detected language '{transcript['language']}', "
          f"{len(transcript['segments'])} segments")
    for segment in transcript['segments']:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return transcript

def _run_gemini_stage(client, model_name, prompt, schema):
    """One schema-enforced Gemini call with transient-error backoff.
    Returns (parsed_dict, cost_analysis)."""
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            gemini_worker.raise_if_blocked(response)
            # Parsing lives inside the retry loop on purpose: Gemini sometimes
            # returns 200 with an empty body, which raises here rather than at
            # the call. Retrying that recovered every occurrence seen in prod
            # (22-jul-2026) — the same payload succeeds on the next attempt.
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                parsed = gemini_worker._parse_json_response_text(
                    gemini_worker._get_response_text(response))
            return parsed, gemini_worker._calculate_cost_analysis(response, model_name)
        except Exception as e:
            msg = str(e)
            transient = any(tok in msg for tok in (
                '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
                '500', 'INTERNAL', 'overloaded', 'Deadline',
                'empty response body', 'did not contain a JSON object',
                'Failed to parse Gemini JSON response'))
            if attempt == max_attempts or not transient:
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"⚠️ Gemini transient error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def get_viral_clips(transcript_result, video_duration):
    """Two-pass clip selection: score transcript windows, then detail the best.

    Windowing gives even coverage on long videos (a single call over the whole
    transcript clusters picks near the start), and the cheap scoring pass keeps
    the expensive detail reasoning focused on the shortlist. Cuts are snapped to
    word boundaries so clips don't start/end mid-word.
    """
    print("\U0001f916  Analyzing with Gemini (2-pass: score → detail)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3-flash-preview'
    language = str(transcript_result.get('language') or 'unknown')
    print(f"\U0001f916  Model: {model_name} | language: {language}")

    # Full word list — ground truth for snapping cut points.
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    try:
        windows = build_transcript_windows(transcript_result, video_duration)
        print(f"   Built {len(windows)} scoring window(s).")
        costs = []

        # --- Pass 1: score windows in batches, keep the highest-scoring ---
        scored = []
        SCORE_BATCH = 8
        for b in range(0, len(windows), SCORE_BATCH):
            batch = windows[b:b + SCORE_BATCH]
            payload = [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in batch]
            prompt = gemini_worker.SCORE_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                windows_json=json.dumps(payload, ensure_ascii=False))
            parsed, cost = _run_gemini_stage(client, model_name, prompt, gemini_worker.ScoreResponse)
            if cost:
                costs.append(cost)
            scored.extend(parsed.get("windows") or [])

        # Shortlist the top windows; scale with duration so long videos surface
        # more candidates without exploding the detail call.
        scored.sort(key=lambda w: w.get("score", 0), reverse=True)
        target = max(3, min(10, int(video_duration // 90) + 2))
        by_id = {w["id"]: w for w in windows}
        shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
        if not shortlist:
            shortlist = windows[:target]  # scoring returned nothing usable
        print(f"   Shortlisted {len(shortlist)} window(s) for detail.")

        # --- Pass 2: detailed clip extraction on the shortlist ---
        payload = [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in shortlist]
        prompt = gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language,
            windows_json=json.dumps(payload, ensure_ascii=False))
        detail, cost = _run_gemini_stage(client, model_name, prompt, gemini_worker.DetailResponse)
        if cost:
            costs.append(cost)

        shorts = detail.get("shorts") or []
        # Snap each proposed clip onto real word boundaries (+ a bit of silence).
        for s in shorts:
            ns, ne = snap_clip_to_words(s.get("start", 0), s.get("end", 0), words, video_duration)
            s["start"], s["end"] = ns, ne

        # Aggregate cost across both passes.
        cost_analysis = None
        if costs:
            cost_analysis = {
                "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
                "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
                "total_cost": sum(c.get("total_cost", 0) for c in costs),
                "model": model_name,
            }
            print(f"\U0001f4b0 Total cost ({model_name}, 2-pass, {len(costs)} calls): ${cost_analysis['total_cost']:.6f}")

        if not shorts:
            print("⚠️ 2-pass returned no clips.")
            return None

        result = {"shorts": shorts}
        if cost_analysis:
            result["cost_analysis"] = cost_analysis
        return result
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT video by vision: Gemini watches the footage and picks the
    most engaging visual moments (no transcript). Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    print("🎥  Silent video — analyzing with Gemini vision (no transcript)...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return None
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3-flash-preview'
    print(f"🎥  Model: {model_name} | uploading {os.path.basename(video_path)}…")

    file_upload = None
    try:
        file_upload = client.files.upload(file=video_path)
        deadline = time.time() + 180
        while True:
            info = client.files.get(name=file_upload.name)
            state = str(getattr(getattr(info, "state", info), "name", "")).upper()
            if state == "ACTIVE":
                break
            if state == "FAILED":
                print("❌ Gemini could not process the video.")
                return None
            if time.time() > deadline:
                print("❌ Gemini video processing timed out.")
                return None
            time.sleep(2)

        prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
            video_duration=video_duration, language=language)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.VisualResponse,
        )
        response = client.models.generate_content(
            model=model_name, contents=[file_upload, prompt], config=config)
        parsed = json.loads(response.text)
        shorts = parsed.get("shorts") or []
        # Clamp to the real duration; drop anything degenerate.
        clean = []
        for s in shorts:
            s["start"] = max(0.0, float(s.get("start", 0)))
            s["end"] = min(float(video_duration), float(s.get("end", 0)))
            if s["end"] - s["start"] >= 1.0:
                clean.append(s)
        if not clean:
            print("⚠️ Vision pass returned no usable clips.")
            return None

        cost = gemini_worker._calculate_cost_analysis(response, model_name)
        if cost:
            print(f"💰 Vision cost ({model_name}): ${cost.get('total_cost', 0):.6f}")
        result = {"shorts": clean}
        if cost:
            result["cost_analysis"] = cost
        return result
    except Exception as e:
        print(f"❌ Gemini vision error: {e}")
        return None
    finally:
        if file_upload is not None:
            try:
                client.files.delete(name=file_upload.name)
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--format', type=str, default="auto", choices=["auto", "vertical", "horizontal", "square"],
                        help="Output aspect: vertical/auto (9:16), horizontal (keep 16:9), square (1:1).")

    args = parser.parse_args()
    output_format = args.format

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
        render_clip(input_video, output_file, output_format)
    else:
        # Get duration (needed by both the transcript and the vision path).
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 3. Transcribe — unless the video has no audio, in which case fall back
        # to Gemini vision (picks clips from the imagery instead of the speech).
        from transcribe_backends import NoAudioError
        transcript = None
        try:
            transcript = transcribe_video(input_video)
        except NoAudioError as e:
            print(f"🔇 {e} — switching to visual analysis.")

        # 4. Gemini Analysis (transcript-driven, or vision for silent videos)
        if transcript is not None:
            clips_data = get_viral_clips(transcript, duration)
        else:
            clips_data = get_visual_clips(input_video, duration)

        if not clips_data or 'shorts' not in clips_data:
            # Deliberately fail instead of reframing the whole video: that path
            # wrote no metadata.json, so app.py marked the job failed anyway
            # (app.py:1087) after burning GPU on a render nobody could see.
            raise RuntimeError(
                "Clip detection failed — Gemini did not return usable clips for this video.")

        shorts = clips_data['shorts']

        # Gemini timestamps are untrusted. Normalize before scoring/rendering:
        # pad speech context, clamp to source, sort, and merge overlaps so the
        # same source frames can never enter the render queue twice.
        shorts = brutal_truth.normalize_intervals(
            shorts, source_duration=duration, min_duration=15.0,
            pre_roll=0.2, post_roll=0.3)
        if not shorts:
            raise RuntimeError("Clip detection returned no valid non-overlapping intervals.")

        # Compute topic terms once and pass the transcript-aware score into the
        # threshold filter. Previously filter_by_score rescored without the
        # transcript, so the threshold ignored hook/pacing/topic evidence.
        top_terms = brutal_truth.top_terms_from_transcript(transcript) if transcript else None

        # --- Score threshold filter ---
        # ponytail: auto-count mode just raises threshold; no separate logic needed.
        score_threshold = int(os.environ.get("VIRALITY_THRESHOLD", "85"))
        clip_mode = os.environ.get("CLIP_COUNT_MODE", "auto")
        if clip_mode == "manual":
            max_count = int(os.environ.get("MANUAL_CLIP_COUNT", "5"))
            shorts = brutal_truth.cap_count(shorts, max_count)
        else:  # auto
            shorts = brutal_truth.filter_by_score(
                shorts, score_threshold, transcript=transcript, top_terms=top_terms)

        clips_data['shorts'] = shorts
        prints_hot = "🔥"
        print(f"{prints_hot} Selected {len(shorts)} clips (threshold={score_threshold}, mode={clip_mode})!")

        # --- White-label metadata for agency mode ---
        # ponytail: cheap per-clip mutation; consumer (XML export, metadata.json) reads same dict.
        agency_client = os.environ.get("AGENCY_CLIENT", "")
        if agency_client:
            brand_tags_str = os.environ.get("AGENCY_HASHTAGS", "")
            brand_tags = [t.strip() for t in brand_tags_str.split(",") if t.strip()]
            caption_prefix = os.environ.get("AGENCY_CAPTION_PREFIX", "")
            for s in shorts:
                brutal_truth.whitelabel_clip_metadata(
                    s, agency_client, brand_tags, caption_prefix)

        # --- Enriched metadata: composite score (Gemini + hook + duration + pacing + topic)
        brutal_truth.enrich_metadata(shorts, transcript=transcript, top_terms=top_terms)

        # Save metadata. Silent videos have no transcript → no subtitles,
        # which is correct (there's no speech to caption).
        clips_data['transcript'] = transcript or {"language": "none", "segments": []}
        metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(clips_data, f, indent=2)
        print(f"   Saved metadata to {metadata_file}")

        # --- XML timeline export (FCP/Premiere compatible) ---
        # ponytail: minimal FCP XML — one sequence per clip,No transitions.
        if os.environ.get("XML_EXPORT", "1") != "0":
            try:
                xml_path = brutal_truth.export_xml_timeline(shorts, video_title, output_dir, duration)
                print(f"   📼 XML timeline exported: {xml_path}")
            except Exception as xml_err:
                print(f"   ⚠️ XML export failed: {xml_err}")

        # 5. Process clips in parallel: each worker cuts + renders one
        # clip. Renders are mostly ffmpeg subprocesses (parallelize well);
        # detector inference is serialized internally via DETECT_LOCK.
        def _process_one_clip(i, clip):
            start = clip['start']
            end = clip['end']
            print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
            print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

            clip_filename = f"{video_title}_clip_{i+1}.mp4"
            clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
            clip_final_path = os.path.join(output_dir, clip_filename)

            try:
                # Accurate decode/re-encode cut with fresh PTS/DTS. Stream-copy
                # cuts were the source of keyframe timestamp corruption and
                # later looping/desync at boundaries.
                if not brutal_truth.extract_clean_segment(
                        input_video, start, end, clip_temp_path):
                    print(f"   ❌ Clip {i+1} failed: clean cut failed")
                    return False

                # --- Silence/filler removal via ffmpeg silencedetect ---
                # ponytail: skip if disabled; runs a quick probe pass then re-cuts.
                if os.environ.get("SILENCE_REMOVAL", "1") == "1" and transcript:
                    if brutal_truth.remove_silence(clip_temp_path):
                        print(f"   ✂️ Silence removed from clip {i+1}")

                # --- Weak-tail trim: cut any dead trailing air that tanks completion rate ---
                if os.environ.get("WEAK_TAIL_TRIM", "1") == "1" and transcript:
                    if brutal_truth.trim_weak_tail(clip_temp_path, start, end, transcript):
                        print(f"   📉 Trimmed weak tail from clip {i+1}")

                # Cold-open duplication is disabled: selecting a hook from
                # inside this same interval and prepending it repeats footage.
                # Use concat_clean_segments only with disjoint source ranges.

                broll_insertions = []
                broll_dir = os.environ.get("BROLL_DIR", "")
                if (transcript and broll_dir
                        and os.environ.get("BROLL_ENABLED", "1") == "1"):
                    try:
                        broll_scenes, broll_fps = detect_scenes(clip_temp_path)
                        scene_ranges = [
                            (scene_start.get_seconds(), scene_end.get_seconds())
                            for scene_start, scene_end in broll_scenes
                        ]
                        # The clip temp timeline starts at zero; transcript words
                        # still use source-video absolute time, so localize them.
                        local_transcript = {"segments": []}
                        for segment in transcript.get("segments", []):
                            local_words = []
                            for word in segment.get("words", []) or []:
                                local_words.append({
                                    **word,
                                    "start": max(0.0, word.get("start", 0) - start),
                                    "end": max(0.0, word.get("end", 0) - start),
                                })
                            local_transcript["segments"].append({"words": local_words})
                        broll_insertions = brutal_truth.find_static_broll_insertions(
                            scene_ranges, local_transcript,
                            min_static_seconds=4.0, insert_duration=2.0)
                    except Exception as broll_err:
                        print(f"   ⚠️ B-roll analysis skipped for clip {i+1}: {broll_err}")

                success = render_clip(clip_temp_path, clip_final_path, output_format)

                if success and broll_insertions:
                    try:
                        if brutal_truth.overlay_broll(
                                clip_final_path, clip_final_path,
                                broll_insertions, broll_dir):
                            print(f"   🎞️ B-roll pattern interrupt applied to clip {i+1}")
                    except Exception as broll_err:
                        print(f"   ⚠️ B-roll overlay skipped for clip {i+1}: {broll_err}")

                # Optional word-level karaoke captions. This is a separate final
                # encode with PTS reset in burn_subtitles(); disable with
                # KARAOKE_SUBTITLES=0 when clean manual captions are preferred.
                if (success and transcript
                        and os.environ.get("KARAOKE_SUBTITLES", "1") == "1"):
                    try:
                        from subtitles import generate_ass, burn_subtitles
                        ass_path = os.path.join(output_dir, f"karaoke_{i+1}.ass")
                        if generate_ass(
                                transcript, start, end, ass_path,
                                max_chars=28, max_duration=2.0,
                                alignment="bottom", fontsize=42,
                                font_name="Arial", font_color="#FFFFFF",
                                highlight_color="#FFD700", border_width=3,
                                effect="pop", base_opacity=0.5):
                            captioned_path = os.path.join(
                                output_dir, f"captioned_{clip_filename}")
                            burn_subtitles(
                                clip_final_path, ass_path, captioned_path,
                                alignment="bottom", fontsize=42)
                            os.replace(captioned_path, clip_final_path)
                            print(f"   🟡 Karaoke captions burned into clip {i+1}")
                    except Exception as caption_err:
                        print(f"   ⚠️ Karaoke caption pass failed for clip {i+1}: {caption_err}")

                # No watermark is ever applied. Branding remains the user's job.

                if success:
                    print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                return success
            finally:
                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

        clip_workers = max(int(os.environ.get("CLIP_WORKERS", "3")), 1)
        with ThreadPoolExecutor(max_workers=min(clip_workers, len(shorts))) as pool:
            futures = {pool.submit(_process_one_clip, i, clip): i
                       for i, clip in enumerate(shorts)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"   ❌ Clip {i+1} failed: render returned False")
                except Exception as e:
                    print(f"   ❌ Clip {i+1} failed: {type(e).__name__}: {e}")

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
