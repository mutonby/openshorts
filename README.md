# AutoCrop-Vertical: A Smart Video Cropper for Social Media (Horizontal -> Vertical)

![Demo of AutoCrop-Vertical](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

AutoCrop-Vertical is a Python script that automatically converts horizontal videos into a vertical format suitable for platforms like TikTok, Instagram Reels, and YouTube Shorts.

Instead of a simple, static center crop, this script analyzes video content scene-by-scene. It uses object detection to locate people and decides whether to tightly crop the frame on the subjects or to apply letterboxing to preserve a wide shot's composition.

---

### Key Features

*   **Content-Aware Cropping:** Uses a YOLOv8 model to detect people and automatically centers the vertical frame on them.
*   **Automatic Letterboxing:** If multiple people are too far apart for a vertical crop, the script automatically adds black bars (letterboxing) to show the full scene.
*   **Scene-by-Scene Processing:** All decisions are made per-scene, ensuring a consistent and logical edit without jarring transitions.
*   **Native Resolution:** The output resolution is dynamically calculated based on the source video's height to prevent quality loss from unnecessary upscaling.
*   **High Performance:** The processing is offloaded to FFmpeg via a direct pipe, resulting in extremely fast encoding and low CPU usage.

---

### Technical Details

This script is built on a pipeline that uses specialized libraries for each step:

*   **Core Libraries:**
    *   `PySceneDetect`: For accurate, content-aware scene cut detection.
    *   `Ultralytics (YOLOv8)`: For fast and reliable person detection.
    *   `OpenCV`: Used for frame manipulation, face detection (as a fallback), and reading video properties.
    *   `FFmpeg`: The backbone of the video encoding. The script pipes raw, processed frames directly to FFmpeg, which handles the final video encoding and audio merging.
    *   `tqdm`: For clean and informative progress bars in the console.

*   **Processing Pipeline:**
    1.  The script first uses `PySceneDetect` to get a list of all scene timestamps.
    2.  It then loops through each scene and uses `OpenCV` to extract a sample frame.
    3.  This frame is passed to a pre-trained `yolov8n.pt` model to get bounding boxes for all detected people.
    4.  A set of rules determines the strategy (`TRACK` or `LETTERBOX`) for each scene based on the number and position of the detected people.
    5.  Finally, the script re-reads the input video, applies the planned transformation to every frame, and pipes the raw `bgr24` pixel data to an `FFmpeg` subprocess for efficient encoding. Audio is handled separately and merged at the end, also via FFmpeg.

*   **Performance & Optimizations:**
    The main performance gain comes from avoiding slow, frame-by-frame processing within a pure Python loop for *writing* the video. By piping frames directly to FFmpeg's optimized C-based `libx264` encoder, we achieve significant speed.

    *   **Example Benchmark:** On a test 5-minute, 640x360 source video, the entire analysis and conversion process completes in **~11 seconds** on an Apple M1 processor. The video encoding itself runs at over 70x real-time speed.

---

### Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kamilstanuch/AutoCrop-Vertical.git
    cd AutoCrop-Vertical
    ```

2.  **Set up the environment:**
    A Python virtual environment is recommended.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    The `yolov8n.pt` model weights will be downloaded automatically on the first run.

3.  **Run the script:**
    Use the `--input` and `--output` arguments to specify the source and destination files.

    ```bash
    python main.py --input path/to/horizontal_video.mp4 --output path/to/vertical_video.mp4
    ```

---

### Prerequisites

*   Python 3.8+
*   **FFmpeg:** This script requires `ffmpeg` to be installed and available in your system's PATH. It can be installed via a package manager (e.g., `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Debian/Ubuntu).
