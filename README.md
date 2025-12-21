# OpenShorts.app 🚀🎬

OpenShorts is an all-in-one open-source solution to automate the creation and distribution of viral vertical content. It transforms long YouTube videos or local files into high-potential short clips optimized for **TikTok**, **Instagram Reels**, and **YouTube Shorts**.

![OpenShorts Demo](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

### 📺 Video Tutorial: How it works
[![OpenShorts Tutorial](https://img.youtube.com/vi/xlyjD1qCaX0/maxresdefault.jpg)](https://www.youtube.com/watch?v=xlyjD1qCaX0 "Click to watch the video on YouTube")

*Click the image above to watch the full walkthrough.*

---

## ✨ Key Features

OpenShorts leverages state-of-the-art AI to handle the entire content lifecycle:

1.  **🧠 Viral Moment Detection:**
    *   **Faster-Whisper**: High-speed, CPU-optimized transcription and word-level timestamps.
    *   **Google Gemini 2.0 Flash**: Advanced AI analysis to identify the 3-15 most viral moments based on hooks and engagement potential.
    *   **Automatic Copywriting**: Generates SEO-optimized titles and descriptions for all platforms.

2.  **✂️ Smart AI Cropping & Tracking (New V2 Engine):**
    *   **Dual-Mode Strategy**: Automatically detects scene composition to apply the best framing strategy.
        *   **TRACK Mode (Single Subject)**: Uses **MediaPipe Face Detection** + **YOLOv8** fallback for ultra-fast, robust subject tracking. Features a **"Heavy Tripod" stabilization engine** that eliminates jitter and unnatural movements, providing smooth, cinematic reframing. Includes **Speaker Identification** to stick to the active speaker and avoid erratic switching.
        *   **GENERAL Mode (Groups/Landscapes)**: For scenes with multiple people or no clear subject, it automatically switches to a professional **blurred-background layout**, preserving the full width of the original shot while filling the 9:16 vertical space.
    *   **Intelligent Scene Analysis**: Pre-scans every scene to determine the optimal strategy before processing.

3.  **📲 Direct Social posting:**
    *   **Upload-Post Integration**: Share your generated clips directly to TikTok, Instagram, and YouTube with a single click.
    *   **Profile Selector**: Manage multiple social accounts easily through the dashboard.

4.  **🎨 Modern Web Dashboard:**
    *   **Real-time Progress**: Watch clips appear as they are generated with a live results feed.
    *   **Log Streaming**: Follow the technical process with real-time log updates.
    *   **Responsive Design**: A premium, dark-mode glassmorphism interface.

---

## 🛠️ Requirements

*   **Docker & Docker Compose**.
*   **Google Gemini API Key** ([Get it for free here](https://aistudio.google.com/app/apikey)).
*   **Upload-Post API Key** (Optional, for direct social posting. **Free tier available, no credit card required**).

### 📲 Social Media Setup (Upload-Post)
To enable direct posting, follow these steps:
1.  **Login/Register**: [app.upload-post.com/login](https://app.upload-post.com/login)
2.  **Create Profile**: Go to [Manage Users](https://app.upload-post.com/manage-users) and create a user profile.
3.  **Connect Accounts**: In the same section, connect your TikTok, Instagram, or YouTube accounts to that profile.
4.  **Get API Key**: Navigate to [API Keys](https://app.upload-post.com/api-keys) and generate your key.
5.  **Use in OpenShorts**: Paste the API Key and select your Profile in the dashboard.


---

## 🚀 Getting Started

The easiest way to run OpenShorts is using Docker Compose.

### 1. Setup
```bash
git clone https://github.com/your-username/OpenShorts.git
cd OpenShorts
```

### 2. Launch the Application
```bash
docker compose up --build
```

### 3. Access the Dashboard
Open your browser and navigate to:
**`http://localhost:5173`**

1.  Enter your **Gemini API Key**.
2.  (Optional) Enter your **Upload-Post API Key** to enable social sharing.
3.  Paste a **YouTube URL** or **Upload a Video**.
4.  Click **"Generate Clips"** and watch the magic happen!

---

## 🏗️ Technical Pipeline

1.  **Ingestion**: Downloads YouTube videos via `yt-dlp` or handles local uploads.
2.  **Transcription**: `faster-whisper` converts audio to text in seconds.
3.  **AI Intelligence**: Gemini reads the transcript and selects periods of high interest.
4.  **Extraction**: FFmpeg precisely cuts the selected segments.
5.  **Reframing**: AI-powered visual tracking crops clips to vertical format.
6.  **Distribution**: One-click posting via Upload-Post API.

---

## 🔒 Security & Performance

*   **Non-Root Execution**: Containers run as a dedicated `appuser` for security.
*   **Concurrency Control**: Configurable job queue (`MAX_CONCURRENT_JOBS`).
*   **Auto-Cleanup**: Automatic purging of old jobs and temporary files.
*   **File Limits**: Built-in protection against oversized uploads.

---

## 🤝 Contributions

Contributions are welcome! Whether it's adding new AI models or improving the cropping engine, feel free to open a PR.

## 📄 License

MIT License. OpenShorts is yours to use, modify, and scale.
