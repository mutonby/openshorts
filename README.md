# OpenShorts.app

**Deutsch** | [English](README.en.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**Kostenlose Open-Source-KI-Videoplattform** mit 3 Tools in einem: **Clip Generator**, **AI Shorts (UGC-Videos mit KI-Darstellern)** und **YouTube Studio**. Selbst gehostet — deine Daten, deine Keys, deine Regeln. Frei konfigurierbares Wasserzeichen (oder gar keins), keine künstlichen Limits.

![OpenShorts Demo](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

### Video-Tutorial: So funktioniert's
[![OpenShorts Tutorial](https://img.youtube.com/vi/xlyjD1qCaX0/maxresdefault.jpg)](https://www.youtube.com/watch?v=xlyjD1qCaX0 "Klicken, um das Video auf YouTube anzusehen")

*Aufs Bild klicken für den kompletten Walkthrough.*

---

## 3 Tools in 1 Plattform

### 1. Clip Generator
Verwandelt lange YouTube-Videos oder lokale Uploads in virale Shorts für TikTok, Instagram Reels und YouTube Shorts — im Format deiner Wahl: **Auto (smart)**, **9:16**, **16:9** oder **1:1**.

![Clip Generator](screenshots/clip-generator.png)

![Clip Results](screenshots/clip-results.png)

### 2. AI Shorts (UGC-Video-Creator)
Erzeugt Marketing-Videos mit KI-Darstellern für **jedes Produkt und jedes Business**. Keine Kamera, kein Studio, kein Influencer-Budget. Einfach Produkt beschreiben oder URL einfügen.

![AI Shorts Setup](screenshots/ai-shorts.png)

- **Zwei Kostenmodi**: Low Cost (~0,65 $/Video) und Premium (~2 $/Video)
- Funktioniert für jedes Business: SaaS, Restaurants, E-Commerce, Coaching, lokale Unternehmen
- KI-generierte Darsteller mit Lippensynchronisation, Voiceover, B-Roll und TikTok-Style-Untertiteln
- Avatar aus der geteilten Galerie wählen oder eigenes Foto hochladen
- Direkt auf TikTok, Instagram und YouTube veröffentlichen

### 3. YouTube Studio
Komplettes kostenloses KI-Toolkit für YouTube: Thumbnails, Titel, Beschreibungen und direktes Publishing.

![YouTube Studio](screenshots/youtube-studio.png)

- KI-Thumbnail-Generator mit Gesichts-Overlay
- 10 virale Titelvorschläge mit Verfeinerungs-Chat
- Automatische Beschreibungen mit Kapitel-Zeitstempeln
- Ein-Klick-Publishing auf YouTube

### UGC-Video-Galerie
Alle generierten Videos und Avatare landen in einer öffentlichen Galerie mit SEO-Seiten pro Video.

![UGC Gallery](screenshots/ugc-gallery.png)

- Öffentliche Galerie-Seite mit Hover-to-Play (`/gallery`)
- Einzelne SEO-Videoseiten mit og:video-Meta-Tags (`/video/{id}`)
- JSON-LD-strukturierte Daten für Suchmaschinen
- Avatar-Galerie mit Prompt-Historie

---

## Die wichtigsten Features

### Clip Generator
- **Virale-Momente-Erkennung**: zweistufige Gemini-Analyse (Scoring → Detail) findet 3–15 Momente mit hohem Potenzial, mit wortgenauem Schnitt-Snapping und Modellwahl pro Aufgabe über die `.env`
- **Ausgabeformate**: Auto (smarte Quell-Erkennung), 9:16, 16:9 Original oder 1:1 Quadrat — pro Job wählbar, wie bei Opus Clip
- **Smartes Reframing**: KI-Cropping in zwei Modi — TRACK (MediaPipe + YOLOv8 Sprecher-Tracking mit „Heavy Tripod"-Stabilisierung) und GENERAL (unscharfer Hintergrund); Quellen, die schon im Zielformat sind, werden unangetastet übernommen
- **Karaoke-Untertitel**: Wort-Highlighting, 11 Preset-Looks (TikTok, Gold Glow, Neon, Beast …), Glow-/Pop-/Box-Effekte, Bulk-Anwendung auf alle Clips, ZIP-Download
- **Auto-Edit v2**: Gemini plant eine Edit-Liste (Zooms, Punch-ins, Farb-Pops, S/W-Momente, Blitze, Vignetten); ein deterministischer Builder rendert sie mit harten Sicherheitslimits — eingebrannte Untertitel werden von Zooms nie abgeschnitten
- **Wasserzeichen**: dezentes, mittiges Wasserzeichen auf jedem Clip (dein Text, dein Logo — oder aus), damit niemand deine Arbeit als seine hochlädt
- **Qualitäts-Gate**: Pre-Flight-Check warnt dich *vor* der Verarbeitung, wenn YouTube nur niedrige Auflösung anbietet — inklusive Cookie-Aktualisierungs-Anleitung
- **Selbstlernende Zeitschätzung**: realistische ETA, kalibriert auf deine Hardware, mit Gesamtprognose direkt beim Start
- **KI-Stimm-Dubbing**: ElevenLabs-Integration für 30+ Sprachen mit Voice Cloning
- **Hook-Text-Overlays**: KI-generierte Aufmerksamkeits-Hooks mit Emoji-Unterstützung

### AI-Shorts-Pipeline
1. **Analyse**: Website-URL scrapen + Web-Recherche, oder manuelle Beschreibung
2. **Skript**: KI schreibt virale Skripte (Hook – Problem – Lösung – CTA)
3. **Darsteller**: KI-Darsteller mit Flux 2 Pro generieren oder aus der Galerie wählen
4. **Stimme**: ElevenLabs-TTS-Voiceover (Englisch/Spanisch, männlich/weiblich)
5. **Video**: Talking-Head-Generierung (Hailuo 2.3 Fast img2video + VEED Lipsync)
6. **B-Roll**: KI-generierte Visuals mit Ken-Burns-Effekt
7. **Compositing**: finale FFmpeg-Montage mit Untertiteln und Hook-Overlays
8. **Publishing**: Direktes Posten auf TikTok, Instagram Reels, YouTube Shorts via Upload-Post

### YouTube Studio
- KI-Titelgenerierung mit 10 viralen Optionen
- Interaktiver Verfeinerungs-Chat für Titel
- KI-Thumbnails mit eigenem Gesicht + Hintergrund
- Automatische Beschreibungen mit Kapitel-Zeitstempeln aus dem Whisper-Transkript
- Direktes YouTube-Publishing via Upload-Post

### Social Publishing
- Ein-Klick-Posting auf TikTok, Instagram Reels und YouTube Shorts
- Posts für später planen
- Upload-Post-Integration mit asynchronen Uploads

### Infrastruktur
- S3-Cloud-Backup (privater Bucket für Clips, öffentlicher Bucket für Galerie/Avatare)
- SEO-Galerie-Seiten aus FastAPI mit JSON-LD-strukturierten Daten
- Asynchrone Job-Queue mit Nebenläufigkeits-Kontrolle, sauberem Abbrechen, Crash-Resume, Keepalive-Heartbeat und Windows-Standby-Verhinderung
- 63 automatisierte Tests + GitHub-Actions-CI (Backend, Frontend, Docker)

---

## Voraussetzungen

- **Google Gemini API Key** ([Kostenlos — hier holen](https://aistudio.google.com/app/apikey)) — für alle KI-Features erforderlich
- **fal.ai API Key** ([Pay-per-use](https://fal.ai)) — erforderlich für AI Shorts (Darsteller, Video, Lipsync)
- **ElevenLabs API Key** ([Free Tier](https://elevenlabs.io)) — erforderlich für Voiceover/Dubbing
- **Upload-Post API Key** (Optional, [Free Tier](https://upload-post.com)) — für direktes Social Posting
- **Docker & Docker Compose** — oder Python 3.11+ / Node 18+ / FFmpeg für die lokale Installation

---

## Loslegen

### 1. Klonen
```bash
git clone https://github.com/Themegaindex/openshorts.git
cd openshorts
```

### 2. Konfigurieren (optional)
```bash
cp .env.example .env
# .env anpassen: AWS-Keys für S3-Backup, Gemini-Modelle, Wasserzeichen, Qualitäts-Gate …
```

### 3. Starten

**Option A — Docker:**
```bash
docker compose up --build
```

**Option B — Lokal ohne Docker (Windows):**
```bash
pip install -r requirements.txt
cd dashboard && npm install && cd ..
start.bat
```
(Linux/macOS: `uvicorn app:app --host 0.0.0.0 --port 8000` starten und `npm run dev` im Ordner `dashboard/`.)

### 4. Dashboard öffnen
Im Browser **`http://localhost:5175`** aufrufen

1. In den **Settings** die API-Keys eintragen (Gemini, fal.ai, ElevenLabs, Upload-Post)
2. **Clip Generator**: YouTube-URL einfügen oder Video hochladen, Ausgabeformat wählen, virale Shorts generieren
3. **AI Shorts**: Produkt beschreiben oder URL einfügen, um UGC-Marketing-Videos zu erzeugen
4. **YouTube Studio**: Thumbnails, Titel und Beschreibungen für YouTube generieren
5. **UGC Gallery**: Alle generierten Videos und Avatare durchstöbern

---

## Technische Pipeline

### Clip Generator
1. **Ingest** — YouTube-Download (yt-dlp, HD-fähige Standard-Clients) oder lokaler Upload
2. **Qualitäts-Gate** — Pre-Flight-Auflösungscheck mit Rückfrage bei niedriger Qualität
3. **Transkription** — faster-whisper mit Zeitstempeln auf Wortebene
4. **Szenenerkennung** — PySceneDetect für Szenengrenzen
5. **Analyse** — zweistufige Gemini-Analyse findet 3–15 virale Momente (je 15–60 s)
6. **Extraktion** — präziser FFmpeg-Schnitt
7. **Reframing** — formatabhängiges Rendering: KI-Cropping (vertikal/quadratisch) mit Subjekt-Tracking oder Original-Passthrough; Wasserzeichen wird direkt im Frame eingeblendet
8. **Effekte** — Karaoke-Untertitel, Hooks, Auto-Edit v2
9. **Publishing** — S3-Backup + Upload-Post-Distribution

### AI Shorts
1. **Analyse** — Website-Scraping + Gemini-Web-Recherche (oder manuelle Beschreibung)
2. **Skript** — Gemini generiert virale Skripte mit Segmenten
3. **Darsteller** — Flux-2-Pro-Portraitgenerierung (oder Galerie/Upload)
4. **Stimme** — ElevenLabs-TTS-Voiceover
5. **Video** — Hailuo 2.3 Fast img2video + VEED Lipsync (Low Cost) oder Kling Avatar v2 (Premium)
6. **B-Roll** — Flux-2-Pro-Bildgenerierung + Ken-Burns-Effekt
7. **Compositing** — FFmpeg-Montage mit ASS-Untertiteln und Hook-Overlays
8. **Galerie** — Upload in den öffentlichen S3 mit Metadaten für SEO-Seiten
9. **Publishing** — Upload-Post zu TikTok, Instagram, YouTube

---

## Tech-Stack

| Ebene | Technologie |
|-------|-------------|
| Backend | Python 3.11, FastAPI, google-genai, faster-whisper, ultralytics (YOLOv8), mediapipe, opencv-python, yt-dlp, FFmpeg, httpx |
| Frontend | React 18, Vite 4, Tailwind CSS 3.4 |
| KI-APIs | Google Gemini, fal.ai (Flux, Hailuo, VEED, Kling), ElevenLabs |
| Infrastruktur | Docker + Docker Compose, AWS S3, GitHub Actions CI |
| Publishing | Upload-Post API (TikTok, Instagram, YouTube) |

---

## Umgebungsvariablen

**Serverseitig (.env)** — die komplette kommentierte Liste steht in `.env.example`:
| Variable | Beschreibung |
|----------|--------------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` / `AWS_S3_BUCKET` / `AWS_S3_PUBLIC_BUCKET` | S3-Backup & öffentliche Galerie |
| `MAX_CONCURRENT_JOBS` | Limit paralleler Jobs (Standard: 5) |
| `GEMINI_MODEL` / `GEMINI_MODEL_ANALYSIS` / `GEMINI_MODEL_EDITOR` / … | Gemini-Modell pro Aufgabe |
| `GEMINI_THINKING_SCORE` | Optionales Thinking-Budget fürs Clip-Scoring |
| `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE` | Transkriptionsqualität/-geschwindigkeit |
| `QUALITY_GATE_MIN_HEIGHT` | Rückfrage unterhalb dieser Auflösung (Standard: 720) |
| `WATERMARK_ENABLED` / `WATERMARK_TEXT` / `WATERMARK_IMAGE` / `WATERMARK_OPACITY` | Wasserzeichen-Konfiguration |

**Clientseitig (verschlüsselt im localStorage):**
| Key | Beschreibung |
|-----|--------------|
| `GEMINI_API_KEY` | Google Gemini — erforderlich |
| `FAL_KEY` | fal.ai — erforderlich für AI Shorts |
| `ELEVENLABS_API_KEY` | ElevenLabs — erforderlich für Voiceover/Dubbing |
| `UPLOAD_POST_API_KEY` | Upload-Post — optional, für Social Posting |

---

## Sicherheit & Performance

- **Non-Root-Ausführung**: Container laufen als dedizierter `appuser`
- **Nebenläufigkeits-Kontrolle**: Semaphore-basierte Job-Queue (`MAX_CONCURRENT_JOBS`)
- **Auto-Cleanup**: Automatisches Aufräumen alter Jobs (1 h Aufbewahrung)
- **Verschlüsselte Keys**: API-Keys clientseitig verschlüsselt, nie serverseitig gespeichert
- **Injection-Härtung**: Fonts, Farben und Zahlen werden vor ASS/FFmpeg bereinigt
- **Upload-Validierung**: Bild-Uploads werden auf Format und Mindestgröße geprüft
- **Datei-Limits**: 2-GB-Upload-Schutz
- **Schnelles Frontend**: Code-Splitting, GZip-API-Antworten, faststart-MP4s

---

## Social-Media-Einrichtung (Upload-Post)

1. **Registrieren**: [app.upload-post.com/login](https://app.upload-post.com/login)
2. **Profil anlegen**: unter [Manage Users](https://app.upload-post.com/manage-users)
3. **Konten verbinden**: TikTok, Instagram und/oder YouTube verknüpfen
4. **API-Key holen**: unter [API Keys](https://app.upload-post.com/api-keys)
5. **In OpenShorts eintragen**: Key in den Settings einfügen

---

## Changelog

Die komplette Release-Historie steht in der [CHANGELOG.md](CHANGELOG.md).

## Mitmachen

Beiträge sind willkommen! Ob neue KI-Modelle, Verbesserungen an der Lipsync-Pipeline oder neue Features — gerne einfach einen PR öffnen.

## Lizenz

MIT-Lizenz. OpenShorts gehört dir — nutzen, verändern, skalieren.
