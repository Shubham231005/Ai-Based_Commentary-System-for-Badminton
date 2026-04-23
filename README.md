<div align="center">

# 🏸 AI-Based Commentary System for Badminton

### *Event-Driven Sports Commentary Engine*

> **CV generates facts. Rules structure meaning. LLM adds emotion. Voice brings it to life.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano-00FFFF?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![ElevenLabs](https://img.shields.io/badge/ElevenLabs-TTS-000000?logo=elevenlabs&logoColor=white)](https://elevenlabs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

*An end-to-end AI pipeline that watches a badminton match video, detects gameplay events using Computer Vision, generates expressive **Hinglish commentary** (Hindi + English) in the style of Star Sports broadcasters, and produces a **final video with dual-voice AI narration and burned-in subtitles**.*

</div>

---

## 🎬 How It Works

```
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Video   │───▶│  YOLOv8n   │───▶│  Event       │───▶│  Hinglish       │───▶│  Dual-Voice TTS  │
│  Input   │    │  Detection │    │  Engine      │    │  Commentary     │    │  + Subtitles     │
│          │    │  + Tracking│    │  (Rule-based)│    │  (Gemini/Local) │    │  (ElevenLabs)    │
└──────────┘    └────────────┘    └──────────────┘    └─────────────────┘    └──────────────────┘
                   Phase 1              Phase 1              Phase 2                Phase 3
              Vision Foundation     Event Detection     Commentary Layer       Audio + Video Export
```

**Key Innovation**: CV extracts structured events (smash, rally, point) as the *truth layer* — no LLM involved in detection. The LLM only adds language, emotion, and personality on top of verified facts.

---

## ✨ Features

### 🔍 Phase 1 — Computer Vision & Event Detection
- **YOLOv8 Nano** object detection for players and shuttle (CPU-optimized)
- **Split FPS processing** — YOLO at 10 FPS for players, optical flow at 30 FPS for shuttle
- **Kalman Filter** tracking for continuous shuttle trajectory
- **Rule-based Event Engine** — detects smashes, drop shots, rallies, points, and more
- **Intensity scoring** — classifies moments as low/medium/high/maximum
- **Debug mode** — annotated video output + per-frame CSV logging

### 🎙️ Phase 2 — AI Commentary Generation
- **Gemini 2.0 Flash** powered Hinglish commentary (when API key is set)
- **Rich template fallback** — 50+ handcrafted Hinglish templates for offline use
- **Sidhu-style delivery** — energetic, Bollywood references, cricket analogies, desi metaphors
- **Devanagari Hindi** mixed with English — `"ओये होये! ये smash नहीं, ये तो Brahmos missile थी!"`
- **Smart event filtering** — commentates only key moments (no spam)
- **Match summary** generation

### 🔊 Phase 3 — Dual-Voice Audio & Video Export
- **ElevenLabs** multilingual TTS with emotional expressiveness
- **Dual commentator system** — Male (Voice A) + Female (Voice B) alternate lines
- **Anti-overlap trimming** — clips are trimmed so commentary never overlaps
- **SRT subtitle generation** with speaker tags
- **FFmpeg video export** — burns subtitles, replaces original audio with commentary
- **gTTS fallback** — free Google TTS if ElevenLabs isn't available

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Shubham231005/Ai-Based_Commentary-System-for-Badminton.git
cd Ai-Based_Commentary-System-for-Badminton

python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
# Required for AI-powered commentary (Phase 2)
GEMINI_API_KEY=your_gemini_api_key_here

# Required for ElevenLabs TTS (Phase 3) — optional, falls back to gTTS
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
```

> **Note**: The system works without API keys — it uses template-based commentary and gTTS as fallbacks.

### 3. Run the Pipeline

```bash
# Full pipeline: CV → Commentary → Audio → Final Video
python main.py --video datasets/sample_match.mp4 --player-a "Lin Dan" --player-b "Lee Chong Wei"

# Event detection only (skip commentary)
python main.py --video match.mp4 --no-commentary

# With debug visualization
python main.py --video match.mp4 --debug --verbose

# Custom persona
python main.py --video match.mp4 --persona analytical

# Live debug window (press 'q' to close)
python main.py --video match.mp4 --debug-live
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📊 Output Examples

### Event Timeline (JSON)

```json
[
  {"timestamp": 12.3, "event": "rally_start", "server": "Lin Dan", "intensity": "low"},
  {"timestamp": 18.9, "event": "smash", "by": "Lee Chong Wei", "intensity": "high", "velocity": 35.2},
  {"timestamp": 21.1, "event": "point_won", "by": "Lee Chong Wei", "rally_length": 18, "intensity": "maximum"}
]
```

### Hinglish Commentary

```
[00:12.3]  चलिए जी! Lin Dan ने serve उछाली — पूरा stadium साँस रोके बैठा है!
[00:18.9]  ओ माँ! Lee Chong Wei ON FIRE है! 35 speed — ये smash नहीं, ये तो बिजली गिरी!
[00:21.1]  SENSATIONAL POINT! Lee Chong Wei ने जीत लिया! 1-0! 18 shots का EPIC rally!
```

---

## 🎯 Detected Event Types

| Event | Detection Method | Commentary Style |
|-------|-----------------|-----------------|
| `rally_start` | Serve motion + shuttle movement | Hype building, match references |
| `smash` | High velocity + downward angle | Power analogies, missile comparisons |
| `drop_shot` | Low velocity + steep angle near net | Finesse appreciation, chess analogies |
| `long_rally` | Hit count > 8 direction reversals | Epic battle references, Mahabharata vibes |
| `point_won` | Shuttle stationary for N frames | Score update + celebration |

---

## 📁 Project Structure

```
Ai-Based-Commentary-System/
├── main.py                           # CLI entry point
├── config/
│   └── config.yaml                   # Event rules, thresholds, TTS & commentary config
├── src/
│   ├── pipeline.py                   # Pipeline orchestrator (Phase 1 → 2 → 3)
│   ├── vision/
│   │   ├── video_processor.py        # Frame extraction (split FPS)
│   │   ├── detector.py               # YOLOv8n detection + optical flow
│   │   ├── tracker.py                # Centroid tracking + Kalman filter
│   │   ├── feature_extractor.py      # Velocity, angle, intensity features
│   │   └── debug_visualizer.py       # Annotated debug video output
│   ├── events/
│   │   ├── event_models.py           # Pydantic data models
│   │   └── event_engine.py           # Rule-based event classification
│   ├── commentary/
│   │   ├── commentary_generator.py   # Hinglish commentary (Gemini + templates)
│   │   └── commentary_models.py      # Commentary data models
│   ├── audio/
│   │   └── tts_engine.py             # ElevenLabs/gTTS dual-voice + FFmpeg export
│   ├── utils/
│   │   └── config.py                 # YAML config loader
│   └── debug_event_logger.py         # Per-frame CSV debug logger
├── tests/
│   └── test_commentary.py            # Commentary unit tests
├── datasets/                         # Input video files (gitignored)
├── output/                           # Generated outputs (gitignored)
│   ├── event_timeline.json
│   ├── commentary.json
│   ├── commentary.txt
│   ├── commentary_audio.mp3
│   ├── commentary.srt
│   └── final_commentary_video.mp4
├── requirements.txt
├── .env                              # API keys (gitignored)
└── .gitignore
```

---

## ⚙️ Configuration

All tunable parameters live in `config/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `video` | `fps: 10`, `resolution: 720`, `max_duration: 120` |
| `vision` | `yolo_model: yolov8n.pt`, `confidence_threshold: 0.35` |
| `event_rules` | Smash velocity, rally length, drop shot proximity |
| `commentary` | `persona: hinglish_excited`, `model: gemini-2.0-flash`, `temperature: 0.85` |
| `tts` | `provider: elevenlabs`, voice IDs, stability, style expressiveness |
| `audio` | Volume levels, ducking settings |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Object Detection** | YOLOv8 Nano (Ultralytics) | Player & shuttle detection |
| **Motion Tracking** | OpenCV + Kalman Filter | Shuttle trajectory tracking |
| **Event Classification** | Rule-based engine (Pydantic) | Structured event detection |
| **Commentary AI** | Google Gemini 2.0 Flash | Natural Hinglish generation |
| **Text-to-Speech** | ElevenLabs Multilingual v2 | Dual-voice expressive audio |
| **TTS Fallback** | gTTS | Free Google TTS backup |
| **Video Processing** | FFmpeg + OpenCV | Frame extraction, audio muxing, subtitle burn-in |
| **Configuration** | PyYAML + python-dotenv | Flexible YAML + env config |

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel i3 / AMD equivalent | Intel i5+ / Apple M1+ |
| **RAM** | 8 GB | 16 GB |
| **GPU** | Not required (CPU mode) | CUDA GPU (optional, faster YOLO) |
| **FFmpeg** | Required for Phase 3 | Install via `choco install ffmpeg` or `brew install ffmpeg` |
| **Python** | 3.10+ | 3.11+ |

**Processing Speed**: ~2-5 min for a 30s video at 10 FPS on CPU

---

## 🗝️ API Keys

| Service | Required? | Free Tier | Get Key |
|---------|----------|-----------|---------|
| **Gemini** | Optional (has template fallback) | ✅ Generous free quota | [Google AI Studio](https://aistudio.google.com/apikey) |
| **ElevenLabs** | Optional (has gTTS fallback) | ✅ 10k chars/month | [ElevenLabs](https://elevenlabs.io/) |

---

## 🛣️ Roadmap

- ✅ **Phase 1**: Vision Foundation — CV-based event detection pipeline
- ✅ **Phase 2**: Commentary Layer — Gemini + template Hinglish commentary
- ✅ **Phase 3**: Audio Integration — ElevenLabs dual-voice TTS + FFmpeg video export
- 📅 **Phase 4**: Enhancement — Web UI, multi-sport support, regional dialects

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for Indian Sports Commentary**

*"ओये होये! ये project नहीं, ये तो revolution है!" — Sidhu Mode ON 🏸🔥*

</div>
