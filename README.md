# Antigravity — Event-Driven Sports Commentary Engine 🎙️🏸

> *CV generates facts. Rules structure meaning. LLM adds emotion and style.*

An event-driven commentary engine for badminton that uses **Computer Vision** to detect gameplay events, **rule-based logic** to structure meaning, and (Phase 2+) **LLM** to generate expressive commentary with audio overlay.

## 🏗️ Architecture

```
Video Input → Frame Extraction → YOLOv8n Detection → Centroid Tracking
    → Feature Extraction → Rule-Based Event Engine → JSON Event Timeline
    → [Phase 2] LLM Commentary → [Phase 3] TTS + Video Export
```

**Key Innovation**: CV extracts structured events (smash, rally, point) as the *truth layer* — no LLM involved in detection. The LLM only adds language and emotion on top of verified facts.

## ⚡ Quick Start

```bash
# Setup
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Run Phase 1 — Event Detection
python main.py --video datasets/sample_match.mp4 --player-a "Lin Dan" --player-b "Lee Chong Wei"

# Run tests
python -m pytest tests/ -v
```

## 📊 Output Example

```json
[
  {"timestamp": 12.3, "event": "rally_start", "server": "Lin Dan", "intensity": "low"},
  {"timestamp": 18.9, "event": "smash", "by": "Lee Chong Wei", "intensity": "high", "velocity": 35.2},
  {"timestamp": 21.1, "event": "point_won", "by": "Lee Chong Wei", "rally_length": 18, "intensity": "maximum"}
]
```

## 🎯 Event Types

| Event | Detection Method |
|-------|-----------------|
| `rally_start` | Serve motion + shuttle movement |
| `smash` | High velocity + downward angle |
| `drop_shot` | Low velocity + steep angle near net |
| `long_rally` | Hit count > 15 direction reversals |
| `point_won` | Shuttle stationary for N frames |

## 📁 Project Structure

```
├── main.py                    # CLI entry point
├── config/config.yaml         # Event rules & thresholds
├── src/
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── vision/
│   │   ├── video_processor.py # Frame extraction
│   │   ├── detector.py        # YOLOv8n detection
│   │   ├── tracker.py         # Centroid tracking
│   │   └── feature_extractor.py
│   └── events/
│       ├── event_models.py    # Pydantic data models
│       └── event_engine.py    # Rule-based classification
└── tests/                     # 27 unit tests
```

## 💻 Hardware Requirements

- **CPU**: i3 or above (no GPU needed)
- **RAM**: 8GB minimum
- **Processing**: ~2-5 min for a 30s video at 10 FPS

## 🛣️ Roadmap

- ✅ **Phase 1**: Vision Foundation — CV event detection pipeline
- 📅 **Phase 2**: Commentary Layer — LLM-powered narrative generation
- 📅 **Phase 3**: Audio Integration — TTS + FFmpeg video export
- 📅 **Phase 4**: Enhancement — Better detection, Hinglish, Web UI

## 📄 License

MIT License
