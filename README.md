# Antigravity - AI Sports Commentary System 🎙️⚡

> *Defying traditional broadcasting with lightweight, AI-driven regional sports commentary*

An AI-powered cognitive sports commentary engine that transforms video into authentic, high-energy commentary in 10+ Indian regional languages and dialects.

## 🎯 Vision

Build a system that understands the "spirit" of sports in regional dialects like Bhojpuri, Tamil, Haryanvi, and more—going beyond simple translation to capture the authentic energy of local sports fans.

## 🏗️ Architecture

**Three-Layer Design:**
1. **Vision Layer** (The "Eyes") - YOLOv8n + Gemini Vision API for event detection
2. **Reasoning Layer** (The "Brain") - Gemini 1.5 Pro for narrative generation
3. **Audio Layer** (The "Voice") - Google Cloud TTS with prosody-aware synthesis

## 🏆 Supported Sports

- 🏏 Cricket
- ⚽ Football
- 🏎️ Formula 1
- 🏸 Badminton
- 🤼 Kabaddi

## 🗣️ Supported Languages

- Hinglish (60/40 Hindi-English)
- Tanglish (Tamil-English)
- Bhojpuri-English
- Haryanvi-Hindi
- ...and more!

## 💡 Personas

Choose your commentary style:
- **Professional**: Formal, technical analysis
- **Desi/Funny**: Humorous, meme-worthy, local flavor
- **Analytical**: Data-driven insights

## 🚀 Quick Start

See [Quick Start Guide](docs/quick_start.md) for detailed setup instructions.

```bash
# Clone the repo
git clone https://github.com/Shubham231005/Ai-Based_Commentary-System-for-MotorSports.git
cd Ai-Based-Commentary-System

# Set up environment
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Add your API keys to .env
cp .env.example .env

# Run Phase 1 demo
jupyter notebook notebooks/phase1_f1_demo.ipynb
```

## 📊 Hardware Requirements

- **RAM**: 8GB minimum
- **Processor**: i3 or better
- **GPU**: Not required (cloud-based processing)
- **Internet**: Required for API calls

## 💰 Cost Estimation

~$0.02-0.03 per 30-second video (Gemini + TTS APIs)

## 🛣️ Roadmap

- ✅ Phase 1: F1 English Commentary
- 🚧 Phase 2: Cricket Hinglish Commentary
- 📅 Phase 3: Multi-Sport Multi-Dialect System

## 📄 License

MIT License - See LICENSE file for details

## 🙌 Credits

Built with ❤️ using Google Gemini, Ultralytics YOLOv8, and Google Cloud TTS
