"""
Audio Synthesiser — Converts commentary text to speech and overlays on video.

Dual Commentator System:
  • Voice A (Male)   — energetic main commentator
  • Voice B (Female) — analytical co-commentator
  Lines alternate between both voices for a natural conversational feel.

Supports two TTS providers:
  • ElevenLabs (default) — high-quality multilingual voices, needs API key
  • gTTS (fallback)     — free Google TTS, lower quality

Also generates SRT subtitles and burns them into the final video.
Clips are trimmed so they never overlap the next commentary line.
"""

import io
import os
import re
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ── Default Indian voice IDs (ElevenLabs) ──────────────────
# Change these in config.yaml → tts.elevenlabs.voice_a_id / voice_b_id
# Browse voices: https://elevenlabs.io/voice-library (filter by Hindi)
DEFAULT_VOICE_A = "pNInz6obpgDQGcFmaJgB"  # Adam — male, energetic
DEFAULT_VOICE_B = "EXAVITQu4vr4xnSDxMaL"  # Sarah — female, warm


class TTSEngine:
    """Generate dual-commentator audio, subtitles, and overlay onto video."""

    def __init__(
        self,
        lang: str = "hi",
        slow: bool = False,
        tts_provider: str = "elevenlabs",
        elevenlabs_voice_a_id: Optional[str] = None,
        elevenlabs_voice_b_id: Optional[str] = None,
        elevenlabs_model_id: str = "eleven_multilingual_v2",
        elevenlabs_stability: float = 0.45,
        elevenlabs_similarity_boost: float = 0.80,
        elevenlabs_style: float = 0.65,
        # Legacy single-voice param (ignored if voice_a/b provided)
        elevenlabs_voice_id: Optional[str] = None,
    ):
        """
        Args:
            lang: gTTS language code ('hi' for Hindi/Hinglish)
            slow: speak slowly if True (gTTS only)
            tts_provider: 'elevenlabs' or 'gtts'
            elevenlabs_voice_a_id: Male commentator voice ID
            elevenlabs_voice_b_id: Female commentator voice ID
            elevenlabs_model_id: ElevenLabs model
            elevenlabs_stability: voice stability (lower = more expressive)
            elevenlabs_similarity_boost: similarity boost (0-1)
            elevenlabs_style: emotional expressiveness (0-1, higher = more emotion)
        """
        self.lang = lang
        self.slow = slow
        self.tts_provider = tts_provider.lower()

        # Dual voices — A is male main, B is female co-commentator
        self.voice_a_id = elevenlabs_voice_a_id or elevenlabs_voice_id or DEFAULT_VOICE_A
        self.voice_b_id = elevenlabs_voice_b_id or DEFAULT_VOICE_B

        self.elevenlabs_model_id = elevenlabs_model_id
        self.elevenlabs_stability = elevenlabs_stability
        self.elevenlabs_similarity_boost = elevenlabs_similarity_boost
        self.elevenlabs_style = elevenlabs_style

        # Validate ElevenLabs availability
        if self.tts_provider == "elevenlabs":
            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            if not api_key:
                logger.warning("ELEVENLABS_API_KEY not set — falling back to gTTS")
                self.tts_provider = "gtts"
            else:
                try:
                    import elevenlabs  # noqa: F401
                    logger.info(
                        f"ElevenLabs dual-commentator ready: "
                        f"Voice A={self.voice_a_id[:8]}… Voice B={self.voice_b_id[:8]}…"
                    )
                except ImportError:
                    logger.warning(
                        "elevenlabs package not installed — falling back to gTTS. "
                        "Install with: pip install elevenlabs"
                    )
                    self.tts_provider = "gtts"

    # ── Main entry point ───────────────────────────────────────

    def generate_audio(
        self,
        commentary_json_path: str,
        output_audio_path: str = "output/commentary_audio.mp3",
    ) -> str:
        """
        Read commentary.json → generate TTS for each line (alternating
        male/female voices) → stitch at timestamps with anti-overlap trimming.

        Returns path to the final commentary audio file.
        """
        from pydub import AudioSegment

        with open(commentary_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = data.get("lines", [])
        if not lines:
            logger.warning("No commentary lines found, skipping audio generation")
            return ""

        # Calculate total duration needed
        last_ts = max(l["timestamp"] for l in lines)
        total_duration_ms = int((last_ts + 15) * 1000)

        # Create silent base track
        base = AudioSegment.silent(duration=total_duration_ms)

        logger.info(
            f"Generating dual-commentator TTS ({self.tts_provider}) "
            f"for {len(lines)} lines..."
        )

        # Generate all clips — alternate voices
        clips: List[Tuple[int, AudioSegment]] = []
        for i, line in enumerate(lines):
            text = line["text"]
            ts = line["timestamp"]
            position_ms = int(ts * 1000)

            clean_text = self._strip_emojis(text)
            if not clean_text.strip():
                continue

            # Alternate: even lines = Voice A (male), odd = Voice B (female)
            use_voice_b = (i % 2 == 1)
            clip = self._synthesise_line(clean_text, i, len(lines), use_voice_b)
            if clip is not None:
                clips.append((position_ms, clip))

        # Trim clips so they don't overlap the next one
        for idx in range(len(clips)):
            pos, clip = clips[idx]
            if idx + 1 < len(clips):
                next_pos = clips[idx + 1][0]
                max_len_ms = max(next_pos - pos - 200, 500)  # 200ms gap
                if len(clip) > max_len_ms:
                    clip = clip[:max_len_ms].fade_out(150)
                    clips[idx] = (pos, clip)

        # Overlay trimmed clips onto base
        for pos, clip in clips:
            base = base.overlay(clip, position=pos)

        # Export
        out_path = Path(output_audio_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        base.export(str(out_path), format="mp3")
        logger.info(f"Commentary audio saved: {out_path}")
        return str(out_path)

    # ── Subtitle generation ────────────────────────────────────

    def generate_srt(
        self,
        commentary_json_path: str,
        output_srt_path: str = "output/commentary.srt",
    ) -> str:
        """Generate an SRT subtitle file from commentary.json with speaker tags."""
        with open(commentary_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = data.get("lines", [])
        if not lines:
            return ""

        srt_lines = []
        for i, line in enumerate(lines):
            ts = line["timestamp"]
            text = line["text"]

            # Tag with speaker
            speaker = "🎙️" if i % 2 == 0 else "🎤"

            # Each subtitle shows for up to 4 seconds or until next line
            start = ts
            if i + 1 < len(lines):
                end = min(ts + 4.0, lines[i + 1]["timestamp"])
            else:
                end = ts + 4.0

            srt_lines.append(
                f"{i + 1}\n"
                f"{self._format_srt_time(start)} --> {self._format_srt_time(end)}\n"
                f"{speaker} {text}\n"
            )

        out_path = Path(output_srt_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        logger.info(f"Subtitles saved: {out_path}")
        return str(out_path)

    # ── Video overlay with subtitles ───────────────────────────

    def overlay_on_video(
        self,
        video_path: str,
        audio_path: str,
        srt_path: Optional[str] = None,
        output_path: str = "output/final_commentary_video.mp4",
        max_duration: Optional[float] = None,
    ) -> str:
        """
        Mux commentary audio onto video using ffmpeg.
        Replaces original audio with commentary (no mixing).
        Optionally burns in subtitles and trims to max_duration.
        """
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found in PATH. Install ffmpeg to merge audio+video.")
            return ""

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Build filter graph
        filters = []
        if srt_path and Path(srt_path).exists():
            # Escape path for ffmpeg filter
            safe_srt = str(Path(srt_path).resolve()).replace("\\", "/")
            safe_srt = safe_srt.replace(":", "\\:")
            filters.append(
                f"subtitles='{safe_srt}':force_style="
                f"'FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
                f"OutlineColour=&H00000000,Outline=2,Shadow=1,"
                f"MarginV=30,Alignment=2'"
            )

        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path]

        if filters:
            cmd += ["-vf", ",".join(filters)]
            cmd += ["-map", "0:v", "-map", "1:a"]
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy"]

        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

        if max_duration:
            cmd += ["-t", str(max_duration)]

        cmd.append(str(out))

        logger.info("Muxing commentary audio + subtitles onto video...")
        logger.debug(f"ffmpeg cmd: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"ffmpeg error:\n{result.stderr[-500:]}")
                # Fallback: try without subtitles
                if filters:
                    logger.info("Retrying without subtitles...")
                    cmd_simple = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-i", audio_path,
                        "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest",
                    ]
                    if max_duration:
                        cmd_simple += ["-t", str(max_duration)]
                    cmd_simple.append(str(out))
                    result = subprocess.run(
                        cmd_simple, capture_output=True, text=True, timeout=300
                    )
                    if result.returncode != 0:
                        logger.error(f"ffmpeg fallback error:\n{result.stderr[-300:]}")
                        return ""
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out")
            return ""

        logger.info(f"Final video with commentary: {out}")
        return str(out)

    # ── Private: synthesise a single line ──────────────────────

    def _synthesise_line(
        self, text: str, index: int, total: int, use_voice_b: bool = False
    ) -> Optional["AudioSegment"]:
        """Generate audio for one commentary line, using voice A or B."""
        from pydub import AudioSegment

        voice_label = "Voice B (female)" if use_voice_b else "Voice A (male)"

        if self.tts_provider == "elevenlabs":
            voice_id = self.voice_b_id if use_voice_b else self.voice_a_id
            clip = self._synthesise_elevenlabs(text, index, total, voice_id, voice_label)
            if clip is not None:
                return clip
            logger.warning(f"  ElevenLabs failed for line {index+1}, trying gTTS...")

        return self._synthesise_gtts(text, index, total)

    def _synthesise_elevenlabs(
        self, text: str, index: int, total: int,
        voice_id: str, voice_label: str = ""
    ) -> Optional["AudioSegment"]:
        """Generate audio using ElevenLabs API with emotional settings."""
        from pydub import AudioSegment

        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings

            api_key = os.getenv("ELEVENLABS_API_KEY", "")
            client = ElevenLabs(api_key=api_key)

            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=self.elevenlabs_model_id,
                voice_settings=VoiceSettings(
                    stability=self.elevenlabs_stability,
                    similarity_boost=self.elevenlabs_similarity_boost,
                    style=self.elevenlabs_style,
                    use_speaker_boost=True,
                ),
            )

            # Collect bytes from generator
            audio_bytes = b"".join(audio_generator)

            buf = io.BytesIO(audio_bytes)
            clip = AudioSegment.from_mp3(buf)
            logger.debug(
                f"  [{index+1}/{total}] {voice_label} ElevenLabs OK — {len(clip)}ms"
            )
            return clip

        except Exception as e:
            logger.warning(f"  ElevenLabs error for line {index+1} ({voice_label}): {e}")
            return None

    def _synthesise_gtts(
        self, text: str, index: int, total: int
    ) -> Optional["AudioSegment"]:
        """Generate audio using gTTS (free fallback)."""
        from pydub import AudioSegment

        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            clip = AudioSegment.from_mp3(buf)
            logger.debug(f"  [{index+1}/{total}] gTTS OK — {len(clip)}ms")
            return clip
        except Exception as e:
            logger.warning(f"  gTTS failed for line {index+1}: {e}")
            return None

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _strip_emojis(text: str) -> str:
        """Remove emoji characters that TTS engines can't handle."""
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002600-\U000026FF"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
