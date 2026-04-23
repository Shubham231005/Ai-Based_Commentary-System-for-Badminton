"""
Pipeline — Orchestrates the full Phase 1.5 CV → Events pipeline.

Split FPS Architecture:
  YOLO (players) runs at 10 FPS — players move slowly, saves CPU
  Optical flow (shuttle) runs at native FPS (30) — captures fast projectile
  Kalman filter bridges detection gaps for continuous trajectory

Flow: Video → Detection (split FPS) → Tracking (Kalman) → Features → Events
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .vision.video_processor import VideoProcessor
from .vision.detector import ShuttlecockDetector
from .vision.tracker import CentroidTracker
from .vision.feature_extractor import FeatureExtractor
from .vision.debug_visualizer import DebugVisualizer
from .events.event_engine import EventEngine
from .events.event_models import EventTimeline
from .debug_event_logger import DebugEventLogger
from .commentary.commentary_generator import CommentaryGenerator

logger = logging.getLogger(__name__)


class CommentaryPipeline:
    """
    Phase 1.5 Pipeline: Video → Structured Event Timeline

    Uses split FPS processing:
      YOLO at target_fps (10) for players
      Optical flow at native_fps (30) for shuttle
    """

    def __init__(
        self,
        config: Dict[str, Any],
        player_a: str = "Player A",
        player_b: str = "Player B",
        debug_mode: bool = False,
        debug_live: bool = False,
        commentary_enabled: bool = True,
    ):
        self.config = config
        self.player_a = player_a
        self.player_b = player_b
        self.debug_mode = debug_mode
        self.debug_live = debug_live
        self.commentary_enabled = commentary_enabled

        # Configuration sections
        video_cfg = config.get("video", {})
        vision_cfg = config.get("vision", {})
        tracker_cfg = config.get("tracker", {})
        court_cfg = config.get("court", {})

        self._target_fps = video_cfg.get("fps", 10)
        self._resolution = video_cfg.get("resolution", 720)
        self._max_duration = video_cfg.get("max_duration", None)
        self._log_interval = config.get("performance", {}).get("log_interval", 100)

        # Initialize components
        self.detector = ShuttlecockDetector(
            model_path=vision_cfg.get("yolo_model", "yolov8n.pt"),
            player_confidence=vision_cfg.get("player_confidence", 0.5),
            shuttle_confidence=vision_cfg.get("confidence_threshold", 0.35),
            iou_threshold=vision_cfg.get("iou_threshold", 0.45),
            device=vision_cfg.get("device", "cpu"),
            court_bounds=court_cfg,
        )

        self.tracker = CentroidTracker(
            max_disappeared=tracker_cfg.get("max_disappeared", 15),
            max_distance=tracker_cfg.get("max_distance", 80),
            court_net_y_ratio=court_cfg.get("net_y", 0.5),
        )

        self.feature_extractor = FeatureExtractor(
            smash_velocity_threshold=config.get("event_rules", {}).get("smash", {}).get("min_velocity", 12.0),
            smash_angle_threshold=config.get("event_rules", {}).get("smash", {}).get("max_angle_from_vertical", 45),
            net_y_ratio=court_cfg.get("net_y", 0.5),
            frame_height=self._resolution,
        )

        self.event_engine = EventEngine(
            config=config,
            player_a_name=player_a,
            player_b_name=player_b,
        )

        # Debug components (initialized later when frame dims are known)
        self._debug_visualizer: Optional[DebugVisualizer] = None
        self._debug_logger: Optional[DebugEventLogger] = None

        mode_str = "Split FPS + DEBUG" if debug_mode else "Split FPS"
        logger.info(
            f"Pipeline initialized: {player_a} vs {player_b} | "
            f"FPS={self._target_fps}, Resolution={self._resolution}p | "
            f"{mode_str} mode enabled"
        )

    def run(self, video_path: str, output_path: Optional[str] = None) -> EventTimeline:
        """
        Run the full pipeline on a video using split FPS processing.

        Args:
            video_path: Path to the input badminton video
            output_path: Path to save the JSON timeline (optional)

        Returns:
            EventTimeline with all detected events
        """
        logger.info(f"Starting pipeline on: {video_path}")
        start_time = time.time()

        # Load video
        video = VideoProcessor(
            video_path=video_path,
            target_fps=self._target_fps,
            target_height=self._resolution,
            max_duration=self._max_duration,
        )
        logger.info(f"Video loaded: {video.metadata}")

        # Set actual frame height for dynamic threshold computation
        actual_height = video._target_h
        actual_width = video._target_w
        self.feature_extractor.update_frame_height(actual_height)
        self.event_engine.set_frame_height(actual_height, actual_width)
        logger.info(f"Frame dims set to {actual_width}x{actual_height}px for threshold computation")

        # Initialize debug components if debug mode is on
        if self.debug_mode:
            self._debug_visualizer = DebugVisualizer(
                output_path="output/debug_output.mp4",
                fps=video.source_fps,
                show_live=self.debug_live,
                player_a_name=self.player_a,
                player_b_name=self.player_b,
            )
            self._debug_logger = DebugEventLogger(
                output_path="output/debug_log.csv",
            )
            logger.info("Debug visualizer + CSV logger initialized")

        # Process each frame with split FPS
        frame_count = 0
        yolo_frames = 0
        shuttle_detections = 0
        events_found = 0

        for frame_idx, timestamp, frame, run_yolo in video.extract_frames_split():
            # Detect objects (YOLO + optical flow, or optical flow only)
            detections = self.detector.detect(
                frame, frame_idx, timestamp, run_yolo=run_yolo
            )

            if run_yolo:
                yolo_frames += 1

            if detections.shuttle is not None:
                shuttle_detections += 1

            # Track objects with Kalman filter
            tracked = self.tracker.update(
                player_detections=detections.players,
                shuttle_detection=detections.shuttle,
                timestamp=timestamp,
                frame_height=frame.shape[0],
            )

            # Get Kalman acceleration estimates for landing detection
            kalman = self.tracker.get_shuttle_kalman()
            ax, ay = kalman.get_acceleration()

            # Extract features with Kalman acceleration
            features = self.feature_extractor.extract(
                frame_index=frame_idx,
                timestamp=timestamp,
                tracked_objects=tracked,
                kalman_ax=ax,
                kalman_ay=ay,
            )

            # Manage rally state
            if not self.feature_extractor.get_rally_state().active:
                if features.shuttle_velocity > 2.0:
                    self.feature_extractor.start_rally(
                        timestamp,
                        features.player_a_position,
                        features.player_b_position,
                    )

            # Detect events
            frame_events = self.event_engine.process_frame(features)
            events_found += len(frame_events)

            for event in frame_events:
                logger.info(
                    f"[{event.timestamp:.1f}s] {event.event.upper()} "
                    f"| by={event.by or event.server or '—'} "
                    f"| intensity={event.intensity}"
                )

            # ── Debug rendering ──
            if self.debug_mode and self._debug_visualizer and self._debug_logger:
                landing_signals = self.event_engine.get_landing_signals()
                last_attr = self.event_engine.get_last_attribution()

                self._debug_visualizer.render_frame(
                    frame=frame,
                    timestamp=timestamp,
                    detections=detections,
                    tracked_objects=tracked,
                    features=features,
                    events=frame_events,
                    score_a=self.event_engine.match_state.player_a_score,
                    score_b=self.event_engine.match_state.player_b_score,
                    landing_signals=landing_signals if landing_signals else None,
                    last_attribution=last_attr,
                )

                self._debug_logger.log_frame(
                    frame_index=frame_idx,
                    features=features,
                    events=frame_events,
                    score_a=self.event_engine.match_state.player_a_score,
                    score_b=self.event_engine.match_state.player_b_score,
                    landing_signals=landing_signals if landing_signals else None,
                    last_attribution=last_attr,
                )

            # Reset rally on point_won
            for event in frame_events:
                if event.event in ("point_won", "net_fault", "out_of_bounds"):
                    self.feature_extractor.end_rally()
                    self.tracker.clear_shuttle_trajectory()

            frame_count += 1

            # Progress logging
            if frame_count % self._log_interval == 0:
                elapsed = time.time() - start_time
                fps_rate = frame_count / elapsed if elapsed > 0 else 0
                detect_rate = shuttle_detections / frame_count * 100 if frame_count > 0 else 0
                logger.info(
                    f"Progress: {frame_count} frames ({yolo_frames} YOLO) | "
                    f"{events_found} events | "
                    f"shuttle detected {detect_rate:.0f}% | "
                    f"{fps_rate:.1f} frames/sec | "
                    f"t={timestamp:.1f}s"
                )

        # Finalize
        video.release()
        elapsed = time.time() - start_time
        timeline = self.event_engine.finalize(
            video_path=video_path,
            video_duration=video.duration,
        )

        detect_rate = shuttle_detections / frame_count * 100 if frame_count > 0 else 0
        logger.info(
            f"\n{'='*60}\n"
            f"Pipeline complete!\n"
            f"  Frames processed: {frame_count} ({yolo_frames} with YOLO)\n"
            f"  Shuttle detected: {shuttle_detections}/{frame_count} ({detect_rate:.0f}%)\n"
            f"  Events detected:  {timeline.total_events}\n"
            f"  Points scored:    {timeline.total_points}\n"
            f"  Time elapsed:     {elapsed:.1f}s\n"
            f"  Processing speed: {frame_count / elapsed:.1f} frames/sec\n"
            f"{'='*60}"
        )

        # Finalize debug outputs
        if self.debug_mode:
            if self._debug_visualizer:
                self._debug_visualizer.finalize()
            if self._debug_logger:
                self._debug_logger.finalize()

        # Save output
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            timeline.save(output_path)
            logger.info(f"Timeline saved to: {output_path}")
        else:
            default_path = Path("output") / "event_timeline.json"
            default_path.parent.mkdir(parents=True, exist_ok=True)
            timeline.save(str(default_path))
            logger.info(f"Timeline saved to: {default_path}")

        print(timeline.summary())

        # ── Phase 2: Commentary Generation ──
        commentary_cfg = self.config.get("commentary", {})
        commentary_json_path = None
        if self.commentary_enabled and commentary_cfg.get("enabled", True):
            logger.info("\n🎙️ Phase 2: Generating Hinglish commentary...")
            try:
                commentary_gen = CommentaryGenerator(self.config)
                commentary = commentary_gen.generate(timeline)
                json_path, txt_path = commentary.save("output")
                commentary_json_path = json_path
                logger.info(f"Commentary saved: {json_path}, {txt_path}")
                logger.info(f"Commentary lines: {commentary.total_lines}")
                print(f"\n🎙️ Commentary generated: {commentary.total_lines} lines")
                print(f"   → {txt_path}")
            except Exception as e:
                logger.error(f"Commentary generation failed: {e}")

        # ── Phase 3: TTS Audio + Video Overlay ──
        if commentary_json_path:
            logger.info("\n🔊 Phase 3: Generating commentary audio...")
            try:
                from .audio.tts_engine import TTSEngine

                # Read TTS config
                tts_cfg = self.config.get("tts", {})
                el_cfg = tts_cfg.get("elevenlabs", {})

                tts = TTSEngine(
                    lang="hi",
                    tts_provider=tts_cfg.get("provider", "elevenlabs"),
                    elevenlabs_voice_a_id=el_cfg.get("voice_a_id"),
                    elevenlabs_voice_b_id=el_cfg.get("voice_b_id"),
                    elevenlabs_model_id=el_cfg.get("model_id", "eleven_multilingual_v2"),
                    elevenlabs_stability=el_cfg.get("stability", 0.45),
                    elevenlabs_similarity_boost=el_cfg.get("similarity_boost", 0.80),
                    elevenlabs_style=el_cfg.get("style", 0.65),
                )

                audio_path = tts.generate_audio(commentary_json_path)
                if audio_path:
                    # Generate subtitles
                    srt_path = tts.generate_srt(commentary_json_path)

                    # Overlay audio + subtitles on video
                    final_video = tts.overlay_on_video(
                        video_path=video_path,
                        audio_path=audio_path,
                        srt_path=srt_path,
                        max_duration=self._max_duration,
                    )
                    if final_video:
                        print(f"\n🔊 Final video with commentary: {final_video}")
            except ImportError as ie:
                logger.warning(f"TTS dependency missing: {ie}. Run: pip install elevenlabs gTTS pydub")
            except Exception as e:
                logger.error(f"Audio generation failed: {e}", exc_info=True)

        return timeline
