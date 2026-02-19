"""
Video Processor — Frame extraction with split FPS support.

Two modes:
1. Standard: Extract at target_fps (default 10)
2. Split FPS: Yield ALL native frames with a run_yolo flag.
   YOLO runs every Nth frame, optical flow runs on every frame.
   This gives 30 FPS for shuttle tracking + 10 FPS for player detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Load a video and extract preprocessed frames."""

    def __init__(
        self,
        video_path: str,
        target_fps: int = 10,
        target_height: int = 720,
        max_duration: Optional[float] = None,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.target_fps = target_fps
        self.target_height = target_height
        self.max_duration = max_duration

        # Open video
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.source_fps if self.source_fps > 0 else 0

        # Compute frame skip interval for standard mode
        self._frame_interval = max(1, int(round(self.source_fps / self.target_fps)))

        # Compute YOLO interval for split mode
        # e.g., source=30fps, target=10fps → run YOLO every 3 frames
        self._yolo_interval = max(1, int(round(self.source_fps / self.target_fps)))

        # Resize
        self._scale = self.target_height / self.height if self.height > self.target_height else 1.0
        self._target_w = int(self.width * self._scale)
        self._target_h = int(self.height * self._scale)

        logger.info(
            f"Video loaded: {self.video_path.name} | "
            f"{self.width}x{self.height} @ {self.source_fps:.1f} FPS | "
            f"Duration: {self.duration:.1f}s | "
            f"YOLO every {self._yolo_interval} frames"
        )

    @property
    def metadata(self) -> dict:
        return {
            "path": str(self.video_path),
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": round(self.duration, 2),
            "resize_to": f"{self._target_w}x{self._target_h}",
        }

    def extract_frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Standard mode: yield frames at target_fps."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        extracted = 0

        max_frame = self.total_frames
        if self.max_duration and self.source_fps > 0:
            max_frame = min(max_frame, int(self.max_duration * self.source_fps))

        while frame_count < max_frame:
            ret, frame = self._cap.read()
            if not ret:
                break

            if frame_count % self._frame_interval == 0:
                timestamp = frame_count / self.source_fps if self.source_fps > 0 else 0
                processed = self._preprocess(frame)
                yield extracted, round(timestamp, 3), processed
                extracted += 1

            frame_count += 1

        logger.info(f"Extracted {extracted} frames from {frame_count} total")

    def extract_frames_split(
        self,
    ) -> Generator[Tuple[int, float, np.ndarray, bool], None, None]:
        """
        Split FPS mode: yield EVERY frame with a run_yolo flag.

        Yields: (frame_index, timestamp, frame, run_yolo)
        - run_yolo=True: Run YOLO (player detection) + optical flow (shuttle)
        - run_yolo=False: Run optical flow only (shuttle detection)

        This gives native FPS for shuttle tracking, target_fps for players.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        extracted = 0

        max_frame = self.total_frames
        if self.max_duration and self.source_fps > 0:
            max_frame = min(max_frame, int(self.max_duration * self.source_fps))

        while frame_count < max_frame:
            ret, frame = self._cap.read()
            if not ret:
                break

            timestamp = frame_count / self.source_fps if self.source_fps > 0 else 0
            processed = self._preprocess(frame)
            run_yolo = (frame_count % self._yolo_interval == 0)

            yield extracted, round(timestamp, 3), processed, run_yolo
            extracted += 1
            frame_count += 1

        logger.info(f"Split FPS: extracted {extracted} frames from {frame_count} total")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if needed."""
        if self._scale < 1.0:
            return cv2.resize(
                frame,
                (self._target_w, self._target_h),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def get_frame_at(self, timestamp: float) -> Optional[np.ndarray]:
        """Get a single frame at a specific timestamp."""
        frame_num = int(timestamp * self.source_fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self._cap.read()
        if ret:
            return self._preprocess(frame)
        return None

    def release(self):
        if self._cap:
            self._cap.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoProcessor('{self.video_path.name}', "
            f"{self.width}x{self.height} @ {self.source_fps:.0f}fps, "
            f"extract={self.target_fps}fps)"
        )
