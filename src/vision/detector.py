"""
Object Detector — YOLOv8n players + Optical Flow shuttle detection.

Architecture:
- Players: YOLOv8n 'person' class (runs at 10 FPS)
- Shuttlecock: Dense optical flow (runs at 30 FPS native)
  Uses cv2.calcOpticalFlowFarneback to find fast-moving small objects
  that are NOT inside player bounding boxes. This works even during
  motion blur because optical flow captures velocity vectors directly.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ultralytics import YOLO
import math
import logging

logger = logging.getLogger(__name__)

PERSON_CLASS = 0


@dataclass
class Detection:
    """A single detected object in a frame."""
    bbox: List[float]          # [x1, y1, x2, y2] in pixels
    center: List[float]        # [cx, cy] center point
    class_id: int              # 0=person, -1=shuttle (flow-detected)
    confidence: float          # Detection confidence
    label: str                 # 'player' or 'shuttle'
    area: float = 0.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_index: int
    timestamp: float
    players: List[Detection] = field(default_factory=list)
    shuttle: Optional[Detection] = None
    raw_detections: List[Detection] = field(default_factory=list)


class ShuttlecockDetector:
    """
    Hybrid detector: YOLO for players + optical flow for shuttle.

    Dense optical flow captures velocity vectors for every pixel.
    We extract the fastest-moving small region that isn't a player
    — that's the shuttlecock.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        player_confidence: float = 0.5,
        shuttle_confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        court_bounds: Optional[dict] = None,
    ):
        self.player_confidence = player_confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.court_bounds = court_bounds or {
            "left": 0.08, "right": 0.92,
            "top": 0.05, "bottom": 0.95,
        }

        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path} (device={device})")
        self.model = YOLO(model_path)

        # Optical flow state
        self._prev_gray: Optional[np.ndarray] = None

        # Flow-based shuttle detection params (tuned for 640x360 badminton)
        self._flow_magnitude_threshold = 3.0    # Minimum flow magnitude to consider
        self._min_shuttle_area = 3              # min contour area in px²
        self._max_shuttle_area = 600            # max contour area
        self._max_shuttle_dim = 35              # max width or height of bounding rect
        self._player_margin = 40               # pixels to expand player bbox exclusion

        # Cache last known player bboxes for motion-only frames
        self._last_player_bboxes: List[List[float]] = []
        self._last_players: List[Detection] = []

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
        run_yolo: bool = True,
    ) -> FrameDetections:
        """
        Run detection on a single frame.

        Args:
            frame: BGR frame
            frame_index: Frame number
            timestamp: Time in seconds
            run_yolo: If True, run YOLO for player detection.
                      If False, only run optical flow for shuttle (split FPS mode).
        """
        h, w = frame.shape[:2]
        result = FrameDetections(frame_index=frame_index, timestamp=timestamp)

        # --- Player detection via YOLO (only on YOLO frames) ---
        player_bboxes = []
        if run_yolo:
            predictions = self.model.predict(
                frame,
                conf=self.player_confidence,
                iou=self.iou_threshold,
                device=self.device,
                classes=[PERSON_CLASS],
                verbose=False,
            )

            if predictions and len(predictions) > 0:
                boxes = predictions[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)

                        if not self._in_court(cx, cy, w, h):
                            continue

                        det = Detection(
                            bbox=[x1, y1, x2, y2],
                            center=[cx, cy],
                            class_id=PERSON_CLASS,
                            confidence=conf,
                            label="player",
                            area=area,
                        )
                        result.raw_detections.append(det)
                        result.players.append(det)

            # Keep top 2 players by confidence (singles)
            result.players.sort(key=lambda d: d.confidence, reverse=True)
            result.players = result.players[:2]
            player_bboxes = [p.bbox for p in result.players]

            # Cache for motion-only frames
            self._last_player_bboxes = player_bboxes
            self._last_players = list(result.players)
        else:
            # Reuse last known player positions
            result.players = list(self._last_players)
            player_bboxes = self._last_player_bboxes

        # --- Shuttle detection via optical flow ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shuttle = self._detect_shuttle_optical_flow(gray, w, h, player_bboxes)
        if shuttle is not None:
            result.shuttle = shuttle
            result.raw_detections.append(shuttle)

        self._prev_gray = gray

        return result

    def _detect_shuttle_optical_flow(
        self,
        gray: np.ndarray,
        w: int, h: int,
        player_bboxes: List[List[float]],
    ) -> Optional[Detection]:
        """
        Detect shuttle using Farneback dense optical flow.

        Dense flow gives a velocity vector (vx, vy) for every pixel.
        We threshold by magnitude to find fast-moving regions, then
        filter for small contours outside player bounding boxes.
        """
        if self._prev_gray is None:
            return None

        # Compute dense optical flow — ~10-15ms on 640x360 i3
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold: only keep fast-moving pixels
        fast_mask = (mag > self._flow_magnitude_threshold).astype(np.uint8) * 255

        # Create player exclusion mask
        exclusion_mask = np.zeros((h, w), dtype=np.uint8)
        m = self._player_margin
        for bbox in player_bboxes:
            x1, y1, x2, y2 = bbox
            rx1 = max(0, int(x1 - m))
            ry1 = max(0, int(y1 - m))
            rx2 = min(w, int(x2 + m))
            ry2 = min(h, int(y2 + m))
            exclusion_mask[ry1:ry2, rx1:rx2] = 255

        # Subtract player motion from fast-moving mask
        shuttle_mask = cv2.bitwise_and(fast_mask, cv2.bitwise_not(exclusion_mask))

        # Morphological cleanup — remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shuttle_mask = cv2.morphologyEx(shuttle_mask, cv2.MORPH_OPEN, kernel)
        shuttle_mask = cv2.dilate(shuttle_mask, kernel, iterations=1)

        # Find contours in the shuttle mask
        contours, _ = cv2.findContours(
            shuttle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_shuttle_area or area > self._max_shuttle_area:
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if cw > self._max_shuttle_dim or ch > self._max_shuttle_dim:
                continue

            cx = x + cw / 2.0
            cy = y + ch / 2.0

            # Must be in court
            if not self._in_court(cx, cy, w, h):
                continue

            # Get average flow magnitude in this region (= speed of object)
            roi_mag = mag[y:y+ch, x:x+cw]
            avg_magnitude = float(np.mean(roi_mag))
            max_magnitude = float(np.max(roi_mag))

            # Get average flow direction
            roi_flow = flow[y:y+ch, x:x+cw]
            avg_vx = float(np.mean(roi_flow[..., 0]))
            avg_vy = float(np.mean(roi_flow[..., 1]))

            # Score: prefer fast, small, central objects
            speed_score = min(avg_magnitude / 20.0, 1.0)  # normalize to ~20px/frame max
            size_score = 1.0 - min(area / self._max_shuttle_area, 1.0)
            centrality = 1.0 - abs(cx / w - 0.5) * 2  # prefer center-court

            # Bonus for consistent motion direction (shuttle moves in one dir)
            direction_consistency = min(
                math.sqrt(avg_vx**2 + avg_vy**2) / (avg_magnitude + 0.001), 1.0
            )

            score = (
                speed_score * 0.45
                + size_score * 0.20
                + centrality * 0.15
                + direction_consistency * 0.20
            )

            candidates.append({
                "score": score,
                "cx": cx, "cy": cy,
                "x": x, "y": y, "w": cw, "h": ch,
                "area": area,
                "magnitude": avg_magnitude,
                "vx": avg_vx, "vy": avg_vy,
            })

        if not candidates:
            return None

        # Pick best candidate
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        return Detection(
            bbox=[best["x"], best["y"], best["x"] + best["w"], best["y"] + best["h"]],
            center=[best["cx"], best["cy"]],
            class_id=-1,
            confidence=min(best["score"], 0.99),
            label="shuttle",
            area=best["area"],
        )

    def _in_court(self, cx: float, cy: float, w: int, h: int) -> bool:
        """Check if a point is within the court boundary."""
        rel_x = cx / w
        rel_y = cy / h
        return (
            self.court_bounds["left"] <= rel_x <= self.court_bounds["right"]
            and self.court_bounds["top"] <= rel_y <= self.court_bounds["bottom"]
        )
