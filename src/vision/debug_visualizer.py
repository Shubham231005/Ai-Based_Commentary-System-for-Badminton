"""
Debug Visualizer — Annotated frame renderer for understanding rule triggers.

Overlays detection results, shuttle phase, rule evaluations, scoring decisions,
and timestamps onto video frames. Outputs an annotated debug video + optional
live cv2.imshow window.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ── Color palette ──────────────────────────────────────────────
COLORS = {
    "player_a":      (0, 255, 100),    # green
    "player_b":      (100, 255, 0),    # lime
    "shuttle":       (255, 255, 0),    # cyan (BGR)
    "shuttle_pred":  (0, 255, 255),    # yellow (predicted)
    "trajectory":    (255, 200, 0),    # cyan trail
    "text":          (255, 255, 255),  # white
    "text_bg":       (30, 30, 30),     # dark bg
    "score_bg":      (40, 20, 60),     # purple-ish
    "event_flash":   (0, 100, 255),    # orange flash
    "hud_bg":        (0, 0, 0),        # black
}

PHASE_COLORS = {
    "idle":     (128, 128, 128),  # gray
    "hit":      (0, 100, 255),    # orange
    "ascent":   (0, 255, 255),    # yellow
    "apex":     (0, 255, 0),      # green
    "descent":  (0, 0, 255),      # red
    "landing":  (255, 0, 255),    # magenta
}

EVENT_COLORS = {
    "rally_start":    (0, 200, 0),
    "smash":          (0, 0, 255),
    "drop_shot":      (255, 150, 0),
    "long_rally":     (0, 255, 255),
    "point_won":      (0, 255, 0),
    "out_of_bounds":  (0, 100, 255),
    "net_fault":      (0, 0, 255),
}


class DebugVisualizer:
    """
    Renders annotated debug frames showing YOLO detections, tracking state,
    phase transitions, rule evaluations, and score attribution.
    """

    def __init__(
        self,
        output_path: str = "output/debug_output.mp4",
        fps: float = 30.0,
        show_live: bool = False,
        player_a_name: str = "Player A",
        player_b_name: str = "Player B",
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.show_live = show_live
        self.player_a_name = player_a_name
        self.player_b_name = player_b_name

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[Tuple[int, int]] = None

        # Event flash state: list of (event_name, frames_remaining, extra_text)
        self._event_flashes: List[List[Any]] = []
        self._flash_duration = 45  # frames to show the event flash

        # Attribution flash
        self._attribution_flash: Optional[Dict] = None
        self._attribution_frames_left = 0

        self._frame_count = 0

    def _init_writer(self, width: int, height: int):
        """Initialize the video writer on first frame."""
        self._frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps,
            (width, height), True,
        )
        logger.info(f"Debug video writer initialized: {width}x{height} @ {self.fps} FPS → {self.output_path}")

    def render_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        detections: Any,
        tracked_objects: Dict,
        features: Any,
        events: List[Any],
        score_a: int = 0,
        score_b: int = 0,
        landing_signals: Optional[List] = None,
        last_attribution: Optional[Dict] = None,
    ):
        """
        Render a single annotated debug frame.

        Args:
            frame: Original BGR frame
            timestamp: Current video time in seconds
            detections: FrameDetections from detector
            tracked_objects: Dict from tracker
            features: FrameFeatures from feature extractor
            events: List of Event objects emitted this frame
            score_a: Player A score
            score_b: Player B score
            landing_signals: Active landing detector signals (if any)
            last_attribution: Latest winner attribution result
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        if self._writer is None:
            self._init_writer(w, h)

        # Register new event flashes
        for evt in events:
            evt_name = evt.event if isinstance(evt.event, str) else evt.event.value
            extra = ""
            if hasattr(evt, "by") and evt.by:
                extra = f" by {evt.by}"
            elif hasattr(evt, "server") and evt.server:
                extra = f" server: {evt.server}"
            self._event_flashes.append([evt_name, self._flash_duration, extra])

            # Check for attribution
            if evt_name in ("point_won", "net_fault", "out_of_bounds") and evt.metadata:
                self._attribution_flash = {
                    "winner": getattr(evt, "by", "?"),
                    "reason": evt.metadata.get("attribution_reason", "?"),
                    "confidence": evt.metadata.get("attribution_confidence", 0),
                    "landing_type": evt.metadata.get("landing_type", "?"),
                    "landing_conf": evt.metadata.get("landing_confidence", 0),
                }
                self._attribution_frames_left = 60

        # ── 1. Draw player bounding boxes ──
        self._draw_players(canvas, detections, tracked_objects)

        # ── 2. Draw shuttle + trajectory ──
        self._draw_shuttle(canvas, tracked_objects, features)

        # ── 3. Draw HUD: timestamp + phase + velocity ──
        self._draw_hud(canvas, timestamp, features, w, h)

        # ── 4. Draw score ──
        self._draw_score(canvas, score_a, score_b, w)

        # ── 5. Draw landing signals (during descent) ──
        if landing_signals and features.shuttle_phase == "descent":
            self._draw_landing_signals(canvas, landing_signals, h)

        # ── 6. Draw event flashes ──
        self._draw_event_flashes(canvas, w, h)

        # ── 7. Draw attribution flash ──
        self._draw_attribution(canvas, w, h)

        # ── 8. Draw dynamics panel (velocity, decay, accel) ──
        self._draw_dynamics_panel(canvas, features, w, h)

        # Write frame
        self._writer.write(canvas)
        self._frame_count += 1

        # Live display
        if self.show_live:
            cv2.imshow("Antigravity Debug", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.show_live = False
                cv2.destroyAllWindows()

    # ── Drawing helpers ────────────────────────────────────────

    def _draw_players(self, canvas: np.ndarray, detections: Any, tracked: Dict):
        """Draw bounding boxes around detected players."""
        for key, color in [("player_a", COLORS["player_a"]), ("player_b", COLORS["player_b"])]:
            obj = tracked.get(key)
            if obj is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = self.player_a_name if key == "player_a" else self.player_b_name
            self._put_text_bg(canvas, label, (x1, y1 - 8), color, scale=0.5)

    def _draw_shuttle(self, canvas: np.ndarray, tracked: Dict, features: Any):
        """Draw shuttle position, prediction marker, and trajectory trail."""
        shuttle = tracked.get("shuttle")

        # Draw trajectory trail
        traj = getattr(features, "trajectory_buffer", [])
        if len(traj) >= 2:
            pts = [(int(p[0]), int(p[1])) for p in traj]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                color = tuple(int(c * alpha) for c in COLORS["trajectory"])
                cv2.line(canvas, pts[i - 1], pts[i], color, thickness)

        if shuttle is None:
            return

        cx, cy = int(shuttle.centroid[0]), int(shuttle.centroid[1])
        is_pred = getattr(shuttle, "is_predicted", False) or getattr(features, "is_predicted", False)

        if is_pred:
            # Predicted: dashed yellow circle
            cv2.circle(canvas, (cx, cy), 12, COLORS["shuttle_pred"], 2)
            cv2.circle(canvas, (cx, cy), 4, COLORS["shuttle_pred"], -1)
            self._put_text_bg(canvas, "PREDICTED", (cx + 15, cy - 5), COLORS["shuttle_pred"], scale=0.4)
        else:
            # Detected: solid cyan circle
            cv2.circle(canvas, (cx, cy), 10, COLORS["shuttle"], 2)
            cv2.circle(canvas, (cx, cy), 3, COLORS["shuttle"], -1)

        # Velocity arrow
        vel = features.shuttle_velocity
        if vel > 2.0 and features.shuttle_position:
            vx = features.shuttle_vx * 0.3
            vy = features.shuttle_vy * 0.3
            end_x = int(cx + vx)
            end_y = int(cy + vy)
            cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)

    def _draw_hud(self, canvas: np.ndarray, timestamp: float, features: Any, w: int, h: int):
        """Draw top-left HUD with timestamp, phase, and frame info."""
        # Background bar
        cv2.rectangle(canvas, (0, 0), (w, 40), COLORS["hud_bg"], -1)
        cv2.rectangle(canvas, (0, 0), (w, 40), (60, 60, 60), 1)

        # Timestamp (center)
        mins = int(timestamp // 60)
        secs = timestamp % 60
        ts_text = f"{mins:02d}:{secs:05.2f}"
        ts_w = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(canvas, ts_text, ((w - ts_w) // 2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2)

        # Phase badge (left)
        phase = features.shuttle_phase
        phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
        badge_text = f" {phase.upper()} "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (8, 6), (16 + tw, 10 + th + 10), phase_color, -1)
        cv2.putText(canvas, badge_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Rally info (right of phase)
        if features.rally_active:
            rally_text = f"Rally: {features.rally_hit_count} hits | {features.rally_duration:.1f}s"
            cv2.putText(canvas, rally_text, (26 + tw, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _draw_score(self, canvas: np.ndarray, score_a: int, score_b: int, w: int):
        """Draw score in top-right corner."""
        score_text = f"{self.player_a_name}: {score_a}  |  {self.player_b_name}: {score_b}"
        (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        x = w - tw - 15
        cv2.rectangle(canvas, (x - 8, 44), (w, 44 + th + 14), COLORS["score_bg"], -1)
        cv2.putText(canvas, score_text, (x, 44 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1)

    def _draw_landing_signals(self, canvas: np.ndarray, signals: List, h: int):
        """Draw active landing detector signals."""
        y_start = h - 120
        self._put_text_bg(canvas, "LANDING SIGNALS:", (10, y_start), (0, 150, 255), scale=0.45)
        for i, (name, weight, conf) in enumerate(signals):
            bar_w = int(conf * 120)
            y = y_start + 18 + i * 20
            cv2.rectangle(canvas, (10, y), (10 + bar_w, y + 14), (0, int(255 * conf), 0), -1)
            cv2.rectangle(canvas, (10, y), (130, y + 14), (100, 100, 100), 1)
            text = f"{name}: {weight * conf:.2f}"
            cv2.putText(canvas, text, (140, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_event_flashes(self, canvas: np.ndarray, w: int, h: int):
        """Draw event trigger flashes at the bottom."""
        active = []
        y_base = h - 35
        x = 10
        for flash in self._event_flashes:
            name, frames_left, extra = flash
            if frames_left <= 0:
                continue

            alpha = min(frames_left / 15.0, 1.0)
            color = EVENT_COLORS.get(name, (255, 255, 255))
            color = tuple(int(c * alpha) for c in color)

            text = f">> {name.upper()}{extra}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

            # Flash background
            cv2.rectangle(canvas, (x - 4, y_base - th - 6), (x + tw + 4, y_base + 6), (0, 0, 0), -1)
            cv2.rectangle(canvas, (x - 4, y_base - th - 6), (x + tw + 4, y_base + 6), color, 2)
            cv2.putText(canvas, text, (x, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            flash[1] -= 1
            x += tw + 20
            active.append(flash)

        self._event_flashes = [f for f in self._event_flashes if f[1] > 0]

    def _draw_attribution(self, canvas: np.ndarray, w: int, h: int):
        """Draw winner attribution flash (large center overlay)."""
        if self._attribution_flash is None or self._attribution_frames_left <= 0:
            self._attribution_flash = None
            return

        attr = self._attribution_flash
        alpha = min(self._attribution_frames_left / 20.0, 1.0)

        # Semi-transparent overlay
        overlay = canvas.copy()
        cx, cy = w // 2, h // 2
        box_w, box_h = 380, 120
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 60), -1)
        cv2.addWeighted(overlay, alpha * 0.85, canvas, 1 - alpha * 0.85, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Text
        winner_text = f"POINT: {attr['winner']}"
        cv2.putText(canvas, winner_text, (x1 + 15, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        reason_text = f"Reason: {attr['reason']} | Type: {attr['landing_type']}"
        cv2.putText(canvas, reason_text, (x1 + 15, y1 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        conf_text = f"Attribution: {attr['confidence']:.0%} | Landing: {attr['landing_conf']:.0%}"
        cv2.putText(canvas, conf_text, (x1 + 15, y1 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        self._attribution_frames_left -= 1

    def _draw_dynamics_panel(self, canvas: np.ndarray, features: Any, w: int, h: int):
        """Draw velocity/acceleration dynamics panel on the right side."""
        panel_x = w - 220
        panel_y = 70
        panel_h = 100

        cv2.rectangle(canvas, (panel_x, panel_y), (w, panel_y + panel_h), (20, 20, 20), -1)
        cv2.rectangle(canvas, (panel_x, panel_y), (w, panel_y + panel_h), (60, 60, 60), 1)

        lines = [
            f"Vel:   {features.shuttle_velocity:6.1f} px/f",
            f"Decay: {features.velocity_decay_rate:6.2f}",
            f"VY:    {features.shuttle_vy:6.1f}",
            f"Phase: {features.shuttle_phase}",
        ]

        for i, line in enumerate(lines):
            color = (180, 180, 180)
            if "Decay" in line and features.velocity_decay_rate < -1.5:
                color = (0, 100, 255)  # orange warning for deceleration
            cv2.putText(canvas, line, (panel_x + 8, panel_y + 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _put_text_bg(self, canvas, text, pos, color, scale=0.5, thickness=1):
        """Put text with a dark background for readability."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = pos
        cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 2, y + 4), COLORS["text_bg"], -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def finalize(self):
        """Release video writer and close windows."""
        if self._writer:
            self._writer.release()
            logger.info(f"Debug video saved: {self.output_path} ({self._frame_count} frames)")
        if self.show_live:
            cv2.destroyAllWindows()
