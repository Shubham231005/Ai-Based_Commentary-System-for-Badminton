"""
Feature Extractor — Computes gameplay features from tracked objects.

Takes tracked player and shuttle positions across frames and derives:
- Shuttle velocity & trajectory angle
- Rally hit count (direction reversals)
- Rally duration
- Player displacement
- Smash indicators
- Point end detection
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class RallyState:
    """State of the current rally being tracked."""
    active: bool = False
    start_time: float = 0.0
    hit_count: int = 0
    last_direction: Optional[str] = None  # 'up' or 'down'
    player_a_start_pos: Optional[Tuple[float, float]] = None
    player_b_start_pos: Optional[Tuple[float, float]] = None
    player_a_displacement: float = 0.0
    player_b_displacement: float = 0.0
    peak_velocity: float = 0.0
    shuttle_positions: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class FrameFeatures:
    """Extracted features for a single frame."""
    frame_index: int
    timestamp: float

    # Shuttle movement
    shuttle_velocity: float = 0.0          # pixels per frame
    shuttle_angle: float = 0.0             # degrees from horizontal
    shuttle_direction: Optional[str] = None  # 'up', 'down', or None
    shuttle_position: Optional[Tuple[float, float]] = None

    # Rally state
    rally_active: bool = False
    rally_duration: float = 0.0
    rally_hit_count: int = 0

    # Player positions
    player_a_position: Optional[Tuple[float, float]] = None
    player_b_position: Optional[Tuple[float, float]] = None
    player_a_displacement: float = 0.0
    player_b_displacement: float = 0.0

    # Indicators
    is_smash: bool = False
    is_point_end: bool = False
    is_direction_reversal: bool = False
    shuttle_stationary_frames: int = 0

    # Raw velocity components
    shuttle_vx: float = 0.0
    shuttle_vy: float = 0.0


class FeatureExtractor:
    """
    Computes game-play features from tracked objects across frames.

    Maintains internal rally state and produces per-frame features
    used by the event engine for classification.
    """

    def __init__(
        self,
        smash_velocity_threshold: float = 15.0,     # Lowered for 360p
        smash_angle_threshold: float = 45.0,
        stationary_threshold: float = 2.0,           # Lowered for motion-detected shuttle
        direction_reversal_threshold: float = 3.0,   # Lowered for 360p
        net_y_ratio: float = 0.5,
        frame_height: int = 360,
    ):
        self.smash_velocity_threshold = smash_velocity_threshold
        self.smash_angle_threshold = smash_angle_threshold
        self.stationary_threshold = stationary_threshold
        self.direction_reversal_threshold = direction_reversal_threshold
        self.net_y = net_y_ratio * frame_height
        self.frame_height = frame_height

        # Internal state
        self._rally = RallyState()
        self._prev_shuttle_pos: Optional[Tuple[float, float]] = None
        self._prev_timestamp: float = 0.0
        self._stationary_count: int = 0
        self._no_shuttle_count: int = 0
        self._prev_player_a: Optional[Tuple[float, float]] = None
        self._prev_player_b: Optional[Tuple[float, float]] = None

        # History for smoothing
        self._velocity_history: List[float] = []
        self._max_velocity_history = 3  # Shorter window for more responsive detection

    def update_frame_height(self, height: int):
        """Dynamically update frame height (in case video res differs from config)."""
        self.frame_height = height
        self.net_y = 0.5 * height

    def extract(
        self,
        frame_index: int,
        timestamp: float,
        tracked_objects: Dict[str, object],
    ) -> FrameFeatures:
        """
        Extract features from a single frame's tracked objects.

        Args:
            frame_index: Index of the current frame
            timestamp: Timestamp in seconds
            tracked_objects: Dict with 'player_a', 'player_b', 'shuttle' TrackedObjects

        Returns:
            FrameFeatures for this frame
        """
        features = FrameFeatures(frame_index=frame_index, timestamp=timestamp)

        # Get positions
        player_a = tracked_objects.get("player_a")
        player_b = tracked_objects.get("player_b")
        shuttle = tracked_objects.get("shuttle")

        if player_a is not None:
            features.player_a_position = player_a.centroid
        if player_b is not None:
            features.player_b_position = player_b.centroid

        # --- Shuttle velocity & direction ---
        if shuttle is not None:
            sx, sy = shuttle.centroid
            features.shuttle_position = (sx, sy)
            self._no_shuttle_count = 0

            if self._prev_shuttle_pos is not None:
                dt = timestamp - self._prev_timestamp
                if dt > 0:
                    dx = sx - self._prev_shuttle_pos[0]
                    dy = sy - self._prev_shuttle_pos[1]
                    velocity = math.sqrt(dx**2 + dy**2)

                    # Smooth velocity with history
                    self._velocity_history.append(velocity)
                    if len(self._velocity_history) > self._max_velocity_history:
                        self._velocity_history.pop(0)
                    smoothed_velocity = np.mean(self._velocity_history)

                    features.shuttle_velocity = smoothed_velocity
                    features.shuttle_vx = dx / dt
                    features.shuttle_vy = dy / dt

                    # Angle from horizontal (0° = horizontal, 90° = vertical)
                    if abs(dx) > 0.001:
                        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                    else:
                        angle = 90.0
                    features.shuttle_angle = angle

                    # Direction: based on Y movement (down = toward near court)
                    if abs(dy) > self.direction_reversal_threshold:
                        features.shuttle_direction = "down" if dy > 0 else "up"

                    # Smash detection — lowered thresholds for 360p
                    if (
                        smoothed_velocity > self.smash_velocity_threshold
                        and dy > 0  # Moving downward
                        and angle > (90 - self.smash_angle_threshold)
                    ):
                        features.is_smash = True

                    # Direction reversal (rally hit counting)
                    if features.shuttle_direction is not None:
                        if (
                            self._rally.last_direction is not None
                            and features.shuttle_direction != self._rally.last_direction
                        ):
                            features.is_direction_reversal = True
                            self._rally.hit_count += 1
                        self._rally.last_direction = features.shuttle_direction

                    # Stationary detection
                    if velocity < self.stationary_threshold:
                        self._stationary_count += 1
                    else:
                        self._stationary_count = 0

                    features.shuttle_stationary_frames = self._stationary_count

            self._prev_shuttle_pos = (sx, sy)
            self._prev_timestamp = timestamp

            # Update rally shuttle positions
            self._rally.shuttle_positions.append((sx, sy, timestamp))
            if features.shuttle_velocity > self._rally.peak_velocity:
                self._rally.peak_velocity = features.shuttle_velocity

        else:
            # No shuttle detected — track how long it's been missing
            # Only count as "stationary" if shuttle has been missing for a while
            # This prevents false point_won from momentary detection gaps
            self._no_shuttle_count += 1
            if self._no_shuttle_count >= 5:  # Only after 5+ frames of no shuttle
                self._stationary_count += 1
            features.shuttle_stationary_frames = self._stationary_count

        # --- Player displacement tracking ---
        if features.player_a_position is not None:
            if self._prev_player_a is not None:
                disp = math.sqrt(
                    (features.player_a_position[0] - self._prev_player_a[0])**2
                    + (features.player_a_position[1] - self._prev_player_a[1])**2
                )
                self._rally.player_a_displacement += disp
                features.player_a_displacement = self._rally.player_a_displacement
            self._prev_player_a = features.player_a_position

        if features.player_b_position is not None:
            if self._prev_player_b is not None:
                disp = math.sqrt(
                    (features.player_b_position[0] - self._prev_player_b[0])**2
                    + (features.player_b_position[1] - self._prev_player_b[1])**2
                )
                self._rally.player_b_displacement += disp
                features.player_b_displacement = self._rally.player_b_displacement
            self._prev_player_b = features.player_b_position

        # --- Rally state ---
        features.rally_active = self._rally.active
        features.rally_hit_count = self._rally.hit_count
        if self._rally.active:
            features.rally_duration = timestamp - self._rally.start_time

        return features

    def start_rally(self, timestamp: float, player_a_pos=None, player_b_pos=None):
        """Mark the start of a new rally."""
        self._rally = RallyState(
            active=True,
            start_time=timestamp,
            player_a_start_pos=player_a_pos,
            player_b_start_pos=player_b_pos,
        )
        self._stationary_count = 0
        self._no_shuttle_count = 0
        self._velocity_history.clear()
        logger.debug(f"Rally started at {timestamp:.2f}s")

    def end_rally(self) -> RallyState:
        """End the current rally and return its final state."""
        rally_snapshot = RallyState(
            active=False,
            start_time=self._rally.start_time,
            hit_count=self._rally.hit_count,
            player_a_displacement=self._rally.player_a_displacement,
            player_b_displacement=self._rally.player_b_displacement,
            peak_velocity=self._rally.peak_velocity,
            shuttle_positions=list(self._rally.shuttle_positions),
        )
        self._rally = RallyState()
        self._stationary_count = 0
        self._no_shuttle_count = 0
        self._velocity_history.clear()
        logger.debug(f"Rally ended: {rally_snapshot.hit_count} hits")
        return rally_snapshot

    def get_rally_state(self) -> RallyState:
        """Get current rally state."""
        return self._rally

    def reset(self):
        """Full reset of extractor state."""
        self._rally = RallyState()
        self._prev_shuttle_pos = None
        self._prev_timestamp = 0.0
        self._stationary_count = 0
        self._no_shuttle_count = 0
        self._prev_player_a = None
        self._prev_player_b = None
        self._velocity_history.clear()
