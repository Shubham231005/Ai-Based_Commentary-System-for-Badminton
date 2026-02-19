"""
Feature Extractor — Phase-aware gameplay features from tracked objects.

Implements a shuttle phase state machine:
  IDLE → HIT → ASCENT → APEX → DESCENT → LANDING

Computes per-frame features including:
- Shuttle velocity, angle, direction
- Velocity decay rate (deceleration signal for landing detection)
- Shuttle phase transitions
- Rally hit count (direction reversals)
- Player displacement
- Smash / drop shot indicators
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)


class ShuttlePhase(str, Enum):
    """Phase of shuttle flight — models projectile physics."""
    IDLE = "idle"
    HIT = "hit"
    ASCENT = "ascent"
    APEX = "apex"
    DESCENT = "descent"
    LANDING = "landing"


@dataclass
class RallyState:
    """State of the current rally being tracked."""
    active: bool = False
    start_time: float = 0.0
    hit_count: int = 0
    last_direction: Optional[str] = None
    last_hitter: Optional[str] = None  # Who hit the shuttle last
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
    shuttle_velocity: float = 0.0
    shuttle_angle: float = 0.0
    shuttle_direction: Optional[str] = None
    shuttle_position: Optional[Tuple[float, float]] = None
    shuttle_vx: float = 0.0
    shuttle_vy: float = 0.0

    # Velocity dynamics (new for physics-based detection)
    velocity_decay_rate: float = 0.0      # d(velocity)/dt — negative = decelerating
    angle_change_rate: float = 0.0        # d(angle)/dt
    shuttle_ax: float = 0.0               # Kalman-estimated acceleration x
    shuttle_ay: float = 0.0               # Kalman-estimated acceleration y

    # Shuttle phase
    shuttle_phase: str = "idle"
    phase_changed: bool = False           # True when phase transition occurred
    is_predicted: bool = False            # True if position is Kalman-predicted

    # Rally state
    rally_active: bool = False
    rally_duration: float = 0.0
    rally_hit_count: int = 0
    last_hitter: Optional[str] = None

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

    # Trajectory buffer (last N positions for landing prediction)
    trajectory_buffer: List[Tuple[float, float, float]] = field(default_factory=list)


class FeatureExtractor:
    """
    Phase-aware feature extractor with velocity dynamics tracking.

    Maintains a shuttle phase state machine and computes physics-based
    signals (velocity decay rate, acceleration) used by the event engine
    for reliable landing detection and winner attribution.
    """

    def __init__(
        self,
        smash_velocity_threshold: float = 12.0,
        smash_angle_threshold: float = 45.0,
        stationary_threshold: float = 2.0,
        direction_reversal_threshold: float = 3.0,
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

        # Velocity smoothing
        self._velocity_history: List[float] = []
        self._max_velocity_history = 3

        # Phase state machine
        self._phase = ShuttlePhase.IDLE
        self._prev_velocity: float = 0.0
        self._prev_angle: float = 0.0
        self._prev_vy: float = 0.0

        # Trajectory ring buffer (last 15 positions for landing prediction)
        self._trajectory_buffer: deque = deque(maxlen=15)

        # Landing signal accumulator
        self._descent_start_time: float = 0.0
        self._descent_peak_vy: float = 0.0

    def update_frame_height(self, height: int):
        """Dynamically update frame height."""
        self.frame_height = height
        self.net_y = 0.5 * height

    def extract(
        self,
        frame_index: int,
        timestamp: float,
        tracked_objects: Dict[str, object],
        kalman_ax: float = 0.0,
        kalman_ay: float = 0.0,
    ) -> FrameFeatures:
        """
        Extract features from a single frame's tracked objects.

        Args:
            frame_index: Index of the current frame
            timestamp: Timestamp in seconds
            tracked_objects: Dict with 'player_a', 'player_b', 'shuttle'
            kalman_ax: Kalman-estimated x acceleration
            kalman_ay: Kalman-estimated y acceleration
        """
        features = FrameFeatures(frame_index=frame_index, timestamp=timestamp)

        player_a = tracked_objects.get("player_a")
        player_b = tracked_objects.get("player_b")
        shuttle = tracked_objects.get("shuttle")

        if player_a is not None:
            features.player_a_position = player_a.centroid
        if player_b is not None:
            features.player_b_position = player_b.centroid

        # Store acceleration from Kalman
        features.shuttle_ax = kalman_ax
        features.shuttle_ay = kalman_ay

        # --- Shuttle velocity & direction ---
        velocity = 0.0
        vy = 0.0
        angle = 0.0

        if shuttle is not None:
            sx, sy = shuttle.centroid
            features.shuttle_position = (sx, sy)
            features.is_predicted = getattr(shuttle, 'is_predicted', False)
            self._no_shuttle_count = 0

            # Add to trajectory buffer
            self._trajectory_buffer.append((sx, sy, timestamp))

            if self._prev_shuttle_pos is not None:
                dt = timestamp - self._prev_timestamp
                if dt > 0:
                    dx = sx - self._prev_shuttle_pos[0]
                    dy = sy - self._prev_shuttle_pos[1]
                    velocity = math.sqrt(dx**2 + dy**2)
                    vy = dy

                    # Smooth velocity
                    self._velocity_history.append(velocity)
                    if len(self._velocity_history) > self._max_velocity_history:
                        self._velocity_history.pop(0)
                    smoothed_velocity = float(np.mean(self._velocity_history))
                    features.shuttle_velocity = smoothed_velocity
                    features.shuttle_vx = dx / dt
                    features.shuttle_vy = dy / dt

                    # Angle from horizontal
                    if abs(dx) > 0.001:
                        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                    else:
                        angle = 90.0
                    features.shuttle_angle = angle

                    # Velocity decay rate (key landing signal)
                    features.velocity_decay_rate = smoothed_velocity - self._prev_velocity

                    # Angle change rate
                    features.angle_change_rate = angle - self._prev_angle

                    # Direction
                    if abs(dy) > self.direction_reversal_threshold:
                        features.shuttle_direction = "down" if dy > 0 else "up"

                    # Smash detection
                    if (
                        smoothed_velocity > self.smash_velocity_threshold
                        and dy > 0
                        and angle > (90 - self.smash_angle_threshold)
                    ):
                        features.is_smash = True

                    # Direction reversal
                    if features.shuttle_direction is not None:
                        if (
                            self._rally.last_direction is not None
                            and features.shuttle_direction != self._rally.last_direction
                        ):
                            features.is_direction_reversal = True
                            self._rally.hit_count += 1

                            # Track who hit last based on shuttle side
                            if features.shuttle_direction == "up":
                                self._rally.last_hitter = "Player A"
                            else:
                                self._rally.last_hitter = "Player B"

                        self._rally.last_direction = features.shuttle_direction

                    # Stationary detection
                    if velocity < self.stationary_threshold:
                        self._stationary_count += 1
                    else:
                        self._stationary_count = 0

                    features.shuttle_stationary_frames = self._stationary_count

            self._prev_shuttle_pos = (sx, sy)
            self._prev_timestamp = timestamp
            self._rally.shuttle_positions.append((sx, sy, timestamp))
            if features.shuttle_velocity > self._rally.peak_velocity:
                self._rally.peak_velocity = features.shuttle_velocity

        else:
            self._no_shuttle_count += 1
            if self._no_shuttle_count >= 5:
                self._stationary_count += 1
            features.shuttle_stationary_frames = self._stationary_count

        # --- Phase State Machine ---
        prev_phase = self._phase
        self._update_phase(features, velocity, vy, timestamp)
        features.shuttle_phase = self._phase.value
        features.phase_changed = (self._phase != prev_phase)

        # Update previous values for next frame
        self._prev_velocity = features.shuttle_velocity
        self._prev_angle = angle
        self._prev_vy = vy

        # --- Player displacement ---
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
        features.last_hitter = self._rally.last_hitter
        if self._rally.active:
            features.rally_duration = timestamp - self._rally.start_time

        # Trajectory buffer snapshot
        features.trajectory_buffer = list(self._trajectory_buffer)

        return features

    def _update_phase(self, features: FrameFeatures, velocity: float, vy: float, timestamp: float):
        """
        Update shuttle phase state machine.

        Transitions:
          IDLE → HIT:     velocity spike > 5.0
          HIT → ASCENT:   vy < 0 (moving upward)
          HIT → DESCENT:  vy > 2.0 (immediate downward = smash)
          ASCENT → APEX:  vy sign change (was negative, now >= 0)
          APEX → DESCENT: vy > 0 and decelerating
          DESCENT → LANDING: complex multi-signal (handled by event engine)
          DESCENT → HIT:  velocity spike (opponent intercepted)
          LANDING → IDLE: after point_won
        """
        vel = features.shuttle_velocity
        decay = features.velocity_decay_rate

        if self._phase == ShuttlePhase.IDLE:
            if vel > 5.0 and self._prev_velocity < 3.0:
                self._phase = ShuttlePhase.HIT
                logger.debug(f"Phase: IDLE → HIT at {timestamp:.2f}s")

        elif self._phase == ShuttlePhase.HIT:
            if vy < -1.0 and vel > 3.0:
                self._phase = ShuttlePhase.ASCENT
                logger.debug(f"Phase: HIT → ASCENT at {timestamp:.2f}s")
            elif vy > 2.0:
                self._phase = ShuttlePhase.DESCENT
                self._descent_start_time = timestamp
                self._descent_peak_vy = vy
                logger.debug(f"Phase: HIT → DESCENT (direct) at {timestamp:.2f}s")

        elif self._phase == ShuttlePhase.ASCENT:
            if self._prev_vy < 0 and vy >= 0:
                self._phase = ShuttlePhase.APEX
                logger.debug(f"Phase: ASCENT → APEX at {timestamp:.2f}s")
            elif vel > 5.0 and decay > 2.0:
                # Opponent intercepted during ascent
                self._phase = ShuttlePhase.HIT
                logger.debug(f"Phase: ASCENT → HIT (intercept) at {timestamp:.2f}s")

        elif self._phase == ShuttlePhase.APEX:
            if vy > 0:
                self._phase = ShuttlePhase.DESCENT
                self._descent_start_time = timestamp
                self._descent_peak_vy = vy
                logger.debug(f"Phase: APEX → DESCENT at {timestamp:.2f}s")

        elif self._phase == ShuttlePhase.DESCENT:
            # Check for intercept (opponent hit it back)
            if vel > 5.0 and decay > 2.0:
                self._phase = ShuttlePhase.HIT
                logger.debug(f"Phase: DESCENT → HIT (intercept) at {timestamp:.2f}s")
            # Track peak descent velocity
            if vy > self._descent_peak_vy:
                self._descent_peak_vy = vy

            # Landing detection is handled by the event engine using
            # the multi-signal algorithm. The phase transition to LANDING
            # is explicitly triggered by the event engine.

        elif self._phase == ShuttlePhase.LANDING:
            # Stays in LANDING until reset by event engine
            pass

    def set_phase(self, phase: ShuttlePhase):
        """Explicitly set phase (called by event engine for LANDING → IDLE)."""
        self._phase = phase

    def get_phase(self) -> ShuttlePhase:
        """Get current shuttle phase."""
        return self._phase

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
        self._phase = ShuttlePhase.IDLE
        logger.debug(f"Rally started at {timestamp:.2f}s")

    def end_rally(self) -> RallyState:
        """End the current rally and return its final state."""
        rally_snapshot = RallyState(
            active=False,
            start_time=self._rally.start_time,
            hit_count=self._rally.hit_count,
            last_hitter=self._rally.last_hitter,
            player_a_displacement=self._rally.player_a_displacement,
            player_b_displacement=self._rally.player_b_displacement,
            peak_velocity=self._rally.peak_velocity,
            shuttle_positions=list(self._rally.shuttle_positions),
        )
        self._rally = RallyState()
        self._stationary_count = 0
        self._no_shuttle_count = 0
        self._velocity_history.clear()
        self._phase = ShuttlePhase.IDLE
        self._trajectory_buffer.clear()
        logger.debug(f"Rally ended: {rally_snapshot.hit_count} hits")
        return rally_snapshot

    def get_rally_state(self) -> RallyState:
        return self._rally

    def get_trajectory_buffer(self) -> List[Tuple[float, float, float]]:
        """Get last N shuttle positions for landing prediction."""
        return list(self._trajectory_buffer)

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
        self._phase = ShuttlePhase.IDLE
        self._prev_velocity = 0.0
        self._prev_angle = 0.0
        self._prev_vy = 0.0
        self._trajectory_buffer.clear()
