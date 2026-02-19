"""
Event Engine — Phase-aware event detection with physics-based landing
and multi-signal winner attribution.

Architecture:
  1. Rally lifecycle (start/end)
  2. Shot classification (smash, drop shot, long rally)
  3. Physics-based landing detection (velocity collapse + descent vanish +
     boundary proximity + net collision)
  4. Multi-signal winner attribution (landing side + trajectory direction +
     last hitter + opponent proximity)
"""

import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

from .event_models import Event, EventType, EventTimeline, MatchState, Intensity

logger = logging.getLogger(__name__)


class LandingDetector:
    """
    Multi-signal landing detection.

    Combines four signals to determine if the shuttle has landed:
    - Velocity collapse (fast → stopped)
    - Descent vanish (disappeared while descending)
    - Boundary proximity (near baseline)
    - Net collision (stopped at net level)

    Requires at least two signals agreeing to prevent false positives.
    """

    def __init__(self, frame_height: int = 360, frame_width: int = 640):
        self.frame_height = frame_height
        self.frame_width = frame_width

        # Court boundaries as fractions of frame (from config)
        self.baseline_a_ratio = 0.95   # A's baseline (bottom)
        self.baseline_b_ratio = 0.05   # B's baseline (top)
        self.sideline_left_ratio = 0.15
        self.sideline_right_ratio = 0.85
        self.net_y_ratio = 0.5

        self._prev_vel: float = 0.0
        self._prev_vy: float = 0.0

    def set_frame_dims(self, height: int, width: int):
        self.frame_height = height
        self.frame_width = width

    def detect(
        self,
        phase: str,
        vel: float,
        vy: float,
        decay_rate: float,
        shuttle_pos: Optional[Tuple[float, float]],
        frames_no_detect: int,
        is_predicted: bool,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check for shuttle landing.

        Returns: (is_landing, landing_type, confidence)
        landing_type: 'ground', 'net', 'out_of_bounds'
        """
        if phase != "descent":
            self._prev_vel = vel
            self._prev_vy = vy
            return False, None, 0.0

        signals = []
        landing_type = "ground"
        net_y = self.net_y_ratio * self.frame_height

        # Signal 1: Velocity collapse (0.30 weight)
        # Shuttle was moving fast, now nearly stopped
        if vel < 2.0 and self._prev_vel > 4.0:
            signals.append(("vel_collapse", 0.30, 1.0))
        elif vel < 3.0 and decay_rate < -1.5:
            signals.append(("vel_collapse", 0.30, 0.7))

        # Signal 2: Descent vanish (0.25 weight)
        # Shuttle disappeared while descending at speed
        if frames_no_detect >= 3 and self._prev_vy > 2.0:
            conf = min(frames_no_detect / 5.0, 1.0)
            signals.append(("descent_vanish", 0.25, conf))

        # Signal 3: Boundary proximity (0.25 weight)
        if shuttle_pos is not None:
            sx, sy = shuttle_pos
            baseline_a_y = self.baseline_a_ratio * self.frame_height
            baseline_b_y = self.baseline_b_ratio * self.frame_height

            dist_a = abs(sy - baseline_a_y) / self.frame_height
            dist_b = abs(sy - baseline_b_y) / self.frame_height
            min_dist = min(dist_a, dist_b)

            if min_dist < 0.15:
                conf = 1.0 - (min_dist / 0.15)
                signals.append(("near_baseline", 0.25, conf))

            # Signal 4: Net collision (0.20 weight)
            net_proximity = abs(sy - net_y) / self.frame_height
            if vel < 1.0 and net_proximity < 0.08 and self._prev_vel > 5.0:
                signals.append(("net_collision", 0.20, 1.0))
                landing_type = "net"

            # Out-of-bounds check
            if landing_type == "ground":
                court_left = self.sideline_left_ratio * self.frame_width
                court_right = self.sideline_right_ratio * self.frame_width
                if sx < court_left or sx > court_right:
                    landing_type = "out_of_bounds"

        self._prev_vel = vel
        self._prev_vy = vy

        if not signals:
            return False, None, 0.0

        total_score = sum(w * c for _, w, c in signals)

        # Requires at least 0.35 (two+ agreeing signals)
        if total_score >= 0.35:
            logger.debug(
                f"Landing detected: type={landing_type}, score={total_score:.2f}, "
                f"signals={[(n, round(w*c, 2)) for n, w, c in signals]}"
            )
            return True, landing_type, total_score

        return False, None, total_score


class WinnerAttribution:
    """
    Multi-signal winner attribution.

    Combines four signals:
    1. Landing side (0.40) — shuttle on your side = opponent scores
    2. Trajectory direction (0.30) — where was shuttle heading
    3. Last hitter (0.20) — who hit it last + where did it land
    4. Opponent proximity (0.10) — far from opponent = unreturnable winner

    Special cases: net collision and out-of-bounds handled deterministically.
    """

    def determine_winner(
        self,
        landing_pos: Optional[Tuple[float, float]],
        landing_type: str,
        last_hitter: Optional[str],
        trajectory_buffer: list,
        player_a_pos: Optional[Tuple[float, float]],
        player_b_pos: Optional[Tuple[float, float]],
        net_y: float,
        player_a_name: str = "Player A",
        player_b_name: str = "Player B",
    ) -> Tuple[str, float, str]:
        """
        Determine point winner.

        Returns: (winner_name, confidence, reason)
        """
        # === SPECIAL CASES ===

        # Net fault: last hitter loses
        if landing_type == "net":
            if last_hitter:
                loser = last_hitter
                winner = player_b_name if loser == player_a_name else player_a_name
                return winner, 0.95, "net_fault"
            # Unknown hitter — fall through to signals
            return self._signal_based(
                landing_pos, trajectory_buffer, player_a_pos, player_b_pos,
                last_hitter, net_y, player_a_name, player_b_name
            )

        # Out of bounds: last hitter loses (their shot went out)
        if landing_type == "out_of_bounds":
            if last_hitter:
                loser = last_hitter
                winner = player_b_name if loser == player_a_name else player_a_name
                return winner, 0.90, "out_of_bounds"
            return self._signal_based(
                landing_pos, trajectory_buffer, player_a_pos, player_b_pos,
                last_hitter, net_y, player_a_name, player_b_name
            )

        # Normal landing
        return self._signal_based(
            landing_pos, trajectory_buffer, player_a_pos, player_b_pos,
            last_hitter, net_y, player_a_name, player_b_name
        )

    def _signal_based(
        self,
        landing_pos, trajectory_buffer, player_a_pos, player_b_pos,
        last_hitter, net_y, player_a_name, player_b_name,
    ) -> Tuple[str, float, str]:
        """Multi-signal winner scoring."""
        scores = {player_a_name: 0.0, player_b_name: 0.0}

        # Signal 1: Landing side (0.40)
        if landing_pos is not None:
            if landing_pos[1] > net_y:  # A's side (bottom)
                scores[player_b_name] += 0.40
            else:
                scores[player_a_name] += 0.40

        # Signal 2: Trajectory direction (0.30)
        if len(trajectory_buffer) >= 3:
            recent = trajectory_buffer[-5:]
            avg_vy = np.mean([
                recent[i][1] - recent[i-1][1]
                for i in range(1, len(recent))
            ])
            if avg_vy > 0:  # Moving toward A's baseline
                scores[player_b_name] += 0.30
            elif avg_vy < 0:  # Moving toward B's baseline
                scores[player_a_name] += 0.30

        # Signal 3: Last hitter cross-validation (0.20)
        if last_hitter and landing_pos is not None:
            if last_hitter == player_a_name:
                if landing_pos[1] < net_y:  # A hit, landed on B's side → A wins
                    scores[player_a_name] += 0.20
                else:  # A hit into own court
                    scores[player_b_name] += 0.20
            elif last_hitter == player_b_name:
                if landing_pos[1] > net_y:  # B hit, landed on A's side → B wins
                    scores[player_b_name] += 0.20
                else:
                    scores[player_a_name] += 0.20

        # Signal 4: Proximity to opponent (0.10)
        if landing_pos is not None and player_a_pos and player_b_pos:
            dist_a = math.sqrt(
                (landing_pos[0] - player_a_pos[0])**2 +
                (landing_pos[1] - player_a_pos[1])**2
            )
            dist_b = math.sqrt(
                (landing_pos[0] - player_b_pos[0])**2 +
                (landing_pos[1] - player_b_pos[1])**2
            )
            # Landing far from a player = point against them
            if dist_a > dist_b:
                scores[player_b_name] += 0.10
            else:
                scores[player_a_name] += 0.10

        # Decision
        winner = max(scores, key=scores.get)
        confidence = scores[winner]
        margin = abs(scores[player_a_name] - scores[player_b_name])

        if margin < 0.1 and landing_pos is not None:
            # Ambiguous — fall back to landing side
            winner = player_b_name if landing_pos[1] > net_y else player_a_name
            confidence = 0.55

        return winner, confidence, "multi_signal"


class IntensityScorer:
    """Score event intensity based on rally metrics."""

    def __init__(self, config: Dict[str, Any]):
        t = config.get("thresholds", {})
        self.low_t = t.get("low", 0.25)
        self.medium_t = t.get("medium", 0.50)
        self.high_t = t.get("high", 0.75)
        self.weights = {
            "rally_length": 0.35,
            "velocity": 0.30,
            "player_movement": 0.20,
            "proximity": 0.15,
        }

    def score(
        self,
        rally_length: int = 0,
        velocity: float = 0.0,
        player_displacement: float = 0.0,
        proximity_factor: float = 0.0,
    ) -> Intensity:
        raw = (
            self.weights["rally_length"] * min(rally_length / 20.0, 1.0)
            + self.weights["velocity"] * min(velocity / 30.0, 1.0)
            + self.weights["player_movement"] * min(player_displacement / 300.0, 1.0)
            + self.weights["proximity"] * min(proximity_factor, 1.0)
        )

        if raw >= self.high_t:
            return Intensity.MAXIMUM
        elif raw >= self.medium_t:
            return Intensity.HIGH
        elif raw >= self.low_t:
            return Intensity.MEDIUM
        return Intensity.LOW


class EventEngine:
    """
    Phase-aware event detection engine.

    Uses shuttle phase state machine and physics-based landing detection
    to produce reliable gameplay events and score attribution.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        player_a_name: str = "Player A",
        player_b_name: str = "Player B",
    ):
        self.player_a_name = player_a_name
        self.player_b_name = player_b_name

        # Config
        rules = config.get("event_rules", {})
        self._smash_cfg = rules.get("smash", {})
        self._long_rally_cfg = rules.get("long_rally", {})
        self._drop_shot_cfg = rules.get("drop_shot", {})
        self._point_won_cfg = rules.get("point_won", {})

        # Intensity
        self._scorer = IntensityScorer(config.get("intensity", {}))

        # Court
        self._net_y_ratio = config.get("court", {}).get("net_y", 0.5)
        self._frame_height: Optional[int] = None
        self._frame_width: Optional[int] = None

        # State
        self.match_state = MatchState(
            player_a_name=player_a_name,
            player_b_name=player_b_name,
        )
        self._timeline = EventTimeline(
            player_a=player_a_name,
            player_b=player_b_name,
            match_state=self.match_state,
        )

        # Event cooldowns
        self._min_event_gap: float = 1.0
        self._last_event_times: Dict[str, float] = {}

        # Rally tracking
        self._rally_active = False
        self._rally_start_time: float = 0.0
        self._shuttle_motion_frames: int = 0
        self._rally_start_motion_threshold: int = 3
        self._long_rally_emitted = False
        self._last_rally_hits: int = 0

        # Landing detection
        self._landing_detector = LandingDetector()
        self._winner_attribution = WinnerAttribution()

    def set_frame_height(self, height: int, width: int = 640):
        self._frame_height = height
        self._frame_width = width
        self._landing_detector.set_frame_dims(height, width)

    def _get_net_y(self) -> float:
        h = self._frame_height or 360
        return self._net_y_ratio * h

    def _can_emit(self, timestamp: float, event_type: str) -> bool:
        last = self._last_event_times.get(event_type, -10.0)
        return (timestamp - last) >= self._min_event_gap

    def _record_emit(self, timestamp: float, event_type: str):
        self._last_event_times[event_type] = timestamp

    def process_frame(self, features) -> List[Event]:
        """
        Process a single frame's features and emit events.

        Uses shuttle phase and physics signals for landing/point detection
        instead of simple stationary frame counting.
        """
        events: List[Event] = []
        ts = features.timestamp
        net_y = self._get_net_y()

        # --- Rally Start Detection ---
        if features.shuttle_velocity > 2.0:
            self._shuttle_motion_frames += 1
        else:
            if self._shuttle_motion_frames < self._rally_start_motion_threshold:
                self._shuttle_motion_frames = 0

        if (
            not self._rally_active
            and self._shuttle_motion_frames >= self._rally_start_motion_threshold
            and self._can_emit(ts, "rally_start")
        ):
            self._rally_active = True
            self._rally_start_time = ts
            self._long_rally_emitted = False
            self._last_rally_hits = 0

            server = "—"
            if features.shuttle_position:
                server = self.player_a_name if features.shuttle_position[1] > net_y else self.player_b_name

            events.append(Event(
                timestamp=ts,
                event=EventType.RALLY_START,
                server=server,
                intensity=Intensity.LOW,
            ))
            self._record_emit(ts, "rally_start")

        if not self._rally_active:
            return events

        rally_duration = ts - self._rally_start_time

        # --- Smash Detection ---
        if (
            features.is_smash
            and self._can_emit(ts, "smash")
            and self._smash_cfg.get("enabled", True)
        ):
            by = features.last_hitter or (
                self.player_a_name if features.shuttle_direction == "down" else self.player_b_name
            )
            intensity = self._scorer.score(
                rally_length=features.rally_hit_count,
                velocity=features.shuttle_velocity,
                player_displacement=features.player_a_displacement + features.player_b_displacement,
            )
            events.append(Event(
                timestamp=ts,
                event=EventType.SMASH,
                by=by,
                intensity=intensity,
                velocity=round(features.shuttle_velocity, 1),
                rally_length=features.rally_hit_count,
            ))
            self._record_emit(ts, "smash")

        # --- Drop Shot Detection ---
        if (
            self._drop_shot_cfg.get("enabled", True)
            and self._can_emit(ts, "drop_shot")
            and features.shuttle_velocity < self._drop_shot_cfg.get("max_velocity", 8.0)
            and features.shuttle_velocity > 1.0
            and features.shuttle_position is not None
        ):
            net_prox = abs(features.shuttle_position[1] - net_y) / (self._frame_height or 360)
            if net_prox < self._drop_shot_cfg.get("net_proximity", 0.15):
                by = features.last_hitter or self.player_b_name
                intensity = self._scorer.score(
                    rally_length=features.rally_hit_count,
                    velocity=features.shuttle_velocity,
                    proximity_factor=1.0 - net_prox,
                )
                events.append(Event(
                    timestamp=ts,
                    event=EventType.DROP_SHOT,
                    by=by,
                    intensity=intensity,
                    velocity=round(features.shuttle_velocity, 1),
                ))
                self._record_emit(ts, "drop_shot")

        # --- Long Rally Detection ---
        hit_threshold = self._long_rally_cfg.get("min_hit_count", 8)
        if (
            not self._long_rally_emitted
            and features.rally_hit_count >= hit_threshold
            and self._can_emit(ts, "long_rally")
            and self._long_rally_cfg.get("enabled", True)
        ):
            intensity = self._scorer.score(
                rally_length=features.rally_hit_count,
                velocity=features.shuttle_velocity,
                player_displacement=features.player_a_displacement + features.player_b_displacement,
            )
            events.append(Event(
                timestamp=ts,
                event=EventType.LONG_RALLY,
                intensity=intensity,
                rally_length=features.rally_hit_count,
                rally_duration=round(rally_duration, 1),
            ))
            self._record_emit(ts, "long_rally")
            self._long_rally_emitted = True

        # --- Physics-Based Landing Detection ---
        if (
            self._point_won_cfg.get("enabled", True)
            and self._can_emit(ts, "point_won")
        ):
            is_landing, landing_type, landing_conf = self._landing_detector.detect(
                phase=features.shuttle_phase,
                vel=features.shuttle_velocity,
                vy=features.shuttle_vy if hasattr(features, 'shuttle_vy') else 0.0,
                decay_rate=features.velocity_decay_rate,
                shuttle_pos=features.shuttle_position,
                frames_no_detect=getattr(features, '_no_shuttle_count', 0)
                    if hasattr(features, '_no_shuttle_count') else
                    (features.shuttle_stationary_frames if features.shuttle_position is None else 0),
                is_predicted=features.is_predicted,
            )

            # Fallback: also check stationary frames for robustness
            stationary_threshold = self._point_won_cfg.get("stationary_frames", 15)
            min_hits = self._point_won_cfg.get("min_rally_hits", 1)
            stationary_landing = (
                features.shuttle_stationary_frames >= stationary_threshold
                and features.rally_hit_count >= min_hits
            )

            if (is_landing or stationary_landing) and self._rally_active:
                landing_t = landing_type or "ground"

                # Multi-signal winner attribution
                winner, confidence, reason = self._winner_attribution.determine_winner(
                    landing_pos=features.shuttle_position,
                    landing_type=landing_t,
                    last_hitter=features.last_hitter,
                    trajectory_buffer=getattr(features, 'trajectory_buffer', []),
                    player_a_pos=features.player_a_position,
                    player_b_pos=features.player_b_position,
                    net_y=net_y,
                    player_a_name=self.player_a_name,
                    player_b_name=self.player_b_name,
                )

                # Update score
                if winner == self.player_a_name:
                    self.match_state.player_a_score += 1
                else:
                    self.match_state.player_b_score += 1

                logger.info(
                    f"Score: {self.player_a_name} {self.match_state.player_a_score} - "
                    f"{self.match_state.player_b_score} {self.player_b_name}"
                )

                intensity = self._scorer.score(
                    rally_length=features.rally_hit_count,
                    velocity=features.shuttle_velocity,
                    player_displacement=features.player_a_displacement + features.player_b_displacement,
                )

                # Determine event type
                if landing_t == "net":
                    event_type = EventType.NET_FAULT
                elif landing_t == "out_of_bounds":
                    event_type = EventType.OUT_OF_BOUNDS
                else:
                    event_type = EventType.POINT_WON

                events.append(Event(
                    timestamp=ts,
                    event=event_type,
                    by=winner,
                    intensity=intensity,
                    rally_length=features.rally_hit_count,
                    rally_duration=round(rally_duration, 1),
                    metadata={
                        "landing_type": landing_t,
                        "attribution_confidence": round(confidence, 2),
                        "attribution_reason": reason,
                        "landing_confidence": round(landing_conf, 2) if is_landing else 0.0,
                    },
                ))
                self._record_emit(ts, "point_won")

                # Reset rally
                self._rally_active = False
                self._shuttle_motion_frames = 0

        # Track hit count
        self._last_rally_hits = features.rally_hit_count

        # Add events to timeline
        for event in events:
            self._timeline.add_event(event)

        return events

    def finalize(self, video_path: str = "", video_duration: float = 0.0) -> EventTimeline:
        """Finalize and return the event timeline."""
        self._timeline.video_path = video_path
        self._timeline.video_duration = round(video_duration, 2)
        self._timeline.match_state = self.match_state
        return self._timeline
