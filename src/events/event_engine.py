"""
Event Engine — Rule-based conversion of CV features into gameplay events.

This is the core intelligence layer. It takes raw features from the
FeatureExtractor and applies configurable rules to classify structured
badminton events: rally_start, smash, drop_shot, long_rally, point_won.

Key design:
  CV generates facts → Rules structure meaning → Events carry truth.
  No LLM is involved at this layer.
"""

from typing import List, Optional, Dict, Any
from .event_models import Event, EventType, Intensity, EventTimeline, MatchState
import logging
import math

logger = logging.getLogger(__name__)


class IntensityScorer:
    """
    Scores event intensity as Low/Medium/High/Maximum.

    Uses weighted combination of:
    - Rally length (hit count)
    - Shuttle velocity
    - Player movement (displacement)
    - Proximity factor (how close to the line/boundary)
    """

    def __init__(self, config: Dict[str, Any]):
        self.rally_w = config.get("rally_length_weight", 0.35)
        self.velocity_w = config.get("velocity_weight", 0.30)
        self.movement_w = config.get("player_movement_weight", 0.20)
        self.proximity_w = config.get("proximity_weight", 0.15)
        self.thresholds = config.get("thresholds", {
            "low": 0.25, "medium": 0.50, "high": 0.75,
        })

        # Normalization caps (expected max values)
        self._max_rally = 30
        self._max_velocity = 50.0
        self._max_displacement = 500.0

    def score(
        self,
        rally_length: int = 0,
        velocity: float = 0.0,
        player_displacement: float = 0.0,
        proximity_factor: float = 0.5,
    ) -> Intensity:
        """Compute intensity from weighted features."""
        # Normalize each feature to [0, 1]
        norm_rally = min(rally_length / self._max_rally, 1.0)
        norm_vel = min(velocity / self._max_velocity, 1.0)
        norm_disp = min(player_displacement / self._max_displacement, 1.0)
        norm_prox = min(proximity_factor, 1.0)

        raw_score = (
            self.rally_w * norm_rally
            + self.velocity_w * norm_vel
            + self.movement_w * norm_disp
            + self.proximity_w * norm_prox
        )

        if raw_score >= self.thresholds.get("high", 0.75):
            return Intensity.MAXIMUM
        elif raw_score >= self.thresholds.get("medium", 0.50):
            return Intensity.HIGH
        elif raw_score >= self.thresholds.get("low", 0.25):
            return Intensity.MEDIUM
        else:
            return Intensity.LOW


class EventEngine:
    """
    Rule-based engine that converts frame features into structured events.

    Processes features frame-by-frame, maintaining state to detect
    multi-frame events (rallies, point endings) and single-frame
    events (smashes, drop shots).
    """

    def __init__(self, config: Dict[str, Any], player_a_name: str = "Player A", player_b_name: str = "Player B"):
        self.config = config
        self.player_a_name = player_a_name
        self.player_b_name = player_b_name

        # Event rules from config
        rules = config.get("event_rules", {})
        self._smash_rules = rules.get("smash", {})
        self._long_rally_rules = rules.get("long_rally", {})
        self._drop_shot_rules = rules.get("drop_shot", {})
        self._point_won_rules = rules.get("point_won", {})

        # Intensity scorer
        intensity_config = config.get("intensity", {})
        self.intensity_scorer = IntensityScorer(intensity_config)

        # Court net Y as RATIO (not hardcoded px)
        self._net_y_ratio = config.get("court", {}).get("net_y", 0.5)

        # Match state
        self.match_state = MatchState(
            player_a_name=player_a_name,
            player_b_name=player_b_name,
        )

        # Frame height — will be set dynamically on first frame
        self._frame_height: Optional[int] = None

        # Internal state
        self._timeline = EventTimeline(
            player_a=player_a_name,
            player_b=player_b_name,
            match_state=self.match_state,
        )
        self._rally_active = False
        self._rally_start_time: float = 0.0
        self._rally_hit_count: int = 0
        self._peak_velocity: float = 0.0
        self._total_player_displacement: float = 0.0
        self._smash_detected_this_rally = False
        self._drop_detected_this_rally = False
        self._long_rally_emitted = False
        self._last_shuttle_side: Optional[str] = None  # 'a' or 'b'

        # Shuttle tracking — require sustained loss, not momentary gaps
        self._frames_without_shuttle: int = 0
        self._frames_with_shuttle_stationary: int = 0

        # Cooldown to prevent duplicate events
        self._last_event_time: float = -10.0
        self._min_event_gap: float = 1.0  # seconds between events of same type
        self._last_event_times: Dict[str, float] = {}

        # Rally start requires sustained motion
        self._shuttle_motion_frames: int = 0
        self._rally_start_motion_threshold: int = 3  # frames of shuttle motion before rally_start

        logger.info("EventEngine initialized")

    def set_frame_height(self, height: int):
        """Set video frame height for dynamic net_y calculation."""
        self._frame_height = height

    def _get_net_y(self) -> float:
        """Get net Y in pixels, dynamically based on frame height."""
        h = self._frame_height or 360
        return self._net_y_ratio * h

    def process_frame(self, features) -> List[Event]:
        """
        Process a single frame's features and emit any detected events.

        Args:
            features: FrameFeatures object from FeatureExtractor

        Returns:
            List of Event objects detected in this frame (usually 0 or 1)
        """
        events: List[Event] = []
        ts = features.timestamp

        # Determine which side of net shuttle is on
        shuttle_side = None
        if features.shuttle_position is not None:
            net_y = self._get_net_y()
            shuttle_side = "a" if features.shuttle_position[1] > net_y else "b"

        # Track shuttle visibility
        has_shuttle = features.shuttle_position is not None and features.shuttle_velocity > 0
        if has_shuttle:
            self._frames_without_shuttle = 0
        else:
            self._frames_without_shuttle += 1

        # --- Rally Start Detection ---
        # Require sustained shuttle motion before declaring rally start
        if features.shuttle_velocity > 2.0:
            self._shuttle_motion_frames += 1
        else:
            self._shuttle_motion_frames = max(0, self._shuttle_motion_frames - 1)

        if (
            not self._rally_active
            and self._shuttle_motion_frames >= self._rally_start_motion_threshold
            and self._can_emit(ts, "rally_start")
        ):
            self._start_rally(ts)
            server = self._determine_server(features)
            event = Event(
                timestamp=round(ts, 2),
                event=EventType.RALLY_START,
                server=server,
                intensity=Intensity.LOW,
            )
            events.append(event)
            self._timeline.add_event(event)
            self._last_event_times["rally_start"] = ts

        if not self._rally_active:
            return events

        # --- Update rally state ---
        self._rally_hit_count = features.rally_hit_count
        if features.shuttle_velocity > self._peak_velocity:
            self._peak_velocity = features.shuttle_velocity
        self._total_player_displacement = features.player_a_displacement + features.player_b_displacement

        # --- Smash Detection ---
        if (
            self._smash_rules.get("enabled", True)
            and features.is_smash
            and not self._smash_detected_this_rally
            and self._can_emit(ts, "smash")
        ):
            self._smash_detected_this_rally = True
            player = self._determine_attacker(features, shuttle_side)
            intensity = self.intensity_scorer.score(
                rally_length=self._rally_hit_count,
                velocity=features.shuttle_velocity,
                player_displacement=self._total_player_displacement,
            )
            event = Event(
                timestamp=round(ts, 2),
                event=EventType.SMASH,
                by=player,
                intensity=intensity,
                velocity=round(features.shuttle_velocity, 1),
                rally_length=self._rally_hit_count,
            )
            events.append(event)
            self._timeline.add_event(event)
            self._last_event_times["smash"] = ts

        # --- Drop Shot Detection ---
        if (
            self._drop_shot_rules.get("enabled", True)
            and not self._drop_detected_this_rally
            and features.shuttle_velocity > 0
            and features.shuttle_velocity < self._drop_shot_rules.get("max_velocity", 8.0)
            and features.shuttle_angle > self._drop_shot_rules.get("min_angle_from_vertical", 30)
            and self._rally_hit_count >= 3
            and self._can_emit(ts, "drop_shot")
        ):
            self._drop_detected_this_rally = True
            player = self._determine_attacker(features, shuttle_side)
            intensity = self.intensity_scorer.score(
                rally_length=self._rally_hit_count,
                velocity=features.shuttle_velocity,
                player_displacement=self._total_player_displacement,
            )
            event = Event(
                timestamp=round(ts, 2),
                event=EventType.DROP_SHOT,
                by=player,
                intensity=intensity,
                velocity=round(features.shuttle_velocity, 1),
                rally_length=self._rally_hit_count,
            )
            events.append(event)
            self._timeline.add_event(event)
            self._last_event_times["drop_shot"] = ts

        # --- Long Rally Detection ---
        min_hits = self._long_rally_rules.get("min_hit_count", 15)
        if (
            self._long_rally_rules.get("enabled", True)
            and not self._long_rally_emitted
            and self._rally_hit_count >= min_hits
            and self._can_emit(ts, "long_rally")
        ):
            self._long_rally_emitted = True
            intensity = self.intensity_scorer.score(
                rally_length=self._rally_hit_count,
                velocity=self._peak_velocity,
                player_displacement=self._total_player_displacement,
            )
            event = Event(
                timestamp=round(ts, 2),
                event=EventType.LONG_RALLY,
                intensity=intensity,
                rally_length=self._rally_hit_count,
                rally_duration=round(ts - self._rally_start_time, 1),
            )
            events.append(event)
            self._timeline.add_event(event)
            self._last_event_times["long_rally"] = ts

        # --- Point Won Detection ---
        # Use shuttle_stationary_frames from feature extractor (which handles
        # both actual stationary shuttle AND no-shuttle-detected cases)
        stationary_threshold = self._point_won_rules.get("stationary_frames", 8)
        min_rally_hits = self._point_won_rules.get("min_rally_hits", 2)

        # Only trigger point_won after sustained inactivity AND minimum rally length
        if (
            features.shuttle_stationary_frames >= stationary_threshold
            and self._rally_hit_count >= min_rally_hits
            and self._can_emit(ts, "point_won")
        ):
            winner = self._determine_point_winner(features, shuttle_side)
            rally_duration = ts - self._rally_start_time

            intensity = self.intensity_scorer.score(
                rally_length=self._rally_hit_count,
                velocity=self._peak_velocity,
                player_displacement=self._total_player_displacement,
            )

            event = Event(
                timestamp=round(ts, 2),
                event=EventType.POINT_WON,
                by=winner,
                intensity=intensity,
                rally_length=self._rally_hit_count,
                rally_duration=round(rally_duration, 1),
                velocity=round(self._peak_velocity, 1),
                metadata={
                    "player_a_displacement": round(features.player_a_displacement, 1),
                    "player_b_displacement": round(features.player_b_displacement, 1),
                },
            )
            events.append(event)
            self._timeline.add_event(event)
            self._update_score(winner)
            self._end_rally()
            self._last_event_times["point_won"] = ts

        # Track shuttle side for next iteration
        if shuttle_side is not None:
            self._last_shuttle_side = shuttle_side

        return events

    def _start_rally(self, timestamp: float):
        """Initialize a new rally."""
        self._rally_active = True
        self._rally_start_time = timestamp
        self._rally_hit_count = 0
        self._peak_velocity = 0.0
        self._total_player_displacement = 0.0
        self._smash_detected_this_rally = False
        self._drop_detected_this_rally = False
        self._long_rally_emitted = False
        self._last_shuttle_side = None
        self._shuttle_motion_frames = 0
        self._frames_without_shuttle = 0

    def _end_rally(self):
        """End the current rally."""
        self._rally_active = False
        self._shuttle_motion_frames = 0

    def _can_emit(self, timestamp: float, event_type: str = "") -> bool:
        """Check if enough time has passed since the last event of this type."""
        if event_type:
            last = self._last_event_times.get(event_type, -10.0)
            return (timestamp - last) >= self._min_event_gap
        return True

    def _determine_server(self, features) -> str:
        """Determine which player is serving."""
        if features.shuttle_position is not None:
            net_y = self._get_net_y()
            if features.shuttle_position[1] > net_y:
                return self.player_a_name
        return self.player_b_name

    def _determine_attacker(self, features, shuttle_side: Optional[str]) -> str:
        """Determine which player hit the shuttle (the attacker)."""
        if shuttle_side == "a":
            return self.player_a_name
        elif shuttle_side == "b":
            return self.player_b_name
        return self.player_a_name

    def _determine_point_winner(self, features, shuttle_side: Optional[str]) -> str:
        """
        Determine who won the point.

        In badminton, the shuttle landing on your side means the opponent wins.
        """
        # Use last known shuttle side if current is unknown
        side = shuttle_side or self._last_shuttle_side
        if side == "a":
            return self.player_b_name
        elif side == "b":
            return self.player_a_name
        return self.player_a_name

    def _update_score(self, winner: str):
        """Update match score."""
        if winner == self.player_a_name:
            self.match_state.player_a_score += 1
        else:
            self.match_state.player_b_score += 1
        self.match_state.serving = winner
        logger.info(
            f"Score: {self.match_state.player_a_name} {self.match_state.player_a_score} - "
            f"{self.match_state.player_b_score} {self.match_state.player_b_name}"
        )

    def get_timeline(self) -> EventTimeline:
        """Get the complete event timeline."""
        self._timeline.match_state = self.match_state
        return self._timeline

    def finalize(self, video_path: str = "", video_duration: float = 0.0) -> EventTimeline:
        """Finalize the timeline with video metadata."""
        self._timeline.video_path = video_path
        self._timeline.video_duration = round(video_duration, 2)
        self._timeline.match_state = self.match_state
        return self._timeline
