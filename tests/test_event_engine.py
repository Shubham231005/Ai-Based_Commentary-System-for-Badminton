"""Tests for EventEngine with physics-based landing detection and winner attribution."""

import pytest
import math
from src.events.event_engine import EventEngine, LandingDetector, WinnerAttribution, IntensityScorer
from src.events.event_models import EventType, Intensity


class MockFeatures:
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get("timestamp", 0.0)
        self.shuttle_velocity = kwargs.get("shuttle_velocity", 0.0)
        self.shuttle_position = kwargs.get("shuttle_position", None)
        self.shuttle_angle = kwargs.get("shuttle_angle", 0.0)
        self.shuttle_direction = kwargs.get("shuttle_direction", None)
        self.shuttle_vx = kwargs.get("shuttle_vx", 0.0)
        self.shuttle_vy = kwargs.get("shuttle_vy", 0.0)
        self.is_smash = kwargs.get("is_smash", False)
        self.is_direction_reversal = kwargs.get("is_direction_reversal", False)
        self.rally_hit_count = kwargs.get("rally_hit_count", 0)
        self.rally_active = kwargs.get("rally_active", False)
        self.rally_duration = kwargs.get("rally_duration", 0.0)
        self.player_a_position = kwargs.get("player_a_position", None)
        self.player_b_position = kwargs.get("player_b_position", None)
        self.player_a_displacement = kwargs.get("player_a_displacement", 0.0)
        self.player_b_displacement = kwargs.get("player_b_displacement", 0.0)
        self.shuttle_stationary_frames = kwargs.get("shuttle_stationary_frames", 0)
        self.last_hitter = kwargs.get("last_hitter", None)
        self.shuttle_phase = kwargs.get("shuttle_phase", "idle")
        self.phase_changed = kwargs.get("phase_changed", False)
        self.velocity_decay_rate = kwargs.get("velocity_decay_rate", 0.0)
        self.angle_change_rate = kwargs.get("angle_change_rate", 0.0)
        self.shuttle_ax = kwargs.get("shuttle_ax", 0.0)
        self.shuttle_ay = kwargs.get("shuttle_ay", 0.0)
        self.is_predicted = kwargs.get("is_predicted", False)
        self.trajectory_buffer = kwargs.get("trajectory_buffer", [])


def make_config():
    return {
        "event_rules": {
            "smash": {"enabled": True, "min_velocity": 12.0, "max_angle_from_vertical": 45},
            "long_rally": {"enabled": True, "min_hit_count": 8},
            "drop_shot": {"enabled": True, "max_velocity": 8.0, "min_angle_from_vertical": 30, "net_proximity": 0.15},
            "point_won": {"enabled": True, "stationary_frames": 15, "min_rally_hits": 1},
        },
        "intensity": {"thresholds": {"low": 0.25, "medium": 0.50, "high": 0.75}},
        "court": {"net_y": 0.5},
    }


class TestLandingDetector:
    def setup_method(self):
        self.detector = LandingDetector(frame_height=360, frame_width=640)

    def test_no_landing_when_not_descent(self):
        """Landing should only trigger during descent phase."""
        is_l, _, _ = self.detector.detect(
            phase="ascent", vel=1.0, vy=5.0, decay_rate=-2.0,
            shuttle_pos=(320, 340), frames_no_detect=0, is_predicted=False,
        )
        assert is_l is False

    def test_velocity_collapse_landing(self):
        """Rapid velocity collapse near baseline should trigger landing."""
        # First call to set prev values
        self.detector.detect(
            phase="descent", vel=10.0, vy=5.0, decay_rate=0.0,
            shuttle_pos=(320, 330), frames_no_detect=0, is_predicted=False,
        )
        # Now velocity collapses near baseline
        is_l, landing_type, conf = self.detector.detect(
            phase="descent", vel=1.0, vy=2.0, decay_rate=-3.0,
            shuttle_pos=(320, 340), frames_no_detect=0, is_predicted=False,
        )
        assert is_l is True
        assert landing_type == "ground"
        assert conf >= 0.35

    def test_descent_vanish_landing(self):
        """Missing shuttle during descent should trigger landing."""
        self.detector.detect(
            phase="descent", vel=8.0, vy=5.0, decay_rate=0.0,
            shuttle_pos=(320, 300), frames_no_detect=0, is_predicted=False,
        )
        is_l, _, _ = self.detector.detect(
            phase="descent", vel=0.0, vy=0.0, decay_rate=-3.0,
            shuttle_pos=(320, 340), frames_no_detect=5, is_predicted=True,
        )
        assert is_l is True

    def test_net_collision(self):
        """Sudden stop near net should be detected as net collision."""
        self.detector.detect(
            phase="descent", vel=10.0, vy=5.0, decay_rate=0.0,
            shuttle_pos=(320, 180), frames_no_detect=0, is_predicted=False,
        )
        is_l, landing_type, _ = self.detector.detect(
            phase="descent", vel=0.5, vy=0.0, decay_rate=-5.0,
            shuttle_pos=(320, 180), frames_no_detect=0, is_predicted=False,
        )
        assert is_l is True
        assert landing_type == "net"

    def test_out_of_bounds(self):
        """Landing outside sidelines should be out of bounds."""
        self.detector.detect(
            phase="descent", vel=10.0, vy=5.0, decay_rate=0.0,
            shuttle_pos=(50, 330), frames_no_detect=0, is_predicted=False,
        )
        is_l, landing_type, _ = self.detector.detect(
            phase="descent", vel=1.0, vy=1.0, decay_rate=-3.0,
            shuttle_pos=(50, 340), frames_no_detect=0, is_predicted=False,
        )
        assert is_l is True
        assert landing_type == "out_of_bounds"

    def test_single_signal_insufficient(self):
        """A single weak signal should NOT trigger landing."""
        self.detector.detect(
            phase="descent", vel=3.0, vy=1.0, decay_rate=0.0,
            shuttle_pos=(320, 200), frames_no_detect=0, is_predicted=False,
        )
        is_l, _, conf = self.detector.detect(
            phase="descent", vel=2.5, vy=0.5, decay_rate=-0.5,
            shuttle_pos=(320, 200), frames_no_detect=0, is_predicted=False,
        )
        assert is_l is False


class TestWinnerAttribution:
    def setup_method(self):
        self.attr = WinnerAttribution()

    def test_net_fault_last_hitter_loses(self):
        """Net fault: last hitter should lose."""
        winner, conf, reason = self.attr.determine_winner(
            landing_pos=(320, 180), landing_type="net",
            last_hitter="Player A", trajectory_buffer=[],
            player_a_pos=(320, 300), player_b_pos=(320, 60),
            net_y=180, player_a_name="Player A", player_b_name="Player B",
        )
        assert winner == "Player B"  # A hit into net → B wins
        assert reason == "net_fault"
        assert conf >= 0.9

    def test_out_of_bounds_last_hitter_loses(self):
        """Out of bounds: last hitter should lose."""
        winner, conf, reason = self.attr.determine_winner(
            landing_pos=(50, 100), landing_type="out_of_bounds",
            last_hitter="Player B", trajectory_buffer=[],
            player_a_pos=(320, 300), player_b_pos=(320, 60),
            net_y=180, player_a_name="Player A", player_b_name="Player B",
        )
        assert winner == "Player A"  # B's shot went out → A wins
        assert reason == "out_of_bounds"

    def test_normal_landing_side_attribution(self):
        """Landing on A's side → B wins."""
        winner, _, reason = self.attr.determine_winner(
            landing_pos=(320, 300), landing_type="ground",
            last_hitter="Player B", trajectory_buffer=[
                (320, 100, 0.1), (320, 150, 0.2), (320, 200, 0.3),
                (320, 250, 0.4), (320, 300, 0.5),
            ],
            player_a_pos=(320, 300), player_b_pos=(320, 60),
            net_y=180, player_a_name="Player A", player_b_name="Player B",
        )
        assert winner == "Player B"
        assert reason == "multi_signal"

    def test_trajectory_cross_validation(self):
        """Trajectory direction should support landing side decision."""
        winner, conf, _ = self.attr.determine_winner(
            landing_pos=(320, 280), landing_type="ground",
            last_hitter="Player B",
            trajectory_buffer=[
                (320, 100, 0.1), (320, 140, 0.2), (320, 180, 0.3),
                (320, 220, 0.4), (320, 280, 0.5),
            ],
            player_a_pos=(320, 300), player_b_pos=(320, 60),
            net_y=180, player_a_name="Player A", player_b_name="Player B",
        )
        assert winner == "Player B"
        # Confidence should be higher with trajectory agreement
        assert conf >= 0.6


class TestEventEngine:
    def setup_method(self):
        self.engine = EventEngine(make_config(), "Player A", "Player B")
        self.engine.set_frame_height(360, 640)

    def test_rally_start(self):
        for i in range(4):
            events = self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
            if events:
                assert events[0].event == EventType.RALLY_START
                return
        assert False, "rally_start not detected"

    def test_smash_event(self):
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        events = self.engine.process_frame(MockFeatures(
            timestamp=2.0, shuttle_velocity=20.0, shuttle_position=(200, 250),
            is_smash=True, rally_hit_count=2,
        ))
        assert any(e.event == EventType.SMASH for e in events)

    def test_stationary_point_won_fallback(self):
        """Stationary detection should still work as fallback."""
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        events = self.engine.process_frame(MockFeatures(
            timestamp=5.0, shuttle_velocity=0.0, shuttle_position=(200, 250),
            shuttle_stationary_frames=20, rally_hit_count=3,
        ))
        point_events = [e for e in events if e.event == EventType.POINT_WON]
        assert len(point_events) == 1

    def test_point_won_has_attribution_metadata(self):
        """Point won should include attribution confidence and reason."""
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        events = self.engine.process_frame(MockFeatures(
            timestamp=5.0, shuttle_velocity=0.0, shuttle_position=(200, 250),
            shuttle_stationary_frames=20, rally_hit_count=2,
            last_hitter="Player A",
            trajectory_buffer=[(200, 200, 4.5), (200, 220, 4.7), (200, 250, 5.0)],
        ))
        point_events = [e for e in events if e.event == EventType.POINT_WON]
        assert len(point_events) == 1
        assert "attribution_confidence" in point_events[0].metadata
        assert "attribution_reason" in point_events[0].metadata

    def test_long_rally_detection(self):
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        events = self.engine.process_frame(MockFeatures(
            timestamp=3.0, shuttle_velocity=5.0, shuttle_position=(200, 100), rally_hit_count=10,
        ))
        assert any(e.event == EventType.LONG_RALLY for e in events)

    def test_finalize_returns_timeline(self):
        tl = self.engine.finalize(video_path="test.mp4", video_duration=120.0)
        assert tl.video_path == "test.mp4"
        assert tl.video_duration == 120.0


class TestIntensityScorer:
    def setup_method(self):
        self.scorer = IntensityScorer({"thresholds": {"low": 0.25, "medium": 0.50, "high": 0.75}})

    def test_low_intensity(self):
        assert self.scorer.score(rally_length=1, velocity=2.0) == Intensity.LOW

    def test_maximum_intensity(self):
        assert self.scorer.score(rally_length=30, velocity=50.0, player_displacement=500.0, proximity_factor=1.0) == Intensity.MAXIMUM
