"""Tests for EventEngine with updated thresholds."""

import pytest
from src.events.event_engine import EventEngine, IntensityScorer
from src.events.event_models import EventType, Intensity
from unittest.mock import MagicMock


class MockFeatures:
    """Minimal FrameFeatures mock."""
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get("timestamp", 0.0)
        self.shuttle_velocity = kwargs.get("shuttle_velocity", 0.0)
        self.shuttle_position = kwargs.get("shuttle_position", None)
        self.shuttle_angle = kwargs.get("shuttle_angle", 0.0)
        self.shuttle_direction = kwargs.get("shuttle_direction", None)
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


def make_config():
    return {
        "event_rules": {
            "smash": {"enabled": True, "min_velocity": 12.0, "max_angle_from_vertical": 45},
            "long_rally": {"enabled": True, "min_hit_count": 8},
            "drop_shot": {"enabled": True, "max_velocity": 8.0, "min_angle_from_vertical": 30},
            "point_won": {"enabled": True, "stationary_frames": 15, "min_rally_hits": 1},
        },
        "intensity": {"thresholds": {"low": 0.25, "medium": 0.50, "high": 0.75}},
        "court": {"net_y": 0.5},
    }


class TestEventEngine:
    def setup_method(self):
        self.engine = EventEngine(make_config(), "Player A", "Player B")
        self.engine.set_frame_height(360)

    def test_rally_start_on_shuttle_motion(self):
        """Rally should start after sustained shuttle motion."""
        # Need 3 frames of motion (rally_start_motion_threshold)
        for i in range(4):
            features = MockFeatures(
                timestamp=i * 0.1,
                shuttle_velocity=5.0,
                shuttle_position=(200, 250),
            )
            events = self.engine.process_frame(features)
            if events:
                assert events[0].event == EventType.RALLY_START
                return
        # Should have detected rally_start by now
        assert False, "rally_start not detected"

    def test_smash_event(self):
        """Smash should be emitted when is_smash flag is set during rally."""
        # Start a rally first
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))

        # Now trigger smash
        events = self.engine.process_frame(MockFeatures(
            timestamp=2.0,
            shuttle_velocity=20.0,
            shuttle_position=(200, 250),
            is_smash=True,
            rally_hit_count=2,
        ))
        smash_events = [e for e in events if e.event == EventType.SMASH]
        assert len(smash_events) == 1

    def test_point_won_after_stationary(self):
        """Point should be won after enough stationary frames."""
        # Start a rally
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))

        # Shuttle goes stationary on one side
        events = self.engine.process_frame(MockFeatures(
            timestamp=5.0,
            shuttle_velocity=0.0,
            shuttle_position=(200, 250),
            shuttle_stationary_frames=20,
            rally_hit_count=3,
        ))
        point_events = [e for e in events if e.event == EventType.POINT_WON]
        assert len(point_events) == 1

    def test_score_updates_on_point_won(self):
        """Score should update when point is won."""
        # Start rally
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        # Point won (shuttle on A's side → B wins)
        self.engine.process_frame(MockFeatures(
            timestamp=5.0, shuttle_velocity=0.0,
            shuttle_position=(200, 250),  # Below net_y=180, so side "a"
            shuttle_stationary_frames=20, rally_hit_count=2,
        ))
        assert self.engine.match_state.player_b_score >= 1

    def test_long_rally_detection(self):
        """Long rally should emit when hit count exceeds threshold."""
        # Start rally
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        # Simulate enough hits
        events = self.engine.process_frame(MockFeatures(
            timestamp=3.0, shuttle_velocity=5.0,
            shuttle_position=(200, 100), rally_hit_count=10,
        ))
        long_rallies = [e for e in events if e.event == EventType.LONG_RALLY]
        assert len(long_rallies) == 1

    def test_event_cooldown(self):
        """Same event type shouldn't emit too quickly."""
        # Two rapid rally starts should be blocked
        self.engine.process_frame(MockFeatures(
            timestamp=0.0, shuttle_velocity=5.0, shuttle_position=(200, 250)
        ))
        for i in range(4):
            self.engine.process_frame(MockFeatures(
                timestamp=i * 0.1, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
        # Events within cooldown should not re-trigger
        all_events = []
        for i in range(5):
            e = self.engine.process_frame(MockFeatures(
                timestamp=0.5 + i * 0.01, shuttle_velocity=5.0, shuttle_position=(200, 250)
            ))
            all_events.extend(e)
        rally_starts = [e for e in all_events if e.event == EventType.RALLY_START]
        # Should not get duplicate rally_starts
        assert len(rally_starts) <= 1

    def test_finalize_returns_timeline(self):
        """Finalize should return timeline with metadata."""
        tl = self.engine.finalize(video_path="test.mp4", video_duration=120.0)
        assert tl.video_path == "test.mp4"
        assert tl.video_duration == 120.0


class TestIntensityScorer:
    def setup_method(self):
        self.scorer = IntensityScorer({
            "thresholds": {"low": 0.25, "medium": 0.50, "high": 0.75}
        })

    def test_low_intensity(self):
        result = self.scorer.score(rally_length=1, velocity=2.0)
        assert result == Intensity.LOW

    def test_medium_intensity(self):
        result = self.scorer.score(rally_length=10, velocity=15.0, player_displacement=100.0)
        assert result in [Intensity.MEDIUM, Intensity.HIGH]

    def test_maximum_intensity(self):
        result = self.scorer.score(rally_length=30, velocity=50.0, player_displacement=500.0, proximity_factor=1.0)
        assert result == Intensity.MAXIMUM
