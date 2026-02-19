"""Tests for FeatureExtractor with lowered thresholds for 360p."""

import pytest
from src.vision.feature_extractor import FeatureExtractor, FrameFeatures, RallyState
from unittest.mock import MagicMock


class MockTrackedObject:
    """Minimal tracked object mock."""
    def __init__(self, cx, cy):
        self.centroid = (cx, cy)
        self.is_predicted = False


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(
            smash_velocity_threshold=12.0,
            stationary_threshold=2.0,
            direction_reversal_threshold=3.0,
            frame_height=360,
        )

    def _tracked(self, player_a=None, player_b=None, shuttle=None):
        """Build tracked_objects dict from optional positions."""
        result = {}
        result["player_a"] = MockTrackedObject(*player_a) if player_a else None
        result["player_b"] = MockTrackedObject(*player_b) if player_b else None
        result["shuttle"] = MockTrackedObject(*shuttle) if shuttle else None
        return result

    def test_shuttle_velocity(self):
        """Velocity should be computed from consecutive positions."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)

        t2 = self._tracked(shuttle=(110, 100))
        features = self.extractor.extract(1, 0.1, t2)
        assert features.shuttle_velocity > 0

    def test_shuttle_direction_up(self):
        """Shuttle moving upward (decreasing Y) should be 'up'."""
        t1 = self._tracked(shuttle=(100, 200))
        self.extractor.extract(0, 0.0, t1)

        t2 = self._tracked(shuttle=(100, 190))
        features = self.extractor.extract(1, 0.1, t2)
        assert features.shuttle_direction == "up"

    def test_shuttle_direction_down(self):
        """Shuttle moving downward (increasing Y) should be 'down'."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)

        t2 = self._tracked(shuttle=(100, 115))
        features = self.extractor.extract(1, 0.1, t2)
        assert features.shuttle_direction == "down"

    def test_direction_reversal_increments_hit_count(self):
        """Direction reversal should increase rally hit count."""
        self.extractor.start_rally(0.0)

        # Moving down
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)
        t2 = self._tracked(shuttle=(100, 120))
        self.extractor.extract(1, 0.1, t2)

        # Now moving up (reversal)
        t3 = self._tracked(shuttle=(100, 100))
        features = self.extractor.extract(2, 0.2, t3)
        assert features.is_direction_reversal is True
        assert features.rally_hit_count >= 1

    def test_smash_detection(self):
        """Fast downward shuttle should trigger smash."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)

        # Large downward movement (velocity > 12, downward angle)
        t2 = self._tracked(shuttle=(105, 130))
        features = self.extractor.extract(1, 0.1, t2)
        assert features.is_smash is True

    def test_no_shuttle_grace_period(self):
        """Missing shuttle shouldn't immediately count as stationary."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)

        # Miss shuttle for 3 frames — should NOT increment stationary yet
        for i in range(3):
            t = self._tracked()
            features = self.extractor.extract(i + 1, 0.1 * (i + 1), t)

        assert features.shuttle_stationary_frames == 0  # Grace period = 5 frames

    def test_stationary_after_grace_period(self):
        """After 5+ frames of missing shuttle, stationary should increment."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)

        for i in range(8):
            t = self._tracked()
            features = self.extractor.extract(i + 1, 0.1 * (i + 1), t)

        assert features.shuttle_stationary_frames > 0

    def test_rally_lifecycle(self):
        """Start and end rally should work correctly."""
        self.extractor.start_rally(1.0)
        state = self.extractor.get_rally_state()
        assert state.active is True
        assert state.start_time == 1.0

        ended = self.extractor.end_rally()
        assert ended.active is False
        assert self.extractor.get_rally_state().active is False

    def test_player_displacement(self):
        """Player movement should accumulate displacement."""
        t1 = self._tracked(player_a=(100, 300))
        self.extractor.extract(0, 0.0, t1)

        t2 = self._tracked(player_a=(120, 300))
        features = self.extractor.extract(1, 0.1, t2)
        assert features.player_a_displacement > 0

    def test_reset(self):
        """Reset should clear all state."""
        t1 = self._tracked(shuttle=(100, 100))
        self.extractor.extract(0, 0.0, t1)
        self.extractor.start_rally(0.0)
        self.extractor.reset()
        assert self.extractor.get_rally_state().active is False
