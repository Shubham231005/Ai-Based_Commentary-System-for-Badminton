"""Tests for FeatureExtractor with shuttle phase state machine."""

import pytest
from src.vision.feature_extractor import FeatureExtractor, FrameFeatures, RallyState, ShuttlePhase


class MockTrackedObject:
    def __init__(self, cx, cy, is_predicted=False):
        self.centroid = (cx, cy)
        self.is_predicted = is_predicted


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor(
            smash_velocity_threshold=12.0, stationary_threshold=2.0,
            direction_reversal_threshold=3.0, frame_height=360,
        )

    def _tracked(self, player_a=None, player_b=None, shuttle=None):
        result = {}
        result["player_a"] = MockTrackedObject(*player_a) if player_a else None
        result["player_b"] = MockTrackedObject(*player_b) if player_b else None
        result["shuttle"] = MockTrackedObject(*shuttle) if shuttle else None
        return result

    def test_shuttle_velocity(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        features = self.extractor.extract(1, 0.1, self._tracked(shuttle=(110, 100)))
        assert features.shuttle_velocity > 0

    def test_shuttle_direction_up(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 200)))
        features = self.extractor.extract(1, 0.1, self._tracked(shuttle=(100, 190)))
        assert features.shuttle_direction == "up"

    def test_shuttle_direction_down(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        features = self.extractor.extract(1, 0.1, self._tracked(shuttle=(100, 115)))
        assert features.shuttle_direction == "down"

    def test_velocity_decay_rate(self):
        """Velocity decay rate should be negative when shuttle decelerates."""
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        self.extractor.extract(1, 0.1, self._tracked(shuttle=(120, 100)))  # fast
        features = self.extractor.extract(2, 0.2, self._tracked(shuttle=(125, 100)))  # slower
        assert features.velocity_decay_rate < 0  # decelerating

    def test_direction_reversal_tracks_last_hitter(self):
        """Direction reversal should track who hit last."""
        self.extractor.start_rally(0.0)
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        self.extractor.extract(1, 0.1, self._tracked(shuttle=(100, 120)))  # down
        features = self.extractor.extract(2, 0.2, self._tracked(shuttle=(100, 100)))  # up → reversal
        assert features.is_direction_reversal is True
        assert features.last_hitter == "Player A"  # Up direction ⇒ A hit it

    def test_smash_detection(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        features = self.extractor.extract(1, 0.1, self._tracked(shuttle=(105, 130)))
        assert features.is_smash is True

    def test_no_shuttle_grace_period(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        for i in range(3):
            features = self.extractor.extract(i + 1, 0.1 * (i + 1), self._tracked())
        assert features.shuttle_stationary_frames == 0

    def test_stationary_after_grace_period(self):
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        for i in range(8):
            features = self.extractor.extract(i + 1, 0.1 * (i + 1), self._tracked())
        assert features.shuttle_stationary_frames > 0

    def test_rally_lifecycle(self):
        self.extractor.start_rally(1.0)
        assert self.extractor.get_rally_state().active is True
        ended = self.extractor.end_rally()
        assert ended.active is False

    def test_player_displacement(self):
        self.extractor.extract(0, 0.0, self._tracked(player_a=(100, 300)))
        features = self.extractor.extract(1, 0.1, self._tracked(player_a=(120, 300)))
        assert features.player_a_displacement > 0

    def test_trajectory_buffer(self):
        """Trajectory buffer should accumulate shuttle positions."""
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        self.extractor.extract(1, 0.1, self._tracked(shuttle=(110, 110)))
        features = self.extractor.extract(2, 0.2, self._tracked(shuttle=(120, 120)))
        assert len(features.trajectory_buffer) == 3

    def test_kalman_acceleration_passthrough(self):
        """Kalman acceleration should be passed through to features."""
        features = self.extractor.extract(
            0, 0.0, self._tracked(shuttle=(100, 100)),
            kalman_ax=1.5, kalman_ay=-2.0
        )
        assert features.shuttle_ax == 1.5
        assert features.shuttle_ay == -2.0


class TestShuttlePhaseStateMachine:
    def setup_method(self):
        self.extractor = FeatureExtractor(frame_height=360)

    def _tracked(self, shuttle=None):
        return {
            "player_a": None, "player_b": None,
            "shuttle": MockTrackedObject(*shuttle) if shuttle else None,
        }

    def test_initial_phase_is_idle(self):
        assert self.extractor.get_phase() == ShuttlePhase.IDLE

    def test_phase_idle_to_hit(self):
        """Velocity spike should trigger HIT phase."""
        self.extractor.extract(0, 0.0, self._tracked(shuttle=(100, 100)))
        # Big movement = velocity spike
        features = self.extractor.extract(1, 0.033, self._tracked(shuttle=(110, 100)))
        assert self.extractor.get_phase() == ShuttlePhase.HIT

    def test_phase_set_externally(self):
        """Phase should be settable by event engine."""
        self.extractor.set_phase(ShuttlePhase.LANDING)
        assert self.extractor.get_phase() == ShuttlePhase.LANDING

    def test_phase_resets_on_rally_end(self):
        self.extractor.set_phase(ShuttlePhase.DESCENT)
        self.extractor.end_rally()
        assert self.extractor.get_phase() == ShuttlePhase.IDLE

    def test_reset_clears_phase(self):
        self.extractor.set_phase(ShuttlePhase.DESCENT)
        self.extractor.reset()
        assert self.extractor.get_phase() == ShuttlePhase.IDLE
