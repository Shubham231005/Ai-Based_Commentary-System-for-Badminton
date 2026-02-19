"""Tests for CentroidTracker with Kalman-filtered shuttle tracking."""

import pytest
from unittest.mock import MagicMock
from src.vision.tracker import CentroidTracker, TrackedObject, ShuttleKalmanFilter


class MockDetection:
    """Minimal detection mock."""
    def __init__(self, cx, cy, bbox=None):
        self.center = [cx, cy]
        self.bbox = bbox or [cx - 10, cy - 10, cx + 10, cy + 10]


# --- CentroidTracker Tests ---

class TestCentroidTracker:
    def setup_method(self):
        self.tracker = CentroidTracker(
            max_disappeared=5,
            max_distance=80.0,
            court_net_y_ratio=0.5,
        )

    def test_register_two_players(self):
        """Two player detections should register as player_a (bottom) and player_b (top)."""
        dets = [
            MockDetection(320, 300),  # bottom → player_a
            MockDetection(320, 100),  # top → player_b
        ]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is not None
        assert result["player_b"] is not None
        assert result["player_a"].label == "player_a"
        assert result["player_b"].label == "player_b"

    def test_single_player_bottom(self):
        """One detection below net → player_a."""
        dets = [MockDetection(320, 250)]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is not None
        assert result["player_b"] is None

    def test_single_player_top(self):
        """One detection above net → player_b."""
        dets = [MockDetection(320, 100)]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is None
        assert result["player_b"] is not None

    def test_shuttle_detection_and_tracking(self):
        """Shuttle detection should register and track."""
        shuttle = MockDetection(200, 150)
        result = self.tracker.update([], shuttle, 0.0, frame_height=360)
        assert result["shuttle"] is not None
        assert result["shuttle"].label == "shuttle"
        assert len(self.tracker.shuttle_trajectory) == 1

    def test_shuttle_kalman_prediction(self):
        """Missing shuttle should be predicted by Kalman for up to max_predict_frames."""
        # Give two observations so Kalman has velocity
        shuttle1 = MockDetection(200, 150)
        self.tracker.update([], shuttle1, 0.0, frame_height=360)
        shuttle2 = MockDetection(210, 155)
        self.tracker.update([], shuttle2, 0.1, frame_height=360)

        # Now miss the shuttle for a few frames — Kalman should predict
        result = self.tracker.update([], None, 0.2, frame_height=360)
        assert result["shuttle"] is not None
        assert result["shuttle"].is_predicted is True
        # Position should advance roughly in the direction of travel
        assert len(self.tracker.shuttle_trajectory) == 3

    def test_shuttle_deregister_after_long_gap(self):
        """Shuttle should be deregistered after exceeding Kalman max + max_disappeared."""
        shuttle = MockDetection(200, 150)
        self.tracker.update([], shuttle, 0.0, frame_height=360)

        # Miss for many frames beyond Kalman bridge + max_disappeared
        for i in range(25):
            self.tracker.update([], None, 0.1 * (i + 1), frame_height=360)

        result = self.tracker.update([], None, 3.0, frame_height=360)
        assert result["shuttle"] is None

    def test_player_identity_persists(self):
        """Player IDs should persist across frames."""
        dets1 = [MockDetection(320, 300), MockDetection(320, 100)]
        r1 = self.tracker.update(dets1, None, 0.0, frame_height=360)
        id_a = r1["player_a"].object_id
        id_b = r1["player_b"].object_id

        dets2 = [MockDetection(325, 305), MockDetection(315, 95)]
        r2 = self.tracker.update(dets2, None, 0.1, frame_height=360)
        assert r2["player_a"].object_id == id_a
        assert r2["player_b"].object_id == id_b

    def test_clear_shuttle_trajectory(self):
        """Clearing trajectory should also reset Kalman filter."""
        shuttle = MockDetection(200, 150)
        self.tracker.update([], shuttle, 0.0, frame_height=360)
        self.tracker.clear_shuttle_trajectory()
        assert len(self.tracker.shuttle_trajectory) == 0

    def test_reset(self):
        """Full reset should clear all state."""
        dets = [MockDetection(320, 300)]
        self.tracker.update(dets, MockDetection(200, 150), 0.0, frame_height=360)
        self.tracker.reset()
        result = self.tracker.update([], None, 0.0, frame_height=360)
        assert result["player_a"] is None
        assert result["shuttle"] is None


# --- Kalman Filter Tests ---

class TestShuttleKalmanFilter:
    def test_initial_correction(self):
        kf = ShuttleKalmanFilter()
        pos = kf.correct(100.0, 200.0)
        assert pos == (100.0, 200.0)

    def test_predict_without_init(self):
        kf = ShuttleKalmanFilter()
        assert kf.predict() is None

    def test_predict_after_correction(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        predicted = kf.predict()
        assert predicted is not None
        # Should be near initial position (velocity ~0)
        assert abs(predicted[0] - 100.0) < 10
        assert abs(predicted[1] - 200.0) < 10

    def test_velocity_estimate(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        kf.predict()
        kf.correct(110.0, 200.0)  # Moved 10px right
        vx, vy = kf.get_velocity()
        # Should estimate positive x velocity
        assert vx > 0

    def test_max_predict_limit(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        # Predict beyond limit
        for _ in range(10):
            result = kf.predict()
        assert result is None  # Should return None after max_predict_frames

    def test_reset(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        kf.reset()
        assert kf.predict() is None
        assert kf.get_velocity() == (0.0, 0.0)
