"""Tests for CentroidTracker with CA-Kalman-filtered shuttle tracking."""

import pytest
from src.vision.tracker import CentroidTracker, TrackedObject, ShuttleKalmanFilter


class MockDetection:
    def __init__(self, cx, cy, bbox=None):
        self.center = [cx, cy]
        self.bbox = bbox or [cx - 10, cy - 10, cx + 10, cy + 10]


class TestCentroidTracker:
    def setup_method(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=80.0, court_net_y_ratio=0.5)

    def test_register_two_players(self):
        dets = [MockDetection(320, 300), MockDetection(320, 100)]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is not None
        assert result["player_b"] is not None
        assert result["player_a"].label == "player_a"
        assert result["player_b"].label == "player_b"

    def test_single_player_bottom(self):
        dets = [MockDetection(320, 250)]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is not None
        assert result["player_b"] is None

    def test_single_player_top(self):
        dets = [MockDetection(320, 100)]
        result = self.tracker.update(dets, None, 0.0, frame_height=360)
        assert result["player_a"] is None
        assert result["player_b"] is not None

    def test_shuttle_detection_and_tracking(self):
        shuttle = MockDetection(200, 150)
        result = self.tracker.update([], shuttle, 0.0, frame_height=360)
        assert result["shuttle"] is not None
        assert result["shuttle"].label == "shuttle"
        assert len(self.tracker.shuttle_trajectory) == 1

    def test_shuttle_kalman_prediction(self):
        self.tracker.update([], MockDetection(200, 150), 0.0, frame_height=360)
        self.tracker.update([], MockDetection(210, 155), 0.1, frame_height=360)
        result = self.tracker.update([], None, 0.2, frame_height=360)
        assert result["shuttle"] is not None
        assert result["shuttle"].is_predicted is True
        assert len(self.tracker.shuttle_trajectory) == 3

    def test_shuttle_deregister_after_long_gap(self):
        self.tracker.update([], MockDetection(200, 150), 0.0, frame_height=360)
        for i in range(25):
            self.tracker.update([], None, 0.1 * (i + 1), frame_height=360)
        result = self.tracker.update([], None, 3.0, frame_height=360)
        assert result["shuttle"] is None

    def test_player_identity_persists(self):
        dets1 = [MockDetection(320, 300), MockDetection(320, 100)]
        r1 = self.tracker.update(dets1, None, 0.0, frame_height=360)
        id_a = r1["player_a"].object_id
        id_b = r1["player_b"].object_id
        dets2 = [MockDetection(325, 305), MockDetection(315, 95)]
        r2 = self.tracker.update(dets2, None, 0.1, frame_height=360)
        assert r2["player_a"].object_id == id_a
        assert r2["player_b"].object_id == id_b

    def test_clear_and_reset(self):
        self.tracker.update([], MockDetection(200, 150), 0.0, frame_height=360)
        self.tracker.clear_shuttle_trajectory()
        assert len(self.tracker.shuttle_trajectory) == 0
        self.tracker.reset()
        result = self.tracker.update([], None, 0.0, frame_height=360)
        assert result["player_a"] is None
        assert result["shuttle"] is None

    def test_get_shuttle_kalman(self):
        """Kalman filter should be accessible for acceleration queries."""
        kf = self.tracker.get_shuttle_kalman()
        assert kf is not None
        assert not kf.initialized


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
        assert abs(predicted[0] - 100.0) < 15
        assert abs(predicted[1] - 200.0) < 15

    def test_velocity_and_acceleration(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        kf.predict()
        kf.correct(110.0, 200.0)
        vx, vy = kf.get_velocity()
        assert vx > 0  # Moving right
        ax, ay = kf.get_acceleration()
        # Acceleration should exist (may be small)
        assert isinstance(ax, float)

    def test_is_decelerating(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        # Initially not decelerating
        assert not kf.is_decelerating()

    def test_max_predict_limit(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        result = None
        for _ in range(10):
            result = kf.predict()
        assert result is None

    def test_get_speed(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        speed = kf.get_speed()
        assert speed >= 0.0

    def test_reset(self):
        kf = ShuttleKalmanFilter()
        kf.correct(100.0, 200.0)
        kf.reset()
        assert kf.predict() is None
        assert kf.get_velocity() == (0.0, 0.0)
        assert kf.get_acceleration() == (0.0, 0.0)
