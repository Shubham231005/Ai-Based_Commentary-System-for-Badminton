"""
Tracker — Centroid tracking + Constant-Acceleration Kalman filter for shuttle.

Players: Centroid-based matching with spatial assignment (A=bottom, B=top).
Shuttle: CA-Kalman filter (state=[x, y, vx, vy, ax, ay]) that models
         deceleration explicitly, predicting through detection gaps and
         providing acceleration estimates for landing detection.
"""

import cv2
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object with persistent identity."""
    object_id: int
    label: str
    centroid: Tuple[float, float]
    bbox: List[float]
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    frames_since_seen: int = 0
    total_frames_tracked: int = 0
    is_predicted: bool = False


class ShuttleKalmanFilter:
    """
    Constant-Acceleration Kalman filter for shuttle tracking.

    State: [x, y, vx, vy, ax, ay] — position + velocity + acceleration
    Measurement: [x, y] — observed position

    The acceleration terms capture shuttle deceleration (drag), enabling:
    - More accurate gap bridging (accounts for slowdown)
    - Direct deceleration readout for landing detection
    - Better trajectory prediction during descent phase
    """

    def __init__(self, dt: float = 0.033):
        """
        Args:
            dt: Time step between frames (1/30 ≈ 0.033s for 30 FPS)
        """
        self.dt = dt
        self.kf = cv2.KalmanFilter(6, 2)  # 6 state dims, 2 measurement dims

        # State transition: constant acceleration model
        # x' = x + vx*dt + 0.5*ax*dt²
        # vx' = vx + ax*dt
        # ax' = ax (assumed constant between frames)
        dt2 = 0.5 * dt * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  dt2,  0  ],
            [0, 1, 0,  dt, 0,    dt2],
            [0, 0, 1,  0,  dt,   0  ],
            [0, 0, 0,  1,  0,    dt ],
            [0, 0, 0,  0,  1,    0  ],
            [0, 0, 0,  0,  0,    1  ],
        ], dtype=np.float32)

        # Measurement: we observe x, y only
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Process noise — tuned for shuttle dynamics
        # Higher noise on acceleration (it changes rapidly)
        q = np.diag([1.0, 1.0, 2.0, 2.0, 8.0, 8.0]).astype(np.float32)
        self.kf.processNoiseCov = q

        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0

        # Initial covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 10.0

        self._initialized = False
        self._frames_since_correction = 0
        self._max_predict_frames = 6  # CA-KF can predict farther than CV-KF

    def correct(self, x: float, y: float) -> Tuple[float, float]:
        """Update filter with an observed measurement."""
        measurement = np.array([[x], [y]], dtype=np.float32)

        if not self._initialized:
            self.kf.statePost = np.array(
                [[x], [y], [0], [0], [0], [0]], dtype=np.float32
            )
            self._initialized = True
            self._frames_since_correction = 0
            return (x, y)

        self.kf.predict()
        self.kf.correct(measurement)
        self._frames_since_correction = 0

        state = self.kf.statePost
        return (float(state[0, 0]), float(state[1, 0]))

    def predict(self) -> Optional[Tuple[float, float]]:
        """Predict next position without a measurement."""
        if not self._initialized:
            return None

        self._frames_since_correction += 1
        if self._frames_since_correction > self._max_predict_frames:
            return None

        predicted = self.kf.predict()
        return (float(predicted[0, 0]), float(predicted[1, 0]))

    def get_velocity(self) -> Tuple[float, float]:
        """Get estimated velocity (vx, vy) from filter state."""
        if not self._initialized:
            return (0.0, 0.0)
        state = self.kf.statePost
        return (float(state[2, 0]), float(state[3, 0]))

    def get_acceleration(self) -> Tuple[float, float]:
        """Get estimated acceleration (ax, ay) from filter state."""
        if not self._initialized:
            return (0.0, 0.0)
        state = self.kf.statePost
        return (float(state[4, 0]), float(state[5, 0]))

    def get_speed(self) -> float:
        """Get scalar speed from velocity vector."""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2))

    def is_decelerating(self) -> bool:
        """Check if shuttle is decelerating (drag slowing it down)."""
        vx, vy = self.get_velocity()
        ax, ay = self.get_acceleration()
        # Deceleration = acceleration opposing velocity direction
        dot = vx * ax + vy * ay
        return dot < -0.5  # Threshold for meaningful deceleration

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def frames_since_correction(self) -> int:
        return self._frames_since_correction

    def reset(self):
        """Reset the filter."""
        self.__init__(self.dt)


class CentroidTracker:
    """
    Multi-object tracker with CA-Kalman-filtered shuttle.

    Players: Centroid distance matching with spatial assignment.
    Shuttle: Constant-acceleration Kalman smooths positions, predicts
             through gaps, and provides velocity + acceleration readouts.
    """

    def __init__(
        self,
        max_disappeared: int = 15,
        max_distance: float = 80.0,
        court_net_y_ratio: float = 0.5,
    ):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.court_net_y_ratio = court_net_y_ratio

        self._next_id = 0
        self._objects: OrderedDict[int, TrackedObject] = OrderedDict()

        self._player_a_id: Optional[int] = None
        self._player_b_id: Optional[int] = None
        self._shuttle_id: Optional[int] = None

        # CA-Kalman for shuttle
        self._shuttle_kalman = ShuttleKalmanFilter()

        # Shuttle trajectory history
        self.shuttle_trajectory: List[Tuple[float, float, float]] = []

    def update(
        self,
        player_detections: list,
        shuttle_detection: Optional[object] = None,
        timestamp: float = 0.0,
        frame_height: int = 720,
    ) -> Dict[str, Optional[TrackedObject]]:
        """Update tracker with new frame detections."""
        net_y = frame_height * self.court_net_y_ratio
        self._update_players(player_detections, timestamp, net_y)
        self._update_shuttle_kalman(shuttle_detection, timestamp)

        return {
            "player_a": self._objects.get(self._player_a_id) if self._player_a_id is not None else None,
            "player_b": self._objects.get(self._player_b_id) if self._player_b_id is not None else None,
            "shuttle": self._objects.get(self._shuttle_id) if self._shuttle_id is not None else None,
        }

    def _update_players(self, detections: list, timestamp: float, net_y: float):
        """Update player tracking with spatial assignment."""
        if len(detections) == 0:
            for pid in [self._player_a_id, self._player_b_id]:
                if pid is not None and pid in self._objects:
                    self._objects[pid].frames_since_seen += 1
                    if self._objects[pid].frames_since_seen > self.max_disappeared:
                        self._deregister(pid)
                        if pid == self._player_a_id:
                            self._player_a_id = None
                        elif pid == self._player_b_id:
                            self._player_b_id = None
            return

        sorted_dets = sorted(detections, key=lambda d: d.center[1])
        if len(sorted_dets) >= 2:
            top_det = sorted_dets[0]
            bot_det = sorted_dets[-1]
        else:
            det = sorted_dets[0]
            if det.center[1] < net_y:
                top_det, bot_det = det, None
            else:
                top_det, bot_det = None, det

        if bot_det is not None:
            if self._player_a_id is not None and self._player_a_id in self._objects:
                self._update_object(self._player_a_id, bot_det.center, bot_det.bbox, timestamp)
            else:
                self._player_a_id = self._register("player_a", bot_det.center, bot_det.bbox, timestamp)

        if top_det is not None:
            if self._player_b_id is not None and self._player_b_id in self._objects:
                self._update_object(self._player_b_id, top_det.center, top_det.bbox, timestamp)
            else:
                self._player_b_id = self._register("player_b", top_det.center, top_det.bbox, timestamp)

    def _update_shuttle_kalman(self, detection: Optional[object], timestamp: float):
        """Update shuttle tracking with CA-Kalman filter."""
        if detection is not None:
            cx, cy = detection.center[0], detection.center[1]
            smoothed_x, smoothed_y = self._shuttle_kalman.correct(cx, cy)

            if self._shuttle_id is not None and self._shuttle_id in self._objects:
                self._update_object(self._shuttle_id, (smoothed_x, smoothed_y), detection.bbox, timestamp)
                self._objects[self._shuttle_id].is_predicted = False
            else:
                self._shuttle_id = self._register("shuttle", (smoothed_x, smoothed_y), detection.bbox, timestamp)

            self.shuttle_trajectory.append((smoothed_x, smoothed_y, timestamp))
        else:
            predicted = self._shuttle_kalman.predict()
            if predicted is not None and self._shuttle_id is not None and self._shuttle_id in self._objects:
                px, py = predicted
                obj = self._objects[self._shuttle_id]
                obj.centroid = (px, py)
                obj.is_predicted = True
                obj.frames_since_seen += 1
                obj.total_frames_tracked += 1
                obj.trajectory.append((px, py, timestamp))
                self.shuttle_trajectory.append((px, py, timestamp))
                if len(obj.trajectory) > 500:
                    obj.trajectory = obj.trajectory[-500:]
            elif self._shuttle_id is not None and self._shuttle_id in self._objects:
                self._objects[self._shuttle_id].frames_since_seen += 1
                if self._objects[self._shuttle_id].frames_since_seen > self.max_disappeared:
                    self._deregister(self._shuttle_id)
                    self._shuttle_id = None

    def get_shuttle_kalman(self) -> ShuttleKalmanFilter:
        """Get the shuttle Kalman filter for external queries."""
        return self._shuttle_kalman

    def _register(self, label, centroid, bbox, timestamp) -> int:
        obj_id = self._next_id
        self._next_id += 1
        self._objects[obj_id] = TrackedObject(
            object_id=obj_id, label=label, centroid=centroid, bbox=bbox,
            trajectory=[(centroid[0], centroid[1], timestamp)],
        )
        logger.debug(f"Registered {label} as ID {obj_id}")
        return obj_id

    def _update_object(self, obj_id, centroid, bbox, timestamp):
        obj = self._objects[obj_id]
        obj.centroid = centroid
        obj.bbox = bbox
        obj.frames_since_seen = 0
        obj.total_frames_tracked += 1
        obj.trajectory.append((centroid[0], centroid[1], timestamp))
        if len(obj.trajectory) > 500:
            obj.trajectory = obj.trajectory[-500:]

    def _deregister(self, obj_id):
        if obj_id in self._objects:
            label = self._objects[obj_id].label
            del self._objects[obj_id]
            logger.debug(f"Deregistered {label} (ID {obj_id})")

    def get_shuttle_trajectory(self, last_n=None):
        if last_n:
            return self.shuttle_trajectory[-last_n:]
        return self.shuttle_trajectory

    def clear_shuttle_trajectory(self):
        self.shuttle_trajectory.clear()
        self._shuttle_kalman.reset()

    def get_player_positions(self):
        return {
            "player_a": self._objects[self._player_a_id].centroid
                if self._player_a_id is not None and self._player_a_id in self._objects else None,
            "player_b": self._objects[self._player_b_id].centroid
                if self._player_b_id is not None and self._player_b_id in self._objects else None,
        }

    def reset(self):
        self._objects.clear()
        self._next_id = 0
        self._player_a_id = None
        self._player_b_id = None
        self._shuttle_id = None
        self.shuttle_trajectory.clear()
        self._shuttle_kalman.reset()
