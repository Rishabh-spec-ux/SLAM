import math
import random
from dataclasses import dataclass

import numpy as np
import pygame

from sensor import LidarMeasurement


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class LandmarkObservation:
    landmark_id: int
    range_m: float
    bearing_rad: float


@dataclass
class LoopClosureMatch:
    frame_idx: int
    matched_idx: int
    score: float
    target_pose: Pose2D


class OdometryModel:
    def __init__(self, linear_std: float = 2.0, angular_std: float = 0.02) -> None:
        self.linear_std = linear_std
        self.angular_std = angular_std

    def sample(self, v_cmd: float, w_cmd: float) -> tuple[float, float]:
        v_noisy = v_cmd + random.gauss(0.0, self.linear_std)
        w_noisy = w_cmd + random.gauss(0.0, self.angular_std)
        return v_noisy, w_noisy


def propagate_pose(pose: Pose2D, v: float, w: float, dt: float) -> Pose2D:
    if abs(w) < 1e-6:
        nx = pose.x + v * dt * math.cos(pose.theta)
        ny = pose.y + v * dt * math.sin(pose.theta)
        ntheta = pose.theta
    else:
        nx = pose.x - (v / w) * math.sin(pose.theta) + (v / w) * math.sin(pose.theta + w * dt)
        ny = pose.y + (v / w) * math.cos(pose.theta) - (v / w) * math.cos(pose.theta + w * dt)
        ntheta = pose.theta + w * dt

    return Pose2D(nx, ny, wrap_angle(ntheta))


class MapLandmarkExtractor:
    @staticmethod
    def _is_obstacle(surface: pygame.Surface, x: int, y: int, threshold: int) -> bool:
        r, g, b, _ = surface.get_at((x, y))
        return r < threshold and g < threshold and b < threshold

    @staticmethod
    def extract(
        surface: pygame.Surface,
        obstacle_threshold: int = 100,
        sample_step: int = 3,
        spacing_px: int = 16,
    ) -> list[tuple[int, float, float]]:
        width, height = surface.get_size()
        landmarks: list[tuple[int, float, float]] = []
        occupied_cells: set[tuple[int, int]] = set()

        for y in range(1, height - 1, sample_step):
            for x in range(1, width - 1, sample_step):
                if not MapLandmarkExtractor._is_obstacle(surface, x, y, obstacle_threshold):
                    continue

                has_free_neighbor = False
                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if nx == x and ny == y:
                            continue
                        if not MapLandmarkExtractor._is_obstacle(surface, nx, ny, obstacle_threshold):
                            has_free_neighbor = True
                            break
                    if has_free_neighbor:
                        break

                if not has_free_neighbor:
                    continue

                key = (x // spacing_px, y // spacing_px)
                if key in occupied_cells:
                    continue

                occupied_cells.add(key)
                landmarks.append((len(landmarks), float(x), float(y)))

        return landmarks


class LidarFrontend:
    def __init__(self, map_landmarks: list[tuple[int, float, float]], association_radius: float = 12.0) -> None:
        self.map_landmarks = map_landmarks
        self.association_radius_sq = association_radius * association_radius

    def _nearest_landmark(self, x: float, y: float) -> int | None:
        best_id: int | None = None
        best_d2 = self.association_radius_sq

        for landmark_id, lx, ly in self.map_landmarks:
            dx = lx - x
            dy = ly - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_id = landmark_id

        return best_id

    def build_observations(
        self,
        measurements: list[LidarMeasurement],
        robot_pose: Pose2D,
        hit_stride: int = 3,
        max_observations: int = 40,
    ) -> list[LandmarkObservation]:
        best_per_landmark: dict[int, LandmarkObservation] = {}

        for i, measurement in enumerate(measurements):
            if not measurement.hit or i % hit_stride != 0:
                continue

            bearing_rad = wrap_angle(math.radians(measurement.angle_deg))
            hit_x = robot_pose.x + measurement.distance * math.cos(robot_pose.theta + bearing_rad)
            hit_y = robot_pose.y + measurement.distance * math.sin(robot_pose.theta + bearing_rad)
            landmark_id = self._nearest_landmark(hit_x, hit_y)
            if landmark_id is None:
                continue

            obs = LandmarkObservation(
                landmark_id=landmark_id,
                range_m=measurement.distance,
                bearing_rad=bearing_rad,
            )

            prev = best_per_landmark.get(landmark_id)
            if prev is None or obs.range_m < prev.range_m:
                best_per_landmark[landmark_id] = obs

        observations = sorted(best_per_landmark.values(), key=lambda o: o.range_m)
        return observations[:max_observations]


class EKFSLAM:
    def __init__(
        self,
        initial_pose: Pose2D,
        range_noise_std: float = 3.0,
        bearing_noise_std: float = 0.04,
        motion_noise_lin: float = 0.08,
        motion_noise_ang: float = 0.08,
    ) -> None:
        self.mu = np.array([initial_pose.x, initial_pose.y, initial_pose.theta], dtype=float)
        self.sigma = np.eye(3, dtype=float) * 1.0

        self.range_noise_std = range_noise_std
        self.bearing_noise_std = bearing_noise_std
        self.motion_noise_lin = motion_noise_lin
        self.motion_noise_ang = motion_noise_ang

        self.landmark_indices: dict[int, int] = {}

    def get_robot_pose(self) -> Pose2D:
        return Pose2D(float(self.mu[0]), float(self.mu[1]), float(wrap_angle(self.mu[2])))

    def get_landmarks(self) -> dict[int, tuple[float, float]]:
        landmarks: dict[int, tuple[float, float]] = {}
        for landmark_id, index in self.landmark_indices.items():
            landmarks[landmark_id] = (float(self.mu[index]), float(self.mu[index + 1]))
        return landmarks

    def predict(self, v: float, w: float, dt: float) -> None:
        x, y, theta = self.mu[0], self.mu[1], self.mu[2]

        if abs(w) < 1e-6:
            nx = x + v * dt * math.cos(theta)
            ny = y + v * dt * math.sin(theta)
            ntheta = theta

            g_theta_x = -v * dt * math.sin(theta)
            g_theta_y = v * dt * math.cos(theta)
        else:
            nx = x - (v / w) * math.sin(theta) + (v / w) * math.sin(theta + w * dt)
            ny = y + (v / w) * math.cos(theta) - (v / w) * math.cos(theta + w * dt)
            ntheta = theta + w * dt

            g_theta_x = -(v / w) * math.cos(theta) + (v / w) * math.cos(theta + w * dt)
            g_theta_y = -(v / w) * math.sin(theta) + (v / w) * math.sin(theta + w * dt)

        self.mu[0], self.mu[1], self.mu[2] = nx, ny, wrap_angle(ntheta)

        n = self.mu.shape[0]
        g = np.eye(n, dtype=float)
        g[0, 2] = g_theta_x
        g[1, 2] = g_theta_y

        sigma_trans = 0.25 + self.motion_noise_lin * abs(v) * dt
        sigma_rot = 0.01 + self.motion_noise_ang * abs(w) * dt
        r3 = np.diag([sigma_trans * sigma_trans, sigma_trans * sigma_trans, sigma_rot * sigma_rot])

        r = np.zeros((n, n), dtype=float)
        r[0:3, 0:3] = r3

        self.sigma = g @ self.sigma @ g.T + r
        self.sigma = 0.5 * (self.sigma + self.sigma.T)

    def _augment_landmark(self, observation: LandmarkObservation) -> None:
        x, y, theta = self.mu[0], self.mu[1], self.mu[2]
        r, b = observation.range_m, observation.bearing_rad

        lx = x + r * math.cos(theta + b)
        ly = y + r * math.sin(theta + b)

        old_n = self.mu.shape[0]
        self.mu = np.concatenate([self.mu, np.array([lx, ly], dtype=float)])

        q = np.diag([self.range_noise_std * self.range_noise_std, self.bearing_noise_std * self.bearing_noise_std])

        jr = np.array(
            [
                [1.0, 0.0, -r * math.sin(theta + b)],
                [0.0, 1.0, r * math.cos(theta + b)],
            ],
            dtype=float,
        )
        jz = np.array(
            [
                [math.cos(theta + b), -r * math.sin(theta + b)],
                [math.sin(theta + b), r * math.cos(theta + b)],
            ],
            dtype=float,
        )

        sigma_new = np.zeros((old_n + 2, old_n + 2), dtype=float)
        sigma_new[:old_n, :old_n] = self.sigma

        sigma_lx = jr @ self.sigma[0:3, :]
        sigma_xl = self.sigma[:, 0:3] @ jr.T
        sigma_ll = jr @ self.sigma[0:3, 0:3] @ jr.T + jz @ q @ jz.T

        sigma_new[old_n:, :old_n] = sigma_lx
        sigma_new[:old_n, old_n:] = sigma_xl
        sigma_new[old_n:, old_n:] = sigma_ll

        self.sigma = sigma_new
        self.landmark_indices[observation.landmark_id] = old_n

    def update(self, observations: list[LandmarkObservation]) -> None:
        if not observations:
            return

        q = np.diag([self.range_noise_std * self.range_noise_std, self.bearing_noise_std * self.bearing_noise_std])

        for observation in observations:
            if observation.landmark_id not in self.landmark_indices:
                self._augment_landmark(observation)
                continue

            index = self.landmark_indices[observation.landmark_id]
            x, y, theta = self.mu[0], self.mu[1], self.mu[2]
            lx, ly = self.mu[index], self.mu[index + 1]

            dx = lx - x
            dy = ly - y
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-9:
                continue

            dist = math.sqrt(dist_sq)
            predicted = np.array([dist, wrap_angle(math.atan2(dy, dx) - theta)], dtype=float)

            n = self.mu.shape[0]
            h = np.zeros((2, n), dtype=float)

            h[0, 0] = -dx / dist
            h[0, 1] = -dy / dist
            h[1, 0] = dy / dist_sq
            h[1, 1] = -dx / dist_sq
            h[1, 2] = -1.0

            h[0, index] = dx / dist
            h[0, index + 1] = dy / dist
            h[1, index] = -dy / dist_sq
            h[1, index + 1] = dx / dist_sq

            innovation = np.array(
                [
                    observation.range_m - predicted[0],
                    wrap_angle(observation.bearing_rad - predicted[1]),
                ],
                dtype=float,
            )

            s = h @ self.sigma @ h.T + q
            k = self.sigma @ h.T @ np.linalg.inv(s)

            self.mu = self.mu + k @ innovation
            self.mu[2] = wrap_angle(self.mu[2])

            i = np.eye(n, dtype=float)
            self.sigma = (i - k @ h) @ self.sigma
            self.sigma = 0.5 * (self.sigma + self.sigma.T)

    def apply_loop_closure(self, target_pose: Pose2D) -> None:
        current = self.get_robot_pose()

        dtheta = wrap_angle(target_pose.theta - current.theta)
        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)

        tx = target_pose.x - (cos_t * current.x - sin_t * current.y)
        ty = target_pose.y - (sin_t * current.x + cos_t * current.y)

        # Transform robot position.
        rx = self.mu[0]
        ry = self.mu[1]
        self.mu[0] = cos_t * rx - sin_t * ry + tx
        self.mu[1] = sin_t * rx + cos_t * ry + ty
        self.mu[2] = wrap_angle(self.mu[2] + dtheta)

        # Apply one rigid transform to all landmarks to reduce global drift.
        for index in self.landmark_indices.values():
            lx = self.mu[index]
            ly = self.mu[index + 1]
            self.mu[index] = cos_t * lx - sin_t * ly + tx
            self.mu[index + 1] = sin_t * lx + cos_t * ly + ty

        self.sigma = self.sigma + np.eye(self.sigma.shape[0], dtype=float) * 0.05


class LoopClosureDetector:
    def __init__(
        self,
        min_separation: int = 120,
        distance_threshold: float = 35.0,
        descriptor_threshold: float = 8.0,
        cooldown_frames: int = 100,
    ) -> None:
        self.min_separation = min_separation
        self.distance_threshold = distance_threshold
        self.descriptor_threshold = descriptor_threshold
        self.cooldown_frames = cooldown_frames

        self.pose_history: list[Pose2D] = []
        self.descriptor_history: list[np.ndarray] = []
        self.last_closure_frame = -10_000

    def _build_descriptor(self, measurements: list[LidarMeasurement], bins: int = 36) -> np.ndarray:
        distances = np.array([m.distance for m in measurements], dtype=float)
        if distances.size == 0:
            return np.zeros(bins, dtype=float)

        chunks = np.array_split(distances, bins)
        descriptor = np.array([float(np.mean(chunk)) if chunk.size else 0.0 for chunk in chunks], dtype=float)
        return descriptor

    def register_and_detect(
        self,
        frame_idx: int,
        pose: Pose2D,
        measurements: list[LidarMeasurement],
    ) -> LoopClosureMatch | None:
        descriptor = self._build_descriptor(measurements)
        best_idx = -1
        best_score = float("inf")

        if frame_idx - self.last_closure_frame > self.cooldown_frames and len(self.pose_history) > self.min_separation:
            search_limit = len(self.pose_history) - self.min_separation
            for idx in range(search_limit):
                old_pose = self.pose_history[idx]
                dx = pose.x - old_pose.x
                dy = pose.y - old_pose.y
                if dx * dx + dy * dy > self.distance_threshold * self.distance_threshold:
                    continue

                score = float(np.mean(np.abs(descriptor - self.descriptor_history[idx])))
                if score < best_score:
                    best_score = score
                    best_idx = idx

        self.pose_history.append(pose)
        self.descriptor_history.append(descriptor)

        if best_idx == -1 or best_score > self.descriptor_threshold:
            return None

        self.last_closure_frame = frame_idx
        matched_pose = self.pose_history[best_idx]
        return LoopClosureMatch(frame_idx, best_idx, best_score, matched_pose)
