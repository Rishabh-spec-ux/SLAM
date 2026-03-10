import math
import random
from dataclasses import dataclass

import pygame


@dataclass
class LidarMeasurement:
    angle_deg: float
    distance: float
    hit_point: tuple[float, float]
    hit: bool


class LidarSensor:
    def __init__(
        self,
        max_range: float = 250.0,
        angle_step_deg: float = 1.0,
        ray_step_px: float = 1.0,
        obstacle_threshold: int = 100,
        distance_noise_std: float = 1.5,
        angle_noise_std_deg: float = 0.35,
    ) -> None:
        self.max_range = max_range
        self.angle_step_deg = angle_step_deg
        self.ray_step_px = ray_step_px
        self.obstacle_threshold = obstacle_threshold

        # Uncertainty model (Gaussian noise on angle and measured distance).
        self.distance_noise_std = distance_noise_std
        self.angle_noise_std_deg = angle_noise_std_deg

        self.measurements: list[LidarMeasurement] = []

    def _is_obstacle(self, surface: pygame.Surface, x: int, y: int) -> bool:
        r, g, b, _ = surface.get_at((x, y))
        return r < self.obstacle_threshold and g < self.obstacle_threshold and b < self.obstacle_threshold

    def _in_bounds(self, surface: pygame.Surface, x: float, y: float) -> bool:
        width, height = surface.get_size()
        return 0 <= int(x) < width and 0 <= int(y) < height

    def scan(
        self,
        robot_pos: tuple[float, float],
        map_surface: pygame.Surface,
        robot_heading_deg: float = 0.0,
    ) -> list[LidarMeasurement]:
        rx, ry = robot_pos
        self.measurements.clear()

        num_rays = int(360 / self.angle_step_deg)

        for i in range(num_rays):
            base_relative_angle_deg = i * self.angle_step_deg
            noisy_relative_angle_deg = base_relative_angle_deg + random.gauss(0.0, self.angle_noise_std_deg)
            world_angle_deg = robot_heading_deg + noisy_relative_angle_deg
            theta = math.radians(world_angle_deg)

            hit = False
            true_distance = self.max_range
            hit_x = rx + self.max_range * math.cos(theta)
            hit_y = ry + self.max_range * math.sin(theta)

            d = 0.0
            while d <= self.max_range:
                x = rx + d * math.cos(theta)
                y = ry + d * math.sin(theta)

                if not self._in_bounds(map_surface, x, y):
                    hit = True
                    true_distance = d
                    hit_x, hit_y = x, y
                    break

                xi, yi = int(x), int(y)
                if self._is_obstacle(map_surface, xi, yi):
                    hit = True
                    true_distance = d
                    hit_x, hit_y = x, y
                    break

                d += self.ray_step_px

            measured_distance = true_distance + random.gauss(0.0, self.distance_noise_std)
            measured_distance = max(0.0, min(measured_distance, self.max_range))

            measured_x = rx + measured_distance * math.cos(theta)
            measured_y = ry + measured_distance * math.sin(theta)

            self.measurements.append(
                LidarMeasurement(
                    angle_deg=noisy_relative_angle_deg % 360,
                    distance=measured_distance,
                    hit_point=(measured_x, measured_y),
                    hit=hit,
                )
            )

        return self.measurements

    def get_hit_points(self) -> list[tuple[float, float]]:
        return [m.hit_point for m in self.measurements if m.hit]

    def draw_rays(
        self,
        screen: pygame.Surface,
        robot_pos: tuple[float, float],
        ray_color: tuple[int, int, int] = (0, 220, 0),
        hit_color: tuple[int, int, int] = (255, 80, 80),
    ) -> None:
        rx, ry = int(robot_pos[0]), int(robot_pos[1])

        for m in self.measurements:
            hx, hy = int(m.hit_point[0]), int(m.hit_point[1])
            pygame.draw.line(screen, ray_color, (rx, ry), (hx, hy), 1)
            if m.hit:
                pygame.draw.circle(screen, hit_color, (hx, hy), 2)
