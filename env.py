from pathlib import Path

import pygame


class Environment:
    def __init__(self, map_filename: str = "map.png") -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.map_path = self.base_dir / map_filename

        if not self.map_path.exists():
            raise FileNotFoundError(f"Map file not found: {self.map_path}")

        pygame.init()

        raw_map_surface = pygame.image.load(str(self.map_path))
        self.width, self.height = raw_map_surface.get_size()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SLAM Environment")
        self.map_surface = raw_map_surface.convert()

        # Stores SLAM point cloud as (x, y) world/map points.
        self.point_cloud: list[tuple[float, float]] = []

    def add_point(self, x: float, y: float) -> None:
        self.point_cloud.append((x, y))

    def extend_point_cloud(self, points: list[tuple[float, float]]) -> None:
        self.point_cloud.extend(points)

    def clear_point_cloud(self) -> None:
        self.point_cloud.clear()

    def draw(self, update_display: bool = True) -> None:
        self.screen.blit(self.map_surface, (0, 0))

        for x, y in self.point_cloud:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), 2)

        if update_display:
            pygame.display.flip()
