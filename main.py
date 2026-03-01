import math
from datetime import datetime
from pathlib import Path

import pygame

from env import Environment
from sensor import LidarSensor
from slam import (
    EKFSLAM,
    LidarFrontend,
    LoopClosureDetector,
    MapLandmarkExtractor,
    OdometryModel,
    Pose2D,
    propagate_pose,
    wrap_angle,
)


def draw_trajectory(screen: pygame.Surface, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    pygame.draw.lines(screen, color, False, [(int(x), int(y)) for x, y in points], 2)


def is_obstacle(surface: pygame.Surface, x: float, y: float, threshold: int) -> bool:
    xi, yi = int(x), int(y)
    width, height = surface.get_size()
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return True

    r, g, b, _ = surface.get_at((xi, yi))
    return r < threshold and g < threshold and b < threshold


def is_pose_blocked(surface: pygame.Surface, x: float, y: float, radius: int, threshold: int) -> bool:
    # Check center and robot footprint ring.
    if is_obstacle(surface, x, y, threshold):
        return True

    for angle_deg in range(0, 360, 20):
        angle_rad = math.radians(angle_deg)
        px = x + radius * math.cos(angle_rad)
        py = y + radius * math.sin(angle_rad)
        if is_obstacle(surface, px, py, threshold):
            return True

    return False


def find_free_start(surface: pygame.Surface, x: float, y: float, radius: int, threshold: int) -> tuple[float, float]:
    if not is_pose_blocked(surface, x, y, radius, threshold):
        return x, y

    for dist in range(3, 220, 3):
        for angle_deg in range(0, 360, 10):
            angle_rad = math.radians(angle_deg)
            px = x + dist * math.cos(angle_rad)
            py = y + dist * math.sin(angle_rad)
            if not is_pose_blocked(surface, px, py, radius, threshold):
                return px, py

    raise RuntimeError("Could not find a collision-free robot start pose on the map")


def propagate_with_collision(
    pose: Pose2D,
    v_cmd: float,
    w_cmd: float,
    dt: float,
    surface: pygame.Surface,
    robot_radius: int,
    obstacle_threshold: int,
) -> tuple[Pose2D, bool]:
    rotated_pose = propagate_pose(pose, 0.0, w_cmd, dt)
    translated_pose = propagate_pose(rotated_pose, v_cmd, 0.0, dt)

    if is_pose_blocked(surface, translated_pose.x, translated_pose.y, robot_radius, obstacle_threshold):
        return rotated_pose, False

    return translated_pose, True


def update_mapping_surface(mapping_surface: pygame.Surface, hit_points: list[tuple[float, float]]) -> None:
    for hx, hy in hit_points:
        pygame.draw.circle(mapping_surface, (0, 0, 0), (int(hx), int(hy)), 1)


def save_slam_map(
    map_surface: pygame.Surface,
    true_traj: list[tuple[float, float]],
    est_traj: list[tuple[float, float]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_map = map_surface.copy()
    if len(true_traj) > 1:
        pygame.draw.lines(final_map, (50, 180, 50), False, [(int(x), int(y)) for x, y in true_traj], 1)
    if len(est_traj) > 1:
        pygame.draw.lines(final_map, (60, 120, 220), False, [(int(x), int(y)) for x, y in est_traj], 1)

    filename = f"slam_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    target = output_dir / filename
    pygame.image.save(final_map, str(target))
    return target


def main() -> None:
    env = Environment("map.png")

    obstacle_threshold = 100
    robot_radius = 6

    lidar = LidarSensor(
        max_range=280.0,
        angle_step_deg=2.0,
        ray_step_px=1.0,
        obstacle_threshold=obstacle_threshold,
        distance_noise_std=1.5,
        angle_noise_std_deg=0.35,
    )

    start_x, start_y = find_free_start(env.map_surface, env.width / 2.0, env.height / 2.0, robot_radius, obstacle_threshold)
    true_pose = Pose2D(start_x, start_y, 0.0)

    odom = OdometryModel(linear_std=2.0, angular_std=0.02)
    ekf = EKFSLAM(initial_pose=Pose2D(true_pose.x + 10.0, true_pose.y - 10.0, 0.1))

    map_landmarks = MapLandmarkExtractor.extract(env.map_surface, obstacle_threshold=obstacle_threshold, sample_step=3, spacing_px=16)
    frontend = LidarFrontend(map_landmarks, association_radius=12.0)
    loop_closure = LoopClosureDetector(
        min_separation=120,
        distance_threshold=35.0,
        descriptor_threshold=8.0,
        cooldown_frames=120,
    )

    true_traj: list[tuple[float, float]] = []
    est_traj: list[tuple[float, float]] = []
    frame_idx = 0

    mapping_surface = pygame.Surface((env.width, env.height))
    mapping_surface.fill((255, 255, 255))
    mapping_active = True
    saved_map_path = ""

    loop_closure_banner_frames = 0
    last_loop_closure_text = ""
    collision_banner_frames = 0

    speed_linear = 90.0
    speed_angular = 1.4

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                if mapping_active:
                    saved_file = save_slam_map(mapping_surface, true_traj, est_traj, Path("maps"))
                    saved_map_path = str(saved_file)
                    mapping_active = False
                else:
                    mapping_active = True

        keys = pygame.key.get_pressed()

        v_cmd = 0.0
        w_cmd = 0.0
        if keys[pygame.K_w]:
            v_cmd += speed_linear
        if keys[pygame.K_s]:
            v_cmd -= speed_linear
        if keys[pygame.K_a]:
            w_cmd -= speed_angular
        if keys[pygame.K_d]:
            w_cmd += speed_angular

        previous_pose = true_pose
        true_pose, moved = propagate_with_collision(
            true_pose,
            v_cmd,
            w_cmd,
            dt,
            env.map_surface,
            robot_radius,
            obstacle_threshold,
        )
        if not moved and abs(v_cmd) > 0.0:
            collision_banner_frames = 20

        dx = true_pose.x - previous_pose.x
        dy = true_pose.y - previous_pose.y
        traveled = math.sqrt(dx * dx + dy * dy)
        direction = 1.0 if v_cmd >= 0.0 else -1.0
        effective_v = direction * (traveled / dt) if dt > 1e-9 else 0.0
        effective_w = wrap_angle(true_pose.theta - previous_pose.theta) / dt if dt > 1e-9 else 0.0

        odom_v, odom_w = odom.sample(effective_v, effective_w)
        ekf.predict(odom_v, odom_w, dt)

        measurements = lidar.scan(
            (true_pose.x, true_pose.y),
            env.map_surface,
            robot_heading_deg=math.degrees(true_pose.theta),
        )

        observations = frontend.build_observations(
            measurements,
            true_pose=true_pose,
            hit_stride=3,
            max_observations=40,
        )
        ekf.update(observations)

        match = loop_closure.register_and_detect(frame_idx, ekf.get_robot_pose(), measurements)
        if match is not None:
            ekf.apply_loop_closure(match.target_pose)
            last_loop_closure_text = f"Loop closure: frame {match.frame_idx} -> {match.matched_idx} (score={match.score:.2f})"
            loop_closure_banner_frames = 160

        estimated_pose = ekf.get_robot_pose()
        true_traj.append((true_pose.x, true_pose.y))
        est_traj.append((estimated_pose.x, estimated_pose.y))

        if len(true_traj) > 2400:
            true_traj.pop(0)
        if len(est_traj) > 2400:
            est_traj.pop(0)

        hit_points = lidar.get_hit_points()
        if mapping_active:
            update_mapping_surface(mapping_surface, hit_points)

        env.clear_point_cloud()
        env.extend_point_cloud(hit_points)

        env.draw(update_display=False)
        lidar.draw_rays(env.screen, (true_pose.x, true_pose.y), ray_color=(0, 180, 0), hit_color=(255, 90, 90))

        draw_trajectory(env.screen, true_traj, (0, 230, 0))
        draw_trajectory(env.screen, est_traj, (80, 140, 255))

        for lx, ly in ekf.get_landmarks().values():
            pygame.draw.circle(env.screen, (255, 220, 40), (int(lx), int(ly)), 2)

        pygame.draw.circle(env.screen, (40, 235, 40), (int(true_pose.x), int(true_pose.y)), robot_radius)
        pygame.draw.circle(env.screen, (40, 120, 255), (int(estimated_pose.x), int(estimated_pose.y)), max(3, robot_radius - 1))

        lines = [
            "Controls: W forward | S backward | A left turn | D right turn",
            "Press M to stop mapping and save. Press M again to resume mapping.",
            f"Mapping: {'RUNNING' if mapping_active else 'STOPPED'}",
            f"Measurements: {len(measurements)} | EKF observations: {len(observations)} | Landmarks: {len(ekf.landmark_indices)}",
            f"True pose: ({true_pose.x:.1f}, {true_pose.y:.1f}, {math.degrees(true_pose.theta):.1f} deg)",
            f"EKF pose:  ({estimated_pose.x:.1f}, {estimated_pose.y:.1f}, {math.degrees(estimated_pose.theta):.1f} deg)",
        ]

        if saved_map_path:
            lines.append(f"Saved map: {saved_map_path}")

        y_offset = 10
        for line in lines:
            text = font.render(line, True, (245, 245, 245))
            bg = pygame.Surface((text.get_width() + 10, text.get_height() + 6), pygame.SRCALPHA)
            bg.fill((15, 15, 15, 170))
            env.screen.blit(bg, (10, y_offset))
            env.screen.blit(text, (15, y_offset + 3))
            y_offset += text.get_height() + 8

        if collision_banner_frames > 0:
            c_text = font.render("Collision: obstacle detected, translation blocked", True, (255, 220, 120))
            c_bg = pygame.Surface((c_text.get_width() + 14, c_text.get_height() + 8), pygame.SRCALPHA)
            c_bg.fill((70, 30, 10, 190))
            env.screen.blit(c_bg, (10, y_offset + 5))
            env.screen.blit(c_text, (17, y_offset + 9))
            collision_banner_frames -= 1

        if loop_closure_banner_frames > 0:
            lc_text = font.render(last_loop_closure_text, True, (255, 245, 120))
            lc_bg = pygame.Surface((lc_text.get_width() + 14, lc_text.get_height() + 8), pygame.SRCALPHA)
            lc_bg.fill((70, 40, 10, 190))
            env.screen.blit(lc_bg, (10, y_offset + 37))
            env.screen.blit(lc_text, (17, y_offset + 41))
            loop_closure_banner_frames -= 1

        pygame.display.flip()
        frame_idx += 1

    pygame.quit()


if __name__ == "__main__":
    main()
