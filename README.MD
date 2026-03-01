# Complete 2D SLAM Pipeline (Pygame)

This project now implements a full end-to-end SLAM simulation loop with:

- robot motion model
- noisy odometry
- 360-degree LiDAR simulation
- scan-to-landmark frontend
- EKF-SLAM state estimation
- loop-closure detection and global drift correction
- realtime visualization of true pose, estimated pose, map points, rays, and trajectory

The implementation is educational and modular. It is complete enough to study how each SLAM block interacts in one runnable system.

## Repository Structure

- `env.py`: loads map image, owns the render surface, and stores a point cloud.
- `sensor.py`: simulates LiDAR range sensing by ray casting over `map.png`.
- `slam.py`: SLAM core (motion model, odometry noise, landmark extraction, LiDAR frontend, EKF-SLAM, loop closure).
- `main.py`: runtime loop connecting all components.
- `map.png`: occupancy-like map image (dark pixels treated as obstacles).

## How To Run

```bash
python3 main.py
```

## Controls

- `W`: move forward
- `S`: move backward
- `A`: rotate left
- `D`: rotate right
- `M`: stop mapping and save the built SLAM map to `maps/`; press `M` again to resume mapping

## System Data Flow

Per frame, `main.py` executes:

1. Compute `W/A/S/D` control command `(v_cmd, w_cmd)`.
2. Update **true pose** with kinematic model and collision checking against black obstacles.
3. Create **noisy odometry** sample `(v_odom, w_odom)` from effective executed motion.
4. EKF predict step with odometry.
5. Run LiDAR scan on map at true pose.
6. Convert LiDAR hits into landmark observations.
7. EKF update step with observations.
8. Run loop-closure detector from scan descriptor + revisited position.
9. If loop closes, apply global rigid correction to robot and landmarks.
10. If mapping is active, write scan hit points into the generated map image.
11. Render map, rays, point cloud, trajectories, poses, and status text.

## Detailed File Documentation

## `env.py`

### Purpose

Environment and rendering abstraction.

### Class: `Environment`

### `__init__(map_filename="map.png")`

- Resolves map path relative to script directory.
- Validates that the map exists.
- Initializes pygame.
- Loads the map image as a surface.
- Sets window size equal to map size.
- Converts map surface to display pixel format for faster drawing.
- Initializes `point_cloud` as list of `(x, y)` tuples.

### `add_point(x, y)`

Appends one point into `point_cloud`.

### `extend_point_cloud(points)`

Appends many points at once.

### `clear_point_cloud()`

Removes current cloud content.

### `draw(update_display=True)`

- Blits base map.
- Draws cloud points as red circles.
- Optionally flips display.

This allows `main.py` to draw overlays (rays, trajectories, UI) before final `pygame.display.flip()`.

## `sensor.py`

### Purpose

Physics-style LiDAR sensor simulation with uncertainty.

### Dataclass: `LidarMeasurement`

One beam result:

- `angle_deg`: final ray angle after angular noise.
- `distance`: final measured range after range noise.
- `hit_point`: endpoint `(x, y)` in world/map coordinates.
- `hit`: whether beam hit obstacle or map boundary before max range.

### Class: `LidarSensor`

### Constructor parameters

- `max_range`: max scan distance in pixels.
- `angle_step_deg`: beam angular spacing.
- `ray_step_px`: ray marching step size.
- `obstacle_threshold`: dark-pixel threshold for occupancy.
- `distance_noise_std`: Gaussian range noise sigma.
- `angle_noise_std_deg`: Gaussian angle noise sigma.

### Internal helpers

- `_is_obstacle(surface, x, y)`: checks pixel darkness.
- `_in_bounds(surface, x, y)`: bounds check.

### `scan(robot_pos, map_surface, robot_heading_deg=0.0)`

For each beam in 360 degrees:

1. Create nominal beam angle.
2. Add Gaussian angle noise.
3. March along ray until obstacle or boundary or max range.
4. Add Gaussian range noise to true range.
5. Clamp noisy distance to valid limits.
6. Build `LidarMeasurement` and append.

Returns full scan measurements list.

### `get_hit_points()`

Returns only hit endpoints. Used as visual point cloud.

### `draw_rays(screen, robot_pos, ...)`

Draws beams and hit markers.

## `slam.py`

`slam.py` is the complete SLAM backend/frontend glue.

## Geometry and utility

### `wrap_angle(angle)`

Normalizes angle to `[-pi, pi]`.

### `Pose2D`

Robot pose container:

- `x`, `y`, `theta`

### `propagate_pose(pose, v, w, dt)`

Unicycle motion model used for true robot movement.

If `|w|` is near zero:

- `x' = x + v*dt*cos(theta)`
- `y' = y + v*dt*sin(theta)`
- `theta' = theta`

Else:

- `x' = x - (v/w)sin(theta) + (v/w)sin(theta + wdt)`
- `y' = y + (v/w)cos(theta) - (v/w)cos(theta + wdt)`
- `theta' = theta + wdt`

## Odometry model

### `OdometryModel`

`sample(v_cmd, w_cmd)` returns noisy odometry controls:

- `v_noisy = v_cmd + N(0, linear_std)`
- `w_noisy = w_cmd + N(0, angular_std)`

These odometry controls drive EKF prediction.

## Landmark extraction from map

### `MapLandmarkExtractor`

Builds a static set of map landmarks from obstacle boundaries.

Algorithm:

1. Sample map pixels with stride (`sample_step`).
2. Keep only dark pixels (obstacles).
3. Keep only boundary-like obstacle pixels with at least one free neighbor.
4. Downsample using coarse grid (`spacing_px`) to avoid dense duplicates.
5. Assign integer IDs.

Output: list of `(landmark_id, x, y)`.

## LiDAR frontend / data association

### `LandmarkObservation`

Observation model packet:

- `landmark_id`
- `range_m`
- `bearing_rad`

### `LidarFrontend`

Transforms raw LiDAR beams into EKF observations.

Steps:

1. Take only hit beams (optionally stride-subsampled).
2. For each hit endpoint, find nearest extracted map landmark within association radius.
3. Convert beam to `(range, bearing)` relative to current true pose.
4. Keep one best observation per landmark (closest range).
5. Return sorted observation list (capped by `max_observations`).

This is the scan frontend for EKF measurement updates.

## EKF-SLAM core

### `EKFSLAM`

State vector:

- `mu = [x, y, theta, l1x, l1y, l2x, l2y, ...]^T`

Covariance:

- `sigma` with matching dimensions.

Landmark bookkeeping:

- `landmark_indices: dict[landmark_id -> state_index]`

### Predict step: `predict(v, w, dt)`

- Propagates robot state using unicycle equations.
- Builds motion Jacobian `G`.
- Injects process noise only on robot block.
- Updates covariance:
  - `Sigma = G * Sigma * G^T + R`

### Landmark initialization: `_augment_landmark(observation)`

For unseen landmark ID:

1. Convert `(range, bearing)` to world `(lx, ly)` from current robot estimate.
2. Append landmark into state vector.
3. Expand covariance matrix.
4. Use Jacobians (`Jr`, `Jz`) to initialize cross-covariances and landmark covariance correctly.

### Measurement update: `update(observations)`

For each known landmark observation:

1. Compute predicted measurement `z_hat` from current state.
2. Build measurement Jacobian `H`.
3. Innovation:
   - `nu = z - z_hat`, with wrapped bearing residual.
4. Kalman gain:
   - `K = Sigma * H^T * inv(H*Sigma*H^T + Q)`
5. Correct state:
   - `mu = mu + K*nu`
6. Correct covariance:
   - `Sigma = (I - K*H)*Sigma`

Bearing is wrapped after every update to prevent angle discontinuities.

### Loop-closure correction: `apply_loop_closure(target_pose)`

When a loop closure is detected, we compute a rigid correction from current estimated robot pose to target historical pose:

- rotation by `dtheta`
- translation `(tx, ty)`

Then apply that transform to:

- robot `(x, y, theta)`
- all landmark positions in state

This reduces global drift in one step. Covariance is slightly inflated afterward to reduce overconfidence after hard correction.

## Loop-closure detector

### `LoopClosureDetector`

Maintains keyframe history:

- pose history
- scan descriptors

### Descriptor

For each scan, build a compact descriptor by splitting beam distances into bins and averaging per bin.

### Detection logic

For current frame:

1. Search only older frames beyond `min_separation`.
2. Keep candidates spatially near current estimated pose.
3. Compute descriptor difference score (mean absolute difference).
4. Accept best candidate if score < threshold and not in cooldown.

Returns `LoopClosureMatch` with:

- current frame
- matched historical frame
- score
- target pose for correction

## `main.py`

### Purpose

Realtime orchestrator combining environment + sensor + SLAM backend + visualization.

### Setup sequence

1. Create `Environment`.
2. Create `LidarSensor`.
3. Initialize true pose.
4. Initialize odometry model.
5. Initialize EKF with slight initial offset (to visualize convergence).
6. Extract map landmarks.
7. Build LiDAR frontend.
8. Create loop-closure detector.
9. Start frame timer and UI font.

### Movement model

- Keyboard only:
  - `W`: positive linear velocity
  - `S`: negative linear velocity
  - `A`: negative angular velocity (left rotation)
  - `D`: positive angular velocity (right rotation)

### Collision model

- Robot has a circular footprint (`robot_radius` in pixels).
- Candidate motion is split into:
  - rotation update
  - translation update
- Translation is rejected if robot center or footprint ring touches:
  - any black obstacle pixel (`R/G/B < obstacle_threshold`)
  - map boundary (treated as blocked)
- This prevents passing through black lines in front or behind the robot.

### Mapping stop/save behavior

- Mapping runs continuously by default.
- On `M` key press while mapping is active:
  - mapping pauses
  - current generated map is saved to `maps/slam_map_YYYYMMDD_HHMMSS.png`
- Press `M` again to resume mapping updates.

### Frame sequence (detailed)

1. Read events and handle mapping stop/save toggle (`M`).
2. Compute controls.
3. Propagate true pose with collision checking.
5. Sample noisy odometry and run EKF predict.
6. Scan LiDAR from true pose.
7. Convert scan to landmark observations.
8. EKF update with observations.
9. Attempt loop closure and apply correction if found.
10. Store true + estimated trajectories.
11. Update point cloud from scan hits.
12. If mapping is active, burn hit points into generated map image.
13. Draw map and overlays:
   - LiDAR beams
   - true trajectory (green)
   - estimated trajectory (blue)
   - estimated landmarks (yellow)
   - true robot (green dot)
   - estimated robot (blue dot)
   - telemetry text, collision warning, loop-closure banner, and saved-map status
14. Flip display.

## Coordinate and Unit Conventions

- World frame is image pixel frame.
- Origin is top-left.
- +x right, +y down.
- Distance unit is pixel.
- Internal heading in radians.
- Sensor API angle output in degrees.

## Noise and Uncertainty Summary

- Motion uncertainty: `OdometryModel` Gaussian noise on `v,w`.
- LiDAR angular uncertainty: Gaussian on beam angle.
- LiDAR range uncertainty: Gaussian on distance.
- EKF process noise: robot block process covariance in prediction.
- EKF measurement noise: range/bearing covariance in update.

These combine to produce realistic drift and correction behavior.

## What “Complete” Means In This Project

Implemented components:

- motion
- odometry
- LiDAR sensing
- landmark frontend
- EKF prediction and correction
- online landmark-state expansion
- loop-closure detection
- global loop-closure correction
- real-time visualization and telemetry

## Known Practical Limitations

This is a complete educational SLAM stack, but not production robotics-grade SLAM. Current simplifications:

- landmarks are extracted from known static map image (not discovered from raw unknown world occupancy only)
- nearest-neighbor association is simple and can fail in highly repetitive geometry
- loop closure uses descriptor + proximity heuristic, not full graph optimization backend
- covariance handling after hard loop correction is approximate
- no dynamic obstacle handling

## Suggested Next Extensions

1. Replace nearest-neighbor association with gated Mahalanobis association + JCBB.
2. Replace rigid loop correction with full pose-graph optimization.
3. Add occupancy-grid mapping from ray inverse model and map update entropy metrics.
4. Add robust kernels for outlier rejection.
5. Add logging and replay support for repeatable experiments.
