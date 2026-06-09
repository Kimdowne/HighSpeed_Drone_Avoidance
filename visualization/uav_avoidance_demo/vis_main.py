# 본 스크립트는 동일한 정적 장애물 맵에서 5회 UAV 회피 실험을 수행하고, 각 실험 GIF와 전체 통합 GIF를 저장한다.
"""Animated 2D UAV avoidance demo.

Run interactively and save GIFs:
    python -m visualization.uav_avoidance_demo

Save GIFs without opening a window:
    python -m visualization.uav_avoidance_demo --no-show

Save GIFs to a custom path:
    python -m visualization.uav_avoidance_demo --save visualization/uav_avoidance_demo/uav_avoidance_demo.gif --no-show

The default run executes five trials with one shared static-obstacle field.
By default, saving creates five trial GIFs and one combined GIF.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle

Vector = np.ndarray


@dataclass
class DemoConfig:
    field_width_m: float = 100.0
    field_height_m: float = 150.0
    start_band_m: float = 8.0
    goal_band_m: float = 8.0
    obstacle_count: int = 12
    obstacle_min_radius_m: float = 2.5
    obstacle_max_radius_m: float = 5.0
    drone_radius_m: float = 1.2
    sensor_range_m: float = 30.0
    main_max_speed_mps: float = 7.2
    enemy_max_speed_mps: float = 6.4
    main_max_accel_mps2: float = 8.5
    enemy_max_accel_mps2: float = 7.5
    static_influence_m: float = 16.0
    dynamic_influence_m: float = 20.0
    boundary_influence_m: float = 8.0
    enemy_behind_margin_m: float = 2.0
    dt_s: float = 0.08
    max_time_s: float = 45.0
    seed: int = 11


@dataclass(frozen=True)
class CircularObstacle:
    center: Vector
    radius: float


@dataclass
class DroneState:
    position: Vector
    velocity: Vector


@dataclass(frozen=True)
class FrameState:
    time_s: float
    main_position: Vector
    main_velocity: Vector
    enemy_position: Vector
    enemy_velocity: Vector
    detected_static: tuple[int, ...]
    enemy_in_range: bool
    enemy_detected: bool
    occluded_by: int | None
    distance_to_enemy: float
    enemy_forward_offset: float
    status: str


def _norm(vector: Vector) -> float:
    return float(np.linalg.norm(vector))


def _unit(vector: Vector, fallback: Vector | None = None) -> Vector:
    length = _norm(vector)
    if length > 1e-9:
        return vector / length
    if fallback is None:
        return np.zeros(2, dtype=float)
    return np.asarray(fallback, dtype=float)


def _limit(vector: Vector, maximum: float) -> Vector:
    length = _norm(vector)
    if length <= maximum or length <= 1e-9:
        return vector
    return vector * (maximum / length)


def _rotate_left(vector: Vector) -> Vector:
    return np.array([-vector[1], vector[0]], dtype=float)


class UAVDemoSimulation:
    """Local-observation UAV and APF pursuer simulation."""

    def __init__(
        self,
        config: DemoConfig,
        encounter_mode: str = "frontal",
        obstacles: list[CircularObstacle] | tuple[CircularObstacle, ...] | None = None,
    ):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.encounter_mode = encounter_mode

        # 1. 고정된 장애물이 전달되었으면 복사해서 사용
        if obstacles is not None:
            self.obstacles = [CircularObstacle(o.center.copy(), o.radius) for o in obstacles]
        else:
            self.obstacles = []

        # 2. 스폰 위치 무작위 생성 (장애물과 겹치지 않을 때까지 반복)
        attempts = 0
        while True:
            attempts += 1
            main_x = float(self.rng.uniform(10.0, config.field_width_m - 10.0))
            self.main_spawn = np.array([main_x, config.start_band_m + 2.0], dtype=float)

            relative_x_range = 15.0
            enemy_x_offset = float(self.rng.uniform(-relative_x_range, relative_x_range))
            enemy_x = float(np.clip(main_x + enemy_x_offset, 8.0, config.field_width_m - 8.0))
            enemy_y = config.field_height_m - config.goal_band_m - 4.0
            self.enemy_spawn = np.array([enemy_x, enemy_y], dtype=float)

            # 이미 공유된 장애물이 있다면, 새 스폰 지점이 장애물과 가까운지 검사
            if self.obstacles:
                collision = False
                for obs in self.obstacles:
                    if _norm(obs.center - self.main_spawn) < obs.radius + 11.0:
                        collision = True
                        break
                    if _norm(obs.center - self.enemy_spawn) < obs.radius + 11.0:
                        collision = True
                        break
                # 겹친다면(collision==True) 루프를 돌며 다시 뽑음
                if collision and attempts < 1000:
                    continue

            # 겹치지 않으면 루프 탈출
            break

        # 3. 외부 장애물이 주어지지 않았다면 여기서 새로 생성
        if not self.obstacles:
            self.obstacles = self._generate_obstacles()

        self.goal_target = np.array([main_x, config.field_height_m + 5.0], dtype=float)
        self.main_pass_side = 1.0 if main_x <= config.field_width_m / 2.0 else -1.0

        self.main = DroneState(self.main_spawn.copy(), np.zeros(2, dtype=float))
        self.enemy = DroneState(self.enemy_spawn.copy(), np.zeros(2, dtype=float))
        self.time_s = 0.0
        self.status = "running"


    def _generate_obstacles(self) -> list[CircularObstacle]:
        obstacles: list[CircularObstacle] = []
        config = self.config

        # 첫 번째 장애물을 메인 스폰과 적 스폰 사이 경로 부근에 생성
        corridor_progress = float(self.rng.uniform(0.48, 0.62))
        corridor_center = (
            self.main_spawn * (1.0 - corridor_progress)
            + self.enemy_spawn * corridor_progress
        )
        corridor_direction = _unit(self.enemy_spawn - self.main_spawn)
        corridor_normal = _rotate_left(corridor_direction)
        corridor_center += corridor_normal * float(self.rng.uniform(-1.2, 1.2))
        corridor_radius = float(
            self.rng.uniform(
                max(3.8, config.obstacle_min_radius_m),
                config.obstacle_max_radius_m,
            )
        )
        obstacles.append(CircularObstacle(corridor_center, corridor_radius))

        attempts = 0
        while len(obstacles) < config.obstacle_count and attempts < 5000:
            attempts += 1
            radius = float(
                self.rng.uniform(
                    config.obstacle_min_radius_m,
                    config.obstacle_max_radius_m,
                )
            )
            center = np.array(
                [
                    self.rng.uniform(radius + 4.0, config.field_width_m - radius - 4.0),
                    self.rng.uniform(
                        config.start_band_m + 12.0,
                        config.field_height_m - config.goal_band_m - 12.0,
                    ),
                ],
                dtype=float,
            )

            # 장애물이 메인 드론 시작점과 너무 가까운지 체크
            if _norm(center - self.main_spawn) < radius + 11.0:
                continue

            # [수정된 부분] 장애물이 확정된 적 드론 시작점과 너무 가까운지 체크
            if _norm(center - self.enemy_spawn) < radius + 11.0:
                continue

            # 장애물끼리 겹치는지 체크
            if any(
                _norm(center - obstacle.center) < radius + obstacle.radius + 7.0
                for obstacle in obstacles
            ):
                continue
            obstacles.append(CircularObstacle(center, radius))

        return obstacles

    def run(self, hold_frames: int = 45) -> list[FrameState]:
        frames = [self._snapshot()]
        steps = int(math.ceil(self.config.max_time_s / self.config.dt_s))
        for _ in range(steps):
            if self.status != "running":
                break
            self.step()
            frames.append(self._snapshot())

        if self.status == "running":
            self.status = "timeout"
            frames.append(self._snapshot())

        frames.extend([frames[-1]] * hold_frames)
        return frames

    def step(self) -> None:
        detected_static = self._detected_static_indices()
        enemy_in_range, enemy_detected, _ = self._enemy_detection()

        main_desired = self._main_desired_velocity(detected_static, enemy_detected)
        enemy_desired = self._enemy_desired_velocity()
        self._integrate(
            self.main,
            main_desired,
            self.config.main_max_accel_mps2,
        )
        self._integrate(
            self.enemy,
            enemy_desired,
            self.config.enemy_max_accel_mps2,
        )
        self._keep_inside_field(self.main, allow_goal=True)
        self._keep_inside_field(self.enemy, allow_goal=False)
        self._resolve_enemy_static_contacts()
        self.time_s += self.config.dt_s

        static_hit = self._colliding_obstacle(self.main.position)
        if static_hit is not None:
            self.status = f"collision_static_{static_hit + 1}"
        elif _norm(self.main.position - self.enemy.position) <= 2.0 * self.config.drone_radius_m:
            self.status = "intercepted"
        elif self._enemy_is_behind():
            self.status = "enemy_behind"
        elif self.main.position[1] >= self.config.field_height_m - self.config.goal_band_m:
            self.status = "goal_reached"

    def _snapshot(self) -> FrameState:
        detected_static = tuple(self._detected_static_indices())
        enemy_in_range, enemy_detected, occluded_by = self._enemy_detection()
        enemy_forward_offset = self._enemy_forward_offset()
        return FrameState(
            time_s=self.time_s,
            main_position=self.main.position.copy(),
            main_velocity=self.main.velocity.copy(),
            enemy_position=self.enemy.position.copy(),
            enemy_velocity=self.enemy.velocity.copy(),
            detected_static=detected_static,
            enemy_in_range=enemy_in_range,
            enemy_detected=enemy_detected,
            occluded_by=occluded_by,
            distance_to_enemy=_norm(self.enemy.position - self.main.position),
            enemy_forward_offset=enemy_forward_offset,
            status=self.status,
        )

    def _enemy_forward_offset(self) -> float:
        forward = _unit(
            self.goal_target - self.main.position,
            np.array([0.0, 1.0]),
        )
        return float(np.dot(self.enemy.position - self.main.position, forward))

    def _enemy_is_behind(self) -> bool:
        return self._enemy_forward_offset() < -self.config.enemy_behind_margin_m

    def _detected_static_indices(self) -> list[int]:
        return [
            index
            for index, obstacle in enumerate(self.obstacles)
            if _norm(obstacle.center - self.main.position) - obstacle.radius
            <= self.config.sensor_range_m
        ]

    def _enemy_detection(self) -> tuple[bool, bool, int | None]:
        distance = _norm(self.enemy.position - self.main.position)
        in_range = distance <= self.config.sensor_range_m
        if not in_range:
            return False, False, None

        occluded_by = self._line_of_sight_blocker(
            self.main.position,
            self.enemy.position,
        )
        return True, occluded_by is None, occluded_by

    def _line_of_sight_blocker(self, start: Vector, end: Vector) -> int | None:
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq <= 1e-9:
            return None

        for index, obstacle in enumerate(self.obstacles):
            progress = float(np.dot(obstacle.center - start, segment) / segment_length_sq)
            if progress <= 0.02 or progress >= 0.98:
                continue
            closest = start + progress * segment
            if _norm(obstacle.center - closest) <= obstacle.radius + 0.25:
                return index
        return None

    def _main_desired_velocity(
        self,
        detected_static: list[int],
        enemy_detected: bool,
    ) -> Vector:
        goal_direction = _unit(self.goal_target - self.main.position, np.array([0.0, 1.0]))
        desired = goal_direction * self.config.main_max_speed_mps
        desired += self._static_apf(
            self.main.position,
            goal_direction,
            detected_static,
            self.config.static_influence_m,
            12.0,
        )
        desired += self._boundary_apf(self.main.position, allow_top=True)

        if enemy_detected:
            desired += self._dynamic_avoidance(goal_direction)

        return _limit(desired, self.config.main_max_speed_mps)

    def _enemy_desired_velocity(self) -> Vector:
        attraction = _unit(
            self.main.position - self.enemy.position,
            np.array([0.0, -1.0]),
        )
        desired = attraction * self.config.enemy_max_speed_mps
        nearby = [
            index
            for index, obstacle in enumerate(self.obstacles)
            if _norm(obstacle.center - self.enemy.position) - obstacle.radius <= 22.0
        ]
        desired += self._static_apf(
            self.enemy.position,
            attraction,
            nearby,
            14.0,
            13.5,
        )
        desired += self._boundary_apf(self.enemy.position, allow_top=False)
        return _limit(desired, self.config.enemy_max_speed_mps)

    def _static_apf(
        self,
        position: Vector,
        travel_direction: Vector,
        obstacle_indices: list[int],
        influence: float,
        weight: float,
    ) -> Vector:
        force = np.zeros(2, dtype=float)
        for index in obstacle_indices:
            obstacle = self.obstacles[index]
            delta = position - obstacle.center
            center_distance = max(_norm(delta), 1e-6)
            clearance = center_distance - obstacle.radius - self.config.drone_radius_m
            if clearance >= influence:
                continue

            proximity = 1.0 - max(clearance, 0.0) / influence
            away = delta / center_distance
            tangent = _rotate_left(away)
            if float(np.dot(tangent, travel_direction)) < float(
                np.dot(-tangent, travel_direction)
            ):
                tangent = -tangent
            force += weight * proximity**2 * (away + 0.72 * tangent)
        return force

    def _dynamic_avoidance(self, goal_direction: Vector) -> Vector:
        relative_position = self.enemy.position - self.main.position
        relative_velocity = self.enemy.velocity - self.main.velocity
        relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
        closest_time = 0.0
        if relative_speed_sq > 1e-6:
            closest_time = float(
                np.clip(
                    -np.dot(relative_position, relative_velocity) / relative_speed_sq,
                    0.0,
                    2.5,
                )
            )
        predicted_separation = relative_position + relative_velocity * closest_time
        predicted_distance = _norm(predicted_separation)
        current_distance = _norm(relative_position)
        effective_distance = min(current_distance, predicted_distance)
        if effective_distance >= self.config.dynamic_influence_m:
            return np.zeros(2, dtype=float)

        proximity = 1.0 - effective_distance / self.config.dynamic_influence_m
        away = -_unit(predicted_separation, -goal_direction)
        lateral = np.array([self.main_pass_side, 0.0], dtype=float)
        return (3.0 + 8.0 * proximity) * lateral + 4.0 * proximity * away

    def _boundary_apf(self, position: Vector, allow_top: bool) -> Vector:
        influence = self.config.boundary_influence_m
        radius = self.config.drone_radius_m
        force = np.zeros(2, dtype=float)
        clearances = (
            (position[0] - radius, np.array([1.0, 0.0])),
            (self.config.field_width_m - position[0] - radius, np.array([-1.0, 0.0])),
            (position[1] - radius, np.array([0.0, 1.0])),
        )
        if not allow_top:
            clearances += (
                (
                    self.config.field_height_m - position[1] - radius,
                    np.array([0.0, -1.0]),
                ),
            )
        for clearance, direction in clearances:
            if clearance < influence:
                proximity = 1.0 - max(clearance, 0.0) / influence
                force += direction * 10.0 * proximity**2
        return force

    def _integrate(self, drone: DroneState, desired: Vector, max_accel: float) -> None:
        velocity_change = _limit(
            desired - drone.velocity,
            max_accel * self.config.dt_s,
        )
        drone.velocity = drone.velocity + velocity_change
        drone.position = drone.position + drone.velocity * self.config.dt_s

    def _keep_inside_field(self, drone: DroneState, allow_goal: bool) -> None:
        radius = self.config.drone_radius_m
        min_x = radius
        max_x = self.config.field_width_m - radius
        min_y = radius
        max_y = (
            self.config.field_height_m + self.config.goal_band_m
            if allow_goal
            else self.config.field_height_m - radius
        )
        clipped = np.clip(drone.position, [min_x, min_y], [max_x, max_y])
        for axis in range(2):
            if not math.isclose(float(clipped[axis]), float(drone.position[axis])):
                drone.velocity[axis] *= -0.25
        drone.position = clipped

    def _resolve_enemy_static_contacts(self) -> None:
        for obstacle in self.obstacles:
            delta = self.enemy.position - obstacle.center
            distance = _norm(delta)
            minimum = obstacle.radius + self.config.drone_radius_m + 0.1
            if distance >= minimum:
                continue
            normal = _unit(delta, np.array([1.0, 0.0]))
            self.enemy.position = obstacle.center + normal * minimum
            normal_speed = float(np.dot(self.enemy.velocity, normal))
            if normal_speed < 0.0:
                self.enemy.velocity -= 1.3 * normal_speed * normal

    def _colliding_obstacle(self, position: Vector) -> int | None:
        for index, obstacle in enumerate(self.obstacles):
            if _norm(position - obstacle.center) <= obstacle.radius + self.config.drone_radius_m:
                return index
        return None


class DemoRenderer:
    def __init__(
        self,
        simulation: UAVDemoSimulation,
        frames: list[FrameState],
        fps: int,
        trial_number: int = 1,
        trial_count: int = 1,
    ):
        self.simulation = simulation
        self.frames = frames
        self.fps = fps
        self.trial_number = trial_number
        self.trial_count = trial_count
        self.config = simulation.config
        self.times = np.array([frame.time_s for frame in frames])
        self.distances = np.array([frame.distance_to_enemy for frame in frames])

        self.figure = plt.figure(figsize=(10.8, 8.6), facecolor="#e8edf0")
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.35, 1.0),
            height_ratios=(2.25, 1.0),
            left=0.055,
            right=0.97,
            bottom=0.07,
            top=0.93,
            wspace=0.22,
            hspace=0.23,
        )
        self.field_ax = self.figure.add_subplot(grid[:, 0])
        self.info_ax = self.figure.add_subplot(grid[0, 1])
        self.range_ax = self.figure.add_subplot(grid[1, 1])
        self._configure_axes()
        self._build_field_artists()
        self._build_info_artists()
        self._build_range_artists()

        self.animation = animation.FuncAnimation(
            self.figure,
            self.update,
            frames=len(self.frames),
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )

    def _configure_axes(self) -> None:
        config = self.config
        self.field_ax.set_xlim(0.0, config.field_width_m)
        self.field_ax.set_ylim(0.0, config.field_height_m)
        self.field_ax.set_aspect("equal", adjustable="box")
        self.field_ax.set_facecolor("#f8fafb")
        self.field_ax.set_xlabel("x position [m]", color="#42515a")
        self.field_ax.set_ylabel("y position [m]", color="#42515a")
        self.field_ax.grid(color="#dbe2e6", linewidth=0.7, alpha=0.8)
        self.field_ax.set_axisbelow(True)
        for spine in self.field_ax.spines.values():
            spine.set_color("#667780")
            spine.set_linewidth(1.2)

        self.info_ax.set_facecolor("#f8fafb")
        self.info_ax.set_xticks([])
        self.info_ax.set_yticks([])
        for spine in self.info_ax.spines.values():
            spine.set_color("#bcc8ce")

        self.range_ax.set_facecolor("#f8fafb")
        self.range_ax.grid(color="#dbe2e6", linewidth=0.7)
        self.range_ax.set_xlabel("time [s]", color="#42515a")
        self.range_ax.set_ylabel("UAV separation [m]", color="#42515a")
        self.range_ax.tick_params(colors="#52636c", labelsize=8)
        for spine in self.range_ax.spines.values():
            spine.set_color("#bcc8ce")

        self.figure.suptitle(
            (
                "UAV LOCAL-OBSERVATION AVOIDANCE DEMO"
                f"  |  TRIAL {self.trial_number}/{self.trial_count}"
                f"  |  {self.simulation.encounter_mode.upper().replace('_', ' ')}"
            ),
            fontsize=16,
            fontweight="bold",
            color="#233139",
        )

    def _build_field_artists(self) -> None:
        config = self.config
        self.field_ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                config.field_width_m,
                config.start_band_m,
                facecolor="#67a9cf",
                edgecolor="none",
                alpha=0.28,
                zorder=0,
            )
        )
        self.field_ax.add_patch(
            Rectangle(
                (0.0, config.field_height_m - config.goal_band_m),
                config.field_width_m,
                config.goal_band_m,
                facecolor="#57b879",
                edgecolor="none",
                alpha=0.35,
                zorder=0,
            )
        )
        self.field_ax.text(
            3.0,
            3.2,
            "START",
            color="#356b88",
            fontsize=8,
            fontweight="bold",
        )
        self.field_ax.text(
            3.0,
            config.field_height_m - 5.0,
            "GOAL ZONE",
            color="#317044",
            fontsize=8,
            fontweight="bold",
        )

        self.obstacle_patches: list[Circle] = []
        for index, obstacle in enumerate(self.simulation.obstacles):
            patch = Circle(
                tuple(obstacle.center),
                obstacle.radius,
                facecolor="#5f6b70",
                edgecolor="#37434a",
                linewidth=1.0,
                alpha=0.86,
                zorder=3,
            )
            self.field_ax.add_patch(patch)
            self.obstacle_patches.append(patch)
            self.field_ax.text(
                obstacle.center[0],
                obstacle.center[1],
                str(index + 1),
                color="#f5f7f8",
                fontsize=6,
                ha="center",
                va="center",
                zorder=4,
            )

        first = self.frames[0]
        self.sensor_patch = Circle(
            tuple(first.main_position),
            config.sensor_range_m,
            facecolor="#37a7d0",
            edgecolor="#2384ad",
            linewidth=1.2,
            linestyle=(0, (4, 4)),
            alpha=0.09,
            zorder=1,
        )
        self.field_ax.add_patch(self.sensor_patch)

        (self.main_trail,) = self.field_ax.plot(
            [],
            [],
            color="#137ea8",
            linewidth=1.7,
            alpha=0.85,
            zorder=5,
        )
        (self.enemy_trail,) = self.field_ax.plot(
            [],
            [],
            color="#d34d36",
            linewidth=1.5,
            alpha=0.72,
            zorder=5,
        )
        self.main_patch = Circle(
            tuple(first.main_position),
            config.drone_radius_m * 1.6,
            facecolor="#1ca4d2",
            edgecolor="#075c7a",
            linewidth=2.0,
            zorder=8,
        )
        self.enemy_patch = Circle(
            tuple(first.enemy_position),
            config.drone_radius_m * 1.6,
            facecolor="#e85d46",
            edgecolor="#8f271b",
            linewidth=2.0,
            zorder=8,
        )
        self.field_ax.add_patch(self.main_patch)
        self.field_ax.add_patch(self.enemy_patch)

        (self.sight_line,) = self.field_ax.plot(
            [],
            [],
            color="#e1a72b",
            linewidth=1.5,
            linestyle="--",
            alpha=0.0,
            zorder=6,
        )
        self.main_velocity = self.field_ax.quiver(
            [first.main_position[0]],
            [first.main_position[1]],
            [first.main_velocity[0]],
            [first.main_velocity[1]],
            color="#075c7a",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            zorder=9,
        )
        self.enemy_velocity = self.field_ax.quiver(
            [first.enemy_position[0]],
            [first.enemy_position[1]],
            [first.enemy_velocity[0]],
            [first.enemy_velocity[1]],
            color="#8f271b",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.006,
            zorder=9,
        )
        self.event_text = self.field_ax.text(
            50.0,
            137.5,
            "",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#233139",
            zorder=10,
        )

    def _build_info_artists(self) -> None:
        self.info_ax.text(
            0.06,
            0.94,
            "LIVE TELEMETRY",
            transform=self.info_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            color="#233139",
            va="top",
        )
        self.telemetry_text = self.info_ax.text(
            0.06,
            0.84,
            "",
            transform=self.info_ax.transAxes,
            fontsize=9.2,
            color="#42515a",
            va="top",
            linespacing=1.48,
            family="monospace",
        )
        self.detection_text = self.info_ax.text(
            0.06,
            0.22,
            "",
            transform=self.info_ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="#7b8790",
            va="top",
        )
        self.result_text = self.info_ax.text(
            0.06,
            0.09,
            "",
            transform=self.info_ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color="#2f7d48",
            va="top",
        )

    def _build_range_artists(self) -> None:
        max_time = max(float(self.times.max()), 1.0)
        max_distance = max(float(self.distances.max()) * 1.05, self.config.sensor_range_m + 10.0)
        self.range_ax.set_xlim(0.0, max_time)
        self.range_ax.set_ylim(0.0, max_distance)
        self.range_ax.axhline(
            self.config.sensor_range_m,
            color="#2384ad",
            linewidth=1.2,
            linestyle="--",
            label="30 m sensor limit",
        )
        (self.range_history,) = self.range_ax.plot(
            [],
            [],
            color="#d34d36",
            linewidth=1.7,
            label="UAV separation",
        )
        (self.detected_history,) = self.range_ax.plot(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=2.8,
            color="#e1a72b",
            label="detected",
        )
        self.range_ax.legend(loc="upper right", fontsize=7, frameon=False)

    def update(self, frame_index: int):
        frame = self.frames[frame_index]
        main_path = np.array([item.main_position for item in self.frames[: frame_index + 1]])
        enemy_path = np.array([item.enemy_position for item in self.frames[: frame_index + 1]])
        self.main_trail.set_data(main_path[:, 0], main_path[:, 1])
        self.enemy_trail.set_data(enemy_path[:, 0], enemy_path[:, 1])
        self.main_patch.center = tuple(frame.main_position)
        self.enemy_patch.center = tuple(frame.enemy_position)
        self.sensor_patch.center = tuple(frame.main_position)

        detected = set(frame.detected_static)
        for index, patch in enumerate(self.obstacle_patches):
            if index in detected:
                patch.set_edgecolor("#f0b62b")
                patch.set_linewidth(2.1)
            else:
                patch.set_edgecolor("#37434a")
                patch.set_linewidth(1.0)

        if frame.enemy_detected:
            self.sight_line.set_data(
                [frame.main_position[0], frame.enemy_position[0]],
                [frame.main_position[1], frame.enemy_position[1]],
            )
            self.sight_line.set_color("#e1a72b")
            self.sight_line.set_linestyle("--")
            self.sight_line.set_alpha(0.9)
            self.enemy_patch.set_alpha(1.0)
            self.enemy_patch.set_edgecolor("#e1a72b")
        elif frame.enemy_in_range:
            self.sight_line.set_data(
                [frame.main_position[0], frame.enemy_position[0]],
                [frame.main_position[1], frame.enemy_position[1]],
            )
            self.sight_line.set_color("#7b8790")
            self.sight_line.set_linestyle(":")
            self.sight_line.set_alpha(0.65)
            self.enemy_patch.set_alpha(0.48)
            self.enemy_patch.set_edgecolor("#8f271b")
        else:
            self.sight_line.set_alpha(0.0)
            self.enemy_patch.set_alpha(0.68)
            self.enemy_patch.set_edgecolor("#8f271b")

        self.main_velocity.set_offsets([frame.main_position])
        self.main_velocity.set_UVC([frame.main_velocity[0]], [frame.main_velocity[1]])
        self.enemy_velocity.set_offsets([frame.enemy_position])
        self.enemy_velocity.set_UVC([frame.enemy_velocity[0]], [frame.enemy_velocity[1]])

        detection_label, detection_color = self._detection_label(frame)
        self.detection_text.set_text(detection_label)
        self.detection_text.set_color(detection_color)
        self.telemetry_text.set_text(
            "\n".join(
                (
                    f"TIME              {frame.time_s:6.2f} s",
                    f"MAIN POSITION     ({frame.main_position[0]:5.1f}, {frame.main_position[1]:5.1f}) m",
                    f"MAIN SPEED        {_norm(frame.main_velocity):6.2f} m/s",
                    f"ENEMY POSITION    ({frame.enemy_position[0]:5.1f}, {frame.enemy_position[1]:5.1f}) m",
                    f"ENEMY SPEED       {_norm(frame.enemy_velocity):6.2f} m/s",
                    f"SEPARATION        {frame.distance_to_enemy:6.2f} m",
                    f"ENEMY FORWARD     {frame.enemy_forward_offset:6.2f} m",
                    f"SENSOR RANGE      {self.config.sensor_range_m:6.1f} m",
                    f"STATIC CONTACTS   {len(frame.detected_static):6d}",
                    "",
                    "ENEMY GUIDANCE     APF pursuit",
                    "MAIN GUIDANCE      local avoidance",
                )
            )
        )
        result_label, result_color = self._result_label(frame.status)
        self.result_text.set_text(result_label)
        self.result_text.set_color(result_color)
        self.event_text.set_text(self._event_label(frame))
        self.event_text.set_color(detection_color if frame.status == "running" else result_color)

        history_times = self.times[: frame_index + 1]
        history_distances = self.distances[: frame_index + 1]
        self.range_history.set_data(history_times, history_distances)
        detected_mask = np.array(
            [item.enemy_detected for item in self.frames[: frame_index + 1]],
            dtype=bool,
        )
        self.detected_history.set_data(
            history_times[detected_mask],
            history_distances[detected_mask],
        )
        return (
            self.main_trail,
            self.enemy_trail,
            self.main_patch,
            self.enemy_patch,
            self.sensor_patch,
            self.sight_line,
            self.telemetry_text,
            self.detection_text,
            self.result_text,
            self.event_text,
            self.range_history,
            self.detected_history,
        )

    def _detection_label(self, frame: FrameState) -> tuple[str, str]:
        if frame.enemy_detected:
            return "DYNAMIC CONTACT: DETECTED", "#c58a09"
        if frame.enemy_in_range:
            blocker = "unknown"
            if frame.occluded_by is not None:
                blocker = f"pillar {frame.occluded_by + 1}"
            return f"DYNAMIC CONTACT: OCCLUDED ({blocker})", "#697981"
        return "DYNAMIC CONTACT: OUT OF RANGE", "#2384ad"

    def _event_label(self, frame: FrameState) -> str:
        if frame.status != "running":
            return self._result_label(frame.status)[0]
        if frame.enemy_detected:
            return "CONTACT ACQUIRED - EVASIVE MANEUVER"
        if frame.enemy_in_range:
            return "CONTACT HIDDEN BY STATIC OBSTACLE"
        return ""

    def _result_label(self, status: str) -> tuple[str, str]:
        if status == "goal_reached":
            return "RESULT: GOAL REACHED", "#2f8a4c"
        if status == "intercepted":
            return "RESULT: INTERCEPTED", "#b33d2e"
        if status == "enemy_behind":
            return "RESULT: ENEMY BEHIND - EPISODE ENDED", "#2f8a4c"
        if status.startswith("collision_static"):
            return "RESULT: STATIC COLLISION", "#b33d2e"
        if status == "timeout":
            return "RESULT: TIMEOUT", "#826619"
        return "RESULT: RUNNING", "#52636c"


def save_animation(
    renderer: DemoRenderer,
    output_path: Path,
    fps: int,
    dpi: int,
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".mp4":
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(
                fps=fps,
                bitrate=2400,
                metadata={"title": "UAV local-observation avoidance demo"},
            )
        else:
            output_path = output_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=fps)
            print(f"ffmpeg is unavailable; saving GIF instead: {output_path}")
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError("Animation output must use .mp4 or .gif")

    renderer.animation.save(str(output_path), writer=writer, dpi=dpi)
    return output_path


def trial_output_path(
    output_path: Path,
    trial_number: int,
    trial_count: int,
    encounter_mode: str,
) -> Path:
    if trial_count <= 1:
        return output_path
    return output_path.with_name(
        f"{output_path.stem}_trial_{trial_number:02d}_{encounter_mode}{output_path.suffix}"
    )


def concatenate_gifs(input_paths: list[Path], output_path: Path) -> Path:
    from PIL import Image

    if not input_paths:
        raise ValueError("At least one trial GIF is required")

    def iter_frames():
        for input_path in input_paths:
            with Image.open(input_path) as image:
                default_duration = int(image.info.get("duration", 100))
                for frame_index in range(image.n_frames):
                    image.seek(frame_index)
                    frame = image.convert("RGBA")
                    frame.info["duration"] = int(
                        image.info.get("duration", default_duration)
                    )
                    yield frame

    output_path = output_path.resolve().with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_iterator = iter_frames()
    first_frame = next(frame_iterator)
    first_frame.save(
        output_path,
        save_all=True,
        append_images=frame_iterator,
        duration=first_frame.info["duration"],
        loop=0,
        disposal=2,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).with_name("uav_avoidance_demo.gif"),
        help=(
            "Combined GIF output path. "
            "Per-trial GIFs are saved next to it. "
            "Default: visualization/uav_avoidance_demo/uav_avoidance_demo.gif"
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the simulation without saving GIF files",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open the interactive window")
    # 수정: default=11 을 default=None 으로 변경
    parser.add_argument("--seed", type=int, default=None, help="Random field seed (leave empty for random)")
    parser.add_argument("--trials", type=int, default=5, help="Number of repeated experiments")
    parser.add_argument("--obstacles", type=int, default=12, help="Number of static obstacles")
    parser.add_argument("--max-time", type=float, default=45.0, help="Maximum simulated seconds")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second")
    parser.add_argument("--dpi", type=int, default=72, help="Saved animation resolution")
    return parser.parse_args()


def main() -> None:
    import secrets

    args = parse_args()
    trial_count = max(1, args.trials)
    current_seed = args.seed if args.seed is not None else secrets.randbelow(1000000)

    print(f"\n[INFO] Current Run Seed: {current_seed}")
    print(f"       (To replay this exact obstacle layout, run with: --seed {current_seed})\n")

    config = DemoConfig(
        seed=current_seed,
        obstacle_count=max(1, args.obstacles),
        max_time_s=max(1.0, args.max_time),
    )

    # 맵(장애물) 생성을 위한 초기 템플릿 시뮬레이션
    encounter_mode = "frontal"
    field_template = UAVDemoSimulation(config, encounter_mode=encounter_mode)
    shared_obstacles = tuple(field_template.obstacles)
    save_enabled = not args.no_save
    combined_output_path = args.save.with_suffix(".gif") if save_enabled else None
    trial_gif_paths: list[Path] = []
    results = []

    if save_enabled:
        combined_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] GIF output directory: {combined_output_path.parent.resolve()}")
        print(f"[INFO] Combined GIF path: {combined_output_path.resolve()}\n")

    for trial_index in range(trial_count):
        trial_number = trial_index + 1

        # [핵심 수정] 매 시도(trial)마다 시드값에 trial_number를 더해줍니다.
        # 이렇게 하면 같은 맵(shared_obstacles) 위에서 5번 모두 다른 스폰 지점이 발생합니다.
        trial_config = replace(config, seed=config.seed + trial_number)

        simulation = UAVDemoSimulation(
            trial_config,
            encounter_mode=encounter_mode,
            obstacles=shared_obstacles,
        )
        frames = simulation.run(hold_frames=max(1, int(args.fps * 1.5)))
        results.append(frames[-1].status)

        print(
            f"trial={trial_number}/{trial_count} encounter={encounter_mode} "
            f"seed={trial_config.seed} obstacles={len(simulation.obstacles)} "
            f"frames={len(frames)} result={frames[-1].status}"
        )

        if not save_enabled and args.no_show:
            continue

        renderer = DemoRenderer(
            simulation,
            frames,
            fps=max(1, args.fps),
            trial_number=trial_number,
            trial_count=trial_count,
        )

        if save_enabled:
            output_path = trial_output_path(
                combined_output_path,
                trial_number,
                trial_count,
                encounter_mode,
            )
            saved_path = save_animation(
                renderer,
                output_path,
                max(1, args.fps),
                max(72, args.dpi),
            )
            trial_gif_paths.append(saved_path)
            print(f"saved={saved_path}")

        if args.no_show:
            plt.close(renderer.figure)
        else:
            plt.show()

    if save_enabled and combined_output_path is not None:
        combined_path = concatenate_gifs(trial_gif_paths, combined_output_path)
        print(f"combined_gif={combined_path}")

    print(f"experiment_summary trials={trial_count} results={results}")


if __name__ == "__main__":
    main()
