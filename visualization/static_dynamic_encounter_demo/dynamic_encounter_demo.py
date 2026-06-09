"""EGO-planner dynamic-obstacle encounter demo on the inferred static map.

Run:
    python -m visualization.static_dynamic_encounter_demo --no-show
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse, Rectangle

from planners.ego_planner import EGOPlanner, EGOPlannerConfig

Vector = np.ndarray


@dataclass(frozen=True)
class RectObstacle:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class DemoConfig:
    field_width_m: float = 120.0
    field_height_m: float = 70.0
    drone_radius_m: float = 1.2
    dynamic_radius_m: float = 1.6
    lidar_range_m: float = 50.0
    drone_speed_mps: float = 8.0
    drone_accel_mps2: float = 14.0
    dynamic_speed_mps: float = 7.5
    dt_s: float = 0.12
    max_time_s: float = 28.0
    waypoint_tolerance_m: float = 3.0
    goal_tolerance_m: float = 4.0
    point_spacing_m: float = 2.0


@dataclass(frozen=True)
class Scenario:
    release_delay_s: float
    label: str


@dataclass(frozen=True)
class FrameState:
    time_s: float
    drone_position: Vector
    drone_velocity: Vector
    yaw: float
    dynamic_position: Vector
    dynamic_active: bool
    dynamic_detected: bool
    distance_to_dynamic: float
    waypoint_index: int
    status: str
    command_blocked: bool
    min_clearance_m: float


@dataclass(frozen=True)
class SimulationResult:
    scenario: Scenario
    frames: list[FrameState]
    first_detection_time_s: float | None
    first_detection_distance_m: float | None
    min_distance_m: float
    final_status: str


STATIC_OBSTACLES = (
    RectObstacle(26.0, 62.5, 94.0, 7.5),
    RectObstacle(26.0, 0.0, 94.0, 45.5),
)

START_POS = np.array([13.5, 12.5], dtype=float)
CORNER_WAYPOINT = np.array([13.5, 52.0], dtype=float)
GATE_WAYPOINT = np.array([32.0, 54.5], dtype=float)
GOAL_POS = np.array([112.5, 54.5], dtype=float)
DYNAMIC_SPAWN = np.array([104.0, 54.5], dtype=float)
WAYPOINTS = (CORNER_WAYPOINT, GATE_WAYPOINT, GOAL_POS)
DEFAULT_DELAYS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 11.5)


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


def _point_rect_distance(point: Vector, rect: RectObstacle) -> float:
    dx = max(rect.x - point[0], 0.0, point[0] - rect.right)
    dy = max(rect.y - point[1], 0.0, point[1] - rect.top)
    return math.hypot(dx, dy)


def _segment_intersects_rect(
    start: Vector,
    end: Vector,
    rect: RectObstacle,
    padding: float = 0.0,
) -> bool:
    x_min = rect.x - padding
    x_max = rect.right + padding
    y_min = rect.y - padding
    y_max = rect.top + padding
    direction = end - start
    t_min = 0.0
    t_max = 1.0

    for axis, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
        value = float(start[axis])
        delta = float(direction[axis])
        if abs(delta) < 1e-9:
            if value < low or value > high:
                return False
            continue

        inv_delta = 1.0 / delta
        near = (low - value) * inv_delta
        far = (high - value) * inv_delta
        if near > far:
            near, far = far, near
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return False

    return t_max > 0.02 and t_min < 0.98


def _line_of_sight_clear(start: Vector, end: Vector) -> bool:
    return not any(
        _segment_intersects_rect(start, end, rect, padding=0.15)
        for rect in STATIC_OBSTACLES
    )


def _rectangle_surface_points(rect: RectObstacle, spacing: float) -> list[Vector]:
    xs = np.arange(rect.x, rect.right + 0.5 * spacing, spacing)
    ys = np.arange(rect.y, rect.top + 0.5 * spacing, spacing)
    points: list[Vector] = []

    for x in xs:
        points.append(np.array([x, rect.y], dtype=float))
        points.append(np.array([x, rect.top], dtype=float))
    for y in ys:
        points.append(np.array([rect.x, y], dtype=float))
        points.append(np.array([rect.right, y], dtype=float))
    return points


def _circle_surface_points(center: Vector, radius: float, count: int = 24) -> list[Vector]:
    return [
        center + radius * np.array([math.cos(theta), math.sin(theta)], dtype=float)
        for theta in np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    ]


def _world_points_to_cloud(
    points: list[Vector],
    position: Vector,
    yaw: float,
    max_range_m: float,
) -> list[float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cloud: list[float] = []

    for point in points:
        delta = point - position
        distance = _norm(delta)
        if distance > max_range_m:
            continue
        local_x = delta[0] * cos_yaw + delta[1] * sin_yaw
        local_y = -delta[0] * sin_yaw + delta[1] * cos_yaw
        cloud.extend([float(local_x), float(local_y), 0.0])
    return cloud


def _planner_config(config: DemoConfig) -> EGOPlannerConfig:
    return EGOPlannerConfig(
        max_vel_mps=config.drone_speed_mps,
        max_acc_mps2=config.drone_accel_mps2,
        grid_resolution_m=0.5,
        safe_radius_m=0.8,
        max_range_m=config.lidar_range_m,
        horizon_time_s=2.4,
        max_horizon_m=34.0,
        min_horizon_m=6.0,
        control_point_distance_m=3.0,
        command_lookahead_s=0.7,
        max_lateral_speed_mps=6.0,
        min_lateral_speed_mps=2.0,
        lambda_collision=0.9,
        optimizer_max_iterations=28,
        refine_max_iterations=24,
    )


class StaticDynamicEncounterSimulation:
    def __init__(self, scenario: Scenario, config: DemoConfig | None = None):
        self.scenario = scenario
        self.config = config or DemoConfig()
        self.planner = EGOPlanner(_planner_config(self.config))
        self.static_points = [
            point
            for rect in STATIC_OBSTACLES
            for point in _rectangle_surface_points(rect, self.config.point_spacing_m)
        ]
        self.position = START_POS.copy()
        self.velocity = np.zeros(2, dtype=float)
        self.yaw = math.pi / 2.0
        self.waypoint_index = 0
        self.time_s = 0.0
        self.status = "running"
        self.dynamic_released = False
        self.spawn_suppressed = False
        self.first_detection_time_s: float | None = None
        self.first_detection_distance_m: float | None = None
        self.min_distance_m = float("inf")

    def run(self, hold_frames: int = 32) -> SimulationResult:
        frames = [self._snapshot(False, False, False, float("inf"))]
        steps = int(math.ceil(self.config.max_time_s / self.config.dt_s))

        for _ in range(steps):
            if self.status != "running":
                break
            frame = self.step()
            frames.append(frame)

        if self.status == "running":
            self.status = "timeout"
            frames.append(self._snapshot(False, False, False, float("inf")))

        frames.extend([frames[-1]] * hold_frames)
        return SimulationResult(
            scenario=self.scenario,
            frames=frames,
            first_detection_time_s=self.first_detection_time_s,
            first_detection_distance_m=self.first_detection_distance_m,
            min_distance_m=self.min_distance_m,
            final_status=self.status,
        )

    def step(self) -> FrameState:
        if not self.dynamic_released and self.time_s >= self.scenario.release_delay_s:
            self.dynamic_released = True
            if self._spawn_position_observable():
                self.spawn_suppressed = True
                self.status = "spawn_suppressed"
                return self._snapshot(False, False, False, float("inf"))

        dynamic_active = self.dynamic_released and not self.spawn_suppressed
        dynamic_position = self._dynamic_position(self.time_s)
        distance_to_dynamic = _norm(dynamic_position - self.position)
        dynamic_detected = (
            dynamic_active
            and distance_to_dynamic <= self.config.lidar_range_m
            and _line_of_sight_clear(self.position, dynamic_position)
        )

        if dynamic_active:
            self.min_distance_m = min(self.min_distance_m, distance_to_dynamic)

        if dynamic_detected and self.first_detection_distance_m is None:
            self.first_detection_time_s = self.time_s
            self.first_detection_distance_m = distance_to_dynamic

        command = self._plan(dynamic_position, dynamic_active, dynamic_detected)
        if command.blocked:
            desired_velocity = self.velocity * 0.35
        else:
            desired_velocity = _limit(
                np.array([command.vx, command.vy], dtype=float),
                self.config.drone_speed_mps,
            )
            desired_velocity = self._apply_corridor_tracking(
                desired_velocity,
                dynamic_detected,
            )

        delta_v = _limit(
            desired_velocity - self.velocity,
            self.config.drone_accel_mps2 * self.config.dt_s,
        )
        self.velocity = _limit(
            self.velocity + delta_v,
            self.config.drone_speed_mps,
        )
        self.position = self.position + self.velocity * self.config.dt_s
        if _norm(self.velocity) > 0.1:
            self.yaw = math.atan2(float(self.velocity[1]), float(self.velocity[0]))

        self._advance_waypoint()
        self._update_status(dynamic_position, dynamic_active)
        self.time_s += self.config.dt_s

        return self._snapshot(
            dynamic_active,
            dynamic_detected,
            command.blocked,
            command.min_clearance,
        )

    def _plan(self, dynamic_position: Vector, dynamic_active: bool, dynamic_detected: bool):
        points = list(self.static_points)
        if dynamic_active and dynamic_detected:
            points.extend(
                _circle_surface_points(
                    dynamic_position,
                    self.config.dynamic_radius_m,
                )
            )
        point_cloud = _world_points_to_cloud(
            points,
            self.position,
            self.yaw,
            self.config.lidar_range_m,
        )
        return self.planner.plan(
            point_cloud=point_cloud,
            position=self.position,
            velocity=self.velocity,
            yaw=self.yaw,
            target_position=WAYPOINTS[self.waypoint_index],
            target_speed=self._target_speed(),
        )

    def _target_speed(self) -> float:
        if self.waypoint_index >= len(WAYPOINTS) - 1:
            return self.config.drone_speed_mps

        distance_to_corner = _norm(WAYPOINTS[self.waypoint_index] - self.position)
        return min(
            self.config.drone_speed_mps,
            max(2.5, 0.7 * distance_to_corner),
        )

    def _apply_corridor_tracking(
        self,
        desired_velocity: Vector,
        dynamic_detected: bool,
    ) -> Vector:
        if self.waypoint_index == 0:
            return desired_velocity

        corrected = desired_velocity.copy()
        centerline_error = GOAL_POS[1] - self.position[1]
        if dynamic_detected:
            corrected[1] += 0.35 * centerline_error - 0.20 * self.velocity[1]
        else:
            corrected[1] += 1.15 * centerline_error - 0.65 * self.velocity[1]
        return _limit(corrected, self.config.drone_speed_mps)

    def _dynamic_position(self, time_s: float) -> Vector:
        if not self.dynamic_released or self.spawn_suppressed:
            return DYNAMIC_SPAWN.copy()
        travel_time = time_s - self.scenario.release_delay_s
        return DYNAMIC_SPAWN + np.array(
            [-self.config.dynamic_speed_mps * travel_time, 0.0],
            dtype=float,
        )

    def _spawn_position_observable(self) -> bool:
        return (
            _norm(DYNAMIC_SPAWN - self.position) <= self.config.lidar_range_m
            and _line_of_sight_clear(self.position, DYNAMIC_SPAWN)
        )

    def _advance_waypoint(self) -> None:
        if self.waypoint_index >= len(WAYPOINTS) - 1:
            return
        if _norm(self.position - WAYPOINTS[self.waypoint_index]) <= self.config.waypoint_tolerance_m:
            self.waypoint_index += 1

    def _update_status(self, dynamic_position: Vector, dynamic_active: bool) -> None:
        for index, rect in enumerate(STATIC_OBSTACLES, start=1):
            if _point_rect_distance(self.position, rect) <= self.config.drone_radius_m:
                self.status = f"collision_static_{index}"
                return

        if dynamic_active:
            separation = _norm(dynamic_position - self.position)
            self.min_distance_m = min(self.min_distance_m, separation)
            if separation <= self.config.drone_radius_m + self.config.dynamic_radius_m:
                self.status = "collision_dynamic"
                return

        if _norm(self.position - GOAL_POS) <= self.config.goal_tolerance_m:
            self.status = "goal_reached"

    def _snapshot(
        self,
        dynamic_active: bool,
        dynamic_detected: bool,
        command_blocked: bool,
        min_clearance_m: float,
    ) -> FrameState:
        dynamic_position = self._dynamic_position(self.time_s)
        distance = _norm(dynamic_position - self.position)
        if dynamic_active:
            self.min_distance_m = min(self.min_distance_m, distance)
        return FrameState(
            time_s=self.time_s,
            drone_position=self.position.copy(),
            drone_velocity=self.velocity.copy(),
            yaw=self.yaw,
            dynamic_position=dynamic_position.copy(),
            dynamic_active=dynamic_active,
            dynamic_detected=dynamic_detected,
            distance_to_dynamic=distance,
            waypoint_index=self.waypoint_index,
            status=self.status,
            command_blocked=command_blocked,
            min_clearance_m=min_clearance_m,
        )


class EncounterRenderer:
    def __init__(
        self,
        results: list[SimulationResult],
        config: DemoConfig,
        fps: int,
        frame_stride: int = 3,
        title: str = "Static-map EGO Planner Dynamic Encounter Demo",
    ):
        self.results = results
        self.config = config
        self.fps = fps
        self.title = title
        self.frame_stride = max(1, int(frame_stride))
        self.timeline: list[tuple[int, int]] = []
        for result_index, result in enumerate(results):
            frame_indices = list(range(0, len(result.frames), self.frame_stride))
            if frame_indices[-1] != len(result.frames) - 1:
                frame_indices.append(len(result.frames) - 1)
            self.timeline.extend((result_index, frame_index) for frame_index in frame_indices)

        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        self.figure = plt.figure(figsize=(13.0, 7.4), facecolor="#eef2f4")
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.7, 1.0),
            height_ratios=(1.0, 1.0),
            left=0.045,
            right=0.97,
            bottom=0.08,
            top=0.91,
            hspace=0.24,
            wspace=0.18,
        )
        self.field_ax = self.figure.add_subplot(grid[:, 0])
        self.info_ax = self.figure.add_subplot(grid[0, 1])
        self.distance_ax = self.figure.add_subplot(grid[1, 1])
        self._configure_axes()
        self._build_static_artists()
        self._build_dynamic_artists()
        self.animation = animation.FuncAnimation(
            self.figure,
            self.update,
            frames=len(self.timeline),
            interval=1000.0 / fps,
            blit=False,
            repeat=True,
        )

    def _configure_axes(self) -> None:
        self.field_ax.set_xlim(0.0, self.config.field_width_m)
        self.field_ax.set_ylim(0.0, self.config.field_height_m)
        self.field_ax.set_aspect("equal", adjustable="box")
        self.field_ax.set_facecolor("white")
        self.field_ax.set_xlabel("x [m]", color="#314049")
        self.field_ax.set_ylabel("y [m]", color="#314049")
        self.field_ax.grid(color="#d7dee2", linewidth=0.7)
        self.field_ax.set_axisbelow(True)

        self.info_ax.set_facecolor("#f8fafb")
        self.info_ax.set_xticks([])
        self.info_ax.set_yticks([])
        for spine in self.info_ax.spines.values():
            spine.set_color("#bdc8ce")

        self.distance_ax.set_facecolor("#f8fafb")
        self.distance_ax.grid(color="#d7dee2", linewidth=0.7)
        self.distance_ax.set_xlabel("time [s]", color="#314049")
        self.distance_ax.set_ylabel("separation [m]", color="#314049")
        self.distance_ax.set_ylim(0.0, 80.0)
        max_t = max(frame.time_s for result in self.results for frame in result.frames)
        self.distance_ax.set_xlim(0.0, max(1.0, max_t))
        self.distance_ax.axhline(
            self.config.lidar_range_m,
            color="#2f80ed",
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
        )
        self.distance_ax.axhline(
            self.config.drone_radius_m + self.config.dynamic_radius_m,
            color="#d33f49",
            linewidth=1.2,
            linestyle=":",
            alpha=0.9,
        )
        self.figure.suptitle(self.title, fontsize=16, fontweight="bold")

    def _build_static_artists(self) -> None:
        for rect in STATIC_OBSTACLES:
            self.field_ax.add_patch(
                Rectangle(
                    (rect.x, rect.y),
                    rect.width,
                    rect.height,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=1.0,
                )
            )

        self.field_ax.text(
            74.0,
            66.3,
            "정적 장애물",
            color="white",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        self.field_ax.text(
            74.0,
            23.0,
            "정적 장애물",
            color="white",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        route = np.vstack([START_POS, *WAYPOINTS])
        self.field_ax.plot(
            route[:, 0],
            route[:, 1],
            color="#6f7880",
            linewidth=1.3,
            linestyle="--",
            alpha=0.75,
        )
        self.field_ax.scatter(
            route[:-1, 0],
            route[:-1, 1],
            s=[42] + [34] * (len(route) - 2),
            c=["#2563eb"] + ["#777777"] * (len(route) - 2),
            zorder=4,
        )
        self.field_ax.annotate(
            "",
            xy=(72.0, DYNAMIC_SPAWN[1]),
            xytext=(DYNAMIC_SPAWN[0], DYNAMIC_SPAWN[1]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#c2410c",
                "linewidth": 2.0,
            },
        )
        self.field_ax.text(
            DYNAMIC_SPAWN[0],
            DYNAMIC_SPAWN[1] + 4.2,
            "동적 장애물 생성 위치",
            color="#9a3412",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        goal = Ellipse(
            GOAL_POS,
            width=9.0,
            height=6.5,
            facecolor="red",
            edgecolor="#303030",
            linewidth=1.2,
            zorder=3,
        )
        self.field_ax.add_patch(goal)
        self.field_ax.text(
            *GOAL_POS,
            "목표",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=4,
        )

    def _build_dynamic_artists(self) -> None:
        self.drone_body = Circle(
            START_POS,
            self.config.drone_radius_m,
            facecolor="#1d4ed8",
            edgecolor="white",
            linewidth=1.4,
            zorder=8,
        )
        self.sensor_ring = Circle(
            START_POS,
            self.config.lidar_range_m,
            fill=False,
            edgecolor="#2f80ed",
            linewidth=1.1,
            linestyle="--",
            alpha=0.35,
            zorder=2,
        )
        self.dynamic_body = Circle(
            DYNAMIC_SPAWN,
            self.config.dynamic_radius_m,
            facecolor="#f97316",
            edgecolor="#7c2d12",
            linewidth=1.5,
            zorder=7,
        )
        self.heading_line, = self.field_ax.plot(
            [],
            [],
            color="white",
            linewidth=1.6,
            zorder=9,
        )
        self.drone_trail, = self.field_ax.plot(
            [],
            [],
            color="#1d4ed8",
            linewidth=2.0,
            alpha=0.85,
            zorder=5,
        )
        self.dynamic_trail, = self.field_ax.plot(
            [],
            [],
            color="#f97316",
            linewidth=1.8,
            alpha=0.85,
            zorder=5,
        )
        self.field_ax.add_patch(self.sensor_ring)
        self.field_ax.add_patch(self.drone_body)
        self.field_ax.add_patch(self.dynamic_body)
        self.info_text = self.info_ax.text(
            0.04,
            0.95,
            "",
            transform=self.info_ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            linespacing=1.45,
            color="#16242c",
        )
        self.distance_line, = self.distance_ax.plot(
            [],
            [],
            color="#111827",
            linewidth=1.8,
        )
        self.current_distance_dot, = self.distance_ax.plot(
            [],
            [],
            marker="o",
            color="#d33f49",
            markersize=5,
        )

    def update(self, timeline_index: int):
        result_index, frame_index = self.timeline[timeline_index]
        result = self.results[result_index]
        frame = result.frames[frame_index]
        frames_so_far = result.frames[: frame_index + 1]

        drone_xy = frame.drone_position
        dynamic_xy = frame.dynamic_position
        self.drone_body.center = tuple(drone_xy)
        self.sensor_ring.center = tuple(drone_xy)
        self.dynamic_body.center = tuple(dynamic_xy)
        self.dynamic_body.set_alpha(1.0 if frame.dynamic_active else 0.0)
        self.dynamic_body.set_edgecolor("#dc2626" if frame.dynamic_detected else "#7c2d12")
        self.dynamic_body.set_linewidth(2.3 if frame.dynamic_detected else 1.5)

        heading = np.array([math.cos(frame.yaw), math.sin(frame.yaw)], dtype=float)
        head_end = drone_xy + heading * (self.config.drone_radius_m * 1.45)
        self.heading_line.set_data([drone_xy[0], head_end[0]], [drone_xy[1], head_end[1]])

        drone_path = np.array([f.drone_position for f in frames_so_far])
        self.drone_trail.set_data(drone_path[:, 0], drone_path[:, 1])

        active_dynamic_path = np.array(
            [f.dynamic_position for f in frames_so_far if f.dynamic_active],
            dtype=float,
        )
        if len(active_dynamic_path):
            self.dynamic_trail.set_data(active_dynamic_path[:, 0], active_dynamic_path[:, 1])
        else:
            self.dynamic_trail.set_data([], [])

        times = np.array([f.time_s for f in frames_so_far], dtype=float)
        distances = np.array([f.distance_to_dynamic for f in frames_so_far], dtype=float)
        self.distance_line.set_data(times, distances)
        self.current_distance_dot.set_data([frame.time_s], [frame.distance_to_dynamic])

        spawn_suppressed = result.final_status == "spawn_suppressed"
        detect_distance = (
            "spawn suppressed"
            if spawn_suppressed
            else (
                "not detected"
                if result.first_detection_distance_m is None
                else f"{result.first_detection_distance_m:5.1f} m at {result.first_detection_time_s:4.1f}s"
            )
        )
        current_separation = (
            "NA"
            if spawn_suppressed
            else f"{frame.distance_to_dynamic:5.1f} m"
        )
        min_separation = (
            "NA"
            if not math.isfinite(result.min_distance_m)
            else f"{result.min_distance_m:5.1f} m"
        )
        min_clearance = (
            "inf"
            if not math.isfinite(frame.min_clearance_m)
            else f"{frame.min_clearance_m:5.1f} m"
        )
        self.info_text.set_text(
            "\n".join(
                (
                    f"trial: {result_index + 1}/{len(self.results)}  ({result.scenario.label})",
                    f"release delay: {result.scenario.release_delay_s:4.1f} s",
                    f"time: {frame.time_s:5.1f} s",
                    f"status: {frame.status}",
                    "",
                    f"drone: ({drone_xy[0]:5.1f}, {drone_xy[1]:5.1f}) m",
                    f"target waypoint: {frame.waypoint_index + 1}/{len(WAYPOINTS)}",
                    f"lidar range: {self.config.lidar_range_m:.0f} m",
                    f"dynamic detected: {frame.dynamic_detected}",
                    f"first detection: {detect_distance}",
                    f"current separation: {current_separation}",
                    f"min separation: {min_separation}",
                    f"EGO min clearance: {min_clearance}",
                    "",
                    "red dotted line: collision radius",
                    "blue dashed line: 50 m lidar range",
                )
            )
        )
        return (
            self.drone_body,
            self.sensor_ring,
            self.dynamic_body,
            self.heading_line,
            self.drone_trail,
            self.dynamic_trail,
            self.info_text,
            self.distance_line,
            self.current_distance_dot,
        )

    def save(self, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=self.fps)
        self.animation.save(save_path, writer=writer, dpi=82)
        plt.close(self.figure)


def _make_scenarios(delays: tuple[float, ...]) -> list[Scenario]:
    return [
        Scenario(release_delay_s=delay, label=f"delay {delay:.1f}s")
        for delay in delays
    ]


def run_scenarios(
    delays: tuple[float, ...],
    config: DemoConfig | None = None,
) -> list[SimulationResult]:
    config = config or DemoConfig()
    results = []
    for scenario in _make_scenarios(delays):
        result = StaticDynamicEncounterSimulation(scenario, config).run()
        results.append(result)
    return results


def summarize_results(results: list[SimulationResult]) -> str:
    lines = [
        "release_delay_s,first_detection_distance_m,min_distance_m,final_status"
    ]
    for result in results:
        first_detection = (
            "NA"
            if result.first_detection_distance_m is None
            else f"{result.first_detection_distance_m:.2f}"
        )
        min_distance = (
            "NA"
            if not math.isfinite(result.min_distance_m)
            else f"{result.min_distance_m:.2f}"
        )
        lines.append(
            (
                f"{result.scenario.release_delay_s:.2f},"
                f"{first_detection},"
                f"{min_distance},"
                f"{result.final_status}"
            )
        )

    collision_distances = [
        result.first_detection_distance_m
        for result in results
        if result.final_status == "collision_dynamic"
        and result.first_detection_distance_m is not None
    ]
    success_distances = [
        result.first_detection_distance_m
        for result in results
        if result.final_status == "goal_reached"
        and result.first_detection_distance_m is not None
    ]

    if collision_distances and success_distances:
        lines.extend(
            (
                "",
                (
                    "empirical threshold band: "
                    f"collision observed at <= {max(collision_distances):.2f} m, "
                    f"success observed at >= {min(success_distances):.2f} m"
                ),
            )
        )
    elif collision_distances:
        lines.extend(
            (
                "",
                f"all detected collision cases were at or below {max(collision_distances):.2f} m",
            )
        )
    elif success_distances:
        lines.extend(
            (
                "",
                f"all detected success cases were at or above {min(success_distances):.2f} m",
            )
        )
    return "\n".join(lines)


def _parse_delays(text: str) -> tuple[float, ...]:
    delays = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not delays:
        raise argparse.ArgumentTypeError("at least one delay is required")
    if any(delay < 0.0 for delay in delays):
        raise argparse.ArgumentTypeError("delays must be non-negative")
    return delays


def parse_args() -> argparse.Namespace:
    default_save = Path(__file__).with_name("static_dynamic_encounter_demo.gif")
    parser = argparse.ArgumentParser(
        description="Create a matplotlib EGO-planner dynamic-encounter GIF on the static map.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=default_save,
        help="combined GIF output path",
    )
    parser.add_argument(
        "--delays",
        type=_parse_delays,
        default=DEFAULT_DELAYS,
        help="comma-separated dynamic obstacle release delays in seconds",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="GIF frame rate",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=3,
        help="render every Nth simulation frame to keep GIF generation lightweight",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="simulate and print result table without rendering GIF",
    )
    parser.add_argument(
        "--save-trials",
        action="store_true",
        help="also save one GIF per release-delay trial",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="kept for compatibility; this script saves files with the Agg backend",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DemoConfig()
    results = run_scenarios(args.delays, config)
    print(summarize_results(results))

    if args.summary_only:
        return

    renderer = EncounterRenderer(results, config, args.fps, args.frame_stride)
    renderer.save(args.save)
    print(args.save)

    if args.save_trials:
        stem = args.save.with_suffix("")
        for index, result in enumerate(results, start=1):
            trial_path = stem.parent / f"{stem.name}_trial_{index:02d}_{result.final_status}.gif"
            EncounterRenderer([result], config, args.fps, args.frame_stride).save(trial_path)
            print(trial_path)


if __name__ == "__main__":
    main()
