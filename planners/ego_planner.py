import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    minimize = None


_TIMING_KEYS = (
    "grid",
    "init_bspline",
    "anchor_astar",
    "rebound_optimize",
    "refine",
    "collision_check",
    "command_generation",
    "total",
)

_REBOUND_TIMING_KEYS = (
    "setup",
    "solver_wall",
    "objective_total",
    "objective_rebuild",
    "objective_smooth",
    "objective_collision",
    "objective_feasibility",
    "objective_assembly",
    "optimizer_overhead",
    "result_apply",
    "total",
)


@dataclass
class EGOPlannerConfig:
    max_vel_mps: float = 20.0
    max_acc_mps2: float = 30.0
    grid_resolution_m: float = 0.5
    safe_radius_m: float = 2.0
    min_range_m: float = 0.2
    max_range_m: float = 50.0
    z_band_m: float = 1.0
    horizon_time_s: float = 2.5
    max_horizon_m: float = 50.0
    min_horizon_m: float = 8.0
    control_point_distance_m: float = 2.0
    command_lookahead_s: float = 0.8
    max_lateral_speed_mps: float = 20.0
    min_lateral_speed_mps: float = 8.0
    lateral_gain: float = 1.4
    lambda_smooth: float = 1.0
    lambda_collision: float = 0.5
    lambda_feasibility: float = 0.1
    lambda_fitness: float = 1.0
    optimizer_max_iterations: int = 80
    refine_max_iterations: int = 60
    rebound_collision_retries: int = 3
    feasibility_tolerance: float = 0.05
    astar_max_expansions: int = 20000


@dataclass
class EGOPlanCommand:
    vx: float
    vy: float
    yaw: float
    blocked: bool
    plan_time_ms: float
    min_clearance: float
    traj_duration: float
    astar_segments: int
    optimizer_success: bool
    command_speed: float
    refined: bool
    command_forward: float = 0.0
    command_lateral: float = 0.0
    maneuver_mode: str = "unknown"
    timing_ms: dict = field(default_factory=dict)
    rebound_timing_ms: dict = field(default_factory=dict)
    rebound_stats: dict = field(default_factory=dict)
    reason: str = "ok"


class _OccupancyGrid2D:
    def __init__(self, resolution, safe_radius):
        self.resolution = resolution
        self.safe_radius = safe_radius
        self.raw_cells = set()
        self.occupied_cells = set()
        self.raw_points = []
        self._inflate_steps = int(math.ceil(safe_radius / resolution))

    def add_point(self, point):
        cell = self.point_to_cell(point)
        self.raw_cells.add(cell)
        self.raw_points.append(np.asarray(point, dtype=float))

        radius_cells = self._inflate_steps
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                inflated = (cell[0] + dx, cell[1] + dy)
                center = self.cell_center(inflated)
                if np.linalg.norm(center - point) <= self.safe_radius + self.resolution:
                    self.occupied_cells.add(inflated)

    def point_to_cell(self, point):
        return (
            int(math.floor(float(point[0]) / self.resolution)),
            int(math.floor(float(point[1]) / self.resolution)),
        )

    def cell_center(self, cell):
        return np.array(
            [
                (cell[0] + 0.5) * self.resolution,
                (cell[1] + 0.5) * self.resolution,
            ],
            dtype=float,
        )

    def is_occupied_point(self, point):
        return self.point_to_cell(point) in self.occupied_cells

    def min_clearance(self, points):
        if not self.raw_points:
            return float("inf")

        obstacle_points = np.vstack(self.raw_points)
        min_clearance = float("inf")
        for point in points:
            distances = np.linalg.norm(obstacle_points - point, axis=1)
            min_clearance = min(min_clearance, float(distances.min()) - self.safe_radius)
        return min_clearance


class _UniformCubicBspline2D:
    order = 3

    def __init__(self, control_points, interval):
        self.control_points = np.asarray(control_points, dtype=float)
        self.interval = float(interval)
        self.knots = self._build_knots(len(self.control_points), self.interval)

    def _build_knots(self, point_count, interval):
        n = point_count - 1
        m = n + self.order + 1
        knots = np.zeros(m + 1, dtype=float)
        for i in range(m + 1):
            if i <= self.order:
                knots[i] = float(-self.order + i) * interval
            else:
                knots[i] = knots[i - 1] + interval
        return knots

    @property
    def duration(self):
        return float(self.knots[len(self.control_points)] - self.knots[self.order])

    def evaluate(self, t):
        u = min(max(float(t) + self.knots[self.order], self.knots[self.order]), self.knots[len(self.control_points)])
        k = self.order
        while k + 1 < len(self.knots) and self.knots[k + 1] < u:
            k += 1

        d = [
            self.control_points[k - self.order + i].copy()
            for i in range(self.order + 1)
        ]
        for r in range(1, self.order + 1):
            for i in range(self.order, r - 1, -1):
                left = self.knots[i + k - self.order]
                right = self.knots[i + 1 + k - r]
                alpha = 0.0 if right == left else (u - left) / (right - left)
                d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
        return d[self.order]

    def sample(self, step):
        duration = max(0.0, self.duration)
        if duration <= 0.0:
            return np.array([self.evaluate(0.0)])
        times = np.arange(0.0, duration + 1e-9, max(step, 1e-3))
        if times[-1] < duration:
            times = np.append(times, duration)
        return np.array([self.evaluate(t) for t in times])


class EGOPlanner:
    def __init__(self, config=None):
        self.config = config or EGOPlannerConfig()
        self.last_debug = {}
        self._last_control_points = None
        self._last_dt = None

    def _new_timing(self):
        return {key: 0.0 for key in _TIMING_KEYS}

    def _new_rebound_timing(self):
        return {key: 0.0 for key in _REBOUND_TIMING_KEYS}

    def _new_rebound_stats(self):
        return {
            "attempts": 0,
            "objective_calls": 0,
            "lbfgs_iterations": 0,
            "function_evals": 0,
            "gradient_evals": 0,
            "last_status": None,
            "solver": "none",
            "success": False,
        }

    def _record_timing(self, timing_ms, stage, started_at):
        timing_ms[stage] += (time.perf_counter() - started_at) * 1000.0

    def _record_rebound_timing(self, rebound_timing_ms, stage, started_at):
        if rebound_timing_ms is not None:
            rebound_timing_ms[stage] += (time.perf_counter() - started_at) * 1000.0

    def _rounded_timing(self, timing_ms):
        return {key: round(float(timing_ms.get(key, 0.0)), 3) for key in _TIMING_KEYS}

    def _rounded_rebound_timing(self, rebound_timing_ms):
        return {
            key: round(float(rebound_timing_ms.get(key, 0.0)), 3)
            for key in _REBOUND_TIMING_KEYS
        }

    def plan(
        self,
        point_cloud,
        position,
        velocity,
        yaw,
        target_position,
        target_speed=None,
    ):
        started_at = time.perf_counter()
        timing_ms = self._new_timing()
        rebound_timing_ms = self._new_rebound_timing()
        rebound_stats = self._new_rebound_stats()
        start = self._xy(position)
        start_velocity = self._xy(velocity)
        target = self._xy(target_position)
        command_speed = self._command_speed(target_speed, start_velocity)

        stage_started = time.perf_counter()
        obstacle_grid = self._build_grid(point_cloud, start, yaw)
        self._record_timing(timing_ms, "grid", stage_started)

        target_vector = target - start
        target_distance = float(np.linalg.norm(target_vector))
        target_heading = math.atan2(target_vector[1], target_vector[0]) if target_distance > 1e-6 else yaw

        if target_distance < 1e-6:
            return self._finish_command(
                started_at,
                0.0,
                0.0,
                yaw,
                blocked=False,
                min_clearance=float("inf"),
                traj_duration=0.0,
                astar_segments=0,
                optimizer_success=True,
                command_speed=0.0,
                refined=False,
                reason="at_target",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )

        if obstacle_grid.is_occupied_point(start):
            return self._blocked_command(
                started_at,
                yaw,
                obstacle_grid,
                "start_in_occupied_cell",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )

        if not obstacle_grid.raw_points:
            return self._straight_command(
                started_at,
                target_heading,
                command_speed,
                float("inf"),
                "no_obstacles",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )

        stage_started = time.perf_counter()
        local_target = self._local_target(start, target, command_speed)
        end_velocity = self._end_velocity(start, target, local_target, command_speed)
        control_points, dt = self._initial_bspline(
            start,
            start_velocity,
            local_target,
            end_velocity,
        )
        self._record_timing(timing_ms, "init_bspline", stage_started)

        stage_started = time.perf_counter()
        anchors, astar_paths = self._make_rebound_anchors(control_points, obstacle_grid)
        self._record_timing(timing_ms, "anchor_astar", stage_started)

        astar_segments = 0 if astar_paths is None else len(astar_paths)
        optimizer_success = True
        refined = False

        if anchors is None:
            return self._blocked_command(
                started_at,
                yaw,
                obstacle_grid,
                "astar_failed",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )

        if anchors:
            collision_weight = self.config.lambda_collision
            for retry in range(self.config.rebound_collision_retries + 1):
                stage_started = time.perf_counter()
                control_points, optimizer_success = self._optimize_rebound(
                    control_points,
                    dt,
                    anchors,
                    lambda_collision=collision_weight,
                    rebound_timing_ms=rebound_timing_ms,
                    rebound_stats=rebound_stats,
                )
                self._record_timing(timing_ms, "rebound_optimize", stage_started)

                stage_started = time.perf_counter()
                collides_after_rebound = (
                    optimizer_success
                    and self._trajectory_collides(control_points, dt, obstacle_grid)
                )
                self._record_timing(timing_ms, "collision_check", stage_started)

                if not optimizer_success or not collides_after_rebound:
                    break
                if retry >= self.config.rebound_collision_retries:
                    break

                stage_started = time.perf_counter()
                anchors, astar_paths = self._make_rebound_anchors(control_points, obstacle_grid)
                self._record_timing(timing_ms, "anchor_astar", stage_started)
                if anchors is None:
                    return self._blocked_command(
                        started_at,
                        yaw,
                        obstacle_grid,
                        "astar_failed_after_rebound",
                        timing_ms=timing_ms,
                        rebound_timing_ms=rebound_timing_ms,
                        rebound_stats=rebound_stats,
                        astar_segments=astar_segments,
                    )
                astar_segments = len(astar_paths) if astar_paths is not None else astar_segments
                if not anchors:
                    break
                collision_weight *= 2.0

        if not optimizer_success:
            return self._blocked_command(
                started_at,
                yaw,
                obstacle_grid,
                "optimizer_failed",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
                astar_segments=astar_segments,
            )

        stage_started = time.perf_counter()
        control_points, dt, refined = self._refine_if_needed(control_points, dt)
        self._record_timing(timing_ms, "refine", stage_started)

        stage_started = time.perf_counter()
        trajectory = _UniformCubicBspline2D(control_points, dt)
        samples = trajectory.sample(max(self.config.grid_resolution_m, command_speed * 0.05))
        min_clearance = obstacle_grid.min_clearance(samples)
        trajectory_collides = self._trajectory_collides(control_points, dt, obstacle_grid)
        self._record_timing(timing_ms, "collision_check", stage_started)

        if trajectory_collides:
            return self._blocked_command(
                started_at,
                yaw,
                obstacle_grid,
                "optimized_trajectory_collides",
                timing_ms=timing_ms,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
                min_clearance=min_clearance,
                astar_segments=astar_segments,
            )

        stage_started = time.perf_counter()
        lookahead = min(max(dt, self.config.command_lookahead_s), max(dt, trajectory.duration))
        lookahead_point = trajectory.evaluate(lookahead)
        vx, vy, command_yaw, command_forward, command_lateral, maneuver_mode = self._forward_lateral_command(
            start,
            lookahead_point,
            target_heading,
            command_speed,
            lookahead,
        )
        self._record_timing(timing_ms, "command_generation", stage_started)

        self._last_control_points = control_points
        self._last_dt = dt
        self.last_debug = {
            "status": "ok",
            "raw_obstacles": len(obstacle_grid.raw_points),
            "occupied_cells": len(obstacle_grid.occupied_cells),
            "astar_segments": astar_segments,
            "optimizer_success": optimizer_success,
            "refined": refined,
            "min_clearance": round(min_clearance, 3),
            "traj_duration": round(trajectory.duration, 3),
            "dt": round(dt, 3),
            "command": (round(float(vx), 3), round(float(vy), 3)),
            "command_forward": round(command_forward, 3),
            "command_lateral": round(command_lateral, 3),
            "maneuver_mode": maneuver_mode,
            "timing_ms": self._rounded_timing(timing_ms),
            "rebound_timing_ms": self._rounded_rebound_timing(rebound_timing_ms),
            "rebound_stats": dict(rebound_stats),
        }

        command = self._finish_command(
            started_at,
            vx,
            vy,
            command_yaw,
            blocked=False,
            min_clearance=min_clearance,
            traj_duration=trajectory.duration,
            astar_segments=astar_segments,
            optimizer_success=optimizer_success,
            command_speed=math.hypot(vx, vy),
            refined=refined,
            command_forward=command_forward,
            command_lateral=command_lateral,
            maneuver_mode=maneuver_mode,
            timing_ms=timing_ms,
            rebound_timing_ms=rebound_timing_ms,
            rebound_stats=rebound_stats,
        )
        self.last_debug["timing_ms"] = self._rounded_timing(command.timing_ms)
        self.last_debug["rebound_timing_ms"] = self._rounded_rebound_timing(command.rebound_timing_ms)
        self.last_debug["rebound_stats"] = dict(command.rebound_stats)
        return command

    def _build_grid(self, point_cloud, position, yaw):
        grid = _OccupancyGrid2D(self.config.grid_resolution_m, self.config.safe_radius_m)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for i in range(0, len(point_cloud) - 2, 3):
            x = float(point_cloud[i])
            y = float(point_cloud[i + 1])
            z = float(point_cloud[i + 2])
            if abs(z) > self.config.z_band_m:
                continue

            distance = math.hypot(x, y)
            if distance < self.config.min_range_m or distance > self.config.max_range_m:
                continue

            world = np.array(
                [
                    position[0] + x * cos_yaw - y * sin_yaw,
                    position[1] + x * sin_yaw + y * cos_yaw,
                ],
                dtype=float,
            )
            grid.add_point(world)
        return grid

    def _initial_bspline(self, start, start_velocity, local_target, end_velocity):
        distance = max(float(np.linalg.norm(local_target - start)), 1e-3)
        accel_distance = self.config.max_vel_mps * self.config.max_vel_mps / self.config.max_acc_mps2
        if accel_distance > distance:
            duration = max(math.sqrt(distance / self.config.max_acc_mps2), 0.5)
        else:
            duration = (
                (distance - accel_distance) / self.config.max_vel_mps
                + 2.0 * self.config.max_vel_mps / self.config.max_acc_mps2
            )

        dt = max(self.config.control_point_distance_m / self.config.max_vel_mps * 1.2, 0.05)
        point_count = max(7, int(math.ceil(duration / dt)) + 1)
        times = np.linspace(0.0, duration, point_count)
        points = np.array(
            [
                self._quintic_point(start, start_velocity, local_target, end_velocity, duration, t)
                for t in times
            ],
            dtype=float,
        )
        dt = duration / max(point_count - 1, 1)
        derivatives = [
            start_velocity,
            end_velocity,
            np.zeros(2, dtype=float),
            np.zeros(2, dtype=float),
        ]
        return self._parameterize_to_bspline(dt, points, derivatives), dt

    def _quintic_point(self, start, start_vel, end, end_vel, duration, t):
        if duration <= 1e-6:
            return end.copy()

        coeffs = []
        for dim in range(2):
            a0 = start[dim]
            a1 = start_vel[dim]
            a2 = 0.0
            matrix = np.array(
                [
                    [duration**3, duration**4, duration**5],
                    [3.0 * duration**2, 4.0 * duration**3, 5.0 * duration**4],
                    [6.0 * duration, 12.0 * duration**2, 20.0 * duration**3],
                ],
                dtype=float,
            )
            rhs = np.array(
                [
                    end[dim] - (a0 + a1 * duration + a2 * duration**2),
                    end_vel[dim] - (a1 + 2.0 * a2 * duration),
                    0.0 - (2.0 * a2),
                ],
                dtype=float,
            )
            a3, a4, a5 = np.linalg.solve(matrix, rhs)
            coeffs.append((a0, a1, a2, a3, a4, a5))

        values = []
        for a0, a1, a2, a3, a4, a5 in coeffs:
            values.append(a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5)
        return np.array(values, dtype=float)

    def _parameterize_to_bspline(self, dt, point_set, start_end_derivatives):
        point_count = len(point_set)
        matrix = np.zeros((point_count + 4, point_count + 2), dtype=float)
        prow = np.array([1.0, 4.0, 1.0], dtype=float) / 6.0
        vrow = np.array([-1.0, 0.0, 1.0], dtype=float) / (2.0 * dt)
        arow = np.array([1.0, -2.0, 1.0], dtype=float) / (dt * dt)

        for i in range(point_count):
            matrix[i, i : i + 3] = prow
        matrix[point_count, 0:3] = vrow
        matrix[point_count + 1, point_count - 1 : point_count + 2] = vrow
        matrix[point_count + 2, 0:3] = arow
        matrix[point_count + 3, point_count - 1 : point_count + 2] = arow

        rhs = np.vstack([point_set, np.asarray(start_end_derivatives, dtype=float)])
        control = np.zeros((point_count + 2, 2), dtype=float)
        for dim in range(2):
            control[:, dim] = np.linalg.lstsq(matrix, rhs[:, dim], rcond=None)[0]
        return control

    def _make_rebound_anchors(self, control_points, grid):
        segments = self._collision_segments(control_points, grid)
        if not segments:
            return {}, []

        anchors = {i: [] for i in range(len(control_points))}
        astar_paths = []
        for start_id, end_id in segments:
            start_id = self._nearest_free_control_point(control_points, grid, start_id, -1)
            end_id = self._nearest_free_control_point(control_points, grid, end_id, 1)
            if start_id is None or end_id is None or start_id >= end_id:
                return None, None

            path = self._astar(control_points[start_id], control_points[end_id], grid)
            if path is None:
                return None, None
            astar_paths.append(path)

            for point_id in range(start_id + 1, end_id):
                guide_point = self._guide_point_from_path(
                    control_points,
                    point_id,
                    path,
                )
                anchor = self._surface_anchor(control_points[point_id], guide_point, grid)
                if anchor is None:
                    obstacle_direction = self._nearest_obstacle_direction(control_points[point_id], grid)
                    if np.linalg.norm(obstacle_direction) < 1e-6:
                        continue
                    anchor = (control_points[point_id].copy(), obstacle_direction)
                anchors[point_id].append(anchor)

        anchors = {idx: values for idx, values in anchors.items() if values}
        return anchors, astar_paths

    def _guide_point_from_path(self, control_points, point_id, path):
        point = control_points[point_id]
        if point_id > 0 and point_id + 1 < len(control_points):
            control_law = control_points[point_id + 1] - control_points[point_id - 1]
        else:
            control_law = path[-1] - path[0]

        if np.linalg.norm(control_law) < 1e-6 or len(path) < 2:
            return path[np.argmin(np.linalg.norm(path - point, axis=1))]

        values = (path - point) @ control_law
        for idx in range(len(path) - 1):
            value_a = values[idx]
            value_b = values[idx + 1]
            if value_a == 0.0:
                return path[idx]
            if value_a * value_b <= 0.0 and abs(value_a) + abs(value_b) > 1e-9:
                ratio = abs(value_a) / (abs(value_a) + abs(value_b))
                return path[idx] + ratio * (path[idx + 1] - path[idx])

        return path[np.argmin(np.linalg.norm(path - point, axis=1))]

    def _surface_anchor(self, control_point, guide_point, grid):
        direction = guide_point - control_point
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return None

        direction = direction / length
        distance = length
        while distance >= 0.0:
            probe = control_point + direction * distance
            if grid.is_occupied_point(probe) or distance < grid.resolution:
                if grid.is_occupied_point(probe):
                    distance = min(length, distance + grid.resolution)
                base_point = control_point + direction * distance
                return base_point, direction
            distance -= grid.resolution

        return control_point.copy(), direction

    def _collision_segments(self, control_points, grid):
        order = _UniformCubicBspline2D.order
        end = len(control_points) - order
        colliding = []
        for idx in range(order, end):
            collides = grid.is_occupied_point(control_points[idx])
            if not collides and idx > 0:
                collides = self._segment_collides(control_points[idx - 1], control_points[idx], grid)
            if not collides and idx + 1 < len(control_points):
                collides = self._segment_collides(control_points[idx], control_points[idx + 1], grid)
            colliding.append((idx, collides))

        segments = []
        segment_start = None
        previous = None
        for idx, is_colliding in colliding:
            if is_colliding and segment_start is None:
                segment_start = idx
            elif not is_colliding and segment_start is not None:
                segments.append((max(order, segment_start - 1), min(end - 1, previous + 1)))
                segment_start = None
            previous = idx
        if segment_start is not None:
            segments.append((max(order, segment_start - 1), min(end - 1, previous + 1)))
        return segments

    def _segment_collides(self, start, end, grid):
        distance = float(np.linalg.norm(end - start))
        if distance <= 1e-9:
            return grid.is_occupied_point(start)
        steps = max(1, int(math.ceil(distance / grid.resolution)))
        for i in range(steps + 1):
            point = start + (end - start) * (i / steps)
            if grid.is_occupied_point(point):
                return True
        return False

    def _nearest_free_control_point(self, control_points, grid, start_id, step):
        idx = start_id
        while 0 <= idx < len(control_points):
            if not grid.is_occupied_point(control_points[idx]):
                return idx
            idx += step
        return None

    def _nearest_obstacle_direction(self, point, grid):
        if not grid.raw_points:
            return np.zeros(2, dtype=float)
        obstacle_points = np.vstack(grid.raw_points)
        nearest = obstacle_points[np.argmin(np.linalg.norm(obstacle_points - point, axis=1))]
        direction = point - nearest
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([0.0, 1.0], dtype=float)
        return direction / norm

    def _astar(self, start, goal, grid):
        start_cell = grid.point_to_cell(start)
        goal_cell = grid.point_to_cell(goal)
        start_cell = self._free_cell_near(start_cell, goal_cell, grid, away=True)
        goal_cell = self._free_cell_near(goal_cell, start_cell, grid, away=True)
        if start_cell is None or goal_cell is None:
            return None

        obstacle_cells = list(grid.occupied_cells) or [start_cell, goal_cell]
        xs = [start_cell[0], goal_cell[0], *[cell[0] for cell in obstacle_cells]]
        ys = [start_cell[1], goal_cell[1], *[cell[1] for cell in obstacle_cells]]
        margin = max(8, int(math.ceil(self.config.safe_radius_m * 4.0 / grid.resolution)))
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin

        open_heap = [(0.0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0.0}
        closed = set()
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        expansions = 0
        while open_heap and expansions < self.config.astar_max_expansions:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_cell:
                return self._reconstruct_path(came_from, current, grid)
            closed.add(current)
            expansions += 1

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor[0] < min_x or neighbor[0] > max_x or neighbor[1] < min_y or neighbor[1] > max_y:
                    continue
                if neighbor in grid.occupied_cells:
                    continue
                step_cost = math.sqrt(dx * dx + dy * dy)
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + self._diag_heuristic(neighbor, goal_cell)
                heapq.heappush(open_heap, (f_score, neighbor))

        return None

    def _free_cell_near(self, cell, other_cell, grid, away):
        if cell not in grid.occupied_cells:
            return cell

        direction = np.array([cell[0] - other_cell[0], cell[1] - other_cell[1]], dtype=float)
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0], dtype=float)
        direction = direction / np.linalg.norm(direction)
        if not away:
            direction = -direction

        for radius in range(1, max(4, grid._inflate_steps * 3) + 1):
            candidate = (
                int(round(cell[0] + direction[0] * radius)),
                int(round(cell[1] + direction[1] * radius)),
            )
            if candidate not in grid.occupied_cells:
                return candidate

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if candidate not in grid.occupied_cells:
                        return candidate
        return None

    def _diag_heuristic(self, cell, goal):
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        diagonal = min(dx, dy)
        straight = abs(dx - dy)
        return (math.sqrt(2.0) * diagonal + straight) * (1.0 + 1.0 / 10000.0)

    def _reconstruct_path(self, came_from, current, grid):
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return np.array([grid.cell_center(cell) for cell in cells], dtype=float)

    def _optimize_rebound(
        self,
        control_points,
        dt,
        anchors,
        lambda_collision=None,
        rebound_timing_ms=None,
        rebound_stats=None,
    ):
        total_started = time.perf_counter()
        setup_started = time.perf_counter()
        if rebound_stats is not None:
            rebound_stats["attempts"] += 1
        if minimize is None:
            self._record_rebound_timing(rebound_timing_ms, "setup", setup_started)
            optimized, success = self._gradient_descent_rebound(
                control_points,
                dt,
                anchors,
                lambda_collision,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )
            if rebound_stats is not None:
                rebound_stats["success"] = success
            self._record_rebound_timing(rebound_timing_ms, "total", total_started)
            return optimized, success

        start_id, end_id = self._variable_bounds(control_points)
        if start_id >= end_id:
            self._record_rebound_timing(rebound_timing_ms, "setup", setup_started)
            if rebound_stats is not None:
                rebound_stats["solver"] = "lbfgs_b"
                rebound_stats["success"] = False
            self._record_rebound_timing(rebound_timing_ms, "total", total_started)
            return control_points, False

        initial = control_points[start_id:end_id].reshape(-1)
        self._record_rebound_timing(rebound_timing_ms, "setup", setup_started)

        objective_before = 0.0
        if rebound_timing_ms is not None:
            objective_before = rebound_timing_ms["objective_total"]
        solver_started = time.perf_counter()
        result = minimize(
            lambda x: self._rebound_cost_and_gradient(
                x,
                control_points,
                dt,
                anchors,
                start_id,
                end_id,
                lambda_collision,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            ),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": self.config.optimizer_max_iterations,
                "gtol": 0.01,
                "ftol": 1e-9,
            },
        )
        solver_wall_ms = (time.perf_counter() - solver_started) * 1000.0
        if rebound_timing_ms is not None:
            rebound_timing_ms["solver_wall"] += solver_wall_ms
            objective_delta = rebound_timing_ms["objective_total"] - objective_before
            rebound_timing_ms["optimizer_overhead"] += max(0.0, solver_wall_ms - objective_delta)

        if rebound_stats is not None:
            rebound_stats["solver"] = "lbfgs_b"
            rebound_stats["lbfgs_iterations"] += int(getattr(result, "nit", 0) or 0)
            rebound_stats["function_evals"] += int(getattr(result, "nfev", 0) or 0)
            rebound_stats["gradient_evals"] += int(getattr(result, "njev", 0) or 0)
            rebound_stats["last_status"] = int(getattr(result, "status", -1))

        result_started = time.perf_counter()
        optimized = control_points.copy()
        optimized[start_id:end_id] = result.x.reshape((-1, 2))
        self._record_rebound_timing(rebound_timing_ms, "result_apply", result_started)

        success = bool(result.success or result.nit > 0)
        if rebound_stats is not None:
            rebound_stats["success"] = success
        self._record_rebound_timing(rebound_timing_ms, "total", total_started)
        return optimized, success

    def _gradient_descent_rebound(
        self,
        control_points,
        dt,
        anchors,
        lambda_collision,
        rebound_timing_ms=None,
        rebound_stats=None,
    ):
        solver_started = time.perf_counter()
        objective_before = 0.0
        if rebound_timing_ms is not None:
            objective_before = rebound_timing_ms["objective_total"]
        if rebound_stats is not None:
            rebound_stats["solver"] = "gradient_descent"

        start_id, end_id = self._variable_bounds(control_points)
        x = control_points[start_id:end_id].reshape(-1)
        step = 0.02
        iterations = 0
        for iterations in range(1, self.config.optimizer_max_iterations + 1):
            _, grad = self._rebound_cost_and_gradient(
                x,
                control_points,
                dt,
                anchors,
                start_id,
                end_id,
                lambda_collision,
                rebound_timing_ms=rebound_timing_ms,
                rebound_stats=rebound_stats,
            )
            x = x - step * grad
            if np.linalg.norm(grad) < 0.01:
                break
        solver_wall_ms = (time.perf_counter() - solver_started) * 1000.0
        if rebound_timing_ms is not None:
            rebound_timing_ms["solver_wall"] += solver_wall_ms
            objective_delta = rebound_timing_ms["objective_total"] - objective_before
            rebound_timing_ms["optimizer_overhead"] += max(0.0, solver_wall_ms - objective_delta)
        if rebound_stats is not None:
            rebound_stats["lbfgs_iterations"] += iterations
            rebound_stats["function_evals"] += iterations
            rebound_stats["gradient_evals"] += iterations
            rebound_stats["last_status"] = 0

        result_started = time.perf_counter()
        optimized = control_points.copy()
        optimized[start_id:end_id] = x.reshape((-1, 2))
        self._record_rebound_timing(rebound_timing_ms, "result_apply", result_started)
        return optimized, True

    def _rebound_cost_and_gradient(
        self,
        x,
        base_control_points,
        dt,
        anchors,
        start_id,
        end_id,
        lambda_collision,
        rebound_timing_ms=None,
        rebound_stats=None,
    ):
        objective_started = time.perf_counter()
        if rebound_stats is not None:
            rebound_stats["objective_calls"] += 1

        stage_started = time.perf_counter()
        q = base_control_points.copy()
        q[start_id:end_id] = x.reshape((-1, 2))
        self._record_rebound_timing(rebound_timing_ms, "objective_rebuild", stage_started)

        stage_started = time.perf_counter()
        smooth_cost, smooth_grad = self._smoothness_cost(q)
        self._record_rebound_timing(rebound_timing_ms, "objective_smooth", stage_started)

        stage_started = time.perf_counter()
        collision_cost, collision_grad = self._collision_cost(q, anchors)
        self._record_rebound_timing(rebound_timing_ms, "objective_collision", stage_started)

        stage_started = time.perf_counter()
        feasibility_cost, feasibility_grad = self._feasibility_cost(q, dt)
        self._record_rebound_timing(rebound_timing_ms, "objective_feasibility", stage_started)

        stage_started = time.perf_counter()
        collision_weight = self.config.lambda_collision if lambda_collision is None else lambda_collision
        cost = (
            self.config.lambda_smooth * smooth_cost
            + collision_weight * collision_cost
            + self.config.lambda_feasibility * feasibility_cost
        )
        grad = (
            self.config.lambda_smooth * smooth_grad
            + collision_weight * collision_grad
            + self.config.lambda_feasibility * feasibility_grad
        )
        objective_result = float(cost), grad[start_id:end_id].reshape(-1)
        self._record_rebound_timing(rebound_timing_ms, "objective_assembly", stage_started)
        self._record_rebound_timing(rebound_timing_ms, "objective_total", objective_started)
        return objective_result

    def _smoothness_cost(self, q):
        grad = np.zeros_like(q)
        if len(q) < 4:
            return 0.0, grad

        jerk = q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]
        cost = float(np.sum(jerk * jerk))
        temp = 2.0 * jerk
        grad[:-3] -= temp
        grad[1:-2] += 3.0 * temp
        grad[2:-1] -= 3.0 * temp
        grad[3:] += temp
        return cost, grad

    def _collision_cost(self, q, anchors):
        grad = np.zeros_like(q)
        cost = 0.0
        clearance = self.config.safe_radius_m
        demarcation = clearance
        a = 3.0 * demarcation
        b = -3.0 * demarcation * demarcation
        c = demarcation**3

        for idx, values in anchors.items():
            for base_point, direction in values:
                distance = float(np.dot(q[idx] - base_point, direction))
                distance_error = clearance - distance
                if distance_error <= 0.0:
                    continue
                if distance_error < demarcation:
                    cost += distance_error**3
                    grad[idx] += -3.0 * distance_error * distance_error * direction
                else:
                    cost += a * distance_error * distance_error + b * distance_error + c
                    grad[idx] += -(2.0 * a * distance_error + b) * direction
        return cost, grad

    def _feasibility_cost(self, q, dt):
        grad = np.zeros_like(q)
        dt = max(dt, 1e-6)
        dt_inv = 1.0 / dt
        dt_inv2 = 1.0 / (dt * dt)

        cost = 0.0
        if len(q) >= 2:
            velocity = np.diff(q, axis=0) * dt_inv
            velocity_diff = np.where(
                velocity > self.config.max_vel_mps,
                velocity - self.config.max_vel_mps,
                np.where(velocity < -self.config.max_vel_mps, velocity + self.config.max_vel_mps, 0.0),
            )
            cost += float(np.sum(velocity_diff * velocity_diff) * dt_inv2)
            value = 2.0 * velocity_diff * dt_inv * dt_inv2
            grad[:-1] -= value
            grad[1:] += value

        if len(q) >= 3:
            acceleration = (q[2:] - 2.0 * q[1:-1] + q[:-2]) * dt_inv2
            acceleration_diff = np.where(
                acceleration > self.config.max_acc_mps2,
                acceleration - self.config.max_acc_mps2,
                np.where(
                    acceleration < -self.config.max_acc_mps2,
                    acceleration + self.config.max_acc_mps2,
                    0.0,
                ),
            )
            cost += float(np.sum(acceleration_diff * acceleration_diff))
            value = 2.0 * acceleration_diff * dt_inv2
            grad[:-2] += value
            grad[1:-1] -= 2.0 * value
            grad[2:] += value

        return cost, grad

    def _refine_if_needed(self, control_points, dt):
        ratio = self._feasibility_ratio(control_points, dt)
        if ratio <= 1.0 + self.config.feasibility_tolerance:
            return control_points, dt, False

        new_dt = dt * max(1.05, ratio * 1.05)
        if minimize is None:
            return control_points, new_dt, True

        reference = control_points.copy()
        start_id, end_id = self._variable_bounds(control_points)
        initial = control_points[start_id:end_id].reshape(-1)
        result = minimize(
            lambda x: self._refine_cost_and_gradient(
                x,
                control_points,
                reference,
                new_dt,
                start_id,
                end_id,
            ),
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": self.config.refine_max_iterations,
                "gtol": 0.001,
                "ftol": 1e-9,
            },
        )
        refined = control_points.copy()
        refined[start_id:end_id] = result.x.reshape((-1, 2))
        return refined, new_dt, True

    def _refine_cost_and_gradient(self, x, base_control_points, reference, dt, start_id, end_id):
        q = base_control_points.copy()
        q[start_id:end_id] = x.reshape((-1, 2))
        smooth_cost, smooth_grad = self._smoothness_cost(q)
        feasibility_cost, feasibility_grad = self._feasibility_cost(q, dt)
        fitness_cost = float(np.sum((q - reference) ** 2))
        fitness_grad = 2.0 * (q - reference)

        cost = (
            self.config.lambda_smooth * smooth_cost
            + self.config.lambda_fitness * fitness_cost
            + self.config.lambda_feasibility * feasibility_cost
        )
        grad = (
            self.config.lambda_smooth * smooth_grad
            + self.config.lambda_fitness * fitness_grad
            + self.config.lambda_feasibility * feasibility_grad
        )
        return float(cost), grad[start_id:end_id].reshape(-1)

    def _feasibility_ratio(self, control_points, dt):
        dt = max(dt, 1e-6)
        max_vel = 0.0
        max_acc = 0.0
        for i in range(len(control_points) - 1):
            max_vel = max(max_vel, float(np.max(np.abs((control_points[i + 1] - control_points[i]) / dt))))
        for i in range(len(control_points) - 2):
            acc = (control_points[i + 2] - 2.0 * control_points[i + 1] + control_points[i]) / (dt * dt)
            max_acc = max(max_acc, float(np.max(np.abs(acc))))

        vel_ratio = max_vel / max(self.config.max_vel_mps, 1e-6)
        acc_ratio = math.sqrt(max_acc / max(self.config.max_acc_mps2, 1e-6))
        return max(vel_ratio, acc_ratio)

    def _trajectory_collides(self, control_points, dt, grid):
        trajectory = _UniformCubicBspline2D(control_points, dt)
        step = max(0.02, self.config.grid_resolution_m / max(self.config.max_vel_mps, 1.0))
        check_duration = trajectory.duration * 2.0 / 3.0
        for t in np.arange(0.0, check_duration + 1e-9, step):
            if grid.is_occupied_point(trajectory.evaluate(t)):
                return True
        return False

    def _local_target(self, start, target, command_speed):
        vector = target - start
        distance = float(np.linalg.norm(vector))
        if distance <= 1e-6:
            return target.copy()
        horizon = min(
            self.config.max_horizon_m,
            max(self.config.min_horizon_m, self.config.horizon_time_s * max(command_speed, 1.0)),
            distance,
        )
        return start + vector / distance * horizon

    def _end_velocity(self, start, target, local_target, command_speed):
        braking_distance = self.config.max_vel_mps * self.config.max_vel_mps / (2.0 * self.config.max_acc_mps2)
        if np.linalg.norm(target - local_target) < braking_distance:
            return np.zeros(2, dtype=float)
        direction = local_target - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=float)
        return direction / norm * min(command_speed, self.config.max_vel_mps)

    def _velocity_from_direction(self, direction, speed, fallback_heading):
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return speed * math.cos(fallback_heading), speed * math.sin(fallback_heading), fallback_heading
        unit = direction / norm
        return speed * unit[0], speed * unit[1], math.atan2(unit[1], unit[0])

    def _frame_axes(self, heading):
        forward_axis = np.array([math.cos(heading), math.sin(heading)], dtype=float)
        lateral_axis = np.array([-math.sin(heading), math.cos(heading)], dtype=float)
        return forward_axis, lateral_axis

    def _forward_lateral_command(self, start, lookahead_point, target_heading, forward_speed, lookahead_time):
        forward_axis, lateral_axis = self._frame_axes(target_heading)
        offset = lookahead_point - start
        lateral_offset = float(np.dot(offset, lateral_axis))
        lateral_speed = self.config.lateral_gain * lateral_offset / max(float(lookahead_time), 1e-3)
        lateral_speed = self._clamp_lateral_speed(lateral_speed)
        velocity = forward_axis * forward_speed + lateral_axis * lateral_speed
        maneuver_mode = "forward_lateral" if abs(lateral_speed) > 1e-6 else "forward"
        return (
            float(velocity[0]),
            float(velocity[1]),
            target_heading,
            float(forward_speed),
            float(lateral_speed),
            maneuver_mode,
        )

    def _clamp_lateral_speed(self, lateral_speed):
        if abs(lateral_speed) <= 1e-6:
            return 0.0
        sign = 1.0 if lateral_speed > 0.0 else -1.0
        magnitude = min(abs(lateral_speed), self.config.max_lateral_speed_mps)
        magnitude = max(magnitude, self.config.min_lateral_speed_mps)
        return sign * magnitude

    def _straight_command(
        self,
        started_at,
        heading,
        command_speed,
        min_clearance,
        reason,
        timing_ms=None,
        rebound_timing_ms=None,
        rebound_stats=None,
    ):
        stage_started = time.perf_counter()
        vx = command_speed * math.cos(heading)
        vy = command_speed * math.sin(heading)
        command_forward = command_speed
        command_lateral = 0.0
        maneuver_mode = "forward"
        if timing_ms is not None:
            self._record_timing(timing_ms, "command_generation", stage_started)

        self.last_debug = {
            "status": reason,
            "raw_obstacles": 0 if math.isinf(min_clearance) else None,
            "astar_segments": 0,
            "optimizer_success": True,
            "refined": False,
            "command": (round(float(vx), 3), round(float(vy), 3)),
            "command_forward": round(command_forward, 3),
            "command_lateral": round(command_lateral, 3),
            "maneuver_mode": maneuver_mode,
        }
        command = self._finish_command(
            started_at,
            vx,
            vy,
            heading,
            blocked=False,
            min_clearance=min_clearance,
            traj_duration=0.0,
            astar_segments=0,
            optimizer_success=True,
            command_speed=command_speed,
            refined=False,
            command_forward=command_forward,
            command_lateral=command_lateral,
            maneuver_mode=maneuver_mode,
            reason=reason,
            timing_ms=timing_ms,
            rebound_timing_ms=rebound_timing_ms,
            rebound_stats=rebound_stats,
        )
        self.last_debug["timing_ms"] = self._rounded_timing(command.timing_ms)
        self.last_debug["rebound_timing_ms"] = self._rounded_rebound_timing(command.rebound_timing_ms)
        self.last_debug["rebound_stats"] = dict(command.rebound_stats)
        return command

    def _blocked_command(
        self,
        started_at,
        yaw,
        grid,
        reason,
        timing_ms=None,
        rebound_timing_ms=None,
        rebound_stats=None,
        min_clearance=0.0,
        astar_segments=0,
    ):
        self.last_debug = {
            "status": "blocked",
            "reason": reason,
            "raw_obstacles": len(grid.raw_points),
            "occupied_cells": len(grid.occupied_cells),
            "astar_segments": astar_segments,
            "optimizer_success": False,
            "refined": False,
            "command_forward": 0.0,
            "command_lateral": 0.0,
            "maneuver_mode": "blocked",
        }
        command = self._finish_command(
            started_at,
            0.0,
            0.0,
            yaw,
            blocked=True,
            min_clearance=min_clearance,
            traj_duration=0.0,
            astar_segments=astar_segments,
            optimizer_success=False,
            command_speed=0.0,
            refined=False,
            command_forward=0.0,
            command_lateral=0.0,
            maneuver_mode="blocked",
            reason=reason,
            timing_ms=timing_ms,
            rebound_timing_ms=rebound_timing_ms,
            rebound_stats=rebound_stats,
        )
        self.last_debug["timing_ms"] = self._rounded_timing(command.timing_ms)
        self.last_debug["rebound_timing_ms"] = self._rounded_rebound_timing(command.rebound_timing_ms)
        self.last_debug["rebound_stats"] = dict(command.rebound_stats)
        return command

    def _finish_command(
        self,
        started_at,
        vx,
        vy,
        yaw,
        blocked,
        min_clearance,
        traj_duration,
        astar_segments,
        optimizer_success,
        command_speed,
        refined,
        command_forward=0.0,
        command_lateral=0.0,
        maneuver_mode="unknown",
        reason="ok",
        timing_ms=None,
        rebound_timing_ms=None,
        rebound_stats=None,
    ):
        plan_time_ms = (time.perf_counter() - started_at) * 1000.0
        if timing_ms is None:
            timing_ms = self._new_timing()
        else:
            timing_ms = {**self._new_timing(), **dict(timing_ms)}
        timing_ms["total"] = plan_time_ms
        if rebound_timing_ms is None:
            rebound_timing_ms = self._new_rebound_timing()
        else:
            rebound_timing_ms = {**self._new_rebound_timing(), **dict(rebound_timing_ms)}
        if rebound_stats is None:
            rebound_stats = self._new_rebound_stats()
        else:
            rebound_stats = {**self._new_rebound_stats(), **dict(rebound_stats)}
        rebound_timing_ms["total"] = rebound_timing_ms.get("total", 0.0)

        return EGOPlanCommand(
            vx=float(vx),
            vy=float(vy),
            yaw=float(yaw),
            blocked=bool(blocked),
            plan_time_ms=plan_time_ms,
            min_clearance=float(min_clearance),
            traj_duration=float(traj_duration),
            astar_segments=int(astar_segments),
            optimizer_success=bool(optimizer_success),
            command_speed=float(command_speed),
            refined=bool(refined),
            command_forward=float(command_forward),
            command_lateral=float(command_lateral),
            maneuver_mode=maneuver_mode,
            timing_ms=timing_ms,
            rebound_timing_ms=rebound_timing_ms,
            rebound_stats=rebound_stats,
            reason=reason,
        )

    def _variable_bounds(self, control_points):
        order = _UniformCubicBspline2D.order
        return order, len(control_points) - order

    def _command_speed(self, target_speed, velocity):
        if target_speed is not None:
            return min(float(target_speed), self.config.max_vel_mps)
        velocity_norm = float(np.linalg.norm(velocity))
        if velocity_norm > 1e-6:
            return min(velocity_norm, self.config.max_vel_mps)
        return self.config.max_vel_mps

    def _xy(self, value):
        return np.array([float(value[0]), float(value[1])], dtype=float)

    def debug_summary(self):
        if not self.last_debug:
            return "none"

        parts = []
        for key, value in self.last_debug.items():
            parts.append(f"{key}={value}")
        return " ".join(parts)
