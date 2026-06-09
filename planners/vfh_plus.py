import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class VFHPlanCommand:
    vx: float
    vy: float
    yaw: float
    selected_heading: Optional[float]
    lateral_speed: float
    command_forward: float
    command_lateral: float
    command_speed: float
    blocked: bool
    reason: str
    maneuver_mode: str


class VFHPlusPlanner:
    def __init__(self):
        self.sector_size = 2.5
        self.num_sectors = int(360 / self.sector_size)
        self.bin_deg = self.sector_size
        self.bin_count = self.num_sectors

        self.range_m = 50.0
        self.min_range_m = 0.2
        self.z_band_m = 1.0
        self.grid_resolution_m = 0.5
        self.r_uav_safe = 2.0

        self.a = 1.0
        self.b = 1.0 / self.range_m
        self.tau_high = 0.2
        self.tau_low = 0.08

        self.m1 = 5.0
        self.m2 = 0.5
        self.m3 = 0.25
        self.smax_sectors = 16

        self.max_lateral_accel_mps2 = 30.0
        self.min_turn_radius_m = 0.5
        self.trajectory_sample_step_m = 0.5
        self.max_lateral_speed_mps = 20.0
        self.min_lateral_speed_mps = 8.0
        self.lateral_gain = 1.4
        self.emergency_lateral_angle_deg = 90.0
        self.emergency_lateral_check_m = 8.0

        self.prev_binary_histogram = [0] * self.num_sectors
        self.prev_heading = 0.0
        self.last_debug = {}

    def plan(self, point_cloud, angle_offset=0.0, speed=0.0):
        current_heading = self.normalize(angle_offset)
        active_cells, grid_lookup, centroid, valid_points = self.build_active_grid(
            point_cloud,
            angle_offset,
        )
        primary_histogram = self.build_primary_histogram(active_cells)
        binary_histogram = self.build_binary_histogram(primary_histogram)
        turn_radius = self.turn_radius(speed)
        masked_histogram = self.build_masked_histogram(
            binary_histogram,
            grid_lookup,
            current_heading,
            turn_radius,
        )

        if valid_points == 0:
            self.prev_heading = 0.0
            self.last_debug = self.build_debug(
                speed,
                turn_radius,
                valid_points,
                len(active_cells),
                centroid,
                binary_histogram,
                masked_histogram,
                [],
                0.0,
            )
            return 0.0

        candidates = self.valley_candidates(masked_histogram)
        if not candidates:
            self.last_debug = self.build_debug(
                speed,
                turn_radius,
                valid_points,
                len(active_cells),
                centroid,
                binary_histogram,
                masked_histogram,
                [],
                None,
            )
            return None

        target_direction = 0.0
        candidate_costs = [
            (candidate, self.cost(candidate, target_direction, current_heading))
            for candidate in candidates
        ]
        selected = min(candidate_costs, key=lambda item: item[1])
        selected_heading = self.normalize(selected[0])
        self.prev_heading = selected_heading
        self.last_debug = self.build_debug(
            speed,
            turn_radius,
            valid_points,
            len(active_cells),
            centroid,
            binary_histogram,
            masked_histogram,
            candidate_costs,
            selected_heading,
        )
        return selected_heading

    def plan_command(self, point_cloud, target_heading=0.0, yaw=0.0, speed=0.0, target_speed=20.0):
        angle_offset = math.degrees(yaw - target_heading)
        selected_heading = self.plan(point_cloud, angle_offset, speed)
        forward_speed = float(target_speed)

        if selected_heading is None:
            selected_heading = self.emergency_lateral_heading(point_cloud, angle_offset)
            if selected_heading is None:
                return VFHPlanCommand(
                    vx=0.0,
                    vy=0.0,
                    yaw=target_heading,
                    selected_heading=None,
                    lateral_speed=0.0,
                    command_forward=0.0,
                    command_lateral=0.0,
                    command_speed=0.0,
                    blocked=True,
                    reason="blocked",
                    maneuver_mode="blocked",
                )
            reason = "emergency_lateral"
        else:
            reason = "ok"

        lateral_speed = self.lateral_speed_from_heading(selected_heading)
        vx, vy = self.world_velocity(target_heading, forward_speed, lateral_speed)
        command_speed = math.hypot(vx, vy)
        maneuver_mode = "forward_lateral" if abs(lateral_speed) > 1e-6 else "forward"
        return VFHPlanCommand(
            vx=vx,
            vy=vy,
            yaw=target_heading,
            selected_heading=selected_heading,
            lateral_speed=lateral_speed,
            command_forward=forward_speed,
            command_lateral=lateral_speed,
            command_speed=command_speed,
            blocked=False,
            reason=reason,
            maneuver_mode=maneuver_mode,
        )

    def lateral_speed_from_heading(self, selected_heading):
        heading = self.normalize(selected_heading)
        if abs(heading) <= self.sector_size / 2.0:
            return 0.0
        raw = self.lateral_gain * math.tan(math.radians(heading)) * self.min_lateral_speed_mps
        if abs(raw) <= 1e-6:
            return 0.0
        sign = 1.0 if raw > 0.0 else -1.0
        magnitude = min(abs(raw), self.max_lateral_speed_mps)
        magnitude = max(magnitude, self.min_lateral_speed_mps)
        return sign * magnitude

    def world_velocity(self, target_heading, forward_speed, lateral_speed):
        forward_x = math.cos(target_heading)
        forward_y = math.sin(target_heading)
        lateral_x = -math.sin(target_heading)
        lateral_y = math.cos(target_heading)
        return (
            forward_speed * forward_x + lateral_speed * lateral_x,
            forward_speed * forward_y + lateral_speed * lateral_y,
        )

    def emergency_lateral_heading(self, point_cloud, angle_offset):
        active_cells, grid_lookup, _, valid_points = self.build_active_grid(point_cloud, angle_offset)
        if valid_points == 0:
            return 0.0

        left_clear = self.straight_segment_is_clear(
            0.0,
            0.0,
            math.radians(self.emergency_lateral_angle_deg),
            self.emergency_lateral_check_m,
            grid_lookup,
        )
        right_clear = self.straight_segment_is_clear(
            0.0,
            0.0,
            math.radians(-self.emergency_lateral_angle_deg),
            self.emergency_lateral_check_m,
            grid_lookup,
        )
        if not left_clear and not right_clear:
            return None
        if left_clear and not right_clear:
            return self.emergency_lateral_angle_deg
        if right_clear and not left_clear:
            return -self.emergency_lateral_angle_deg

        left_density = sum(cell["certainty"] for cell in active_cells if cell["y"] > 0.0)
        right_density = sum(cell["certainty"] for cell in active_cells if cell["y"] < 0.0)
        return self.emergency_lateral_angle_deg if left_density <= right_density else -self.emergency_lateral_angle_deg

    def build_active_grid(self, point_cloud, angle_offset):
        cell_sums = {}
        valid_points = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        offset_rad = math.radians(angle_offset)
        cos_offset = math.cos(offset_rad)
        sin_offset = math.sin(offset_rad)

        for i in range(0, len(point_cloud) - 2, 3):
            x, y, z = point_cloud[i], point_cloud[i + 1], point_cloud[i + 2]
            if abs(z) > self.z_band_m:
                continue

            distance = math.hypot(x, y)
            if distance < self.min_range_m or distance > self.range_m:
                continue

            target_x = x * cos_offset - y * sin_offset
            target_y = x * sin_offset + y * cos_offset
            key = self.grid_key(target_x, target_y)

            if key not in cell_sums:
                cell_sums[key] = [0.0, 0.0, 0.0, 0]
            cell_sums[key][0] += target_x
            cell_sums[key][1] += target_y
            cell_sums[key][2] += z
            cell_sums[key][3] += 1

            valid_points += 1
            sum_x += target_x
            sum_y += target_y
            sum_z += z

        if not cell_sums:
            return [], {}, None, valid_points

        max_count = max(value[3] for value in cell_sums.values())
        active_cells = []
        grid_lookup = {}

        for key, (sx, sy, sz, count) in cell_sums.items():
            x = sx / count
            y = sy / count
            z = sz / count
            distance = math.hypot(x, y)
            certainty = min(1.0, count / max_count)
            cell = {
                "key": key,
                "x": x,
                "y": y,
                "z": z,
                "count": count,
                "certainty": certainty,
                "distance": distance,
                "angle": math.degrees(math.atan2(y, x)),
                "sector": self.angle_to_index(math.degrees(math.atan2(y, x))),
            }
            active_cells.append(cell)
            grid_lookup[key] = cell

        centroid = (
            sum_x / valid_points,
            sum_y / valid_points,
            sum_z / valid_points,
        )

        return active_cells, grid_lookup, centroid, valid_points

    def build_primary_histogram(self, active_cells):
        histogram = [0.0] * self.num_sectors

        for cell in active_cells:
            distance_weight = max(0.0, self.a - self.b * cell["distance"])
            magnitude = (cell["certainty"] ** 2) * distance_weight
            histogram[cell["sector"]] += magnitude

        return histogram

    def build_binary_histogram(self, primary_histogram):
        binary_histogram = [0] * self.num_sectors

        for sector, magnitude in enumerate(primary_histogram):
            if magnitude > self.tau_high:
                binary_histogram[sector] = 1
            elif magnitude < self.tau_low:
                binary_histogram[sector] = 0
            else:
                binary_histogram[sector] = self.prev_binary_histogram[sector]

        self.prev_binary_histogram = binary_histogram[:]
        return binary_histogram

    def build_masked_histogram(
        self,
        binary_histogram,
        grid_lookup,
        current_heading,
        turn_radius,
    ):
        masked_histogram = binary_histogram[:]
        if not grid_lookup:
            return masked_histogram

        for sector, blocked in enumerate(binary_histogram):
            if blocked:
                continue

            direction = self.index_to_angle(sector)
            if not self.trajectory_is_clear(
                direction,
                current_heading,
                turn_radius,
                grid_lookup,
            ):
                masked_histogram[sector] = 1

        return masked_histogram

    def trajectory_is_clear(self, direction, current_heading, turn_radius, grid_lookup):
        delta = self.normalize(direction - current_heading)
        delta_rad = math.radians(delta)
        current_rad = math.radians(current_heading)
        direction_rad = math.radians(direction)

        if abs(delta) <= self.sector_size / 2.0:
            return self.straight_segment_is_clear(
                0.0,
                0.0,
                current_rad,
                self.range_m,
                grid_lookup,
            )

        turn_sign = 1.0 if delta_rad > 0.0 else -1.0
        center_x = -turn_sign * turn_radius * math.sin(current_rad)
        center_y = turn_sign * turn_radius * math.cos(current_rad)
        start_radius_angle = current_rad - turn_sign * math.pi / 2.0
        arc_angle = abs(delta_rad)
        arc_length = turn_radius * arc_angle
        samples = max(2, int(math.ceil(arc_length / self.trajectory_sample_step_m)))

        end_x = 0.0
        end_y = 0.0
        for step in range(1, samples + 1):
            progress = step / samples
            radius_angle = start_radius_angle + turn_sign * arc_angle * progress
            x = center_x + turn_radius * math.cos(radius_angle)
            y = center_y + turn_radius * math.sin(radius_angle)
            if self.point_collides(x, y, grid_lookup):
                return False
            end_x = x
            end_y = y

        remaining = max(0.0, self.range_m - math.hypot(end_x, end_y))
        return self.straight_segment_is_clear(
            end_x,
            end_y,
            direction_rad,
            remaining,
            grid_lookup,
        )

    def straight_segment_is_clear(self, start_x, start_y, heading_rad, length, grid_lookup):
        if length <= 0.0:
            return True

        samples = max(1, int(math.ceil(length / self.trajectory_sample_step_m)))
        for step in range(1, samples + 1):
            distance = length * step / samples
            x = start_x + distance * math.cos(heading_rad)
            y = start_y + distance * math.sin(heading_rad)
            if self.point_collides(x, y, grid_lookup):
                return False

        return True

    def point_collides(self, x, y, grid_lookup):
        radius = self.r_uav_safe + math.sqrt(2.0) * self.grid_resolution_m / 2.0
        radius_cells = int(math.ceil(radius / self.grid_resolution_m))
        center_key = self.grid_key(x, y)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = grid_lookup.get((center_key[0] + dx, center_key[1] + dy))
                if cell is None:
                    continue
                if math.hypot(cell["x"] - x, cell["y"] - y) <= radius:
                    return True

        return False

    def valley_candidates(self, masked_histogram):
        free = [value == 0 for value in masked_histogram]
        candidates = []

        if all(free):
            return [0.0]
        if not any(free):
            return []

        for start, end in self.find_valleys(free):
            candidates.extend(self.candidates_for_valley(start, end))

        return list(dict.fromkeys(self.normalize(candidate) for candidate in candidates))

    def find_valleys(self, free):
        start_from = next(index for index, is_free in enumerate(free) if not is_free)
        valleys = []
        in_valley = False
        start = None

        for step in range(1, self.num_sectors + 1):
            index = (start_from + step) % self.num_sectors
            if free[index] and not in_valley:
                start = index
                in_valley = True
            elif not free[index] and in_valley:
                valleys.append((start, (index - 1) % self.num_sectors))
                in_valley = False

        return valleys

    def candidates_for_valley(self, start, end):
        width = self.valley_width(start, end)
        if width <= self.smax_sectors:
            center = (start + (width - 1) / 2.0) % self.num_sectors
            return [self.index_to_angle(center)]

        half_smax = self.smax_sectors / 2.0
        candidates = [
            self.index_to_angle((start + half_smax) % self.num_sectors),
            self.index_to_angle((end - half_smax) % self.num_sectors),
        ]

        target_sector = self.angle_to_index(0.0)
        if self.index_in_valley(target_sector, start, end):
            candidates.append(0.0)

        return candidates

    def index_in_valley(self, index, start, end):
        if end >= start:
            return start <= index <= end
        return index >= start or index <= end

    def valley_width(self, start, end):
        if end >= start:
            return end - start + 1
        return self.num_sectors - start + end + 1

    def cost(self, candidate, target_direction, current_heading):
        return (
            self.m1 * self.angle_distance(target_direction, candidate)
            + self.m2 * self.angle_distance(current_heading, candidate)
            + self.m3 * self.angle_distance(self.prev_heading, candidate)
        )

    def turn_radius(self, speed):
        speed = max(0.0, speed)
        if self.max_lateral_accel_mps2 <= 0.0:
            return self.min_turn_radius_m
        return max(
            self.min_turn_radius_m,
            speed * speed / self.max_lateral_accel_mps2,
        )

    def grid_key(self, x, y):
        return (
            math.floor(x / self.grid_resolution_m),
            math.floor(y / self.grid_resolution_m),
        )

    def angle_distance(self, a, b):
        return abs(self.normalize(a - b))

    def angle_delta(self, a, b):
        return self.normalize(a - b)

    def normalize(self, angle):
        return (angle + 180.0) % 360.0 - 180.0

    def angle_to_index(self, angle):
        return int(((self.normalize(angle) + 180.0) % 360.0) / self.sector_size)

    def index_to_angle(self, index):
        return self.normalize(index * self.sector_size - 180.0 + self.sector_size / 2.0)

    def build_debug(
        self,
        speed,
        turn_radius,
        valid_points,
        active_cell_count,
        centroid,
        binary_histogram,
        masked_histogram,
        candidate_costs,
        selected,
    ):
        return {
            "sector_size": self.sector_size,
            "speed": round(speed, 3),
            "turn_radius": round(turn_radius, 3),
            "valid_points": valid_points,
            "active_cells": active_cell_count,
            "centroid": self.centroid_debug(centroid),
            "binary_blocked": self.histogram_debug(binary_histogram),
            "masked_blocked": self.histogram_debug(masked_histogram),
            "candidates": [
                {
                    "angle": round(self.normalize(candidate), 2),
                    "cost": round(cost, 2),
                }
                for candidate, cost in candidate_costs
            ],
            "selected": None if selected is None else round(selected, 2),
        }

    def centroid_debug(self, centroid):
        if centroid is None:
            return None

        x, y, z = centroid
        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "z": round(z, 3),
            "distance": round(math.hypot(x, y), 3),
            "angle": round(self.normalize(math.degrees(math.atan2(y, x))), 2),
        }

    def histogram_debug(self, histogram):
        blocked = [bool(value) for value in histogram]
        return {
            "count": sum(blocked),
            "ranges": self.sector_ranges(blocked),
        }

    def sector_ranges(self, sectors):
        if not any(sectors):
            return []

        if all(sectors):
            return [
                {
                    "start": round(self.index_to_angle(0), 2),
                    "end": round(self.index_to_angle(self.num_sectors - 1), 2),
                    "count": self.num_sectors,
                },
            ]

        start_from = next(index for index, blocked in enumerate(sectors) if not blocked)
        ranges = []
        in_range = False
        start = None

        for step in range(1, self.num_sectors + 1):
            index = (start_from + step) % self.num_sectors
            if sectors[index] and not in_range:
                start = index
                in_range = True
            elif not sectors[index] and in_range:
                end = (index - 1) % self.num_sectors
                ranges.append(
                    {
                        "start": round(self.index_to_angle(start), 2),
                        "end": round(self.index_to_angle(end), 2),
                        "count": self.valley_width(start, end),
                    },
                )
                in_range = False

        return ranges

    def debug_summary(self):
        if not self.last_debug:
            return "none"

        centroid = self.last_debug["centroid"]
        if centroid is None:
            centroid_text = "none"
        else:
            centroid_text = (
                "x={x:.3f},y={y:.3f},z={z:.3f},d={distance:.3f},angle={angle:.2f}"
            ).format(**centroid)

        candidates = ",".join(
            f"{candidate['angle']:.2f}:{candidate['cost']:.2f}"
            for candidate in self.last_debug["candidates"]
        )
        if not candidates:
            candidates = "none"

        return (
            "sector={sector_size:.1f} speed={speed:.2f} turn_radius={turn_radius:.2f} "
            "points={valid_points} cells={active_cells} centroid=({centroid}) "
            "binary={binary} masked={masked} candidates=[{candidates}] selected={selected}"
        ).format(
            sector_size=self.last_debug["sector_size"],
            speed=self.last_debug["speed"],
            turn_radius=self.last_debug["turn_radius"],
            valid_points=self.last_debug["valid_points"],
            active_cells=self.last_debug["active_cells"],
            centroid=centroid_text,
            binary=self.last_debug["binary_blocked"],
            masked=self.last_debug["masked_blocked"],
            candidates=candidates,
            selected=self.last_debug["selected"],
        )
