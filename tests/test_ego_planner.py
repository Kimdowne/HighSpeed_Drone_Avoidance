import math
import unittest

import numpy as np

from planners.ego_planner import EGOPlanner
from planners.vfh_plus import VFHPlusPlanner


TIMING_KEYS = {
    "grid",
    "init_bspline",
    "anchor_astar",
    "rebound_optimize",
    "refine",
    "collision_check",
    "command_generation",
    "total",
}

REBOUND_TIMING_KEYS = {
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
}


def _reference_smoothness_cost(q):
    grad = np.zeros_like(q)
    cost = 0.0
    for i in range(len(q) - 3):
        jerk = q[i + 3] - 3.0 * q[i + 2] + 3.0 * q[i + 1] - q[i]
        cost += float(np.dot(jerk, jerk))
        temp = 2.0 * jerk
        grad[i] += -temp
        grad[i + 1] += 3.0 * temp
        grad[i + 2] += -3.0 * temp
        grad[i + 3] += temp
    return cost, grad


def _reference_feasibility_cost(planner, q, dt):
    grad = np.zeros_like(q)
    cost = 0.0
    dt = max(dt, 1e-6)
    dt_inv2 = 1.0 / (dt * dt)

    for i in range(len(q) - 1):
        velocity = (q[i + 1] - q[i]) / dt
        for dim in range(2):
            if velocity[dim] > planner.config.max_vel_mps:
                diff = velocity[dim] - planner.config.max_vel_mps
            elif velocity[dim] < -planner.config.max_vel_mps:
                diff = velocity[dim] + planner.config.max_vel_mps
            else:
                continue
            cost += diff * diff * dt_inv2
            value = 2.0 * diff / dt * dt_inv2
            grad[i, dim] += -value
            grad[i + 1, dim] += value

    for i in range(len(q) - 2):
        acceleration = (q[i + 2] - 2.0 * q[i + 1] + q[i]) * dt_inv2
        for dim in range(2):
            if acceleration[dim] > planner.config.max_acc_mps2:
                diff = acceleration[dim] - planner.config.max_acc_mps2
            elif acceleration[dim] < -planner.config.max_acc_mps2:
                diff = acceleration[dim] + planner.config.max_acc_mps2
            else:
                continue
            cost += diff * diff
            value = 2.0 * diff * dt_inv2
            grad[i, dim] += value
            grad[i + 1, dim] += -2.0 * value
            grad[i + 2, dim] += value

    return cost, grad


class EGOPlannerTest(unittest.TestCase):
    def test_vectorized_smoothness_matches_scalar_reference(self):
        planner = EGOPlanner()
        rng = np.random.default_rng(7)

        for point_count in (3, 4, 8, 17):
            control_points = rng.normal(size=(point_count, 2)) * 12.0
            expected_cost, expected_grad = _reference_smoothness_cost(control_points)
            actual_cost, actual_grad = planner._smoothness_cost(control_points)

            self.assertAlmostEqual(actual_cost, expected_cost)
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_vectorized_feasibility_matches_scalar_reference(self):
        planner = EGOPlanner()
        rng = np.random.default_rng(11)
        cases = [
            np.zeros((6, 2), dtype=float),
            np.array([[0.0, 0.0], [8.0, 0.0], [16.0, 0.0], [24.0, 0.0]], dtype=float),
            np.array([[0.0, 0.0], [40.0, 0.0], [80.0, 0.0], [120.0, 0.0]], dtype=float),
            np.array([[0.0, 0.0], [0.0, 0.0], [35.0, -35.0], [-35.0, 35.0]], dtype=float),
            rng.normal(size=(9, 2)) * 25.0,
        ]

        for control_points in cases:
            expected_cost, expected_grad = _reference_feasibility_cost(planner, control_points, 0.2)
            actual_cost, actual_grad = planner._feasibility_cost(control_points, 0.2)

            self.assertAlmostEqual(actual_cost, expected_cost)
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_empty_cloud_goes_straight_to_target(self):
        planner = EGOPlanner()

        command = planner.plan(
            point_cloud=[],
            position=(0.0, 0.0),
            velocity=(20.0, 0.0),
            yaw=0.0,
            target_position=(100.0, 0.0),
            target_speed=20.0,
        )

        self.assertFalse(command.blocked)
        self.assertAlmostEqual(command.vx, 20.0, places=6)
        self.assertAlmostEqual(command.vy, 0.0, places=6)
        self.assertAlmostEqual(command.yaw, 0.0, places=6)
        self.assertAlmostEqual(command.command_forward, 20.0, places=6)
        self.assertAlmostEqual(command.command_lateral, 0.0, places=6)
        self.assertEqual(command.maneuver_mode, "forward")
        self.assertTrue(command.optimizer_success)
        self.assertEqual(set(command.timing_ms), TIMING_KEYS)
        self.assertGreaterEqual(command.timing_ms["grid"], 0.0)
        self.assertGreater(command.timing_ms["total"], 0.0)
        self.assertEqual(command.timing_ms["anchor_astar"], 0.0)
        self.assertEqual(command.timing_ms["rebound_optimize"], 0.0)
        self.assertEqual(set(command.rebound_timing_ms), REBOUND_TIMING_KEYS)
        self.assertEqual(command.rebound_timing_ms["total"], 0.0)
        self.assertEqual(command.rebound_stats["attempts"], 0)
        self.assertEqual(command.rebound_stats["objective_calls"], 0)

    def test_center_obstacle_generates_lateral_command(self):
        planner = EGOPlanner()
        cloud = []
        for x in (19.5, 20.0, 20.5):
            for y in (-0.5, 0.0, 0.5):
                cloud.extend([x, y, 0.0])

        command = planner.plan(
            point_cloud=cloud,
            position=(0.0, 0.0),
            velocity=(20.0, 0.0),
            yaw=0.0,
            target_position=(100.0, 0.0),
            target_speed=20.0,
        )

        self.assertFalse(command.blocked)
        self.assertGreater(command.astar_segments, 0)
        self.assertTrue(command.optimizer_success)
        self.assertAlmostEqual(command.command_forward, 20.0, places=6)
        self.assertGreaterEqual(abs(command.command_lateral), 8.0)
        self.assertAlmostEqual(command.yaw, 0.0, places=6)
        self.assertEqual(command.maneuver_mode, "forward_lateral")
        self.assertEqual(set(command.timing_ms), TIMING_KEYS)
        self.assertGreater(command.timing_ms["anchor_astar"], 0.0)
        self.assertGreater(command.timing_ms["rebound_optimize"], 0.0)
        self.assertEqual(set(command.rebound_timing_ms), REBOUND_TIMING_KEYS)
        self.assertGreater(command.rebound_timing_ms["total"], 0.0)
        self.assertGreater(command.rebound_timing_ms["solver_wall"], 0.0)
        self.assertGreater(command.rebound_timing_ms["objective_total"], 0.0)
        self.assertGreater(command.rebound_stats["attempts"], 0)
        self.assertGreater(command.rebound_stats["objective_calls"], 0)

    def test_center_obstacle_lateral_command_rotates_with_target_heading(self):
        planner = EGOPlanner()
        cloud = []
        for x in (19.5, 20.0, 20.5):
            for y in (-0.5, 0.0, 0.5):
                cloud.extend([x, y, 0.0])

        command = planner.plan(
            point_cloud=cloud,
            position=(0.0, 0.0),
            velocity=(0.0, 20.0),
            yaw=math.pi / 2.0,
            target_position=(0.0, 100.0),
            target_speed=20.0,
        )

        forward_axis = np.array([0.0, 1.0])
        lateral_axis = np.array([-1.0, 0.0])
        velocity = np.array([command.vx, command.vy])

        self.assertFalse(command.blocked)
        self.assertAlmostEqual(float(np.dot(velocity, forward_axis)), 20.0, places=6)
        self.assertAlmostEqual(float(np.dot(velocity, lateral_axis)), command.command_lateral, places=6)
        self.assertGreaterEqual(abs(command.command_lateral), 8.0)
        self.assertAlmostEqual(command.yaw, math.pi / 2.0, places=6)

    def test_rebound_anchor_uses_obstacle_surface_base_point(self):
        planner = EGOPlanner()
        grid = planner._build_grid(
            point_cloud=[10.0, 0.0, 0.0],
            position=np.array([0.0, 0.0]),
            yaw=0.0,
        )
        control_point = np.array([10.0, 0.0])
        guide_point = np.array([15.0, 0.0])

        base_point, direction = planner._surface_anchor(control_point, guide_point, grid)

        self.assertTrue(grid.is_occupied_point(control_point))
        self.assertGreater(base_point[0], control_point[0])
        self.assertGreater(direction[0], 0.0)

    def test_infeasible_control_points_trigger_time_refine(self):
        planner = EGOPlanner()
        control_points = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [40.0, 0.0],
                [80.0, 0.0],
                [100.0, 0.0],
                [120.0, 0.0],
                [140.0, 0.0],
            ],
            dtype=float,
        )

        ratio_before = planner._feasibility_ratio(control_points, 0.1)
        refined_points, refined_dt, refined = planner._refine_if_needed(control_points, 0.1)

        self.assertGreater(ratio_before, 1.0)
        self.assertTrue(refined)
        self.assertGreater(refined_dt, 0.1)
        self.assertLessEqual(planner._feasibility_ratio(refined_points, refined_dt), 1.05)

    def test_surrounded_start_blocks_command(self):
        planner = EGOPlanner()
        cloud = []
        for index in range(24):
            angle = 2.0 * math.pi * index / 24.0
            cloud.extend([math.cos(angle), math.sin(angle), 0.0])

        command = planner.plan(
            point_cloud=cloud,
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
            yaw=0.0,
            target_position=(100.0, 0.0),
            target_speed=20.0,
        )

        self.assertTrue(command.blocked)
        self.assertAlmostEqual(command.vx, 0.0, places=6)
        self.assertAlmostEqual(command.vy, 0.0, places=6)
        self.assertAlmostEqual(command.command_forward, 0.0, places=6)
        self.assertAlmostEqual(command.command_lateral, 0.0, places=6)
        self.assertEqual(command.maneuver_mode, "blocked")
        self.assertFalse(command.optimizer_success)
        self.assertEqual(set(command.timing_ms), TIMING_KEYS)
        self.assertEqual(set(command.rebound_timing_ms), REBOUND_TIMING_KEYS)
        self.assertAlmostEqual(command.timing_ms["total"], command.plan_time_ms)


class VFHPlusPlannerTest(unittest.TestCase):
    def test_vfh_no_obstacle_goes_forward_without_yaw_turn(self):
        planner = VFHPlusPlanner()

        command = planner.plan_command(
            point_cloud=[],
            target_heading=math.pi / 6.0,
            yaw=math.pi / 6.0,
            speed=20.0,
            target_speed=20.0,
        )

        self.assertFalse(command.blocked)
        self.assertAlmostEqual(command.command_forward, 20.0, places=6)
        self.assertAlmostEqual(command.command_lateral, 0.0, places=6)
        self.assertAlmostEqual(command.yaw, math.pi / 6.0, places=6)
        self.assertEqual(command.maneuver_mode, "forward")

    def test_vfh_center_obstacle_uses_lateral_speed(self):
        planner = VFHPlusPlanner()
        cloud = []
        for x in (19.5, 20.0, 20.5):
            for y in (-0.5, 0.0, 0.5):
                cloud.extend([x, y, 0.0])

        command = planner.plan_command(
            point_cloud=cloud,
            target_heading=0.0,
            yaw=0.0,
            speed=20.0,
            target_speed=20.0,
        )

        self.assertFalse(command.blocked)
        self.assertAlmostEqual(command.command_forward, 20.0, places=6)
        self.assertGreaterEqual(abs(command.command_lateral), 8.0)
        self.assertAlmostEqual(command.yaw, 0.0, places=6)
        self.assertEqual(command.maneuver_mode, "forward_lateral")

    def test_vfh_surrounded_blocks_when_no_lateral_escape_exists(self):
        planner = VFHPlusPlanner()
        cloud = []
        for index in range(36):
            angle = 2.0 * math.pi * index / 36.0
            cloud.extend([2.0 * math.cos(angle), 2.0 * math.sin(angle), 0.0])

        command = planner.plan_command(
            point_cloud=cloud,
            target_heading=0.0,
            yaw=0.0,
            speed=20.0,
            target_speed=20.0,
        )

        self.assertTrue(command.blocked)
        self.assertAlmostEqual(command.vx, 0.0, places=6)
        self.assertAlmostEqual(command.vy, 0.0, places=6)
        self.assertEqual(command.maneuver_mode, "blocked")


if __name__ == "__main__":
    unittest.main()
