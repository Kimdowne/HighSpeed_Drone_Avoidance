import logging
import math
import time

import airsim

from planners.ego_planner import EGOPlanner
from planners.vfh_plus import VFHPlusPlanner


# 전체 로직:
# 1. Drone1을 이륙시켜 500m 전방 목표점을 향해 가속한다.
# 2. 목표 속도 조건이 유지되면 지정 거리 앞에 StaticObstacle을 배치한다.
# 3. 장애물 생성 후 선택된 planner(VFH+ 또는 EGO-Planner)가 라이다 점군을 해석한다.
# 4. 충돌하면 실패, 지정 시간 생존하면 성공으로 기록하고 같은 실험을 반복한다.

PLANNER_TYPES = ("ego", "vfh")  # 각 planner를 OBSTACLE_DISTANCES마다 순서대로 실행
DRONE = "Drone1"  # 조종할 메인 드론 이름
OBSTACLE = "StaticObstacle"  # 충돌 대상으로 배치할 장애물 차량 이름
LIDAR = "Lidar2D"  # Drone1에 장착된 라이다 센서 이름

ALTITUDE_Z = -50.0  # 드론이 유지할 NED z 좌표. -50은 고도 50m
COMMAND_SPEED = 40.0  # 실제 목표 속도 도달을 위해 넣는 X+ 방향 속도 명령값
TARGET_SPEED = 20.0  # 회피 중 유지할 속도이자 장애물을 배치할 실제 전진 속도 기준값
TARGET_DISTANCE = 500.0  # 시작 위치 기준 전방 목표 지점 거리
SPAWN_DELAY_SEC = 0.1  # 목표 속도 도달 후 장애물 배치까지 기다릴 시간
OBSTACLE_DISTANCES = (35.0, 36.0, 36.0, 36.5, 37.0)  # 차례로 테스트할 장애물 배치 거리
TRIALS = 10 # 각 planner와 장애물 거리 조합마다 반복할 실험 횟수
SUCCESS_SEC = 10.0  # 장애물 생성 후 이 시간 동안 생존하면 성공

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airsim_loop")

for planner_type in PLANNER_TYPES:
    if planner_type not in ("ego", "vfh"):
        raise ValueError(f"Unsupported PLANNER_TYPE: {planner_type}")

client: airsim.MultirotorClient = airsim.MultirotorClient()
client.confirmConnection()

success_count = {planner_type: 0 for planner_type in PLANNER_TYPES}
fail_count = {planner_type: 0 for planner_type in PLANNER_TYPES}
experiment_results = {
    planner_type: {
        distance_index: {"distance": distance, "results": []}
        for distance_index, distance in enumerate(OBSTACLE_DISTANCES, start=1)
    }
    for planner_type in PLANNER_TYPES
}

for distance_index, OBSTACLE_DISTANCE in enumerate(OBSTACLE_DISTANCES, start=1):
    logger.info(
        "starting distance_index=%d distance=%.1fm planners=%s trials_per_planner=%d",
        distance_index,
        OBSTACLE_DISTANCE,
        PLANNER_TYPES,
        TRIALS,
    )
    distance_success_count = {planner_type: 0 for planner_type in PLANNER_TYPES}
    distance_fail_count = {planner_type: 0 for planner_type in PLANNER_TYPES}

    for run_index in range(1, TRIALS * len(PLANNER_TYPES) + 1):
        planner_type = PLANNER_TYPES[(run_index - 1) // TRIALS]
        trial = (run_index - 1) % TRIALS + 1
        planner = EGOPlanner() if planner_type == "ego" else VFHPlusPlanner()
        client.reset()

        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(0, -1000, ALTITUDE_Z), airsim.Quaternionr()),
            True,
            vehicle_name=OBSTACLE,
        )

        client.enableApiControl(True, vehicle_name=DRONE)
        client.armDisarm(True, vehicle_name=DRONE)
        client.takeoffAsync(timeout_sec=20, vehicle_name=DRONE).join()
        client.moveToZAsync(ALTITUDE_Z, 10.0, vehicle_name=DRONE).join()

        start_position = client.simGetVehiclePose(vehicle_name=DRONE).position
        target_x = start_position.x_val + TARGET_DISTANCE
        target_y = start_position.y_val

        run_started = time.monotonic()
        target_speed_reached_at = None
        obstacle_placed_at = None
        obstacle_placed = False
        last_lidar_seen_at = None
        last_vfh_heading = 0.0
        last_vfh_command = None
        last_ego_command = None

        while True:
            now = time.monotonic()
            elapsed = now - run_started
            state = client.getMultirotorState(vehicle_name=DRONE)
            position = state.kinematics_estimated.position
            velocity = state.kinematics_estimated.linear_velocity

            target_dx = target_x - position.x_val
            target_dy = target_y - position.y_val
            target_distance = math.hypot(target_dx, target_dy)
            target_heading = math.atan2(target_dy, target_dx)

            speed = math.sqrt(velocity.x_val**2 + velocity.y_val**2 + velocity.z_val**2)
            horizontal_speed = math.hypot(velocity.x_val, velocity.y_val)
            yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
            target_forward_speed = (
                velocity.x_val * math.cos(target_heading)
                + velocity.y_val * math.sin(target_heading)
            )
            altitude = -position.z_val
            speed_hold = 0.0
            event = "-"

            lidar_data = client.getLidarData(lidar_name=LIDAR, vehicle_name=DRONE)
            point_count = len(lidar_data.point_cloud) // 3
            vfh_heading = 0.0
            lidar_active = False
            planner_debug_log = "-"
            plan_time_ms = 0.0
            min_clearance = float("nan")
            traj_duration = 0.0
            astar_segments = 0
            optimizer_success = True
            refined = False
            ego_command = None
            ego_planner_invoked = False
            vfh_command = None
            vfh_planner_invoked = False
            command_forward = 0.0
            command_lateral = 0.0
            maneuver_mode = "-"

            if planner_type == "vfh":
                if obstacle_placed and point_count > 0:
                    last_lidar_seen_at = now
                    plan_started = time.perf_counter()
                    vfh_command = planner.plan_command(
                        lidar_data.point_cloud,
                        target_heading,
                        yaw,
                        horizontal_speed,
                        TARGET_SPEED,
                    )
                    plan_time_ms = (time.perf_counter() - plan_started) * 1000.0
                    last_vfh_command = vfh_command
                    vfh_planner_invoked = True
                    vfh_heading = vfh_command.selected_heading
                    if vfh_heading is not None:
                        last_vfh_heading = vfh_heading
                    lidar_active = True

                elif (
                    obstacle_placed
                    and last_lidar_seen_at is not None
                    and now - last_lidar_seen_at <= 0.5
                    and last_vfh_command is not None
                ):
                    vfh_command = last_vfh_command
                    vfh_heading = vfh_command.selected_heading
                    lidar_active = True
                elif obstacle_placed:
                    planner.prev_heading = 0.0
                    vfh_command = None
                else:
                    vfh_command = None

                if vfh_command is not None and vfh_command.blocked:
                    vx = 0.0
                    vy = 0.0
                    command_yaw = target_heading
                    command_forward = 0.0
                    command_lateral = 0.0
                    maneuver_mode = vfh_command.maneuver_mode
                    event = "blocked"
                elif vfh_command is not None and lidar_active:
                    vx = vfh_command.vx
                    vy = vfh_command.vy
                    command_yaw = vfh_command.yaw
                    command_forward = vfh_command.command_forward
                    command_lateral = vfh_command.command_lateral
                    maneuver_mode = vfh_command.maneuver_mode
                else:
                    vx = COMMAND_SPEED * math.cos(target_heading)
                    vy = COMMAND_SPEED * math.sin(target_heading)
                    command_yaw = target_heading
                    command_forward = COMMAND_SPEED
                    command_lateral = 0.0
                    maneuver_mode = "forward"

                vfh_heading_log = f"{vfh_heading:.1f}" if vfh_heading is not None else "blocked"
                planner_debug_log = planner.debug_summary() if lidar_active else "-"

            else:
                vfh_heading_log = "-"
                if obstacle_placed:
                    if (
                        point_count == 0
                        and last_ego_command is not None
                        and last_lidar_seen_at is not None
                        and now - last_lidar_seen_at <= 0.5
                    ):
                        ego_command = last_ego_command
                        planner_debug_log = "cached_last_ego_command"
                    else:
                        ego_command = planner.plan(
                            lidar_data.point_cloud,
                            (position.x_val, position.y_val),
                            (velocity.x_val, velocity.y_val),
                            yaw,
                            (target_x, target_y),
                            TARGET_SPEED,
                        )
                        last_ego_command = ego_command
                        ego_planner_invoked = True

                    if point_count > 0:
                        last_lidar_seen_at = now

                    plan_time_ms = ego_command.plan_time_ms
                    min_clearance = ego_command.min_clearance
                    traj_duration = ego_command.traj_duration
                    astar_segments = ego_command.astar_segments
                    optimizer_success = ego_command.optimizer_success
                    refined = ego_command.refined
                    if planner_debug_log == "-":
                        planner_debug_log = planner.debug_summary()

                    if ego_command.blocked:
                        vx = 0.0
                        vy = 0.0
                        command_yaw = target_heading
                        command_forward = ego_command.command_forward
                        command_lateral = ego_command.command_lateral
                        maneuver_mode = ego_command.maneuver_mode
                        event = "blocked"
                    else:
                        vx = ego_command.vx
                        vy = ego_command.vy
                        command_yaw = ego_command.yaw
                        command_forward = ego_command.command_forward
                        command_lateral = ego_command.command_lateral
                        maneuver_mode = ego_command.maneuver_mode
                elif last_ego_command is not None:
                    vx = last_ego_command.vx
                    vy = last_ego_command.vy
                    command_yaw = last_ego_command.yaw
                    command_forward = last_ego_command.command_forward
                    command_lateral = last_ego_command.command_lateral
                    maneuver_mode = last_ego_command.maneuver_mode
                else:
                    vx = COMMAND_SPEED * math.cos(target_heading)
                    vy = COMMAND_SPEED * math.sin(target_heading)
                    command_yaw = target_heading
                    command_forward = COMMAND_SPEED
                    command_lateral = 0.0
                    maneuver_mode = "forward"

            command_speed = math.hypot(vx, vy)
            yaw_error_deg = math.degrees(
                math.atan2(
                    math.sin(command_yaw - target_heading),
                    math.cos(command_yaw - target_heading),
                )
            )

            client.moveByVelocityZAsync(
                vx,
                vy,
                ALTITUDE_Z,
                0.05,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, math.degrees(command_yaw)),
                vehicle_name=DRONE,
            )

            if target_forward_speed >= TARGET_SPEED:
                if target_speed_reached_at is None:
                    target_speed_reached_at = now
                speed_hold = now - target_speed_reached_at
            else:
                target_speed_reached_at = None

            if not obstacle_placed and speed_hold >= SPAWN_DELAY_SEC:
                obstacle_position = airsim.Vector3r(
                    position.x_val + OBSTACLE_DISTANCE - 99.0,
                    position.y_val - 99.0,
                    0.0,
                )

                client.simSetVehiclePose(
                    airsim.Pose(obstacle_position, airsim.Quaternionr()),
                    True,
                    vehicle_name=OBSTACLE,
                )

                obstacle_placed = True
                obstacle_placed_at = now
                event = "placed"

            collision = client.simGetCollisionInfo(vehicle_name=DRONE)
            collided = obstacle_placed and collision.has_collided

            if collided:
                fail_count[planner_type] += 1
                distance_fail_count[planner_type] += 1
                experiment_results[planner_type][distance_index]["results"].append(False)
                event = "fail"

            if obstacle_placed_at is not None and now - obstacle_placed_at >= SUCCESS_SEC and not collided:
                success_count[planner_type] += 1
                distance_success_count[planner_type] += 1
                experiment_results[planner_type][distance_index]["results"].append(True)
                event = "success"

            if planner_type == "ego" and ego_planner_invoked and ego_command is not None:
                timing = ego_command.timing_ms
                timed_stages = {
                    key: value
                    for key, value in timing.items()
                    if key != "total"
                }
                max_stage = max(timed_stages, key=timed_stages.get)
                max_stage_ms = timed_stages[max_stage]
                logger.info(
                    "EGO_TIMING planner=ego distance=%.1fm trial=%d/%d t=%.2fs lidar_points=%d result=%s reason=%s "
                    "grid_ms=%.3f init_bspline_ms=%.3f anchor_astar_ms=%.3f rebound_optimize_ms=%.3f refine_ms=%.3f "
                    "collision_check_ms=%.3f command_generation_ms=%.3f total_ms=%.3f max_stage=%s max_stage_ms=%.3f "
                    "astar_segments=%d optimizer_success=%s refined=%s min_clearance=%.3f traj_duration=%.3f "
                    "command_forward=%.2f command_lateral=%.2f command_speed=%.2f yaw_error_deg=%.2f maneuver_mode=%s",
                    OBSTACLE_DISTANCE,
                    trial,
                    TRIALS,
                    elapsed,
                    point_count,
                    event,
                    ego_command.reason,
                    timing["grid"],
                    timing["init_bspline"],
                    timing["anchor_astar"],
                    timing["rebound_optimize"],
                    timing["refine"],
                    timing["collision_check"],
                    timing["command_generation"],
                    timing["total"],
                    max_stage,
                    max_stage_ms,
                    astar_segments,
                    optimizer_success,
                    refined,
                    min_clearance,
                    traj_duration,
                    command_forward,
                    command_lateral,
                    command_speed,
                    yaw_error_deg,
                    maneuver_mode,
                )

                rebound_timing = ego_command.rebound_timing_ms
                rebound_stats = ego_command.rebound_stats
                if rebound_stats.get("attempts", 0) > 0:
                    rebound_inner_stages = {
                        "rebuild": rebound_timing.get("objective_rebuild", 0.0),
                        "smooth": rebound_timing.get("objective_smooth", 0.0),
                        "collision": rebound_timing.get("objective_collision", 0.0),
                        "feasibility": rebound_timing.get("objective_feasibility", 0.0),
                        "assembly": rebound_timing.get("objective_assembly", 0.0),
                        "optimizer_overhead": rebound_timing.get("optimizer_overhead", 0.0),
                    }
                    max_rebound_stage = max(rebound_inner_stages, key=rebound_inner_stages.get)
                    max_rebound_stage_ms = rebound_inner_stages[max_rebound_stage]
                    logger.info(
                        "EGO_REBOUND_TIMING planner=ego distance=%.1fm trial=%d/%d t=%.2fs lidar_points=%d result=%s reason=%s "
                        "setup_ms=%.3f solver_wall_ms=%.3f objective_total_ms=%.3f rebuild_ms=%.3f smooth_ms=%.3f "
                        "collision_ms=%.3f feasibility_ms=%.3f assembly_ms=%.3f optimizer_overhead_ms=%.3f "
                        "result_apply_ms=%.3f total_ms=%.3f max_inner_stage=%s max_inner_stage_ms=%.3f "
                        "attempts=%d objective_calls=%d lbfgs_iterations=%d function_evals=%d gradient_evals=%d "
                        "solver=%s optimizer_success=%s",
                        OBSTACLE_DISTANCE,
                        trial,
                        TRIALS,
                        elapsed,
                        point_count,
                        event,
                        ego_command.reason,
                        rebound_timing.get("setup", 0.0),
                        rebound_timing.get("solver_wall", 0.0),
                        rebound_timing.get("objective_total", 0.0),
                        rebound_timing.get("objective_rebuild", 0.0),
                        rebound_timing.get("objective_smooth", 0.0),
                        rebound_timing.get("objective_collision", 0.0),
                        rebound_timing.get("objective_feasibility", 0.0),
                        rebound_timing.get("objective_assembly", 0.0),
                        rebound_timing.get("optimizer_overhead", 0.0),
                        rebound_timing.get("result_apply", 0.0),
                        rebound_timing.get("total", 0.0),
                        max_rebound_stage,
                        max_rebound_stage_ms,
                        rebound_stats.get("attempts", 0),
                        rebound_stats.get("objective_calls", 0),
                        rebound_stats.get("lbfgs_iterations", 0),
                        rebound_stats.get("function_evals", 0),
                        rebound_stats.get("gradient_evals", 0),
                        rebound_stats.get("solver", "none"),
                        rebound_stats.get("success", False),
                    )

            if planner_type == "vfh" and vfh_planner_invoked and vfh_command is not None:
                logger.info(
                    "VFH_TIMING planner=vfh distance=%.1fm trial=%d/%d t=%.2fs lidar_points=%d result=%s reason=%s "
                    "plan_time_ms=%.3f selected_heading=%s command_forward=%.2f command_lateral=%.2f "
                    "command_speed=%.2f yaw_error_deg=%.2f maneuver_mode=%s blocked=%s",
                    OBSTACLE_DISTANCE,
                    trial,
                    TRIALS,
                    elapsed,
                    point_count,
                    event,
                    vfh_command.reason,
                    plan_time_ms,
                    "-" if vfh_command.selected_heading is None else f"{vfh_command.selected_heading:.2f}",
                    command_forward,
                    command_lateral,
                    command_speed,
                    yaw_error_deg,
                    maneuver_mode,
                    vfh_command.blocked,
                )

            logger.info(
                "planner=%s distance=%.1fm trial=%d/%d t=%.2fs target_dist=%.1fm speed=%.2fm/s "
                "target_forward=%.2fm/s target_hold=%.2fs alt=%.2fm z=%.2f yaw=%.1f target_yaw=%.1f "
                "yaw_error=%.1f vfh_heading=%s lidar_points=%d plan_time_ms=%.3f min_clearance=%.3f "
                "traj_duration=%.3f astar_segments=%d optimizer_success=%s refined=%s command_forward=%.2f "
                "command_lateral=%.2f command_speed=%.2f maneuver_mode=%s planner_debug=%s obstacle=%s "
                "collision=%s result=%s distance_success=%d distance_fail=%d total_success=%d total_fail=%d",
                planner_type,
                OBSTACLE_DISTANCE,
                trial,
                TRIALS,
                elapsed,
                target_distance,
                speed,
                target_forward_speed,
                speed_hold,
                altitude,
                position.z_val,
                math.degrees(yaw),
                math.degrees(target_heading),
                yaw_error_deg,
                vfh_heading_log,
                point_count,
                plan_time_ms,
                min_clearance,
                traj_duration,
                astar_segments,
                optimizer_success,
                refined,
                command_forward,
                command_lateral,
                command_speed,
                maneuver_mode,
                planner_debug_log,
                obstacle_placed,
                collision.object_name if collided else False,
                event,
                distance_success_count[planner_type],
                distance_fail_count[planner_type],
                success_count[planner_type],
                fail_count[planner_type],
            )

            if event in ("success", "fail"):
                client.cancelLastTask(vehicle_name=DRONE)
                break

            time.sleep(0.05)

    for planner_type in PLANNER_TYPES:
        logger.info(
            "planner=%s distance_index=%d distance %.1fm complete: success=%d fail=%d trials=%d",
            planner_type,
            distance_index,
            OBSTACLE_DISTANCE,
            distance_success_count[planner_type],
            distance_fail_count[planner_type],
            TRIALS,
        )

for planner_type in PLANNER_TYPES:
    logger.info(
        "%s experiment complete: success=%d fail=%d trials=%d distances=%s",
        planner_type,
        success_count[planner_type],
        fail_count[planner_type],
        TRIALS * len(OBSTACLE_DISTANCES),
        OBSTACLE_DISTANCES,
    )
logger.info("EXPERIMENT_RESULTS = %s", experiment_results)
