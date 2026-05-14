import logging
import math
import time

import airsim

from planners.vfh_plus import VFHPlusPlanner


# 전체 로직:
# 1. Drone1을 이륙시켜 500m 전방 목표점을 향해 가속한다.
# 2. 목표 속도 조건이 유지되면 지정 거리 앞에 StaticObstacle을 배치한다.
# 3. 장애물 생성 후 라이다 점군을 VFH+로 해석하되 yaw는 목표 지점에 고정한다.
# 4. 충돌하면 실패, 지정 시간 생존하면 성공으로 기록하고 같은 실험을 반복한다.

DRONE = "Drone1"  # 조종할 메인 드론 이름
OBSTACLE = "StaticObstacle"  # 충돌 대상으로 배치할 장애물 차량 이름
LIDAR = "Lidar2D"  # Drone1에 장착된 라이다 센서 이름

ALTITUDE_Z = -50.0  # 드론이 유지할 NED z 좌표. -50은 고도 50m
COMMAND_SPEED = 40.0  # 실제 목표 속도 도달을 위해 넣는 X+ 방향 속도 명령값
TARGET_SPEED = 20.0  # 회피 중 유지할 속도이자 장애물을 배치할 실제 전진 속도 기준값
TARGET_DISTANCE = 500.0  # 시작 위치 기준 전방 목표 지점 거리
SPAWN_DELAY_SEC = 0.1  # 목표 속도 도달 후 장애물 배치까지 기다릴 시간
OBSTACLE_DISTANCES = (26.0, )  # 차례로 테스트할 장애물 배치 거리
TRIALS = 100 # 각 장애물 거리마다 반복할 실험 횟수
SUCCESS_SEC = 10.0  # 장애물 생성 후 이 시간 동안 생존하면 성공

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airsim_loop")


client: airsim.MultirotorClient = airsim.MultirotorClient()
client.confirmConnection()

success_count = 0
fail_count = 0
experiment_results = {distance: [] for distance in OBSTACLE_DISTANCES}

for OBSTACLE_DISTANCE in OBSTACLE_DISTANCES:
    distance_success_count = 0
    distance_fail_count = 0

    for trial in range(1, TRIALS + 1):
        planner = VFHPlusPlanner()
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

            if obstacle_placed and point_count > 0:
                last_lidar_seen_at = now
                angle_offset = math.degrees(yaw - target_heading)
                vfh_heading = planner.plan(
                    lidar_data.point_cloud,
                    angle_offset,
                    horizontal_speed,
                )

                if vfh_heading is not None:
                    last_vfh_heading = vfh_heading
                lidar_active = True

            elif (
                obstacle_placed
                and last_lidar_seen_at is not None
                and now - last_lidar_seen_at <= 0.5
            ):
                vfh_heading = last_vfh_heading
                lidar_active = True
            elif obstacle_placed:
                planner.prev_heading = 0.0

            if vfh_heading is None:
                vx = 0.0
                vy = 0.0
                command_yaw = target_heading
                event = "blocked"
            elif lidar_active:
                world_heading = target_heading + math.radians(vfh_heading)
                vx = TARGET_SPEED * math.cos(world_heading)
                vy = TARGET_SPEED * math.sin(world_heading)
                command_yaw = world_heading
            else:
                vx = COMMAND_SPEED * math.cos(target_heading)
                vy = COMMAND_SPEED * math.sin(target_heading)
                command_yaw = target_heading

            vfh_heading_log = f"{vfh_heading:.1f}" if vfh_heading is not None else "blocked"
            vfh_debug_log = planner.debug_summary() if lidar_active else "-"

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
                fail_count += 1
                distance_fail_count += 1
                experiment_results[OBSTACLE_DISTANCE].append(False)
                event = "fail"

            if obstacle_placed_at is not None and now - obstacle_placed_at >= SUCCESS_SEC and not collided:
                success_count += 1
                distance_success_count += 1
                experiment_results[OBSTACLE_DISTANCE].append(True)
                event = "success"

            logger.info(
                "distance=%.1fm trial=%d/%d t=%.2fs target_dist=%.1fm speed=%.2fm/s target_forward=%.2fm/s target_hold=%.2fs alt=%.2fm z=%.2f yaw=%.1f target_yaw=%.1f vfh_heading=%s lidar_points=%d vfh_debug=%s obstacle=%s collision=%s result=%s distance_success=%d distance_fail=%d total_success=%d total_fail=%d",
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
                vfh_heading_log,
                point_count,
                vfh_debug_log,
                obstacle_placed,
                collision.object_name if collided else False,
                event,
                distance_success_count,
                distance_fail_count,
                success_count,
                fail_count,
            )

            if event in ("success", "fail"):
                client.cancelLastTask(vehicle_name=DRONE)
                break

            time.sleep(0.05)

    logger.info(
        "distance %.1fm complete: success=%d fail=%d trials=%d",
        OBSTACLE_DISTANCE,
        distance_success_count,
        distance_fail_count,
        TRIALS,
    )

logger.info(
    "VFH+ experiment complete: success=%d fail=%d trials=%d distances=%s",
    success_count,
    fail_count,
    TRIALS * len(OBSTACLE_DISTANCES),
    OBSTACLE_DISTANCES,
)
logger.info("EXPERIMENT_RESULTS = %s", experiment_results)
