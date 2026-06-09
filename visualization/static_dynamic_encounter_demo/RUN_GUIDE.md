# Static Dynamic Encounter Demo 실행 가이드

## 생성 전략

기준 이미지를 10배 스케일의 2D meter 맵으로 해석했다.

- 필드 크기: `120 m x 70 m`
- 정적 장애물 1: 상단 검은 직사각형 `(x=26, y=62.5, w=94, h=7.5)`
- 정적 장애물 2: 하단 검은 직사각형 `(x=26, y=0, w=94, h=45.5)`
- 드론 시작점: `(13.5, 12.5)`
- 코너 및 통로 waypoint: `(13.5, 52.0) -> (32.0, 54.5)`
- 목표: `(112.5, 54.5)`
- 동적 장애물 생성 위치: `(104.0, 54.5)`
- 동적 장애물 이동 방향: 왼쪽, 즉 `-x` 방향

드론은 매 step마다 다음 point cloud를 구성해 `planners.ego_planner.EGOPlanner`에 전달한다.

- 정적 장애물 직사각형 표면을 2m 간격 point cloud로 샘플링
- 동적 장애물이 활성화되고, 50m 라이다 범위 안이며, 정적 장애물에 의해 line-of-sight가 가려지지 않을 때만 동적 장애물 원형 표면 point cloud를 추가
- EGOPlanner 내부 라이다 범위는 `50 m`

동적 장애물은 모든 trial에서 같은 위치에 생성된 뒤 왼쪽으로 이동한다. 달라지는 값은 `release_delay_s`뿐이다. 이 값이 작을수록 동적 장애물은 더 일찍 이동을 시작하므로, 드론이 코너를 돌아 통로를 볼 수 있게 되는 시점에는 장애물이 코너/게이트 쪽에 더 가까워져 있다.

release 시점에는 생성 위치 `DYNAMIC_SPAWN`이 드론의 실제 관측 가능 조건 안인지 먼저 검사한다.

- 드론과 생성 위치 사이 거리가 `50 m` 이하
- 정적 장애물이 line-of-sight를 가리지 않음

두 조건을 모두 만족하면, 이는 현실적으로 이미 보이는 위치에 갑자기 장애물이 생성되는 무효 trial이다. 이 경우 동적 장애물은 스폰하지 않고 결과를 `spawn_suppressed`로 기록한다.

기본 trial은 다음 지연 시간을 사용한다.

```text
0s, 2s, 4s, 6s, 8s, 10s, 11s, 11.5s
```

현재 기본값은 모두 생성 위치가 관측 가능해지기 전에 스폰되는 케이스다. 따라서 비교의 의미는 "늦게 생성되어 늦게 보임"이 아니라 "이미 이동 중이지만 코너/정적 장애물 때문에 늦게 보임"이다.

기존처럼 `12s` 이후 지연 시간을 직접 넣으면, 현재 맵과 속도 기준으로 생성 위치가 이미 라이다 범위와 line-of-sight 안에 들어와 `spawn_suppressed`가 발생할 수 있다.

이 값은 데모 파라미터 기준의 경험적 결과이므로, 속도, 반경, waypoint, planner 설정을 바꾸면 다시 측정해야 한다.

## 실행

요약 표만 확인:

```powershell
python -m visualization.static_dynamic_encounter_demo --summary-only
```

통합 GIF 생성:

```powershell
python -m visualization.static_dynamic_encounter_demo --no-show
```

기본 출력:

```text
visualization/static_dynamic_encounter_demo/static_dynamic_encounter_demo.gif
```

trial별 GIF까지 같이 저장:

```powershell
python -m visualization.static_dynamic_encounter_demo --no-show --save-trials
```

GIF 렌더링이 느리면 더 큰 frame stride를 지정한다.

```powershell
python -m visualization.static_dynamic_encounter_demo --no-show --frame-stride 5
```

지연 시간 직접 지정:

```powershell
python -m visualization.static_dynamic_encounter_demo --no-show --delays 8,11.5,12
```

출력 경로 지정:

```powershell
python -m visualization.static_dynamic_encounter_demo --no-show --save visualization/static_dynamic_encounter_demo/custom.gif
```

## 결과 해석

콘솔에는 다음 CSV 형식의 요약이 출력된다.

```text
release_delay_s,first_detection_distance_m,min_distance_m,final_status
```

- `release_delay_s`: 동적 장애물이 생성 및 이동을 시작한 시간
- `first_detection_distance_m`: 드론 라이다가 동적 장애물을 처음 관측한 거리
- `min_distance_m`: 해당 trial 전체에서 드론과 동적 장애물 사이의 최소 거리
- `final_status`: `goal_reached`, `collision_dynamic`, `collision_static_*`, `timeout`, `spawn_suppressed`

`spawn_suppressed`는 release 시점에 생성 위치가 이미 관측 가능해, 실험 의도에 맞지 않아 동적 장애물 스폰을 막았다는 뜻이다. 이 경우 `first_detection_distance_m`과 `min_distance_m`은 `NA`로 표시된다.

GIF의 오른쪽 아래 그래프에는 시간에 따른 드론-동적 장애물 거리가 표시된다.

- 파란 점선: 50m 라이다 관측 범위
- 빨간 점선: 드론 반경 + 동적 장애물 반경의 충돌 거리
