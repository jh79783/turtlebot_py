![Alt text](https://github.com/jh79783/turtlebot3/blob/rearrange-repository/rosgraph.png?raw=true)
# **표지판 인식은 구동에 실패**
# 1. cam node
cam 노드로 부터 왼쪽,오른쪽 이미지를 받아 신호등 확인(/traffic_light), 차단바 확인(/cam_tutorial)한다.
/traffic_light노드가 /traffic_msg를 통하여 /mode_ch노드에 전달한다.
/cam_tutorial노드가 /def_msg를 통하여 /mode_ch노드에 전달한다.

**신호등 초록불과 차단바를 확인 및 발견시 출발/정지 하여야 하므로 우선순위를 위해 노드를 앞에다 지정함**

## /mode_ch
* /mode_ch노드는 각각의 토픽으로부터 값을 받아 어떤 모드인지 확인을 한다.
 *  /mode_ch노드에서 /mode_msg를 통해 각 모드의 번호와카운트가 지정되어 있어 상황에 따라 모드가 변하며 그에 맞는 작동을 수행하게 된다.
  1. 기본 모드는 0번으로 초기화 되어 있다.
  2. 1번 모드 및 카운트 2일때 차단바를 위한 동작을 한다.
  3. 1번 모드 및 카운트 1일때 신호등을 위한 동작을 한다.
  4. 2번 모드 및 카운트 0일때 장애물 회피를 위한 수행을 한다.
  5. 5번 모드 및 카운트 5일때 Lane tracking을 수행한다.
## /laser_move
* LIDAR SENSOR에 값이 들어오게 되면 /msg_laser를 통해 /mode_ch로 장애물 회피를 위한 모드로 진입하게 한다.
* 그 후 다시 /mode_msg를 통해 /laser_move로 장애물 회피 기동을 위한 신호를 주어 이때 /mode_twist를 통해 장애물 회피기동을 수행하게 된다.

## /det_move
* 차단바 검출시 정지를 위한 신호를 /twist_ch로 /mode_twist를 통하여 전달함

## /nav_goal
* 터널 표지판 검출시 navigation을 자동으로 실행시키기 위한 노드

**표지판 검출 실패로 진행(X)**

## /lane
* LANE이 검출되면 /lane_msg를 통해 /mode_ch에서 lane주행을 위한 모드 번호를 변경한다.
* /lane노드에서 /mode_msg를 통해 lane주행을 위한 모드가 들어오게 되면 /lane_msg를 통해 /lane_move로 전달하게 된다.
* /lane_move는 /lane_msg뿐만 아니라 /mode_msg도 받아 라인 주행을 위한 조건이 준비 되었는지 확인을 한다.
* 조건이 충족 되었을경우 /lane_move에 있는 주행 기동을 위한 동작을 /mode_twist를 통해 /twist_ch로 전달하여 주행을 신호를 준다.
