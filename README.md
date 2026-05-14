# RAICOM 平安城市安防救援仿真

面向睿抗机器人大赛 2026 CAIR 其他结构机器人（安防救援）平安城市赛道的 ROS/Gazebo 仿真项目。项目基于 Ubuntu 20.04、ROS Noetic 和 Gazebo 11，实现了自研安防巡逻小车、4m x 4m 平安城市场地、SLAM 建图、地图保存、路径规划导航、中心线平滑巡航、墙面照片识别展示和到点识别结果终端输出。

## 项目亮点

- 自研 `raicom_guard_rover` 四轮安防巡逻车，不直接套用 TurtleBot 默认整车模型。
- Gazebo 平安城市场地包含 80cm 跑道墙体、起点标识、内侧墙面识别照片和比赛目标区域。
- 支持 `/scan` 激光雷达、里程计、TF、侧向摄像头 `/camera/rgb/image_raw` 和 RViz 图像显示。
- 支持 `gmapping` SLAM 建图、`map_server` 地图保存，以及默认地图/新建图地图切换导航。
- 支持 `move_base` 单点/多点导航，并提供中心线跟踪模式解决窄弯道贴墙和转弯不顺问题。
- 到达识别目标附近时，终端只输出比赛要求的识别结果：垃圾桶类型、人群类型及数量、楼宇灾害情况。
- 参考 2025 识别示例与模型资源，保留后续接入真实摄像头推理的扩展空间。

## 环境要求

- Ubuntu 20.04
- ROS Noetic
- Gazebo 11
- Catkin 工作空间
- 常用依赖：`gazebo_ros`、`xacro`、`gmapping`、`map_server`、`amcl`、`move_base`、`teleop_twist_keyboard`、`cv_bridge`

## 快速启动

```bash
source /opt/ros/noetic/setup.bash
source ~/raicom_ws/devel/setup.bash
roslaunch raicom_safe_city gui_demo.launch rviz:=true
```

默认会启动 Gazebo、导航后台、RViz 和中心线跟踪节点。RViz 中可以看到地图、机器人、LaserScan、路径/中心线以及侧向摄像头画面。

只打开场景和导航后台，不自动巡航：

```bash
roslaunch raicom_safe_city gui_demo.launch rviz:=true run_centerline_follower:=false run_multi_goal_patrol:=false
```

使用传统 `move_base` 多点导航：

```bash
roslaunch raicom_safe_city gui_demo.launch rviz:=true run_centerline_follower:=false run_multi_goal_patrol:=true
```

## 建图与地图切换

启动 SLAM 建图：

```bash
source /opt/ros/noetic/setup.bash
source ~/raicom_ws/devel/setup.bash
roslaunch raicom_safe_city slam_mapping.launch rviz:=true
```

保存地图：

```bash
rosrun map_server map_saver -f ~/raicom_ws/src/raicom_safe_city/maps/my_slam_map
```

使用默认地图导航：

```bash
roslaunch raicom_safe_city gui_demo.launch rviz:=true
```

使用新保存地图导航：

```bash
roslaunch raicom_safe_city gui_demo.launch rviz:=true map_file:=/home/zcy/raicom_ws/src/raicom_safe_city/maps/my_slam_map.yaml
```

## 常用命令

键盘控制：

```bash
roslaunch raicom_safe_city keyboard.launch
```

单点导航：

```bash
rosrun raicom_safe_city single_goal_driver.py _goal_timeout:=100.0
```

多点导航：

```bash
rosrun raicom_safe_city multi_goal_patrol.py _goal_timeout:=85.0
```

摄像头识别扩展：

```bash
roslaunch raicom_safe_city camera_recognition.launch
```

## 墙面照片识别

场地内侧墙面放置了三类 visual-only 识别照片，不添加碰撞体，不影响机器人行驶、激光建图或代价地图：

- 垃圾桶照片：上方内侧墙
- 人群照片：左侧内侧墙
- 楼宇火灾照片：下方内侧墙

小车侧向摄像头经过对应区域时可以在 RViz 中看到墙面照片。当前稳定演示方式为到达目标附近后由识别节点输出比赛要求结果：

```text
识别结果：垃圾桶类型=厨余垃圾桶
识别结果：人群类型=普通行人，数量=1
识别结果：楼宇灾害情况=电子超市多楼层火灾，浓烟明显
```

`/home/zcy/raicom_ws/2025源码/` 下保留了垃圾桶、人群、楼宇火灾识别模型与示例代码，可作为后续接入真实摄像头推理的参考。

## 目录结构

```text
raicom_ws/
├── src/raicom_safe_city/
│   ├── launch/          # Gazebo、导航、SLAM、演示入口
│   ├── config/          # move_base、costmap、RViz、gmapping 配置
│   ├── maps/            # 默认地图与验证保存地图
│   ├── models/          # 墙面照片目标模型和贴图资源
│   ├── scripts/         # 巡航、导航、识别、地图生成脚本
│   ├── urdf/            # 自研安防巡逻车 URDF/Xacro
│   └── worlds/          # Gazebo 平安城市场地
├── 2025源码/            # 识别模型与旧示例参考资源
└── 任务完成报告.md      # 完整实现记录、验证结果和问题说明
```

## 当前完成情况

- Gazebo 场地能正常打开。
- 自研机器人模型能正常显示和运动。
- 键盘控制、`/scan`、摄像头图像均可用。
- SLAM 建图和地图保存可用。
- 支持默认地图与新建图地图导航。
- 支持单点导航、多点导航和中心线平滑巡航。
- 到点后终端输出垃圾桶、人群、楼宇灾害识别结果。
- 已整理完整任务完成报告，见 [任务完成报告.md](./任务完成报告.md)。

