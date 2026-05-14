# RAICOM Safe City Model Notes

## Default robot path

The default verified robot path is now TurtleBot3 Waffle Pi for stability on ROS Noetic + Gazebo 11.

- default startup entrypoints (`gui_demo.launch`, `navigation.launch`, `slam_mapping.launch`, `patrol_demo.launch`) all route to the TB3 stack;
- the legacy custom rover file `urdf/raicom_guard_rover.urdf.xacro` is still kept in the repo but is no longer the default launch target;
- Gazebo model spawning is guarded by `scripts/spawn_robot_model.py`, which deletes stale robot models before spawning the selected one.

## Field scope

`worlds/safe_city.world` models the first-stage 4000mm x 4000mm Safe City wall track.

It currently includes:

- 4m x 4m field base;
- 80cm drivable wall-bounded lane;
- right-side zebra crossing and transverse start markings;
- collision geometry aligned with the navigation map.

It currently does **not** include:

- trash bin实体模型;
- crowd实体模型;
- building实体模型.

Those are intentionally deferred in the current stage; recognition output is implemented as navigation-zone simulation.

## Verified feature paths

- SLAM path: `slam_mapping.launch` starts Gazebo, TB3, gmapping and optional RViz.
- Navigation path: `navigation.launch` starts Gazebo, TB3, map server, AMCL, move_base and optional RViz.
- Single-goal demo: `single_goal_demo.launch` sends one `move_base` target.
- Multi-goal demo: `patrol_demo.launch` runs continuous waypoint patrol and prints recognition results.
- Camera extension: `camera_recognition.launch` starts a simple color detector on `/camera/rgb/image_raw`.
