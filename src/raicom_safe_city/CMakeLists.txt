cmake_minimum_required(VERSION 3.0.2)
project(raicom_safe_city)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/proximity_recognition_node.py
  scripts/centerline_follower.py
  scripts/multi_goal_patrol.py
  scripts/waypoint_driver.py
  scripts/single_goal_driver.py
  scripts/camera_color_detector.py
  scripts/generate_safe_city_map.py
  scripts/spawn_robot_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch urdf worlds config maps models docs
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
