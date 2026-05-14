#!/usr/bin/env python3
import math
import sys

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

try:
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
except ImportError:
    MoveBaseAction = None
    MoveBaseGoal = None

GOAL = {
    "name": "右侧车道中线点",
    "xyyaw": (1.38, 0.82, math.pi / 2.0),
    "result": "单点导航结果：已到达右侧车道中线点",
}


def make_pose(x, y, yaw):
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def main():
    rospy.init_node("single_goal_driver")

    if MoveBaseAction is None or MoveBaseGoal is None:
        rospy.logerr("缺少 move_base_msgs。请安装 ros-noetic-navigation 或 ros-noetic-move-base-msgs 后重试。")
        sys.exit(2)

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    wait_timeout = float(rospy.get_param("~server_timeout", 20.0))
    goal_timeout = float(rospy.get_param("~goal_timeout", 90.0))
    goal_name = rospy.get_param("~goal_name", GOAL["name"])
    goal_x = float(rospy.get_param("~goal_x", GOAL["xyyaw"][0]))
    goal_y = float(rospy.get_param("~goal_y", GOAL["xyyaw"][1]))
    goal_yaw = float(rospy.get_param("~goal_yaw", GOAL["xyyaw"][2]))

    if not client.wait_for_server(rospy.Duration(wait_timeout)):
        rospy.logerr("[单点导航] %.0f 秒内未连接到 move_base，请先启动 navigation.launch。", wait_timeout)
        sys.exit(3)

    goal = MoveBaseGoal()
    goal.target_pose = make_pose(goal_x, goal_y, goal_yaw)

    rospy.loginfo("[单点导航] 发送目标：%s (x=%.2f, y=%.2f, yaw=%.2f)", goal_name, goal_x, goal_y, goal_yaw)
    client.send_goal(goal)

    finished = client.wait_for_result(rospy.Duration(goal_timeout))
    if not finished:
        client.cancel_goal()
        rospy.logwarn("[单点导航] 目标超时，已取消：%s", goal_name)
        sys.exit(4)

    state = client.get_state()
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("[单点导航] 已到达。%s", GOAL["result"])
        return

    rospy.logwarn("[单点导航] 未成功到达：%s，move_base state=%s", goal_name, state)
    sys.exit(5)


if __name__ == "__main__":
    main()
