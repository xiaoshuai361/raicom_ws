#!/usr/bin/env python3
import math
import sys

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import quaternion_from_euler

try:
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
except ImportError:
    MoveBaseAction = None
    MoveBaseGoal = None


PATROL_GOALS = [
    {"name": "右侧直道 1/3", "xyyaw": (1.38, 0.45, math.pi / 2.0), "result": "巡航结果：沿右侧中心线前进"},
    {"name": "右侧直道 2/3", "xyyaw": (1.38, 0.86, math.pi / 2.0), "result": "巡航结果：沿右侧中心线前进"},
    {"name": "右上弯道入口", "xyyaw": (1.32, 1.12, 2.05), "result": "巡航结果：进入右上中心弯道"},
    {"name": "右上弯道中段", "xyyaw": (1.12, 1.32, 2.65), "result": "巡航结果：通过右上中心弯道"},
    {"name": "上方直道右段", "xyyaw": (0.65, 1.38, math.pi), "result": "巡航结果：进入上方中心直道"},
    {"name": "上方直道中段", "xyyaw": (0.00, 1.38, math.pi), "result": "巡航结果：沿上方中心线巡航"},
    {"name": "上方直道左段", "xyyaw": (-0.65, 1.38, math.pi), "result": "巡航结果：沿上方中心线巡航"},
    {"name": "左上弯道入口", "xyyaw": (-0.98, 1.38, math.pi), "result": "巡航结果：进入左上中心弯道"},
    {"name": "左上弯道前段", "xyyaw": (-1.13, 1.35, -2.90), "result": "巡航结果：沿左上中心弧线巡航"},
    {"name": "左上弯道后段", "xyyaw": (-1.35, 1.13, -2.05), "result": "巡航结果：通过左上中心弧线"},
    {"name": "左侧直道上段", "xyyaw": (-1.38, 0.65, -math.pi / 2.0), "result": "巡航结果：进入左侧中心直道"},
    {"name": "左侧直道中段", "xyyaw": (-1.38, 0.00, -math.pi / 2.0), "result": "巡航结果：沿左侧中心线巡航"},
    {"name": "左侧直道下段", "xyyaw": (-1.38, -0.65, -math.pi / 2.0), "result": "巡航结果：沿左侧中心线巡航"},
    {"name": "左下弯道前段", "xyyaw": (-1.35, -1.13, -1.10), "result": "巡航结果：进入左下中心弯道"},
    {"name": "左下弯道后段", "xyyaw": (-1.13, -1.35, -0.45), "result": "巡航结果：通过左下中心弧线"},
    {"name": "下方直道左段", "xyyaw": (-0.65, -1.38, 0.0), "result": "巡航结果：进入下方中心直道"},
    {"name": "下方直道中段", "xyyaw": (0.00, -1.38, 0.0), "result": "巡航结果：沿下方中心线巡航"},
    {"name": "下方直道右段", "xyyaw": (0.65, -1.38, 0.0), "result": "巡航结果：沿下方中心线巡航"},
    {"name": "右下弯道入口", "xyyaw": (0.98, -1.38, 0.20), "result": "巡航结果：进入右下中心弯道"},
    {"name": "右下弯道前段", "xyyaw": (1.13, -1.35, 0.45), "result": "巡航结果：沿右下中心弧线巡航"},
    {"name": "右下弯道后段", "xyyaw": (1.35, -1.13, 1.10), "result": "巡航结果：通过右下中心弧线"},
    {"name": "右侧回程中线", "xyyaw": (1.38, -0.65, math.pi / 2.0), "result": "巡航结果：沿右侧中心线回到起点方向"},
    {"name": "回到起点中心", "xyyaw": (1.38, 0.0, math.pi / 2.0), "result": "巡航结果：已沿跑道中心线完成完整一圈并回到原点"},
]


class PoseTracker:
    def __init__(self):
        self.xy = None
        self.sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=10)

    def odom_callback(self, msg):
        self.xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def distance_to(self, x, y):
        if self.xy is None:
            return None
        return math.hypot(self.xy[0] - x, self.xy[1] - y)


def make_pose(x, y, yaw):
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


def publish_waypoint_path(pub):
    path = Path()
    path.header.frame_id = "map"
    path.header.stamp = rospy.Time.now()
    path.poses.append(make_pose(1.38, 0.0, math.pi / 2.0))
    for item in PATROL_GOALS:
        x, y, yaw = item["xyyaw"]
        path.poses.append(make_pose(x, y, yaw))
    path.poses.append(make_pose(1.38, 0.0, math.pi / 2.0))
    pub.publish(path)


def send_goal(client, item, timeout, pose_tracker, switch_radius):
    x, y, yaw = item["xyyaw"]
    goal = MoveBaseGoal()
    goal.target_pose = make_pose(x, y, yaw)
    rospy.loginfo("[巡航] 发送目标：%s (x=%.2f, y=%.2f, yaw=%.2f)", item["name"], x, y, yaw)
    client.send_goal(goal)

    deadline = rospy.Time.now() + rospy.Duration(timeout)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        dist = pose_tracker.distance_to(x, y)
        if dist is not None and dist <= switch_radius:
            rospy.loginfo("[巡航] 已接近：%s，距离 %.2fm。%s", item["name"], dist, item["result"])
            return True

        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[巡航] 已到达：%s。%s", item["name"], item["result"])
            return True
        if state in (actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED):
            rospy.logwarn("[巡航] 目标未成功：%s，move_base state=%s", item["name"], state)
            return False

        rate.sleep()

    client.cancel_goal()
    rospy.logwarn("[巡航] 目标超时，已取消：%s", item["name"])
    return False


def wait_for_pose(pose_tracker):
    deadline = rospy.Time.now() + rospy.Duration(10.0)
    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        if pose_tracker.xy is not None:
            return True
        rate.sleep()
    return False


def main():
    rospy.init_node("multi_goal_patrol")

    if MoveBaseAction is None or MoveBaseGoal is None:
        rospy.logerr("缺少 move_base_msgs。请安装 ros-noetic-navigation 或 ros-noetic-move-base-msgs 后重试。")
        sys.exit(2)

    auto_start = rospy.get_param("~auto_start", True)
    goal_timeout = float(rospy.get_param("~goal_timeout", 70.0))
    switch_radius = float(rospy.get_param("~switch_radius", 0.24))
    path_pub = rospy.Publisher("/patrol/waypoints", Path, queue_size=1, latch=True)
    pose_tracker = PoseTracker()

    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.loginfo("[巡航] 等待 move_base action server...")
    if not client.wait_for_server(rospy.Duration(90.0)):
        rospy.logerr("[巡航] 90秒内没有等到 move_base，请先检查 navigation.launch。")
        sys.exit(1)

    publish_waypoint_path(path_pub)
    rospy.loginfo("[巡航] 已发布预设巡航航线 /patrol/waypoints。")
    if not wait_for_pose(pose_tracker):
        rospy.logwarn("[巡航] 10秒内未收到 /odom，仍将尝试按 move_base 状态巡航。")

    if not auto_start:
        try:
            text = input("输入 start 后开始多点巡航：").strip().lower()
        except EOFError:
            text = ""
        if text != "start":
            rospy.loginfo("[巡航] 未输入 start，退出。")
            return

    rospy.sleep(1.0)
    ok_count = 0
    for index, item in enumerate(PATROL_GOALS):
        publish_waypoint_path(path_pub)
        if rospy.is_shutdown():
            return
        radius = 0.18 if index == len(PATROL_GOALS) - 1 else switch_radius
        ok = send_goal(client, item, goal_timeout, pose_tracker, radius)
        if ok:
            ok_count += 1
        rospy.sleep(0.05)

    rospy.loginfo("[巡航] 多点导航完成：%d/%d 个目标到达。", ok_count, len(PATROL_GOALS))


if __name__ == "__main__":
    main()
