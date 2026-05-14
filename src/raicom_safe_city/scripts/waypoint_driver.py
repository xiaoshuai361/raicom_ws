#!/usr/bin/env python3
import math
import sys

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler


WAYPOINTS = [
    {
        "name": "右侧上方车道点",
        "xyyaw": (1.38, 1.05, math.pi / 2.0),
        "result": "巡航结果：已通过右侧墙体车道上段",
    },
    {
        "name": "上方车道点",
        "xyyaw": (-1.05, 1.38, math.pi),
        "result": "巡航结果：已通过上方 80cm 车道",
    },
    {
        "name": "左侧车道点",
        "xyyaw": (-1.38, -1.05, -math.pi / 2.0),
        "result": "巡航结果：已通过左侧墙体车道",
    },
    {
        "name": "下方车道点",
        "xyyaw": (0.92, -1.22, 0.0),
        "result": "巡航结果：已通过下方 80cm 车道",
    },
    {
        "name": "回到右侧起点标志",
        "xyyaw": (1.38, 0.0, math.pi / 2.0),
        "result": "巡航结果：已完成一圈墙体赛道覆盖",
    },
]


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_to_quaternion(yaw):
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    return qx, qy, qz, qw


def pose_stamped(x, y, yaw, frame_id="odom"):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    pose.pose.position.x = x
    pose.pose.position.y = y
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose


class WaypointDriver:
    def __init__(self):
        self.pose = None
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.13))
        self.yaw_tolerance = float(rospy.get_param("~yaw_tolerance", 0.25))
        self.max_linear = float(rospy.get_param("~max_linear", 0.16))
        self.max_angular = float(rospy.get_param("~max_angular", 0.75))
        self.goal_timeout = float(rospy.get_param("~goal_timeout", 55.0))
        self.auto_start = bool(rospy.get_param("~auto_start", True))

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.path_pub = rospy.Publisher("/patrol/waypoints", Path, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=10)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (p.x, p.y, yaw)

    def publish_path(self):
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()
        if self.pose:
            x, y, yaw = self.pose
            path.poses.append(pose_stamped(x, y, yaw))
        else:
            path.poses.append(pose_stamped(1.38, 0.0, math.pi / 2.0))
        for item in WAYPOINTS:
            x, y, yaw = item["xyyaw"]
            path.poses.append(pose_stamped(x, y, yaw))
        self.path_pub.publish(path)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def wait_for_odom(self):
        rospy.loginfo("[导航] 等待 /odom...")
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.pose is None:
            if (rospy.Time.now() - start).to_sec() > 20.0:
                rospy.logerr("[导航] 20秒内没有收到 /odom。")
                return False
            rate.sleep()
        return True

    def drive_to(self, item):
        goal_x, goal_y, goal_yaw = item["xyyaw"]
        rospy.loginfo("[导航] 前往 %s: x=%.2f y=%.2f yaw=%.2f", item["name"], goal_x, goal_y, goal_yaw)
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.pose is None:
                rate.sleep()
                continue

            x, y, yaw = self.pose
            dx = goal_x - x
            dy = goal_y - y
            distance = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            heading_error = wrap_angle(target_yaw - yaw)
            final_yaw_error = wrap_angle(goal_yaw - yaw)

            if distance <= self.goal_tolerance:
                if abs(final_yaw_error) <= self.yaw_tolerance:
                    self.stop()
                    rospy.loginfo("[导航] 已到达 %s。%s", item["name"], item["result"])
                    return True
                cmd = Twist()
                cmd.angular.z = clamp(1.4 * final_yaw_error, -self.max_angular, self.max_angular)
                self.cmd_pub.publish(cmd)
            else:
                cmd = Twist()
                if abs(heading_error) > 0.20:
                    cmd.linear.x = 0.0
                    cmd.angular.z = clamp(1.8 * heading_error, -self.max_angular, self.max_angular)
                else:
                    cmd.linear.x = clamp(0.95 * distance, 0.06, self.max_linear)
                    cmd.angular.z = clamp(0.8 * heading_error, -0.35, 0.35)
                self.cmd_pub.publish(cmd)

            if (rospy.Time.now() - start_time).to_sec() > self.goal_timeout:
                self.stop()
                rospy.logwarn("[导航] 目标超时：%s，当前位置 x=%.2f y=%.2f", item["name"], x, y)
                return False

            rate.sleep()

    def run(self):
        if not self.wait_for_odom():
            sys.exit(1)
        self.publish_path()
        rospy.loginfo("[导航] 已发布巡航线 /patrol/waypoints。")

        if not self.auto_start:
            try:
                text = input("输入 start 后开始稳定多点导航：").strip().lower()
            except EOFError:
                text = ""
            if text != "start":
                rospy.loginfo("[导航] 未输入 start，退出。")
                return

        success = 0
        for item in WAYPOINTS:
            self.publish_path()
            if self.drive_to(item):
                success += 1
            rospy.sleep(0.5)

        self.stop()
        rospy.loginfo("[导航] 多点导航完成：%d/%d。", success, len(WAYPOINTS))


def main():
    rospy.init_node("waypoint_driver")
    node = WaypointDriver()
    node.run()


if __name__ == "__main__":
    main()
