#!/usr/bin/env python3
import math

import rospy
from nav_msgs.msg import Odometry


TARGETS = [
    {
        "name": "start_marker",
        "xy": (1.38, 0.0),
        "radius": 0.45,
        "message": "识别结果：起点区域已确认，赛道标识=斑马线+双横杠，识别模式=导航联调",
    },
    {
        "name": "upper_lane",
        "xy": (0.0, 1.38),
        "radius": 0.45,
        "message": "识别结果：垃圾桶类型=可回收垃圾桶，位置=上方识别区，状态=模拟识别输出",
    },
    {
        "name": "left_lane",
        "xy": (-1.38, 0.0),
        "radius": 0.55,
        "message": "识别结果：人群类型=疏散人群，数量=4，位置=左侧识别区，状态=模拟识别输出",
    },
    {
        "name": "lower_lane",
        "xy": (0.92, -1.22),
        "radius": 0.50,
        "message": "识别结果：楼宇灾害情况=南侧楼宇外立面冒烟，风险等级=中，状态=模拟识别输出",
    },
]


class ProximityRecognitionNode:
    def __init__(self):
        self.detected = set()
        self.reset_distance = rospy.get_param("~reset_distance", 0.85)
        self.last_xy = None
        self.sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=10)
        rospy.loginfo("[识别] 近距离目标识别节点已启动，监听 /odom。")

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_xy = (x, y)

        for target in TARGETS:
            tx, ty = target["xy"]
            dist = math.hypot(x - tx, y - ty)
            if dist <= target["radius"] and target["name"] not in self.detected:
                self.detected.add(target["name"])
                rospy.loginfo("[识别] 到达 %s 附近，距离 %.2fm。%s", target["name"], dist, target["message"])
            elif dist > self.reset_distance and target["name"] in self.detected:
                # Allow a repeated print if the robot leaves and later returns during manual tests.
                self.detected.remove(target["name"])


def main():
    rospy.init_node("proximity_recognition_node")
    ProximityRecognitionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
