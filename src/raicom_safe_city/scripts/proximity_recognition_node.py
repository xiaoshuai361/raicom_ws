#!/usr/bin/env python3
import math

import rospy
from nav_msgs.msg import Odometry


TARGETS = [
    {
        "name": "trash_food_waste",
        "xy": (0.0, 1.38),
        "radius": 0.45,
        "message": "识别结果：垃圾桶类型=厨余垃圾桶",
    },
    {
        "name": "crowd_people",
        "xy": (-1.38, 0.0),
        "radius": 0.55,
        "message": "识别结果：人群类型=普通行人，数量=1",
    },
    {
        "name": "building_fire",
        "xy": (0.15, -1.38),
        "radius": 0.50,
        "message": "识别结果：楼宇灾害情况=电子超市多楼层火灾，浓烟明显",
    },
]


class ProximityRecognitionNode:
    def __init__(self):
        self.detected = set()
        self.reset_distance = rospy.get_param("~reset_distance", 0.85)
        self.last_xy = None
        self.sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=10)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_xy = (x, y)

        for target in TARGETS:
            tx, ty = target["xy"]
            dist = math.hypot(x - tx, y - ty)
            if dist <= target["radius"] and target["name"] not in self.detected:
                self.detected.add(target["name"])
                rospy.loginfo("%s", target["message"])
            elif dist > self.reset_distance and target["name"] in self.detected:
                # Allow a repeated print if the robot leaves and later returns during manual tests.
                self.detected.remove(target["name"])


def main():
    rospy.init_node("proximity_recognition_node")
    ProximityRecognitionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
