#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image


class CameraColorDetector:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        self.min_area = float(rospy.get_param("~min_area", 280.0))
        try:
            import cv2
            import numpy as np
            from cv_bridge import CvBridge
        except ImportError as exc:
            rospy.logerr("摄像头识别需要 cv_bridge/opencv/numpy：%s", exc)
            rospy.logerr("可安装 ros-noetic-cv-bridge python3-opencv python3-numpy。")
            raise

        self.cv2 = cv2
        self.np = np
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo("[视觉识别] 已启动，订阅 %s。", self.image_topic)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[视觉识别] 图像转换失败：%s", exc)
            return

        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        detections = []
        masks = {
            "蓝色垃圾桶/窗体": ((95, 80, 60), (135, 255, 255)),
            "绿色垃圾桶": ((38, 70, 50), (85, 255, 255)),
            "红色火情/警示": ((0, 80, 60), (10, 255, 255)),
            "黄色/橙色人群服装": ((15, 80, 80), (35, 255, 255)),
        }

        for label, (lo, hi) in masks.items():
            lower = self.np.array(lo, dtype=self.np.uint8)
            upper = self.np.array(hi, dtype=self.np.uint8)
            mask = self.cv2.inRange(hsv, lower, upper)
            area = float(self.cv2.countNonZero(mask))
            if area >= self.min_area:
                detections.append((label, area))

        if detections:
            text = "；".join(f"{label} 面积={area:.0f}" for label, area in detections)
            rospy.loginfo_throttle(1.5, "[视觉识别] %s", text)


def main():
    rospy.init_node("camera_color_detector")
    CameraColorDetector()
    rospy.spin()


if __name__ == "__main__":
    main()
