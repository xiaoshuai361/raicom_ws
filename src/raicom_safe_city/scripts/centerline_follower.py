#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler


ROAD_HALF_X = 1.38
ROAD_HALF_Y = 1.38
ROAD_RADIUS = 0.40


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def sample_line(points, x1, y1, x2, y2, step):
    length = math.hypot(x2 - x1, y2 - y1)
    count = max(1, int(math.ceil(length / step)))
    for i in range(count):
        t = i / float(count)
        points.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))


def sample_arc(points, cx, cy, radius, start, end, step):
    length = abs(end - start) * radius
    count = max(2, int(math.ceil(length / step)))
    for i in range(count):
        t = i / float(count)
        angle = start + (end - start) * t
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))


def build_centerline(step):
    hx = ROAD_HALF_X
    hy = ROAD_HALF_Y
    r = ROAD_RADIUS
    points = []

    # Exact 80cm road centerline generated from the same geometry as the Gazebo
    # walls and floor tiles. The controller follows this line directly.
    sample_line(points, hx, 0.0, hx, hy - r, step)
    sample_arc(points, hx - r, hy - r, r, 0.0, math.pi / 2.0, step)
    sample_line(points, hx - r, hy, -hx + r, hy, step)
    sample_arc(points, -hx + r, hy - r, r, math.pi / 2.0, math.pi, step)
    sample_line(points, -hx, hy - r, -hx, -hy + r, step)
    sample_arc(points, -hx + r, -hy + r, r, math.pi, 3.0 * math.pi / 2.0, step)
    sample_line(points, -hx + r, -hy, hx - r, -hy, step)
    sample_arc(points, hx - r, -hy + r, r, 3.0 * math.pi / 2.0, 2.0 * math.pi, step)
    sample_line(points, hx, -hy + r, hx, 0.0, step)
    points.append((hx, 0.0))
    return points


def path_lengths(points):
    lengths = [0.0]
    for p1, p2 in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return lengths


def make_pose(x, y, yaw=0.0):
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


class CenterlineFollower:
    def __init__(self):
        self.path_step = rospy.get_param("~path_step", 0.035)
        self.lookahead = rospy.get_param("~lookahead", 0.34)
        self.base_speed = rospy.get_param("~base_speed", 0.155)
        self.corner_speed = rospy.get_param("~corner_speed", 0.090)
        self.max_angular = rospy.get_param("~max_angular", 1.35)
        self.heading_gain = rospy.get_param("~heading_gain", 2.40)
        self.cross_track_gain = rospy.get_param("~cross_track_gain", 0.32)
        self.max_cross_track_for_full_slowdown = rospy.get_param("~max_cross_track_for_full_slowdown", 0.16)
        self.finish_tolerance = rospy.get_param("~finish_tolerance", 0.16)
        self.auto_start = rospy.get_param("~auto_start", True)
        self.verbose = rospy.get_param("~verbose", False)
        self.last_debug_time = rospy.Time(0)

        self.points = build_centerline(self.path_step)
        self.lengths = path_lengths(self.points)
        self.progress_index = 0
        self.pose = None
        self.started = self.auto_start
        self.finished = False

        self.path_pub = rospy.Publisher("/patrol/waypoints", Path, queue_size=1, latch=True)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=20)
        self.publish_path()

        if self.verbose:
            rospy.loginfo(
                "[中心线跟踪] 已生成车体安全中心轨迹：%d 个点，总长 %.2fm，lookahead=%.2fm。",
                len(self.points),
                self.lengths[-1],
                self.lookahead,
            )

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def publish_path(self):
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()
        for index, (x, y) in enumerate(self.points):
            if index + 1 < len(self.points):
                nx, ny = self.points[index + 1]
                yaw = math.atan2(ny - y, nx - x)
            else:
                yaw = math.pi / 2.0
            path.poses.append(make_pose(x, y, yaw))
        self.path_pub.publish(path)

    def nearest_index(self, x, y):
        start = max(0, self.progress_index - 8)
        end = min(len(self.points), self.progress_index + 90)
        best_index = start
        best_dist = float("inf")
        for index in range(start, end):
            px, py = self.points[index]
            dist = (x - px) * (x - px) + (y - py) * (y - py)
            if dist < best_dist:
                best_dist = dist
                best_index = index
        self.progress_index = max(self.progress_index, best_index)
        return best_index

    def lookahead_index(self, nearest):
        target_s = self.lengths[nearest] + self.lookahead
        for index in range(nearest, len(self.lengths)):
            if self.lengths[index] >= target_s:
                return index
        return len(self.points) - 1

    def path_yaw_at(self, index):
        if index + 1 < len(self.points):
            x1, y1 = self.points[index]
            x2, y2 = self.points[index + 1]
        else:
            x1, y1 = self.points[index - 1]
            x2, y2 = self.points[index]
        return math.atan2(y2 - y1, x2 - x1)

    def signed_cross_track_error(self, x, y, index):
        px, py = self.points[index]
        yaw = self.path_yaw_at(index)
        dx = x - px
        dy = y - py
        return -math.sin(yaw) * dx + math.cos(yaw) * dy

    def upcoming_curvature(self, nearest, sample_count=18):
        start = max(1, nearest)
        end = min(len(self.points) - 2, nearest + sample_count)
        if end <= start:
            return 0.0
        total = 0.0
        samples = 0
        for index in range(start, end):
            yaw1 = self.path_yaw_at(index - 1)
            yaw2 = self.path_yaw_at(index + 1)
            ds = max(0.001, self.lengths[index + 1] - self.lengths[index - 1])
            total += abs(normalize_angle(yaw2 - yaw1)) / ds
            samples += 1
        return total / float(samples)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def control_once(self):
        self.publish_path()
        if self.pose is None or not self.started or self.finished:
            self.stop()
            return

        x, y, yaw = self.pose
        nearest = self.nearest_index(x, y)
        final_x, final_y = self.points[-1]
        final_dist = math.hypot(x - final_x, y - final_y)
        near_finish = nearest > len(self.points) - 18
        if near_finish and final_dist <= self.finish_tolerance:
            self.finished = True
            self.stop()
            if self.verbose:
                rospy.loginfo("[中心线跟踪] 已沿跑道中心线完整一圈并回到起点，终点距离 %.2fm。", final_dist)
            return

        target = self.lookahead_index(nearest)
        tx, ty = self.points[target]
        target_angle = math.atan2(ty - y, tx - x)
        heading_error = normalize_angle(target_angle - yaw)
        cross_track = self.signed_cross_track_error(x, y, nearest)

        progress_left = max(0.0, self.lengths[-1] - self.lengths[nearest])
        speed = self.base_speed
        upcoming_curvature = self.upcoming_curvature(nearest)
        if upcoming_curvature > 0.65 or abs(heading_error) > 0.36:
            speed = self.corner_speed
        if abs(cross_track) > 0.08:
            ratio = min(1.0, abs(cross_track) / self.max_cross_track_for_full_slowdown)
            speed = min(speed, self.corner_speed * (1.0 - 0.25 * ratio))
        if progress_left < 0.45:
            speed = min(speed, 0.09)

        cmd = Twist()
        cmd.linear.x = speed
        # Pure-pursuit heading plus a small signed lateral correction. Positive
        # cross-track means left of the path, so subtracting it steers back
        # toward the road center without over-fighting the curve heading.
        angular = self.heading_gain * heading_error - self.cross_track_gain * cross_track
        cmd.angular.z = max(-self.max_angular, min(self.max_angular, angular))
        self.cmd_pub.publish(cmd)

        now = rospy.Time.now()
        if self.verbose and (now - self.last_debug_time).to_sec() > 1.5:
            self.last_debug_time = now
            rospy.loginfo(
                "[中心线跟踪] 横向误差=%.3fm, 航向误差=%.2frad, 前方曲率=%.2f, v=%.2f, w=%.2f",
                cross_track,
                heading_error,
                upcoming_curvature,
                cmd.linear.x,
                cmd.angular.z,
            )

    def spin(self):
        if not self.auto_start:
            try:
                text = input("输入 start 后开始中心线巡航：").strip().lower()
            except EOFError:
                text = ""
            self.started = text == "start"
            if not self.started:
                if self.verbose:
                    rospy.loginfo("[中心线跟踪] 未输入 start，退出。")
                return

        rate = rospy.Rate(20.0)
        while not rospy.is_shutdown() and not self.finished:
            self.control_once()
            rate.sleep()
        self.stop()


def main():
    rospy.init_node("centerline_follower")
    follower = CenterlineFollower()
    follower.spin()


if __name__ == "__main__":
    main()
