#!/usr/bin/env python3

import rospy
import time
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

class NavigationPointsPublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('navigation_points_publisher', anonymous=True)
        
        # 创建move_base动作客户端
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # 等待move_base服务器启动
        rospy.loginfo("等待move_base服务器...")
        self.client.wait_for_server()
        rospy.loginfo("move_base服务器已连接")
        
        # 导航点列表，每个点包含[x, y, z, qx, qy, qz, qw]
        # qx, qy, qz, qw是四元数表示的方向
        self.navigation_points = [
            [1.8781311511993408, 1.6326346397399902, 0.0, 0.0, 0.0, -0.9999505169677922, 0.009948045830463183], 
            [0.13985848426818848, 5.300715446472168, 0.0, 0.0, 0.0, -0.7019644429977726, 0.7122119914511598],  
            [-2.3496360778808594, 5.279265403747559, 0.0, 0.0, 0.0, -0.707701070070283, 0.706511992411577],  
            [-1.5643374919891357, 2.111314058303833, 0.0, 0.0, 0.0, 0.9998570368713661, 0.016908749770805052], 
            [-3.7587618827819824, 0.8777589797973633, 0.0, 0.0, 0.0, 0.7073405886846846, 0.7068728963535126],  
            [-6.224272727966309, 5.082666397094727, 0.0, 0.0, 0.0, -0.6865675019288319, 0.7270660666646488],  
            [-6.6703715324401855, -0.012180328369140625, 0.0, 0.0, 0.0, 0.3885171508068985, 0.921441492189759], 
            [0.021111249923706055, 0.08142292499542236, 0.0, 0.0, 0.0, 0.014876240667111343, 0.999889342609278], 
        ]
        
        # 每个导航点到达后要输出的信息
        self.arrival_messages = [
            "已到达第一个导航点：任务A位置",
            "已到达第二个导航点：任务B位置",
            "已到达第三个导航点：任务C位置",
            "已到达第四个导航点：任务D位置",
            "已到达第五个导航点：任务E位置",
            "已到达第六个导航点：任务F位置",
            "已到达第七个导航点：任务G位置",
            "已到达第八个导航点：全部任务完成",

        ]
        
        # 发布频率(Hz)
        self.rate = rospy.Rate(10)
        
    def create_goal(self, point):
        """创建导航目标点"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # 设置位置
        goal.target_pose.pose.position.x = point[0]
        goal.target_pose.pose.position.y = point[1]
        goal.target_pose.pose.position.z = point[2]
        
        # 设置方向(四元数)
        goal.target_pose.pose.orientation.x = point[3]
        goal.target_pose.pose.orientation.y = point[4]
        goal.target_pose.pose.orientation.z = point[5]
        goal.target_pose.pose.orientation.w = point[6]
        
        return goal
    
    def send_point(self, point_index):
        """发送单个导航点"""
        point = self.navigation_points[point_index]
        goal = self.create_goal(point)
        rospy.loginfo(f"发送导航点 {point_index+1}/{len(self.navigation_points)}: x={point[0]}, y={point[1]}, qw={point[6]}")
        
        # 发送目标点
        self.client.send_goal(goal)
        
        # 等待导航结果
        wait = self.client.wait_for_result(timeout=rospy.Duration(60.0))
        
        if not wait:
            rospy.logerr("导航超时，未能到达目标点")
            return False
        else:
            result = self.client.get_result()
            if result:
                rospy.loginfo("成功到达目标点")
                # 输出到达信息
                self.print_arrival_message(point_index)
                return True
            else:
                rospy.logerr("导航失败")
                return False
    
    def print_arrival_message(self, point_index):
        """打印导航点到达信息"""
        if 0 <= point_index < len(self.arrival_messages):
            rospy.loginfo(f"===== {self.arrival_messages[point_index]} =====")
        else:
            rospy.loginfo("===== 已到达导航点 =====")
    
    def run(self):
        """运行导航点发布器"""
        try:
            # 遍历所有导航点
            for i in range(len(self.navigation_points)):
                rospy.loginfo(f"前往导航点 {i+1}/{len(self.navigation_points)}")
                success = self.send_point(i)
                
                if not success:
                    rospy.logwarn("导航到当前点失败，尝试下一个点")
                
                # 在每个点之间等待一小段时间
                rospy.sleep(1.0)
                
        except rospy.ROSInterruptException:
            rospy.loginfo("程序被中断")

if __name__ == '__main__':
    try:
        publisher = NavigationPointsPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("程序已退出")
