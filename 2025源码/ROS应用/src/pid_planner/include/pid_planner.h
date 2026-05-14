#ifndef PID_PLANNER_H_
#define PID_PLANNER_H_

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>


namespace pid_planner
{
    class PIDPlanner : public nav_core::BaseLocalPlanner
    {
        public:
            PIDPlanner();
            ~PIDPlanner();

            void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);
            bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);
            bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);
            bool isGoalReached();

    };
}

#endif  // PID_PLANNER_H_
