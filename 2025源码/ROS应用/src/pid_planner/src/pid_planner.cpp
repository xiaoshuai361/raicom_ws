// //// 完整版
#include "pid_planner.h"
#include <pluginlib/class_list_macros.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>


PLUGINLIB_EXPORT_CLASS( pid_planner::PIDPlanner, nav_core::BaseLocalPlanner)


// PID控制参数定义
double Kp = 1.4;  // 比例系数
double Ki = 0.0;  // 积分系数
double Kd = 1.1;  // 微分系数

// PID控制所需变量
double angular_error = 0;   // 当前误差
double last_error = 0;      // 上一次误差
double error_sum = 0;       // 误差累积
double error_diff = 0;      // 误差变化率
double output = 0;          // PID输出值

namespace pid_planner
{
    PIDPlanner::PIDPlanner()
    {
        setlocale(LC_ALL, "");
    }
    PIDPlanner::~PIDPlanner()
    {}

    tf::TransformListener* tf_listener_;
    costmap_2d::Costmap2DROS* costmap_ros_;
    void PIDPlanner::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros)
    {

        tf_listener_ = new tf::TransformListener();
        costmap_ros_ = costmap_ros;
    }

    std::vector<geometry_msgs::PoseStamped> global_plan_;
    int target_index_;
    bool pose_adjusting_;
    bool goal_reached_;
    bool PIDPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
    {
        target_index_ = 0;
        global_plan_ = plan;
        pose_adjusting_ = false;
        goal_reached_ = false;
        return true;
    }

    bool PIDPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        //获取代价地图数据
        costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
        unsigned char* map_data = costmap->getCharMap();
        unsigned int size_x = costmap->getSizeInCellsX();
        unsigned int size_y = costmap->getSizeInCellsY();

        //绘制代价地图
        cv::Mat map_image(size_y,size_x,CV_8UC3,cv::Scalar(128,128,128));
        for(unsigned int y=0;y<size_y;y++)
        {
            for (unsigned int x = 0; x < size_x; x++)
            {
                int map_index=y*size_x+x;
                unsigned char cost = map_data[map_index];
                cv::Vec3b& pixel = map_image.at<cv::Vec3b>(map_index);
                if (cost ==0)//可通行
                {
                    pixel = cv::Vec3b(128,128,128);
                }
                else if (cost == 254)//障碍物
                {
                    pixel = cv::Vec3b(0,0,0);
                }
                else if (cost == 253)//禁止通行
                {
                    pixel = cv::Vec3b(255,255,0);
                }
                else
                {
                    unsigned char blue = 255-cost;
                    unsigned char red = cost;
                    pixel = cv::Vec3b(blue,0,red);
                }
            }
        }
        //代价地图上遍历导航路径点
        for(int i=0;i<global_plan_.size();i++)
        {
            geometry_msgs::PoseStamped pose_odom;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("odom", global_plan_[i], pose_odom);
            double odom_x = pose_odom.pose.position.x;
            double odom_y = pose_odom.pose.position.y;

            double origin_x = costmap->getOriginX();
            double origin_y = costmap->getOriginY();
            double local_x = odom_x - origin_x;
            double local_y = odom_y - origin_y;
            int x = local_x / costmap->getResolution();
            int y = local_y / costmap->getResolution();
            cv::circle(map_image, cv::Point(x,y), 0, cv::Scalar(255,0,255));//导航路径点


            //检测前方路径点是否在禁行区域或者障碍物里
            if(i >= target_index_ && i < target_index_ +10)//当前路径点往后加上10个路径点的范围之内,即为0.5m
            {
                cv::circle(map_image, cv::Point(x,y), 0, cv::Scalar(0,255,255));//检测路径点
                int map_index = y*size_x + x;
                unsigned char cost = map_data[map_index];
                if(cost >= 253)
                {
                    return false;
                }


            }
        }


        int final_index = global_plan_.size()-1;
        geometry_msgs::PoseStamped pose_final;
        tf_listener_->transformPose("base_link",global_plan_[final_index],pose_final);
        if (pose_adjusting_ == false)
        {
            double dx = pose_final.pose.position.x;
            double dy = pose_final.pose.position.y;
            double dist = std::sqrt(dx*dx+dy*dy);
            if(dist< 0.05)
                pose_adjusting_ = true;
        }
        if(pose_adjusting_ == true)
        {
            double final_yaw = tf::getYaw(pose_final.pose.orientation);

            cmd_vel.linear.x = pose_final.pose.position.x *3.5;
            cmd_vel.angular.z = final_yaw * 5.0;
            if(abs(final_yaw) < 0.1)
            {
                goal_reached_ = true;

                ROS_INFO("Goal Reached");
                cmd_vel.linear.x = 0;
                cmd_vel.angular.z = 0;
            }
            return true;
        }
        
        




        geometry_msgs::PoseStamped target_pose;
        for(int i= target_index_;i<global_plan_.size();i++)
        {
            geometry_msgs::PoseStamped pose_base;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("base_link",global_plan_[i],pose_base);
            double dx = pose_base.pose.position.x;
            double dy = pose_base.pose.position.y;
            double dist = std::sqrt(dx*dx+dy*dy);

            if (dist >0.25)
            {
                target_pose = pose_base;
                target_index_ = i;

                break;
            }
            if (i == global_plan_.size()-1)
            {
                target_pose = pose_base;

            }
        }
        cmd_vel.linear.x = target_pose.pose.position.x * 5.0;//修改前进速度
        cmd_vel.linear.y = target_pose.pose.position.y * 4.0;//修改平移速度

        cmd_vel.angular.z = 0.0;//修改旋转速度

        // 计算误差
        angular_error = target_pose.pose.position.y;  // 目标值和当前值的差
        // 计算积分项
        error_sum += angular_error ;
        // 计算微分项
        error_diff = angular_error - last_error;
        // 计算PID控制器最终输出
        output = Kp * angular_error + Ki * error_sum + Kd * error_diff;

        // 角速度使用PID最终输出
        cmd_vel.angular.z = output;
        // 记录误差数值
        last_error = angular_error;



        cv::Mat plan_image(600,600,CV_8UC3,cv::Scalar(0,0,0));
        for(int i=0;i<global_plan_.size();i++)
        {
            geometry_msgs::PoseStamped pose_base;
            global_plan_[i].header.stamp = ros::Time(0);
            tf_listener_->transformPose("base_link",global_plan_[i],pose_base);
            int cv_x=300-pose_base.pose.position.y*100;
            int cv_y=300-pose_base.pose.position.x*100;

            cv::circle(plan_image,cv::Point(300,300),15,cv::Scalar(0,255,0));
            cv::line(plan_image,cv::Point(65,300),cv::Point(510,300),cv::Scalar(0,255,0),1);
            cv::line(plan_image,cv::Point(300,45),cv::Point(300,555),cv::Scalar(0,255,0),1);

            cv::circle(plan_image,cv::Point(cv_x,cv_y),1,cv::Scalar(255,0,255));
        }

        cv::waitKey(1);


        return true;
    }

    bool PIDPlanner::isGoalReached()
    {
        return goal_reached_;
    }


}










