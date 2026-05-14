#!/usr/bin/env python3
import math

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_euler


def make_pose():
    x = float(rospy.get_param("~x", 0.0))
    y = float(rospy.get_param("~y", 0.0))
    z = float(rospy.get_param("~z", 0.0))
    yaw = float(rospy.get_param("~yaw", 0.0))
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def delete_existing_models(model_names, get_world_properties, delete_model):
    world = get_world_properties()
    existing = set(world.model_names)
    for model_name in model_names:
        if model_name in existing:
            try:
                result = delete_model(model_name)
                if result.success:
                    rospy.loginfo("[spawn] 已删除旧模型: %s", model_name)
                else:
                    rospy.logwarn("[spawn] 删除旧模型失败: %s, %s", model_name, result.status_message)
            except rospy.ServiceException as exc:
                rospy.logwarn("[spawn] 删除旧模型异常: %s, %s", model_name, exc)


def main():
    rospy.init_node("spawn_robot_model")

    model_name = rospy.get_param("~model_name")
    robot_param = rospy.get_param("~robot_param", "robot_description")
    reference_frame = rospy.get_param("~reference_frame", "world")
    cleanup_models = rospy.get_param("~cleanup_models", [])

    if not cleanup_models:
        cleanup_models = [model_name]
    elif model_name not in cleanup_models:
        cleanup_models.append(model_name)

    timeout = float(rospy.get_param("~service_timeout", 15.0))
    rospy.wait_for_service("/gazebo/get_world_properties", timeout=timeout)
    rospy.wait_for_service("/gazebo/delete_model", timeout=timeout)
    rospy.wait_for_service("/gazebo/spawn_urdf_model", timeout=timeout)

    get_world_properties = rospy.ServiceProxy("/gazebo/get_world_properties", GetWorldProperties)
    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)

    delete_existing_models(cleanup_models, get_world_properties, delete_model)

    xml = rospy.get_param(robot_param, "")
    if not xml:
        rospy.logerr("[spawn] 参数 %s 为空，无法生成模型。", robot_param)
        raise SystemExit(2)

    pose = make_pose()
    try:
        result = spawn_model(model_name, xml, rospy.get_namespace(), pose, reference_frame)
    except rospy.ServiceException as exc:
        rospy.logerr("[spawn] 生成模型失败: %s", exc)
        raise SystemExit(3)

    if not result.success:
        rospy.logerr("[spawn] 生成模型失败: %s", result.status_message)
        raise SystemExit(4)

    rospy.loginfo("[spawn] 已生成模型 %s 于 x=%.2f y=%.2f yaw=%.2f", model_name, pose.position.x, pose.position.y, math.atan2(2.0 * (pose.orientation.w * pose.orientation.z), 1.0 - 2.0 * pose.orientation.z * pose.orientation.z))


if __name__ == "__main__":
    main()