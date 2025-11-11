#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import geometry_msgs.msg
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, RobotCommander
import moveit_commander

def main():
    rclpy.init()
    moveit_commander.roscpp_initialize([])

    robot = RobotCommander()
    scene = PlanningSceneInterface()
    group = MoveGroupCommander("panda_arm")

    # Print basic info
    print("Reference frame:", group.get_planning_frame())
    print("End effector:", group.get_end_effector_link())
    print("Available groups:", robot.get_group_names())

    # Define a target pose
    pose_target = geometry_msgs.msg.Pose()
    pose_target.orientation.w = 1.0
    pose_target.position.x = 0.4
    pose_target.position.y = 0.1
    pose_target.position.z = 0.4

    group.set_pose_target(pose_target)

    plan = group.plan()
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    moveit_commander.roscpp_shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

