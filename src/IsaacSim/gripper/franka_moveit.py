#!/usr/bin/env python3
import rclpy
import sys
import moveit_commander
from geometry_msgs.msg import Pose
from rclpy.node import Node

class MoveItPoseGoalNode(Node):
    def __init__(self):
        super().__init__('moveit_pose_goal_node')
        
        # MoveIt Commander 초기화
        moveit_commander.roscpp_initialize(sys.argv)
        
        # 로봇 플래닝 그룹(arm)에 대한 MoveGroupCommander 인스턴스 생성
        # 그룹 이름이 다르면 ('panda_arm' 대신) 수정해야 합니다.
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
        # 로봇의 현재 상태 표시 (디버깅용)
        self.get_logger().info(f"End effector link: {self.move_group.get_end_effector_link()}")
        self.get_logger().info(f"Planning frame: {self.move_group.get_planning_frame()}")

    def send_pose_goal(self, x, y, z, roll, pitch, yaw):
        self.get_logger().info(f"Sending goal: P(x,y,z) = ({x}, {y}, {z}), O(R,P,Y) = ({roll}, {pitch}, {yaw})")

        # 1. Pose 메시지 생성
        pose_goal = Pose()
        
        # 2. 위치 설정 (x, y, z)
        pose_goal.position.x = float(x)
        pose_goal.position.y = float(y)
        pose_goal.position.z = float(z)

        # 3. 방향 설정 (Roll, Pitch, Yaw -> Quaternion)
        # RPY (0, 0, 0)은 Quaternion (x=0, y=0, z=0, w=1)에 해당합니다.
        # 이 스크립트는 RPY를 Quaternion으로 변환합니다.
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(float(roll), float(pitch), float(yaw))
        
        pose_goal.orientation.x = q[0]
        pose_goal.orientation.y = q[1]
        pose_goal.orientation.z = q[2]
        pose_goal.orientation.w = q[3]

        # 4. MoveIt!에 목표 설정
        self.move_group.set_pose_target(pose_goal)

        # 5. 계획 및 실행
        self.get_logger().info("Planning and executing...")
        plan = self.move_group.go(wait=True)
        
        if plan:
            self.get_logger().info("Execution successful!")
        else:
            self.get_logger().error("Execution failed!")

        # 6. 정리
        self.move_group.stop()
        self.move_group.clear_pose_targets()

def main(args=None):
    rclpy.init(args=args)
    
    # 이 스크립트를 실행할 때 명령줄 인자를 받도록 설정
    if len(sys.argv) < 7:
        print("Usage: python3 move_pose.py <x> <y> <z> <roll> <pitch> <yaw>")
        return

    x, y, z, roll, pitch, yaw = sys.argv[1:7]
    
    pose_goal_node = MoveItPoseGoalNode()
    
    try:
        pose_goal_node.send_pose_goal(x, y, z, roll, pitch, yaw)
    except Exception as e:
        pose_goal_node.get_logger().error(f"Error during execution: {e}")
    finally:
        pose_goal_node.get_logger().info("Shutting down MoveIt commander.")
        moveit_commander.roscpp_shutdown()
        pose_goal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
