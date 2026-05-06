# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script for Piper controller.

This script tests the Piper dual-arm controller functionality:
- Connection to ROS topics
- Reading joint states and end-effector poses
- Basic motion commands (joint control)
- Gripper control

Prerequisites:
    IMPORTANT: You MUST start ROS master and Piper nodes BEFORE running this script!
    
    Terminal 1: Start roscore
        roscore
    
    Terminal 2: Start Piper ROS nodes
        source /opt/ros/noetic/setup.bash
        source /opt/venv/piper/setup_piper_ros.sh
        roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1
    
    Wait for the launch to complete (you should see "使能状态: True"), then run this script.

Usage:
    python test_piper_controller.py

Commands:
    q              - Quit
    getpos         - Get current TCP pose (quaternion) for both arms
    getpos_euler   - Get current TCP pose (euler angles) for both arms
    getjoint       - Get current joint positions for both arms
    gripper_open   - Open grippers on both arms
    gripper_close  - Close grippers on both arms
    move_left <j1> <j2> <j3> <j4> <j5> <j6>  - Move left arm to joint positions (rad)
    move_right <j1> <j2> <j3> <j4> <j5> <j6> - Move right arm to joint positions (rad)
    move_delta <dj>  - Move all joints by delta (rad)
    home           - Move both arms to home position (all zeros)
"""

import os
import sys
import time

# Add piper_ros devel Python packages to path (for piper_msgs.srv)
piper_ros_python_paths = [
    "/opt/venv/piper/piper_ws/Piper_ros_private-ros-noetic-interrupt/devel/lib/python3/dist-packages",
]
for piper_ros_python_path in piper_ros_python_paths:
    if os.path.exists(piper_ros_python_path) and piper_ros_python_path not in sys.path:
        sys.path.insert(0, piper_ros_python_path)
        break

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.piper.piper_controller import PiperController


def print_help():
    """Print available commands."""
    print("\n" + "=" * 60)
    print("Available Commands:")
    print("=" * 60)
    print("  q              - Quit")
    print("  getpos         - Get current TCP pose (quaternion)")
    print("  getpos_euler   - Get current TCP pose (euler angles)")
    print("  getjoint       - Get current joint positions")
    print("  gripper_open   - Open grippers")
    print("  gripper_close  - Close grippers")
    print("  move_left <j1> <j2> <j3> <j4> <j5> <j6>  - Move left arm to joint positions (rad)")
    print("  move_right <j1> <j2> <j3> <j4> <j5> <j6> - Move right arm to joint positions (rad)")
    print("  move_delta <dj>  - Move all joints by delta (rad)")
    print("  home           - Move both arms to home position (all zeros)")
    print("  help           - Show this help message")
    print("=" * 60 + "\n")


def main():
    """Main test function."""
    # Get ROS namespaces from environment or use defaults
    ns_left = os.environ.get("PIPER_NS_LEFT", "/puppet_left")
    ns_right = os.environ.get("PIPER_NS_RIGHT", "/puppet_right")
    
    print("\n" + "=" * 60)
    print("Piper Controller Test")
    print("=" * 60)
    print(f"Left arm namespace:  {ns_left}")
    print(f"Right arm namespace: {ns_right}")
    print()
    print("IMPORTANT: Make sure roscore and roslaunch are running!")
    print("  Terminal 1: roscore")
    print("  Terminal 2: roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1")
    print()
    print("Initializing controller...")
    
    # Initialize controller
    controller = PiperController(
        ns_left=ns_left,
        ns_right=ns_right,
        use_robot_base=False,
        joint_speed_pct=50,
    )
    
    # Wait for robot to be ready
    print("Waiting for Piper robot to be ready...")
    start_time = time.time()
    
    while not controller.is_robot_up():
        elapsed = time.time() - start_time
        if elapsed > 30:
            print(f"\n✗ ERROR: Robot not ready after {elapsed:.1f} seconds")
            print("Please check:")
            print("  1. Is roscore running?")
            print("     Terminal 1: roscore")
            print("  2. Is roslaunch running in another terminal?")
            print("     Terminal 2: roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1")
            print("  3. Did you wait for it to fully start? (看到 '使能状态: True')")
            print("  4. Are the ROS topics publishing?")
            print(f"     rostopic list | grep {ns_left}")
            print(f"     rostopic list | grep {ns_right}")
            print("  5. Check ROS_MASTER_URI:")
            print(f"     echo $ROS_MASTER_URI")
            return 1
        
        if int(elapsed) % 5 == 0 and elapsed > 0:
            print(f"  Still waiting... ({elapsed:.0f}s)")
        
        time.sleep(0.5)
    
    print(f"✓ Robot ready! (took {time.time() - start_time:.1f}s)\n")
    
    # Show help
    print_help()
    
    # Interactive command loop
    while True:
        try:
            cmd_str = input("piper> ").strip().lower()
            
            if cmd_str == "q" or cmd_str == "quit":
                print("Exiting...")
                break
            
            elif cmd_str == "help" or cmd_str == "h":
                print_help()
            
            elif cmd_str == "getpos":
                state_left = controller.get_left_state()
                state_right = controller.get_right_state()
                
                print("\nLeft arm TCP pose (x, y, z, qx, qy, qz, qw):")
                print(f"  {state_left.tcp_pose}")
                print("\nRight arm TCP pose (x, y, z, qx, qy, qz, qw):")
                print(f"  {state_right.tcp_pose}")
            
            elif cmd_str == "getpos_euler":
                state_left = controller.get_left_state()
                state_right = controller.get_right_state()
                
                # Convert quaternion to euler angles
                tcp_pose_left = state_left.tcp_pose
                r_left = R.from_quat(tcp_pose_left[3:].copy())
                euler_left = r_left.as_euler("xyz")
                
                tcp_pose_right = state_right.tcp_pose
                r_right = R.from_quat(tcp_pose_right[3:].copy())
                euler_right = r_right.as_euler("xyz")
                
                print("\nLeft arm TCP pose (x, y, z, roll, pitch, yaw):")
                print(f"  {np.concatenate([tcp_pose_left[:3], euler_left])}")
                print("\nRight arm TCP pose (x, y, z, roll, pitch, yaw):")
                print(f"  {np.concatenate([tcp_pose_right[:3], euler_right])}")
            
            elif cmd_str == "getjoint":
                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()
                
                print("\nLeft arm joint positions (6 joints):")
                print(f"  Joints: {state_left['arm_joint_position']}")
                print(f"  Gripper: {state_left['gripper_position']}")
                print("\nRight arm joint positions (6 joints):")
                print(f"  Joints: {state_right['arm_joint_position']}")
                print(f"  Gripper: {state_right['gripper_position']}")
            
            elif cmd_str.startswith("move_left "):
                # Parse joint positions: move_left j1 j2 j3 j4 j5 j6
                parts = cmd_str.split()[1:]
                if len(parts) != 6:
                    print("Usage: move_left <j1> <j2> <j3> <j4> <j5> <j6>")
                    print("  Example: move_left 0.0 0.5 0.0 0.0 0.0 0.0")
                    continue
                try:
                    joints = np.array([float(p) for p in parts])
                    state_left = controller.get_left_state().snapshot()
                    gripper = state_left['gripper_position']
                    left_action = np.append(joints, gripper)
                    
                    print(f"Moving left arm to: {joints}")
                    controller.move_left_arm(left_action)
                    time.sleep(1.0)
                    
                    # Verify position
                    new_state = controller.get_left_state().snapshot()
                    print(f"✓ Left arm moved to: {new_state['arm_joint_position']}")
                except ValueError:
                    print("Error: Invalid joint values. Please provide 6 float numbers.")
            
            elif cmd_str.startswith("move_right "):
                # Parse joint positions: move_right j1 j2 j3 j4 j5 j6
                parts = cmd_str.split()[1:]
                if len(parts) != 6:
                    print("Usage: move_right <j1> <j2> <j3> <j4> <j5> <j6>")
                    print("  Example: move_right 0.0 0.5 0.0 0.0 0.0 0.0")
                    continue
                try:
                    joints = np.array([float(p) for p in parts])
                    state_right = controller.get_right_state().snapshot()
                    gripper = state_right['gripper_position']
                    right_action = np.append(joints, gripper)
                    
                    print(f"Moving right arm to: {joints}")
                    controller.move_right_arm(right_action)
                    time.sleep(1.0)
                    
                    # Verify position
                    new_state = controller.get_right_state().snapshot()
                    print(f"✓ Right arm moved to: {new_state['arm_joint_position']}")
                except ValueError:
                    print("Error: Invalid joint values. Please provide 6 float numbers.")
            
            elif cmd_str.startswith("move_delta "):
                # Move all joints by a delta value
                parts = cmd_str.split()[1:]
                if len(parts) != 1:
                    print("Usage: move_delta <delta_rad>")
                    print("  Example: move_delta 0.1  (move all joints by 0.1 rad)")
                    continue
                try:
                    delta = float(parts[0])
                    state_left = controller.get_left_state().snapshot()
                    state_right = controller.get_right_state().snapshot()
                    
                    left_joints = state_left['arm_joint_position'] + delta
                    right_joints = state_right['arm_joint_position'] + delta
                    
                    left_action = np.append(left_joints, state_left['gripper_position'])
                    right_action = np.append(right_joints, state_right['gripper_position'])
                    
                    print(f"Moving both arms by delta={delta} rad...")
                    controller.move_arm(left_action, right_action)
                    time.sleep(1.0)
                    
                    new_left = controller.get_left_state().snapshot()
                    new_right = controller.get_right_state().snapshot()
                    print(f"✓ Left arm:  {new_left['arm_joint_position']}")
                    print(f"✓ Right arm: {new_right['arm_joint_position']}")
                except ValueError:
                    print("Error: Invalid delta value. Please provide a float number.")
            
            elif cmd_str == "home":
                print("Moving both arms to home position (all zeros)...")
                home_joints = np.zeros(6)
                
                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()
                
                left_action = np.append(home_joints, state_left['gripper_position'])
                right_action = np.append(home_joints, state_right['gripper_position'])
                
                controller.move_arm(left_action, right_action)
                time.sleep(2.0)
                
                new_left = controller.get_left_state().snapshot()
                new_right = controller.get_right_state().snapshot()
                print(f"✓ Left arm:  {new_left['arm_joint_position']}")
                print(f"✓ Right arm: {new_right['arm_joint_position']}")
            
            elif cmd_str == "gripper_open":
                print("Opening grippers...")
                # 通过关节话题控制夹爪：获取当前关节位置，修改夹爪值后发送
                # 夹爪值: 0.07m = fully open (对应 gripper_angle)
                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()
                
                left_action = np.append(state_left['arm_joint_position'], 0.07)
                right_action = np.append(state_right['arm_joint_position'], 0.07)
                
                controller.move_arm(left_action, right_action)
                time.sleep(0.5)
                print("✓ Grippers opened (via joint topic)")
            
            elif cmd_str == "gripper_close":
                print("Closing grippers...")
                # 夹爪值: 0.0m = fully closed
                state_left = controller.get_left_state().snapshot()
                state_right = controller.get_right_state().snapshot()
                
                left_action = np.append(state_left['arm_joint_position'], 0.0)
                right_action = np.append(state_right['arm_joint_position'], 0.0)
                
                controller.move_arm(left_action, right_action)
                time.sleep(0.5)
                print("✓ Grippers closed (via joint topic)")
            
            elif cmd_str == "":
                # Empty input, just continue
                continue
            
            else:
                print(f"Unknown command: '{cmd_str}'")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n✗ Error executing command: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nTest completed.")
    return 0


if __name__ == "__main__":
    exit(main())
