# Copyright 2026 The GIGA Authors.
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

"""Piper dual-arm robot controller.

Communicates with piper_ros nodes via ROS topics and services:
- Subscribes to dual-arm joint states and end-effector poses
- Publishes joint control commands
- Calls enable/gripper/go_zero/stop/reset ROS services

Architecture aligned with ``rlinf.envs.realworld.franka.franka_controller.FrankaController``.
"""

import time
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from rlinf.envs.realworld.common.ros import ROSController
from rlinf.utils.logging import get_logger

from .piper_robot_state import PiperRobotState
from .utils import clip_gripper, clip_joint_positions


class PiperController:
    """Piper dual-arm robot controller.

    Communicates with piper_ros nodes via ROS topics and services.
    Aligned with the data flow in ``collect_data_jeff.py`` (RosOperator)
    and the architecture of ``FrankaController``.

    ROS topic mapping (dual-arm mode, one piper_ctrl_single_node per arm):

    **Subscriptions (state reading):**

    - ``{ns_left}/joint_states_single``  -> left arm JointState (7: 6 joints + 1 gripper)
    - ``{ns_right}/joint_states_single`` -> right arm JointState
    - ``{ns_left}/end_pose``             -> left arm TCP pose PoseStamped (quaternion)
    - ``{ns_right}/end_pose``            -> right arm TCP pose PoseStamped (quaternion)
    - ``/odom`` (optional)               -> base odometry

    **Publications (command output):**

    - ``{ns_left}/joint_ctrl_single``    -> left arm joint control JointState
    - ``{ns_right}/joint_ctrl_single``   -> right arm joint control JointState

    **Service calls:**

    - ``{ns}/enable_srv``   -> enable/disable
    - ``{ns}/gripper_srv``  -> gripper control
    - ``{ns}/go_zero_srv``  -> go to zero position
    - ``{ns}/stop_srv``     -> emergency stop
    - ``{ns}/reset_srv``    -> reset (clear errors)

    Args:
        ns_left: Left arm ROS namespace, default ``/puppet_left``.
        ns_right: Right arm ROS namespace, default ``/puppet_right``.
        use_robot_base: Whether to subscribe to base odometry topic.
        robot_base_topic: Base odometry topic name.
        joint_speed_pct: Joint motion speed percentage (0-100).
    """

    def __init__(
        self,
        ns_left: str = "/puppet_left",
        ns_right: str = "/puppet_right",
        use_robot_base: bool = False,
        robot_base_topic: str = "/odom",
        joint_speed_pct: int = 50,
    ) -> None:
        self._logger = get_logger()
        self._ns_left = ns_left
        self._ns_right = ns_right
        self._use_robot_base = use_robot_base
        self._robot_base_topic = robot_base_topic
        self._joint_speed_pct = joint_speed_pct

        # ---- Dual-arm state ----
        self._state_left = PiperRobotState()
        self._state_right = PiperRobotState()

        # ---- Base velocity ----
        self._base_vel: np.ndarray = np.zeros(2, dtype=np.float64)  # [linear.x, angular.z]

        # ---- ROS controller ----
        self._ros = ROSController()

        # ---- Initialize all ROS channels ----
        self._init_ros_channels()

        self._logger.info(
            f"PiperController initialized: left={ns_left}, right={ns_right}, "
            f"base={use_robot_base}"
        )

    # ==================================================================
    # ROS channel initialization
    # ==================================================================

    def _init_ros_channels(self) -> None:
        """Initialize all ROS subscriptions and publishers.

        Topic layout depends on whether the master-slave launch file is used.
        """
        if self._ns_left == "/puppet_left" and self._ns_right == "/puppet_right":
            # Master-slave mode (start_ms_piper.launch)
            self._left_joint_state_topic = "/puppet/joint_left"
            self._right_joint_state_topic = "/puppet/joint_right"
            self._left_end_pose_topic = "/puppet/end_pose_left"
            self._right_end_pose_topic = "/puppet/end_pose_right"
            self._left_joint_ctrl_topic = "/master/joint_left"
            self._right_joint_ctrl_topic = "/master/joint_right"
        else:
            # Single-node mode
            self._left_joint_state_topic = f"{self._ns_left}/joint_states_single"
            self._right_joint_state_topic = f"{self._ns_right}/joint_states_single"
            self._left_end_pose_topic = f"{self._ns_left}/end_pose"
            self._right_end_pose_topic = f"{self._ns_right}/end_pose"
            self._left_joint_ctrl_topic = f"{self._ns_left}/joint_ctrl_single"
            self._right_joint_ctrl_topic = f"{self._ns_right}/joint_ctrl_single"

        # ---- Subscribe: left arm state ----
        self._ros.connect_ros_channel(
            self._left_joint_state_topic, JointState, self._on_left_joint_state,
        )
        self._ros.connect_ros_channel(
            self._left_end_pose_topic, PoseStamped, self._on_left_end_pose,
        )

        # ---- Subscribe: right arm state ----
        self._ros.connect_ros_channel(
            self._right_joint_state_topic, JointState, self._on_right_joint_state,
        )
        self._ros.connect_ros_channel(
            self._right_end_pose_topic, PoseStamped, self._on_right_end_pose,
        )

        # ---- Subscribe: base odometry (optional) ----
        if self._use_robot_base:
            self._ros.connect_ros_channel(
                self._robot_base_topic, Odometry, self._on_robot_base,
            )

        # ---- Publish: joint control commands ----
        self._ros.create_ros_channel(self._left_joint_ctrl_topic, JointState, queue_size=1)
        self._ros.create_ros_channel(self._right_joint_ctrl_topic, JointState, queue_size=1)

    # ==================================================================
    # ROS callbacks (called from ROS subscriber thread)
    # ==================================================================

    def _on_left_joint_state(self, msg: JointState) -> None:
        """Left arm joint state callback.

        piper_ctrl_single_node publishes joint_states_single at 200Hz,
        containing 7 values: 6 joints + 1 gripper (position/velocity/effort).
        """
        self._state_left.update_joint_state(
            position=np.array(msg.position, dtype=np.float64),
            velocity=np.array(msg.velocity, dtype=np.float64),
            effort=np.array(msg.effort, dtype=np.float64),
        )

    def _on_right_joint_state(self, msg: JointState) -> None:
        """Right arm joint state callback."""
        self._state_right.update_joint_state(
            position=np.array(msg.position, dtype=np.float64),
            velocity=np.array(msg.velocity, dtype=np.float64),
            effort=np.array(msg.effort, dtype=np.float64),
        )

    def _on_left_end_pose(self, msg: PoseStamped) -> None:
        """Left arm TCP pose callback (quaternion).

        piper_ctrl_single_node publishes end_pose at 200Hz.
        Position in meters, orientation as quaternion (qx, qy, qz, qw).
        """
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        self._state_left.update_tcp_pose(pos, quat)

    def _on_right_end_pose(self, msg: PoseStamped) -> None:
        """Right arm TCP pose callback (quaternion)."""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        quat = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ], dtype=np.float64)
        self._state_right.update_tcp_pose(pos, quat)

    def _on_robot_base(self, msg: Odometry) -> None:
        """Base odometry callback.

        Extracts [linear.x, angular.z] aligned with collect_data_jeff.py.
        """
        self._base_vel = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.angular.z],
            dtype=np.float64,
        )

    # ==================================================================
    # State getters
    # ==================================================================

    def get_left_state(self) -> PiperRobotState:
        """Return current left arm state."""
        return self._state_left

    def get_right_state(self) -> PiperRobotState:
        """Return current right arm state."""
        return self._state_right

    def get_base_vel(self) -> np.ndarray:
        """Return base velocity [linear.x, angular.z]."""
        return self._base_vel.copy()

    def get_qpos(self) -> np.ndarray:
        """Return concatenated dual-arm qpos (14D).

        Each arm: 6 joint positions + 1 gripper position.
        """
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        left_qpos = np.append(left_snap["arm_joint_position"], left_snap["gripper_position"])
        right_qpos = np.append(right_snap["arm_joint_position"], right_snap["gripper_position"])
        return np.concatenate([left_qpos, right_qpos])

    def get_qvel(self) -> np.ndarray:
        """Return concatenated dual-arm qvel (14D).

        Gripper velocity is 0.0 (not reported by piper_ros).
        """
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        left_qvel = np.append(left_snap["arm_joint_velocity"], 0.0)
        right_qvel = np.append(right_snap["arm_joint_velocity"], 0.0)
        return np.concatenate([left_qvel, right_qvel])

    def get_effort(self) -> np.ndarray:
        """Return concatenated dual-arm effort (14D)."""
        left_snap = self._state_left.snapshot()
        right_snap = self._state_right.snapshot()
        left_effort = np.append(left_snap["arm_joint_effort"], left_snap["gripper_effort"])
        right_effort = np.append(right_snap["arm_joint_effort"], right_snap["gripper_effort"])
        return np.concatenate([left_effort, right_effort])

    # ==================================================================
    # Control command publishing
    # ==================================================================

    def move_arm(
        self,
        left_action: np.ndarray,
        right_action: np.ndarray,
        left_speed_pct: Optional[int] = None,
        right_speed_pct: Optional[int] = None,
    ) -> None:
        """Publish dual-arm joint control commands.

        Message format expected by piper_start_ms_node_double_agilex_dyn_pos.py:
        - ``position[0:6]``: 6 joint angles (rad)
        - ``position[6]``: gripper angle (rad)
        - ``velocity[6]``: global speed percentage (0-100)
        - ``effort[6]``: gripper torque (N·m)
        - ``header.frame_id``: "Master_is_me" enables gravity compensation on slave arm

        Args:
            left_action: Left arm action, length 7 (6 joints + 1 gripper), unit rad.
            right_action: Right arm action, length 7 (6 joints + 1 gripper), unit rad.
            left_speed_pct: Left arm speed percentage (0-100), None uses default.
            right_speed_pct: Right arm speed percentage (0-100), None uses default.
        """
        left_action = np.asarray(left_action, dtype=np.float64)
        right_action = np.asarray(right_action, dtype=np.float64)

        assert len(left_action) == 7, f"Left arm action must be 7D, got {len(left_action)}"
        assert len(right_action) == 7, f"Right arm action must be 7D, got {len(right_action)}"

        left_action[:6] = clip_joint_positions(left_action[:6])
        right_action[:6] = clip_joint_positions(right_action[:6])
        left_action[6] = clip_gripper(left_action[6])
        right_action[6] = clip_gripper(right_action[6])

        self._publish_joint_ctrl(
            self._left_joint_ctrl_topic,
            left_action,
            left_speed_pct or self._joint_speed_pct,
        )
        self._publish_joint_ctrl(
            self._right_joint_ctrl_topic,
            right_action,
            right_speed_pct or self._joint_speed_pct,
        )

    def move_left_arm(self, action: np.ndarray, speed_pct: Optional[int] = None) -> None:
        """Publish left arm joint control command.

        Args:
            action: Length-7 array (6 joints + 1 gripper), unit rad.
            speed_pct: Speed percentage (0-100), None uses default.
        """
        action = np.asarray(action, dtype=np.float64)
        assert len(action) == 7, f"Action must be 7D, got {len(action)}"
        action[:6] = clip_joint_positions(action[:6])
        action[6] = clip_gripper(action[6])
        self._publish_joint_ctrl(
            self._left_joint_ctrl_topic, action, speed_pct or self._joint_speed_pct,
        )

    def move_right_arm(self, action: np.ndarray, speed_pct: Optional[int] = None) -> None:
        """Publish right arm joint control command.

        Args:
            action: Length-7 array (6 joints + 1 gripper), unit rad.
            speed_pct: Speed percentage (0-100), None uses default.
        """
        action = np.asarray(action, dtype=np.float64)
        assert len(action) == 7, f"Action must be 7D, got {len(action)}"
        action[:6] = clip_joint_positions(action[:6])
        action[6] = clip_gripper(action[6])
        self._publish_joint_ctrl(
            self._right_joint_ctrl_topic, action, speed_pct or self._joint_speed_pct,
        )

    def _publish_joint_ctrl(self, topic: str, action: np.ndarray, speed_pct: int) -> None:
        """Build and publish a JointState control message.

        Message format aligned with joint_callback in piper_start_ms_node_double_agilex_dyn_pos.py:
        - ``position``: [j1..j6, gripper] (rad)
        - ``velocity``: [0..0, speed_pct] — index 6 is global speed percentage
        - ``effort``:   [0..0, gripper_effort] — index 6 is gripper torque (N·m)

        Args:
            topic: Target ROS topic.
            action: Length-7 action array.
            speed_pct: Speed percentage (0-100).
        """
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        msg.position = action.tolist()
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(speed_pct)]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        self._ros.put_channel(topic, msg)

    # ==================================================================
    # ROS service calls
    # ==================================================================

    def enable_arm(self, ns: Optional[str] = None) -> bool:
        """Enable the arm. Enable is handled by the launch file; this is a no-op placeholder."""
        self._logger.info("Arm enable is handled by launch file, skipping enable_arm call.")
        return True

    def disable_arm(self, ns: Optional[str] = None) -> bool:
        """Disable the arm. Disable is handled externally; this is a no-op placeholder."""
        self._logger.info("Arm disable is handled externally, skipping disable_arm call.")
        return True

    def gripper_ctrl(
        self,
        gripper_angle: float = 0.0,
        gripper_effort: float = 1.0,
        gripper_code: int = 0x01,
        set_zero: int = 0x00,
        ns: Optional[str] = None,
    ) -> bool:
        """Control gripper. Gripper is controlled via joint_states; this is a no-op placeholder."""
        self._logger.info("Gripper is controlled via joint_states, skipping gripper_ctrl call.")
        return True

    def go_zero(self, is_mit_mode: bool = False, ns: Optional[str] = None) -> bool:
        """Send both arms to zero joint position with gripper open.

        Args:
            is_mit_mode: Unused.
            ns: Namespace to zero; None zeros both arms.

        Returns:
            True on success.
        """
        import numpy as np

        home_joints = np.zeros(6)
        gripper_open = 0.07

        try:
            if ns is None:
                left_action = np.append(home_joints, gripper_open)
                right_action = np.append(home_joints, gripper_open)
                self.move_arm(left_action, right_action)
                self._logger.info("Both arms zeroed (joints=0, gripper open).")
            elif ns == self._ns_left:
                state_right = self.get_right_state().snapshot()
                left_action = np.append(home_joints, gripper_open)
                right_action = np.append(
                    state_right["arm_joint_position"], state_right["gripper_position"]
                )
                self.move_arm(left_action, right_action)
                self._logger.info("Left arm zeroed.")
            elif ns == self._ns_right:
                state_left = self.get_left_state().snapshot()
                left_action = np.append(
                    state_left["arm_joint_position"], state_left["gripper_position"]
                )
                right_action = np.append(home_joints, gripper_open)
                self.move_arm(left_action, right_action)
                self._logger.info("Right arm zeroed.")
            else:
                self._logger.warning(f"Unknown namespace: {ns}")
                return False
            return True
        except Exception as e:
            self._logger.error(f"go_zero failed: {e}")
            return False

    def stop_arm(self, ns: Optional[str] = None) -> bool:
        """Emergency stop via piper_ros ``stop_srv`` service.

        Args:
            ns: Namespace to stop; None stops both arms.

        Returns:
            True if all stop calls succeeded.
        """
        from std_srvs.srv import Trigger

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/stop_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                resp = rospy.ServiceProxy(service_name, Trigger)()
                if not resp.success:
                    self._logger.warning(f"Stop failed: {service_name}, msg={resp.message}")
                    all_ok = False
                else:
                    self._logger.info(f"Stop succeeded: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"Stop service error {service_name}: {e}")
                all_ok = False
        return all_ok

    def reset_arm(self, ns: Optional[str] = None) -> bool:
        """Reset arm (clear error state) via piper_ros ``reset_srv`` service.

        Args:
            ns: Namespace to reset; None resets both arms.

        Returns:
            True if all reset calls succeeded.
        """
        from std_srvs.srv import Trigger

        namespaces = [ns] if ns else [self._ns_left, self._ns_right]
        all_ok = True
        for arm_ns in namespaces:
            service_name = f"{arm_ns}/reset_srv"
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                resp = rospy.ServiceProxy(service_name, Trigger)()
                if not resp.success:
                    self._logger.warning(f"Reset failed: {service_name}, msg={resp.message}")
                    all_ok = False
                else:
                    self._logger.info(f"Reset succeeded: {service_name}")
            except (rospy.ServiceException, rospy.ROSException) as e:
                self._logger.error(f"Reset service error {service_name}: {e}")
                all_ok = False
        return all_ok

    # ==================================================================
    # High-level interface
    # ==================================================================

    def is_robot_up(self) -> bool:
        """Check whether key ROS topics have received data.

        Returns:
            True if joint state and end-pose topics for both arms have data.
        """
        return (
            self._ros.get_input_channel_status(self._left_joint_state_topic)
            and self._ros.get_input_channel_status(self._right_joint_state_topic)
            and self._ros.get_input_channel_status(self._left_end_pose_topic)
            and self._ros.get_input_channel_status(self._right_end_pose_topic)
        )

    def wait_for_robot(self, timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
        """Block until robot is ready or timeout.

        Args:
            timeout: Maximum wait time (seconds).
            poll_interval: Poll interval (seconds).

        Returns:
            True if robot is ready, False if timed out.
        """
        start_time = time.time()
        while not self.is_robot_up():
            if time.time() - start_time > timeout:
                self._logger.warning(
                    f"Robot ready timeout ({timeout}s). "
                    f"left_joint={self._ros.get_input_channel_status(self._left_joint_state_topic)}, "
                    f"right_joint={self._ros.get_input_channel_status(self._right_joint_state_topic)}, "
                    f"left_pose={self._ros.get_input_channel_status(self._left_end_pose_topic)}, "
                    f"right_pose={self._ros.get_input_channel_status(self._right_end_pose_topic)}"
                )
                return False
            time.sleep(poll_interval)
        self._logger.info("Dual-arm robot ready.")
        return True

    def _wait_for_joint(
        self,
        target_pos: np.ndarray,
        side: str = "left",
        timeout: float = 30.0,
        atol: float = 1e-2,
    ) -> bool:
        """Wait until the specified arm reaches the target joint position.

        Args:
            target_pos: Target joint positions, length 6 (rad).
            side: ``"left"`` or ``"right"``.
            timeout: Maximum wait time (seconds).
            atol: Position tolerance (rad).

        Returns:
            True if target reached, False if timed out.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        state = self._state_left if side == "left" else self._state_right
        wait_step = 0.01
        waited = 0.0
        while waited < timeout:
            current = state.snapshot()["arm_joint_position"]
            if np.allclose(current, target_pos, atol=atol):
                self._logger.debug(f"{side} arm reached target: {current}")
                return True
            time.sleep(wait_step)
            waited += wait_step
        self._logger.warning(f"{side} arm joint wait timed out ({timeout}s).")
        return False
