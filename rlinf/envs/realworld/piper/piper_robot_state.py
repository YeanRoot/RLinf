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

import threading
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class PiperRobotState:
    """Piper 机械臂的完整状态表示。

    包含关节空间（Joint space）和笛卡尔空间（Cartesian space）的状态信息。
    所有角度值使用弧度制，位置值使用米制（SI 单位）。

    Attributes:
        arm_joint_position: 6个关节的角度位置 (rad)，来自 joint_states_single.position[0:6]。
        arm_joint_velocity: 6个关节的角速度 (rad/s)，来自 joint_states_single.velocity[0:6]。
        arm_joint_effort: 6个关节的力矩 (Nm)，来自 joint_states_single.effort[0:6]。
        gripper_position: 夹爪开合角度 (rad)，来自 joint_states_single.position[6]。
        gripper_effort: 夹爪力矩 (Nm)，来自 joint_states_single.effort[6]。
        tcp_pose: 末端执行器位姿 (x, y, z, qx, qy, qz, qw)，来自 end_pose 话题。
                  位置单位为米，姿态为四元数。
        tcp_pose_euler: 末端执行器欧拉角位姿 (x, y, z, roll, pitch, yaw)，
                        来自 end_pose_euler 话题。位置单位为米，角度单位为弧度。
    """

    # ---- Joint space ----
    arm_joint_position: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # joint_states_single.position[0:6]  (rad)
    arm_joint_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # joint_states_single.velocity[0:6]  (rad/s)
    arm_joint_effort: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # joint_states_single.effort[0:6]    (Nm)

    # ---- Gripper ----
    gripper_position: float = 0.0  # joint_states_single.position[6]  (rad)
    gripper_effort: float = 0.0  # joint_states_single.effort[6]    (Nm)

    # ---- Cartesian space (end-effector) ----
    tcp_pose: np.ndarray = field(
        default_factory=lambda: np.zeros(7)
    )  # (x, y, z, qx, qy, qz, qw) from end_pose topic
    tcp_pose_euler: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )  # (x, y, z, roll, pitch, yaw) from end_pose_euler topic

    # ---- Thread safety ----
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Thread-safe update helpers (called from ROS subscriber callbacks)
    # ------------------------------------------------------------------

    def update_joint_state(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        effort: np.ndarray,
    ) -> None:
        """线程安全地更新关节空间状态和夹爪状态。

        Args:
            position: 长度为 7 的数组，前 6 个为关节角度 (rad)，第 7 个为夹爪角度。
            velocity: 长度为 7 的数组，前 6 个为关节角速度 (rad/s)。
            effort: 长度为 7 的数组，前 6 个为关节力矩 (Nm)，第 7 个为夹爪力矩。
        """
        with self._lock:
            self.arm_joint_position = np.array(position[:6], dtype=np.float64)
            self.arm_joint_velocity = np.array(velocity[:6], dtype=np.float64)
            self.arm_joint_effort = np.array(effort[:6], dtype=np.float64)
            self.gripper_position = float(position[6])
            self.gripper_effort = float(effort[6])

    def update_tcp_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        """线程安全地更新末端执行器四元数位姿。

        Args:
            position: (x, y, z) 位置，单位为米。
            quaternion: (qx, qy, qz, qw) 四元数姿态。
        """
        with self._lock:
            self.tcp_pose = np.concatenate(
                [np.asarray(position, dtype=np.float64),
                 np.asarray(quaternion, dtype=np.float64)]
            )

    def update_tcp_pose_euler(
        self,
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
    ) -> None:
        """线程安全地更新末端执行器欧拉角位姿。

        Args:
            x: X 轴位置 (m)。
            y: Y 轴位置 (m)。
            z: Z 轴位置 (m)。
            roll: 绕 X 轴旋转 (rad)。
            pitch: 绕 Y 轴旋转 (rad)。
            yaw: 绕 Z 轴旋转 (rad)。
        """
        with self._lock:
            self.tcp_pose_euler = np.array(
                [x, y, z, roll, pitch, yaw], dtype=np.float64
            )

    # ------------------------------------------------------------------
    # Thread-safe snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """线程安全地获取当前状态的完整快照（可序列化字典）。

        Returns:
            dict: 包含所有状态字段的字典副本。
        """
        with self._lock:
            return {
                "arm_joint_position": self.arm_joint_position.copy(),
                "arm_joint_velocity": self.arm_joint_velocity.copy(),
                "arm_joint_effort": self.arm_joint_effort.copy(),
                "gripper_position": self.gripper_position,
                "gripper_effort": self.gripper_effort,
                "tcp_pose": self.tcp_pose.copy(),
                "tcp_pose_euler": self.tcp_pose_euler.copy(),
            }

    def to_dict(self) -> dict:
        """Convert the dataclass to a serializable dictionary."""
        return self.snapshot()
