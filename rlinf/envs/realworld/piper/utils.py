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

"""Piper 机械臂专用工具函数。

包含 Piper SDK 原始 CAN 数据与 SI 单位之间的双向转换、
关节软限位裁剪、双臂 qpos 拼接/拆分等硬件专属逻辑。

通用几何工具（四元数插值、齐次变换等）请直接使用:
    ``rlinf.envs.realworld.franka.utils`` 中的
    ``normalize``, ``quat_slerp``, ``construct_adjoint_matrix``,
    ``construct_homogeneous_matrix`` 等函数，避免重复造轮子。
"""

import numpy as np

# ---------------------------------------------------------------------------
# Piper SDK 单位转换常量
# 参考: piper_ros/src/piper/scripts/piper_ctrl_single_node.py
# ---------------------------------------------------------------------------

# 关节角度: SDK 原始值 = 度 * 1000
# 转弧度: raw / 1000 * (π / 180) ≈ raw * 0.017453293e-3
_DEG_TO_RAD: float = np.pi / 180.0
_JOINT_RAW_TO_RAD: float = _DEG_TO_RAD / 1000.0  # raw → rad
_JOINT_RAD_TO_RAW: float = 1000.0 / _DEG_TO_RAD  # rad → raw  (= 1000 * 180 / π)

# 夹爪角度: SDK 原始值 = 角度(rad) * 1e6
_GRIPPER_RAW_TO_RAD: float = 1e-6  # raw → rad
_GRIPPER_RAD_TO_RAW: float = 1e6  # rad → raw

# 末端位置: SDK 原始值 = 米 * 1e6
_POS_RAW_TO_M: float = 1e-6  # raw → m
_POS_M_TO_RAW: float = 1e6  # m → raw

# 末端姿态(欧拉角): SDK 原始值 = 度 * 1000
_EULER_RAW_TO_RAD: float = _DEG_TO_RAD / 1000.0  # raw → rad
_EULER_RAD_TO_RAW: float = 1000.0 / _DEG_TO_RAD  # rad → raw

# 关节速度: SDK 原始值 = speed * 1000
_VEL_RAW_TO_SI: float = 1e-3  # raw → SI
_VEL_SI_TO_RAW: float = 1e3  # SI → raw

# 关节力矩: SDK 原始值 = effort * 1000
_EFFORT_RAW_TO_SI: float = 1e-3  # raw → SI

# ---------------------------------------------------------------------------
# Piper 6-DOF 关节物理限位 (rad)
# 参考: piper_env.py 中的 min_qpos / max_qpos（前6个为关节，后2个为夹爪）
# ---------------------------------------------------------------------------

PIPER_JOINT_LIMITS_LOW: np.ndarray = np.array(
    [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944], dtype=np.float64
)
PIPER_JOINT_LIMITS_HIGH: np.ndarray = np.array(
    [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944], dtype=np.float64
)

# 夹爪范围 (m): 0 ~ 0.07m（由 Gripper.srv 定义）
PIPER_GRIPPER_MIN: float = 0.0
PIPER_GRIPPER_MAX: float = 0.07


# =========================================================================
# SDK 原始值 → SI 单位 (用于 ROS 回调中解析状态)
# =========================================================================


def joint_raw_to_rad(raw_values: np.ndarray) -> np.ndarray:
    """将 Piper SDK 关节角原始值转换为弧度。

    SDK 原始值 = 度 * 1000，转换公式: raw / 1000 * (π/180)

    Args:
        raw_values: 长度为 6 的 SDK 原始关节角数组。

    Returns:
        长度为 6 的弧度数组。
    """
    return np.asarray(raw_values, dtype=np.float64) * _JOINT_RAW_TO_RAD


def gripper_raw_to_rad(raw_value: float) -> float:
    """将 Piper SDK 夹爪原始值转换为弧度。

    SDK 原始值 = rad * 1e6

    Args:
        raw_value: SDK 夹爪原始值。

    Returns:
        夹爪角度 (rad)。
    """
    return float(raw_value) * _GRIPPER_RAW_TO_RAD


def endpose_raw_to_si(
    raw_xyz: np.ndarray, raw_rpy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """将 Piper SDK 末端位姿原始值转换为 SI 单位。

    位置: raw / 1e6 → m
    姿态: raw / 1000 → deg → rad

    Args:
        raw_xyz: (X_axis, Y_axis, Z_axis) SDK 原始位置值。
        raw_rpy: (RX_axis, RY_axis, RZ_axis) SDK 原始姿态值。

    Returns:
        (xyz_m, rpy_rad): 位置 (m) 和欧拉角 (rad) 的元组。
    """
    xyz_m = np.asarray(raw_xyz, dtype=np.float64) * _POS_RAW_TO_M
    rpy_rad = np.asarray(raw_rpy, dtype=np.float64) * _EULER_RAW_TO_RAD
    return xyz_m, rpy_rad


def vel_raw_to_si(raw_values: np.ndarray) -> np.ndarray:
    """将 Piper SDK 关节速度原始值转换为 SI 单位。

    Args:
        raw_values: 长度为 6 的 SDK 原始速度数组。

    Returns:
        长度为 6 的速度数组 (SI)。
    """
    return np.asarray(raw_values, dtype=np.float64) * _VEL_RAW_TO_SI


def effort_raw_to_si(raw_values: np.ndarray) -> np.ndarray:
    """将 Piper SDK 关节力矩原始值转换为 SI 单位。

    Args:
        raw_values: 长度为 6 或 7 的 SDK 原始力矩数组。

    Returns:
        相同长度的力矩数组 (SI)。
    """
    return np.asarray(raw_values, dtype=np.float64) * _EFFORT_RAW_TO_SI


# =========================================================================
# SI 单位 → SDK 原始值 (用于 Controller 发送指令)
# =========================================================================


def joint_rad_to_raw(rad_values: np.ndarray) -> np.ndarray:
    """将弧度关节角转换为 Piper SDK 原始值（用于发送指令）。

    Args:
        rad_values: 长度为 6 的弧度数组。

    Returns:
        长度为 6 的 SDK 原始值数组（整数）。
    """
    return np.round(
        np.asarray(rad_values, dtype=np.float64) * _JOINT_RAD_TO_RAW
    ).astype(np.int64)


def gripper_rad_to_raw(rad_value: float) -> int:
    """将弧度夹爪角度转换为 Piper SDK 原始值。

    Args:
        rad_value: 夹爪角度 (rad)。

    Returns:
        SDK 夹爪原始值（整数）。
    """
    return int(round(float(rad_value) * _GRIPPER_RAD_TO_RAW))


def endpose_si_to_raw(
    xyz_m: np.ndarray, rpy_rad: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """将 SI 单位末端位姿转换为 Piper SDK 原始值（用于发送指令）。

    Args:
        xyz_m: (x, y, z) 位置 (m)。
        rpy_rad: (roll, pitch, yaw) 欧拉角 (rad)。

    Returns:
        (raw_xyz, raw_rpy): SDK 原始值元组（整数数组）。
    """
    raw_xyz = np.round(
        np.asarray(xyz_m, dtype=np.float64) * _POS_M_TO_RAW
    ).astype(np.int64)
    raw_rpy = np.round(
        np.asarray(rpy_rad, dtype=np.float64) * _EULER_RAD_TO_RAW
    ).astype(np.int64)
    return raw_xyz, raw_rpy


# =========================================================================
# 关节限位与安全裁剪
# =========================================================================


def clip_joint_positions(
    joint_positions: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    """将关节角度裁剪到 Piper 物理限位范围内。

    Args:
        joint_positions: 长度为 6 的关节角度数组 (rad)。
        low: 下限数组，默认使用 ``PIPER_JOINT_LIMITS_LOW``。
        high: 上限数组，默认使用 ``PIPER_JOINT_LIMITS_HIGH``。

    Returns:
        裁剪后的关节角度数组。
    """
    if low is None:
        low = PIPER_JOINT_LIMITS_LOW
    if high is None:
        high = PIPER_JOINT_LIMITS_HIGH
    return np.clip(joint_positions, low, high)


def clip_gripper(
    gripper_value: float,
    low: float = PIPER_GRIPPER_MIN,
    high: float = PIPER_GRIPPER_MAX,
) -> float:
    """将夹爪值裁剪到有效范围内。

    Args:
        gripper_value: 夹爪值。
        low: 下限，默认 0.0。
        high: 上限，默认 0.07m。

    Returns:
        裁剪后的夹爪值。
    """
    return float(np.clip(gripper_value, low, high))


def check_joint_limits(
    joint_positions: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    tolerance: float = 0.0,
) -> tuple[bool, np.ndarray]:
    """检查关节角度是否在限位范围内。

    Args:
        joint_positions: 长度为 6 的关节角度数组 (rad)。
        low: 下限数组，默认使用 ``PIPER_JOINT_LIMITS_LOW``。
        high: 上限数组，默认使用 ``PIPER_JOINT_LIMITS_HIGH``。
        tolerance: 容差 (rad)，允许超出限位的范围。

    Returns:
        (is_safe, violations): 是否安全的布尔值，以及每个关节的越限量数组
        （正值表示超出上限，负值表示低于下限，零表示在范围内）。
    """
    if low is None:
        low = PIPER_JOINT_LIMITS_LOW
    if high is None:
        high = PIPER_JOINT_LIMITS_HIGH
    pos = np.asarray(joint_positions, dtype=np.float64)
    violations = np.zeros_like(pos)
    violations = np.where(pos > high + tolerance, pos - high, violations)
    violations = np.where(pos < low - tolerance, pos - low, violations)
    is_safe = bool(np.all(np.abs(violations) == 0))
    return is_safe, violations


# =========================================================================
# 双臂 qpos 拼接与拆分
# =========================================================================


def concat_dual_arm_qpos(
    left_position: np.ndarray, right_position: np.ndarray
) -> np.ndarray:
    """拼接双臂关节状态为统一的 qpos 向量。

    对应 collect_data_jeff.py 第 348 行的逻辑:
    ``np.concatenate((puppet_arm_left.position, puppet_arm_right.position))``

    每条臂为 7 维 (6 关节 + 1 夹爪)，拼接后为 14 维。

    Args:
        left_position: 左臂关节状态，长度为 7。
        right_position: 右臂关节状态，长度为 7。

    Returns:
        长度为 14 的拼接数组。
    """
    left = np.asarray(left_position, dtype=np.float64)
    right = np.asarray(right_position, dtype=np.float64)
    return np.concatenate([left, right])


def split_dual_arm_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 14 维的双臂动作拆分为左右臂。

    Args:
        action: 长度为 14 的动作数组。

    Returns:
        (left_action, right_action): 各长度为 7 的左右臂动作。
    """
    action = np.asarray(action, dtype=np.float64)
    return action[:7], action[7:14]


# =========================================================================
# 数据质量校验
# =========================================================================


def validate_qpos_trajectory(
    qpos_array: np.ndarray, std_threshold: float = 1e-4
) -> bool:
    """校验 qpos 轨迹数据是否有效（非静止）。

    对应 collect_data_jeff.py 第 24-26 行的逻辑:
    检查左右臂的 qpos 标准差是否过低（表示机械臂未运动）。

    Args:
        qpos_array: 形状为 (T, 14) 的轨迹数组。
        std_threshold: 标准差阈值，低于此值认为数据无效。

    Returns:
        True 表示数据有效，False 表示存在静止臂。
    """
    qpos = np.asarray(qpos_array, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 14:
        return False
    qpos_std = qpos.std(axis=0)
    left_static = np.any(qpos_std[:7] < std_threshold)
    right_static = np.any(qpos_std[7:14] < std_threshold)
    return not (left_static and right_static)
