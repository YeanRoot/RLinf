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

"""Piper dual-arm robot Gym environment.

Unified RLinf interface coordinating ``PiperController`` and ``PiperRobotState``,
providing standard ``gymnasium.Env`` API (step / reset / observation_space / action_space).

Architecture aligned with ``rlinf.envs.realworld.franka.franka_env.FrankaEnv``.

**Action Space (Joint space):**

- 14D absolute joint positions: left arm 7D (6 joints + 1 gripper) + right arm 7D

**Observation Space:**

- ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
- ``frames``: cam_high(480,640,3), cam_left_wrist(480,640,3), cam_right_wrist(480,640,3)

**Human-in-the-loop:**

- ``page_down``: toggle teleoperation (handled by ROS node via ``/enable_message_publish`` param)
- ``page_up``: toggle policy output; when disabled, env holds current qpos
"""

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

from rlinf.utils.logging import get_logger

from .piper_controller import PiperController
from .piper_robot_state import PiperRobotState
from .utils import (
    PIPER_GRIPPER_MAX,
    PIPER_GRIPPER_MIN,
    PIPER_JOINT_LIMITS_HIGH,
    PIPER_JOINT_LIMITS_LOW,
    clip_gripper,
    clip_joint_positions,
    split_dual_arm_action,
)


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class PiperRobotConfig:
    """Piper dual-arm robot environment configuration.

    Attributes:
        ns_left: Left arm ROS namespace.
        ns_right: Right arm ROS namespace.
        use_robot_base: Whether to use mobile base.
        robot_base_topic: Mobile base odometry topic.
        camera_names: Camera name list.
        img_topics: Camera ROS topic list, one-to-one with camera_names.
        img_resolution: Camera image resolution (H, W).
        obs_img_resolution: Observation image resolution (H, W).
        step_frequency: Control frequency (Hz).
        publish_rate: Publishing frequency (Hz).
        joint_speed_pct: Joint motion speed percentage (0-100).
        is_dummy: Whether in dummy mode (no real hardware).
        max_num_steps: Maximum steps per episode.
        min_qpos: Per-arm joint lower limits (7D: 6 joints + 1 gripper).
        max_qpos: Per-arm joint upper limits.
        enable_camera_player: Whether to enable camera display.
        target_qpos: Target joint positions for reward (14D).
        reward_threshold: Per-joint error threshold for target zone.
        use_dense_reward / dense_reward_scale / success_hold_steps: See ``_calc_step_reward``.
        enable_human_intervention: Whether to enable human-in-the-loop intervention.
        intervention_trigger_key: Key that toggles teleoperation in the ROS node (page_down).
        policy_enable_key: Key that toggles policy output in piper_env (page_up).
        master_joint_topic: ROS topic publishing master arm absolute joint targets.
    """

    # ---- ROS namespaces ----
    ns_left: str = "/puppet_left"
    ns_right: str = "/puppet_right"

    # ---- Mobile base ----
    use_robot_base: bool = False
    robot_base_topic: str = "/odom"

    # ---- Camera configuration ----
    camera_names: list[str] = field(
        default_factory=lambda: ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    )
    img_topics: list[str] = field(
        default_factory=lambda: [
            "/camera_f/color/image_raw",
            "/camera_l/color/image_raw",
            "/camera_r/color/image_raw",
        ]
    )
    img_resolution: tuple[int, int] = (480, 640)  # (H, W) raw image resolution
    obs_img_resolution: tuple[int, int] = (480, 640)  # (H, W) observation resolution

    # ---- Control parameters ----
    step_frequency: float = 30.0  # Hz
    publish_rate: int = 30
    joint_speed_pct: int = 50
    pos_lookahead_step: int = 50
    chunk_size: int = 50

    # ---- Environment parameters ----
    task_name: str = "task"
    is_dummy: bool = False
    max_num_steps: int = 10000
    enable_camera_player: bool = False

    # ---- Joint limits (per arm: 6 joints + 1 gripper) ----
    min_qpos: list[float] = field(
        default_factory=lambda: [
            -2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0, 0.0,
        ]
    )
    max_qpos: list[float] = field(
        default_factory=lambda: [
            2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 1.0, 1.0,
        ]
    )

    # ---- Human-in-the-loop configuration ----
    enable_human_intervention: bool = True
    # page_down: handled by ROS node, toggles /enable_message_publish param
    intervention_trigger_key: str = "Key.page_down"
    # page_up: toggles policy output inside piper_env
    policy_enable_key: str = "Key.page_up"
    # Topic where ROS node publishes master arm absolute joint targets (14D)
    master_joint_topic: str = "/master/joint_states"

    # ---- ZMQ inference service (reserved) ----
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080

    # ---- Joint action semantics ----
    # absolute: policy outputs absolute joint positions
    # delta: policy outputs [-1,1]^14, scaled by delta_action_scale and added to current qpos
    joint_action_mode: str = "absolute"
    delta_action_scale: float = 0.05

    # ---- Reward (joint space, aligned with FrankaEnv) ----
    # Target joint positions (14D: left 7 + right 7), calibrate per task.
    target_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(14, dtype=np.float64)
    )
    # Per-joint absolute error threshold; all joints within threshold = target zone.
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.full(14, 0.1, dtype=np.float64)
    )
    use_dense_reward: bool = False
    # Hold success_hold_steps consecutive steps in target zone to terminate with success.
    success_hold_steps: int = 1
    # Dense reward: exp(-dense_reward_scale * sum_i (qpos_i - target_qpos_i)^2)
    dense_reward_scale: float = 50.0


# =========================================================================
# Environment
# =========================================================================


class PiperEnv(gym.Env):
    """Piper dual-arm robot Gymnasium environment.

    Aligned with ``FrankaEnv`` interface. Communicates with piper_ros via
    ``PiperController``. Images are received via ROS topics using ``cv_bridge``.

    **Data flow:**

    1. ``__init__``: create PiperController → subscribe joint/pose/image topics → wait for robot
    2. ``step(action)``: split 14D action → clip → publish joint command → rate control → return obs
    3. ``reset()``: enable → go_zero → wait for joint → return initial obs
    4. ``_get_observation()``: concatenate qpos/qvel/effort/base_vel + camera images

    Args:
        config: PiperRobotConfig instance.
        worker_info: RLinf WorkerInfo (optional).
        hardware_info: Hardware info (optional, reserved).
        env_idx: Environment instance index.
    """

    def __init__(
        self,
        config: PiperRobotConfig,
        worker_info: Any = None,
        hardware_info: Any = None,
        env_idx: int = 0,
    ) -> None:
        super().__init__()
        self._logger = get_logger()
        self.config = config
        self.env_idx = env_idx
        self._num_steps = 0
        self._success_hold_counter = 0
        self._ensure_reward_config_arrays()

        # ---- Initialize controller ----
        if not self.config.is_dummy:
            self._controller = PiperController(
                ns_left=config.ns_left,
                ns_right=config.ns_right,
                use_robot_base=config.use_robot_base,
                robot_base_topic=config.robot_base_topic,
                joint_speed_pct=config.joint_speed_pct,
            )
        else:
            self._controller = None

        # ---- Image buffer (thread-safe, written by ROS callback thread) ----
        self._bridge = CvBridge()
        self._img_lock = threading.Lock()
        self._latest_images: dict[str, np.ndarray] = {}
        for cam_name in config.camera_names:
            h, w = config.obs_img_resolution
            self._latest_images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        # ---- Subscribe camera topics ----
        if not self.config.is_dummy:
            self._img_subscribers: list[rospy.Subscriber] = []
            for cam_name, topic in zip(config.camera_names, config.img_topics):
                sub = rospy.Subscriber(
                    topic,
                    Image,
                    callback=self._make_img_callback(cam_name),
                    queue_size=1,
                    tcp_nodelay=True,
                )
                self._img_subscribers.append(sub)

        # ---- Initialize action/observation spaces ----
        self._init_action_obs_spaces()
        self._joint_limit_low: np.ndarray | None = None
        self._joint_limit_high: np.ndarray | None = None
        if self.config.joint_action_mode == "delta":
            min_q = np.array(self.config.min_qpos, dtype=np.float32)
            max_q = np.array(self.config.max_qpos, dtype=np.float32)
            if len(min_q) < 14:
                self._joint_limit_low = np.tile(min_q[:7], 2).astype(np.float64)
                self._joint_limit_high = np.tile(max_q[:7], 2).astype(np.float64)
            else:
                self._joint_limit_low = min_q[:14].astype(np.float64)
                self._joint_limit_high = max_q[:14].astype(np.float64)

        # ---- Human-in-the-loop: keyboard + master arm topic ----
        if config.enable_human_intervention and not config.is_dummy:
            from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
            self._keyboard = KeyboardListener()
            self._policy_enabled = True
            self._master_action: np.ndarray | None = None
            self._master_action_lock = threading.Lock()
            rospy.Subscriber(
                config.master_joint_topic,
                JointState,
                self._on_master_joint_state,
                queue_size=1,
            )
        else:
            self._keyboard = None
            self._policy_enabled = True
            self._master_action = None
            self._master_action_lock = threading.Lock()

        if self.config.is_dummy:
            self._logger.info("PiperEnv initialized in dummy mode.")
            return

        # ---- Wait for robot ready ----
        self._logger.info("Waiting for Piper dual-arm robot to be ready...")
        ready = self._controller.wait_for_robot(timeout=30.0)
        if not ready:
            self._logger.warning("Robot wait timed out; some topics may not be ready.")

        self._logger.info(
            f"PiperEnv initialized: env_idx={env_idx}, "
            f"cameras={config.camera_names}, freq={config.step_frequency}Hz, "
            f"intervention={config.enable_human_intervention}"
        )

    # ==================================================================
    # ROS callbacks: master arm + images
    # ==================================================================

    def _on_master_joint_state(self, msg: JointState) -> None:
        """Store latest master arm absolute joint targets (14D).

        The ROS node publishes slave_ref + (current_master - master_ref) to
        ``/master/joint_states`` when teleoperation is active.
        """
        with self._master_action_lock:
            self._master_action = np.array(msg.position[:14], dtype=np.float64)

    def _make_img_callback(self, cam_name: str):
        """Create a ROS image callback closure for the given camera name."""

        def _callback(msg: Image) -> None:
            try:
                cv_image = self._bridge.imgmsg_to_cv2(msg, "passthrough")
                h, w = self.config.obs_img_resolution
                if cv_image.shape[0] != h or cv_image.shape[1] != w:
                    cv_image = cv2.resize(cv_image, (w, h))
                with self._img_lock:
                    self._latest_images[cam_name] = cv_image
            except Exception as e:
                self._logger.warning(f"Image callback error ({cam_name}): {e}")

        return _callback

    # ==================================================================
    # Action / observation spaces
    # ==================================================================

    def _init_action_obs_spaces(self) -> None:
        """Initialize action and observation spaces.

        Action space: 14D absolute joint positions (left 7 + right 7),
        each arm: 6 joints + 1 gripper.

        Observation space:
        - ``state``: qpos(14), qvel(14), effort(14), base_vel(2)
        - ``frames``: one (H, W, 3) image per camera
        """
        min_qpos = np.array(self.config.min_qpos, dtype=np.float32)
        max_qpos = np.array(self.config.max_qpos, dtype=np.float32)

        if self.config.joint_action_mode == "delta":
            # Policy outputs normalized deltas; scaled and added to current qpos in step()
            self.action_space = gym.spaces.Box(
                low=-np.ones(14, dtype=np.float32),
                high=np.ones(14, dtype=np.float32),
                dtype=np.float32,
            )
        elif len(min_qpos) < 14:
            # Single-arm limits, tile to dual-arm
            action_low = np.tile(min_qpos[:7], 2).astype(np.float32)
            action_high = np.tile(max_qpos[:7], 2).astype(np.float32)
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        else:
            action_low = min_qpos[:14].astype(np.float32)
            action_high = max_qpos[:14].astype(np.float32)
            self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        h, w = self.config.obs_img_resolution
        state_space = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "qvel": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "effort": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float64),
                "base_vel": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
            }
        )
        frames_space = gym.spaces.Dict(
            {
                cam_name: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
                for cam_name in self.config.camera_names
            }
        )
        self.observation_space = gym.spaces.Dict(
            {"state": state_space, "frames": frames_space}
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # ==================================================================
    # Gym API: step
    # ==================================================================

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: 14D joint position array (left 7 + right 7),
                    each arm: 6 joint angles (rad) + 1 gripper (rad).

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
            info["intervene_action"] is set to the master arm's 14D joint target
            when teleoperation is active.
        """
        start_time = time.time()

        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # ---- Delta mode: policy output [-1,1]^14 -> absolute joint target ----
        if self.config.joint_action_mode == "delta":
            assert self._joint_limit_low is not None and self._joint_limit_high is not None
            current_qpos = (
                self._controller.get_qpos()
                if not self.config.is_dummy and self._controller is not None
                else np.zeros(14, dtype=np.float64)
            )
            delta = action * float(self.config.delta_action_scale)
            action = np.clip(
                current_qpos + delta,
                self._joint_limit_low,
                self._joint_limit_high,
            )

        # ---- page_up: toggle policy output ----
        if self._keyboard is not None and self._keyboard.consume_press(self.config.policy_enable_key):
            self._policy_enabled = not self._policy_enabled
            self._logger.info(f"Policy output {'enabled' if self._policy_enabled else 'disabled'}.")

        # ---- If policy disabled, hold current position ----
        if not self._policy_enabled and not self.config.is_dummy and self._controller is not None:
            action = self._controller.get_qpos()

        # ---- Split into left/right arm actions ----
        left_action, right_action = split_dual_arm_action(action)

        # ---- Check teleop state once (ROS node owns channel when active) ----
        teleop_active = (
            not self.config.is_dummy
            and rospy.get_param("/enable_message_publish", False)
        )

        # ---- Publish control command ----
        # Skip move_arm during teleoperation: the ROS node writes to the same
        # /master/joint_left|right topics; publishing here would conflict with it.
        if not self.config.is_dummy and not teleop_active:
            self._controller.move_arm(left_action, right_action)
        elif self.config.is_dummy:
            self._logger.debug(f"Dummy step: left={left_action}, right={right_action}")

        self._num_steps += 1

        # ---- Rate control ----
        step_time = time.time() - start_time
        sleep_time = max(0.0, (1.0 / self.config.step_frequency) - step_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

        # ---- Get observation ----
        observation = self._get_observation()

        # ---- Compute reward ----
        reward = self._calc_step_reward(observation)

        # ---- Termination (aligned with FrankaEnv) ----
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps

        # ---- Build info: record master arm action when teleop is active ----
        info: dict = {}
        if teleop_active:
            with self._master_action_lock:
                if self._master_action is not None:
                    info["intervene_action"] = self._master_action.copy()

        return observation, reward, terminated, truncated, info

    # ==================================================================
    # Gym API: reset
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reset environment to initial state.

        Aligned with ``FrankaEnv.reset()``:
        1. Enable arms
        2. Go to zero position
        3. Wait for joints to reach zero
        4. Return initial observation

        Args:
            seed: Random seed (reserved).
            options: Extra options (reserved).

        Returns:
            (observation, info) tuple.
        """
        self._num_steps = 0
        self._success_hold_counter = 0

        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # ---- Enable ----
        self._controller.enable_arm()
        time.sleep(0.5)

        # ---- Go to reset pose: left arm all zeros, right arm preset pose ----
        left_reset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        right_reset = np.array([
            0.7386661800000001, 1.549620296, -1.284977372, 0.047343016, 1.066229612, 0.176812384, 0.0289
        ], dtype=np.float64)
        self._controller.move_arm(left_reset, right_reset)

        # ---- Wait for joints to reach reset pose ----
        self._controller._wait_for_joint(
            target_pos=left_reset[:6],
            side="left",
            timeout=10.0,
            atol=0.05,
        )
        self._controller._wait_for_joint(
            target_pos=right_reset[:6],
            side="right",
            timeout=10.0,
            atol=0.05,
        )

        time.sleep(0.5)

        observation = self._get_observation()
        self._logger.info("PiperEnv reset complete.")
        return observation, {}

    def _ensure_reward_config_arrays(self) -> None:
        """Normalize target_qpos / reward_threshold to 14D float64 (compatible with YAML lists)."""
        cfg = self.config
        cfg.target_qpos = np.asarray(cfg.target_qpos, dtype=np.float64).reshape(14)
        cfg.reward_threshold = np.asarray(cfg.reward_threshold, dtype=np.float64).reshape(14)

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_observation(self) -> dict:
        """Build the full observation dictionary.

        - ``state.qpos``: 14D dual-arm joint positions (6 joints + 1 gripper) x2
        - ``state.qvel``: 14D dual-arm joint velocities
        - ``state.effort``: 14D dual-arm joint efforts
        - ``state.base_vel``: 2D base velocity [linear.x, angular.z]
        - ``frames``: camera images

        Returns:
            Dict conforming to observation_space.
        """
        if not self.config.is_dummy:
            state = {
                "qpos": self._controller.get_qpos(),
                "qvel": self._controller.get_qvel(),
                "effort": self._controller.get_effort(),
                "base_vel": self._controller.get_base_vel(),
            }
            frames = self._get_camera_frames()
            return copy.deepcopy({"state": state, "frames": frames})
        else:
            return self._base_observation_space.sample()

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """Thread-safe retrieval of the latest frame from each camera."""
        frames = {}
        with self._img_lock:
            for cam_name in self.config.camera_names:
                frames[cam_name] = self._latest_images[cam_name].copy()
        return frames

    # ==================================================================
    # Reward
    # ==================================================================

    def _calc_step_reward(self, observation: dict, **kwargs: Any) -> float:
        """Compute per-step reward.

        Aligned with ``FrankaEnv._calc_step_reward``: compares current ``qpos``
        against ``target_qpos`` in joint space. All per-joint absolute errors
        within ``reward_threshold`` = target zone (reward 1.0, increment hold counter);
        otherwise reset counter and optionally return dense reward.

        Returns 0.0 in dummy mode.

        Args:
            observation: Current observation (must contain ``state.qpos``).

        Returns:
            Scalar reward.
        """
        if self.config.is_dummy:
            return 0.0

        qpos = np.asarray(observation["state"]["qpos"], dtype=np.float64).reshape(14)
        target = np.asarray(self.config.target_qpos, dtype=np.float64).reshape(14)
        thr = np.asarray(self.config.reward_threshold, dtype=np.float64).reshape(14)

        target_delta = np.abs(qpos - target)
        is_in_target_zone = bool(np.all(target_delta <= thr))

        if is_in_target_zone:
            self._success_hold_counter += 1
            reward = 1.0
        else:
            self._success_hold_counter = 0
            if self.config.use_dense_reward:
                reward = float(
                    np.exp(-self.config.dense_reward_scale * np.sum(np.square(target_delta)))
                )
            else:
                reward = 0.0
            self._logger.debug(
                "Joint target not met: max_delta=%s, threshold=%s, reward=%s",
                float(np.max(target_delta)),
                thr,
                reward,
            )

        return reward

    # ==================================================================
    # Properties and utilities
    # ==================================================================

    @property
    def num_steps(self) -> int:
        """Number of steps executed in the current episode."""
        return self._num_steps

    @property
    def task_description(self) -> str:
        """Task description string, used by RealWorldEnv wrapper."""
        return self.config.task_name

    def close(self) -> None:
        """Close environment and release resources."""
        if not self.config.is_dummy and hasattr(self, "_img_subscribers"):
            for sub in self._img_subscribers:
                sub.unregister()
            self._img_subscribers = []
        self._logger.info("PiperEnv closed.")
