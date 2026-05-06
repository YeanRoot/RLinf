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

"""Test script for Piper human-in-the-loop intervention.

Tests the PiperEnv intervention logic based on the actual implementation:
- Teleoperation state: read from ROS param /enable_message_publish
- Master arm data: stored in env._master_action (set by /master/joint_states callback)
- Policy toggle: env._policy_enabled, toggled by page_up key inside env.step()
- Intervene action: returned in info["intervene_action"] when teleop is active

Prerequisites:
    1. ROS master running: roscore
    2. Piper ROS nodes launched:
       roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch

Usage:
    python test_piper_interrupt.py

Controls:
    page_up    - Toggle policy output (ON/OFF), handled inside env.step()
    page_down  - Toggle teleoperation (ON/OFF), handled by ROS node via /enable_message_publish
    Ctrl+C     - Quit

Control Priority (inside env.step):
    1. Policy disabled (page_up toggled OFF): action = current_qpos (hold)
    2. Policy enabled: action = policy_action passed to step()
    Teleoperation state (page_down) is managed by the ROS node and reflected
    in info["intervene_action"] when active.

Test Flow:
    1. Initialize PiperEnv with intervention enabled
    2. Reset environment
    3. Loop:
       - Generate policy action (sinusoidal, 1Hz)
       - step(policy_action)
       - Read teleop state from /enable_message_publish param
       - Read master data from env._master_action
       - Display status every 10 steps
"""

import os
import signal
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

piper_ros_python_path = "/opt/venv/piper/piper_ws/Piper_ros_private-ros-noetic-interrupt/devel/lib/python3/dist-packages"
if os.path.exists(piper_ros_python_path) and piper_ros_python_path not in sys.path:
    sys.path.insert(0, piper_ros_python_path)

import rospy

from rlinf.envs.realworld.piper import PiperEnv, PiperRobotConfig

shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    print("\n\n[SIGNAL] Ctrl+C detected, initiating graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)


def print_separator(char="=", length=80):
    print(char * length)


def print_header(title):
    print_separator()
    print(f" {title}")
    print_separator()


def get_teleop_state() -> bool:
    """Read teleoperation state from ROS param set by the ROS node."""
    try:
        return bool(rospy.get_param("/enable_message_publish", False))
    except Exception:
        return False


def get_master_action(env: PiperEnv) -> np.ndarray | None:
    """Thread-safe read of latest master arm joint targets from env."""
    with env._master_action_lock:
        if env._master_action is not None:
            return env._master_action.copy()
    return None


def print_status(
    step: int,
    teleop_active: bool,
    policy_enabled: bool,
    master_action: np.ndarray | None,
    qpos: np.ndarray,
    action_source: str,
    intervene_action: np.ndarray | None,
) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    master_ready = master_action is not None

    print(f"\n[{timestamp}] Step {step:04d}")
    print(
        f"  Teleoperation: {'ON ' if teleop_active else 'OFF'} | "
        f"Policy: {'ON ' if policy_enabled else 'OFF'} | "
        f"Action: {action_source}"
    )
    print(f"  Master Data Ready: {master_ready}")
    print(f"  Left  Arm qpos: {np.round(qpos[:7], 4)}")
    print(f"  Right Arm qpos: {np.round(qpos[7:14], 4)}")

    if intervene_action is not None:
        print(f"  Intervene Action (L): {np.round(intervene_action[:7], 4)}")
        print(f"  Intervene Action (R): {np.round(intervene_action[7:14], 4)}")
    elif teleop_active and master_action is not None:
        print(f"  Master Action (L): {np.round(master_action[:7], 4)}")
        print(f"  Master Action (R): {np.round(master_action[7:14], 4)}")


def generate_policy_action(
    step: int,
    base_qpos: np.ndarray,
    amplitude: float = 0.05,
    frequency: float = 1.0,
) -> np.ndarray:
    t = step / 30.0
    delta = amplitude * np.sin(2 * np.pi * frequency * t)
    action = base_qpos.copy()
    action[0] += delta
    action[1] += delta * 0.5
    action[7] += delta
    action[8] += delta * 0.5
    return action


def main():
    print_header("Piper Human-in-the-Loop Intervention Test")

    if "ROS_MASTER_URI" not in os.environ:
        print("WARNING: ROS_MASTER_URI not set. Using default: http://localhost:11311")
        os.environ["ROS_MASTER_URI"] = "http://localhost:11311"
    print(f"  ROS_MASTER_URI: {os.environ.get('ROS_MASTER_URI')}")

    config = PiperRobotConfig(
        ns_left="/puppet_left",
        ns_right="/puppet_right",
        enable_human_intervention=True,
        master_joint_topic="/master/joint_states",
        intervention_trigger_key="Key.page_down",
        policy_enable_key="Key.page_up",
        camera_names=["cam_high"],
        img_topics=["/camera_f/color/image_raw"],
        step_frequency=30.0,
        joint_speed_pct=50,
        is_dummy=False,
        max_num_steps=10000,
        min_qpos=[-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944, 0.0],
        max_qpos=[2.618, 3.14, 0.0, 1.745, 1.22, 2.0944, 0.08],
        target_qpos=np.array([10.0] * 14),
        success_hold_steps=999999,
    )

    print("\nInitializing PiperEnv...")
    print(f"  Control frequency : {config.step_frequency} Hz")
    print(f"  Intervention       : {config.enable_human_intervention}")
    print(f"  master_joint_topic : {config.master_joint_topic}")
    print(f"  intervention_key   : {config.intervention_trigger_key} (ROS node)")
    print(f"  policy_key         : {config.policy_enable_key} (env.step)")

    try:
        env = PiperEnv(config=config, worker_info=None, hardware_info=None, env_idx=0)
    except Exception as e:
        print(f"\nERROR: Failed to initialize environment: {e}")
        print("\nPlease check:")
        print("  1. roscore is running")
        print("  2. Piper launch file is running:")
        print("     roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch")
        return

    print("\n✓ Environment initialized successfully!")

    # Verify intervention-related attributes exist
    assert hasattr(env, "_master_action"), "env._master_action not found"
    assert hasattr(env, "_master_action_lock"), "env._master_action_lock not found"
    assert hasattr(env, "_policy_enabled"), "env._policy_enabled not found"
    assert hasattr(env, "_keyboard"), "env._keyboard not found"
    print("✓ Intervention attributes verified")
    print(f"  _keyboard      : {env._keyboard}")
    print(f"  _policy_enabled: {env._policy_enabled}")

    print("\nResetting environment...")
    try:
        obs, info = env.reset()
    except Exception as e:
        print(f"\nERROR: Failed to reset environment: {e}")
        env.close()
        return

    print("✓ Environment reset successful")
    print(f"  Observation keys : {list(obs.keys())}")
    print(f"  State keys       : {list(obs['state'].keys())}")
    print(f"  qpos shape       : {obs['state']['qpos'].shape}")
    print(f"  qvel shape       : {obs['state']['qvel'].shape}")
    print(f"  effort shape     : {obs['state']['effort'].shape}")

    initial_qpos = obs["state"]["qpos"].copy()
    print(f"\nInitial qpos: {np.round(initial_qpos, 4)}")

    print_header("Test Instructions")
    print("  Initial State: Policy=ON, Teleoperation=OFF")
    print("  Robot executes sinusoidal motions (1Hz) from policy")
    print()
    print("  Keyboard Controls:")
    print("    [page_up]   - Toggle policy output (env.step handles this)")
    print("                  OFF: robot holds current qpos")
    print("    [page_down] - Toggle teleoperation (ROS node sets /enable_message_publish)")
    print("                  ON: info['intervene_action'] populated each step")
    print("    [Ctrl+C]    - Quit program")
    print()
    print("  Verification points:")
    print("    1. Policy=ON,  Teleop=OFF  -> sinusoidal motion; info has no intervene_action")
    print("    2. Policy=OFF, Teleop=OFF  -> hold position; info has no intervene_action")
    print("    3. Any,        Teleop=ON   -> info['intervene_action'] populated from master arm")
    print_separator()

    print("\nStarting test in 3 seconds...")
    time.sleep(3)

    step = 0
    print("\n" + "=" * 80)
    print("Starting test loop... (Press Ctrl+C to stop)")
    print("=" * 80)

    try:
        while not shutdown_requested:
            policy_action = generate_policy_action(
                step=step,
                base_qpos=initial_qpos,
                amplitude=0.05,
                frequency=1.0,
            )

            obs, reward, terminated, truncated, info = env.step(policy_action)

            # Read state after step (policy toggle is handled inside env.step)
            policy_enabled = env._policy_enabled
            teleop_active = get_teleop_state()
            master_action = get_master_action(env)
            intervene_action = info.get("intervene_action", None)

            # Determine effective action source
            if teleop_active and intervene_action is not None:
                action_source = "TELEOPERATION (intervene_action in info)"
            elif teleop_active:
                action_source = "TELEOPERATION (waiting for master data)"
            elif policy_enabled:
                action_source = "POLICY (sinusoidal)"
            else:
                action_source = "HOLD (current_qpos)"

            if step % 10 == 0:
                qpos = obs["state"]["qpos"]
                print_status(
                    step,
                    teleop_active,
                    policy_enabled,
                    master_action,
                    qpos,
                    action_source,
                    intervene_action,
                )

            if terminated or truncated:
                print(f"\n{'!' * 80}")
                print(f"Episode terminated at step {step}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
                print(f"{'!' * 80}\n")
                obs, info = env.reset()
                initial_qpos = obs["state"]["qpos"].copy()
                step = 0
                continue

            step += 1

    except KeyboardInterrupt:
        print("\n\n[EXCEPTION] Test interrupted by KeyboardInterrupt")
    except Exception as e:
        print(f"\n\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        try:
            env.close()
            print("✓ Environment closed successfully")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

        print_header("Test Summary")
        print(f"  Total steps executed: {step}")
        if "policy_enabled" in dir():
            print(f"  Final policy_enabled  : {policy_enabled}")
        if "teleop_active" in dir():
            print(f"  Final teleop_active   : {teleop_active}")
        if "master_action" in dir():
            print(f"  Master data ready     : {master_action is not None}")
        print_separator()
        print("\nTest completed.")


if __name__ == "__main__":
    main()
