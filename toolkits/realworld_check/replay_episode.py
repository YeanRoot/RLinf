"""Replay actions from a collected episode .pt file on the Piper dual-arm robot.

Usage:
    python replay_episode.py [episode.pt] [--hz 30] [--dry-run]

Prerequisites:
    Terminal 1: roscore
    Terminal 2: roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1
"""

import argparse
import os
import sys
import time

piper_ros_python_paths = [
    "/opt/venv/piper/piper_ws/Piper_ros_private-ros-noetic-interrupt/devel/lib/python3/dist-packages",
]
for p in piper_ros_python_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rlinf.envs.realworld.piper.piper_controller import PiperController  # noqa: E402

DEFAULT_EPISODE = (
    "/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/"
    "collect_piper_gigawa_intervention_test/offline_collection/"
    "rank_0/success/episode_000023.pt"
)

ACTION_DIM = 14
DEFAULT_HZ = 30.0
EXEC_ACTION_KEYS = ("action_exec", "action")


def _select_action_tensor(forward_inputs: dict, action_field: str) -> tuple[str, torch.Tensor]:
    if action_field != "auto":
        if action_field not in forward_inputs:
            raise KeyError(f"forward_inputs does not contain action field '{action_field}'")
        return action_field, forward_inputs[action_field]

    for key in EXEC_ACTION_KEYS:
        value = forward_inputs.get(key)
        if torch.is_tensor(value):
            return key, value
    raise KeyError(
        "No executable action found in forward_inputs. "
        "Expected one of: action_exec, action."
    )


def _reshape_action_chunks(action_tensor: torch.Tensor) -> torch.Tensor:
    """Return action tensor as [num_chunks, chunk_size, 14] for env 0."""
    action_tensor = action_tensor.detach().cpu()
    if action_tensor.ndim == 4:
        return action_tensor[:, 0, :, :ACTION_DIM]
    if action_tensor.ndim == 3:
        num_chunks, num_envs, flat_dim = action_tensor.shape
        if flat_dim % ACTION_DIM != 0:
            raise ValueError(
                f"Flat action dimension must be divisible by {ACTION_DIM}, got {flat_dim}"
            )
        chunk_size = flat_dim // ACTION_DIM
        return action_tensor.reshape(num_chunks, num_envs, chunk_size, ACTION_DIM)[:, 0]
    if action_tensor.ndim == 2:
        if action_tensor.shape[-1] != ACTION_DIM:
            raise ValueError(f"2D action tensor must have last dim {ACTION_DIM}")
        return action_tensor[:, None, :]
    raise ValueError(f"Unsupported action tensor shape: {tuple(action_tensor.shape)}")


def _load_valid_mask(forward_inputs: dict, num_chunks: int, chunk_size: int) -> torch.Tensor:
    mask = forward_inputs.get("action_valid_mask")
    if mask is None:
        return torch.ones((num_chunks, chunk_size), dtype=torch.bool)
    mask = mask.detach().cpu().to(torch.bool)
    if mask.ndim == 3:
        mask = mask[:, 0, :]
    elif mask.ndim != 2:
        raise ValueError(f"Unsupported action_valid_mask shape: {tuple(mask.shape)}")
    if tuple(mask.shape) != (num_chunks, chunk_size):
        raise ValueError(
            "action_valid_mask shape does not match actions: "
            f"mask={tuple(mask.shape)}, actions={(num_chunks, chunk_size)}"
        )
    return mask


def load_actions(pt_path: str, action_field: str = "auto") -> tuple[np.ndarray, str]:
    """Load executable per-step actions. Returns ([T_total, 14], source_field)."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    fi = data["forward_inputs"]
    action_key, action_tensor = _select_action_tensor(fi, action_field)
    action_chunks = _reshape_action_chunks(action_tensor)
    valid_mask = _load_valid_mask(fi, action_chunks.shape[0], action_chunks.shape[1])
    steps = []
    for chunk, mask in zip(action_chunks, valid_mask):
        for action, valid in zip(chunk, mask):
            if bool(valid.item()):
                steps.append(action.numpy().astype(np.float64, copy=False))
    if not steps:
        raise ValueError(f"No valid executable actions found in {pt_path}")
    return np.stack(steps), action_key  # [T_total, 14]


def replay_actions(
    controller: PiperController,
    actions: np.ndarray,
    hz: float,
    speed_pct: int | None = None,
) -> None:
    """Replay absolute dual-arm qpos actions at a fixed frequency."""
    period = 1.0 / hz
    next_tick = time.monotonic()
    total = len(actions)
    for i, action in enumerate(actions):
        left_action = action[:7]
        right_action = action[7:]
        controller.move_arm(
            left_action,
            right_action,
            left_speed_pct=speed_pct,
            right_speed_pct=speed_pct,
        )
        print(f"  step {i + 1}/{total}", end="\r")
        next_tick += period
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(description="Replay episode actions on Piper robot")
    parser.add_argument("episode", nargs="?", default=DEFAULT_EPISODE, help="Path to episode .pt file")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="Replay frequency in Hz (default: 30)")
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Deprecated: seconds between steps. Overrides --hz when provided.",
    )
    parser.add_argument(
        "--action-field",
        default="auto",
        choices=("auto", "action_exec", "action", "ref_action"),
        help="forward_inputs action field to replay. Default auto uses action_exec/action.",
    )
    parser.add_argument("--speed-pct", type=int, default=50, help="Joint speed percentage (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving robot")
    args = parser.parse_args()

    if args.step_delay is not None and args.step_delay <= 0:
        raise ValueError(f"Step delay must be positive, got {args.step_delay}")
    hz = (1.0 / args.step_delay) if args.step_delay is not None else args.hz
    if hz <= 0:
        raise ValueError(f"Replay frequency must be positive, got {hz}")

    actions, action_key = load_actions(args.episode, action_field=args.action_field)
    print(f"Loaded {len(actions)} executable steps from {args.episode}")
    print(f"Action source: forward_inputs['{action_key}']")
    print(f"Replay frequency: {hz:.2f} Hz")

    if args.dry_run:
        for i, a in enumerate(actions):
            print(f"  step {i:3d}: left={a[:7]}, right={a[7:]}")
        return 0

    ns_left = os.environ.get("PIPER_NS_LEFT", "/puppet_left")
    ns_right = os.environ.get("PIPER_NS_RIGHT", "/puppet_right")

    controller = PiperController(
        ns_left=ns_left,
        ns_right=ns_right,
        use_robot_base=False,
        joint_speed_pct=args.speed_pct,
    )

    print("Waiting for robot...")
    start = time.time()
    if not controller.wait_for_robot(timeout=30.0, poll_interval=0.5):
        print("ERROR: Robot not ready after 30s")
        print("Please check roscore, roslaunch, namespaces, and ROS_MASTER_URI.")
        return 1
    print(f"Robot ready ({time.time() - start:.1f}s)\n")

    print(f"Replaying {len(actions)} steps at {hz:.2f} Hz...")
    try:
        replay_actions(controller, actions, hz=hz, speed_pct=args.speed_pct)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1

    print(f"\nReplay complete ({len(actions)} steps).")
    return 0


if __name__ == "__main__":
    exit(main())
