"""Replay actions from a LeRobot dataset episode on the Piper dual-arm robot.

Usage:
    python replay_episode_lerobot.py [--dataset-path PATH] [--episode 0] [--hz 30] [--dry-run]

Prerequisites:
    Terminal 1: roscore
    Terminal 2: roslaunch piper start_ms_piper_double_agilex_delta_qpose.launch auto_enable:=1
"""

import argparse
import os
import sys
import time
import threading

piper_ros_python_paths = [
    "/opt/venv/piper/piper_ws/Piper_ros_private-ros-noetic-interrupt/devel/lib/python3/dist-packages",
]
for p in piper_ros_python_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        break

import numpy as np  # noqa: E402

from rlinf.envs.realworld.piper.piper_controller import PiperController  # noqa: E402

SAVE_ROOT = "/home/ubuntu/users/angen.ye/gwp/repla_images"
CAMERA_TOPICS = {
    "cam_high": "/camera_f/color/image_raw",
    "cam_left_wrist": "/camera_l/color/image_raw",
    "cam_right_wrist": "/camera_r/color/image_raw",
}


class CameraRecorder:
    """Subscribe to camera topics and save frames on demand."""

    def __init__(self, save_root: str):
        import rospy
        from sensor_msgs.msg import Image
        import cv2
        self._cv2 = cv2
        self._lock = threading.Lock()
        self._latest: dict[str, np.ndarray | None] = {k: None for k in CAMERA_TOPICS}

        for cam_name, topic in CAMERA_TOPICS.items():
            cam_dir = os.path.join(save_root, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            rospy.Subscriber(topic, Image, self._make_cb(cam_name), queue_size=1)

        self._save_root = save_root

    def _make_cb(self, cam_name: str):
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            def cb(msg):
                img = bridge.imgmsg_to_cv2(msg, "bgr8")
                with self._lock:
                    self._latest[cam_name] = img
        except ImportError:
            import struct
            def cb(msg):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                with self._lock:
                    self._latest[cam_name] = arr[..., :3].copy()
        return cb

    def save_frame(self, frame_idx: int) -> None:
        with self._lock:
            frames = {k: v.copy() if v is not None else None for k, v in self._latest.items()}
        for cam_name, img in frames.items():
            if img is None:
                continue
            path = os.path.join(self._save_root, cam_name, f"{frame_idx:06d}.png")
            self._cv2.imwrite(path, img)


def _dataset_task_name(dataset_path: str) -> str:
    """Use the dataset parent directory as the replay task name."""
    normalized_path = os.path.normpath(dataset_path)
    task_name = os.path.basename(os.path.dirname(normalized_path))
    if not task_name:
        raise ValueError(f"Cannot infer task name from dataset path: {dataset_path}")
    return task_name


def prepare_trajectory_save_root(save_root: str, dataset_path: str) -> str:
    """Create and return the next trajectory directory for one dataset task."""
    task_name = _dataset_task_name(dataset_path)
    task_root = os.path.join(save_root, task_name)
    os.makedirs(task_root, exist_ok=True)

    max_trajectory_index = 0
    for dirname in os.listdir(task_root):
        path = os.path.join(task_root, dirname)
        if os.path.isdir(path) and dirname.isdigit():
            max_trajectory_index = max(max_trajectory_index, int(dirname))

    trajectory_root = os.path.join(task_root, f"{max_trajectory_index + 1:03d}")
    for cam_name in CAMERA_TOPICS:
        os.makedirs(os.path.join(trajectory_root, cam_name), exist_ok=False)
    return trajectory_root

DEFAULT_DATASET_PATH = "/home/ubuntu/users/angen.ye/gwp/repaly/dianyuan/260524155410_8c85"
ACTION_DIM = 14
DEFAULT_HZ = 30.0
DEFAULT_CHUNK_SIZE = 12
DEFAULT_CHUNK_PAUSE_MS = 1100.0
DEFAULT_NOISE_RANGE = 0.015
DEFAULT_NOISE_MIDDLE_COUNT = 8
DEFAULT_REDUCED_NOISE_SCALE = 0.0
GRIPPER_ACTION_INDICES = (6, 13)
DEFAULT_GRIPPER_THRESHOLD = 0.03
DEFAULT_GRIPPER_COMPENSATION = 0.01


def load_actions(dataset_path: str, episode_index: int) -> np.ndarray:
    """Load arm actions for one episode from a LeRobot parquet file. Returns [T, 14]."""
    import json
    with open(os.path.join(dataset_path, "meta", "info.json")) as f:
        info = json.load(f)
    data_path_tpl = info["data_path"]
    chunk_size = info["chunks_size"]
    episode_chunk = episode_index // chunk_size
    parquet_path = os.path.join(
        dataset_path,
        data_path_tpl.format(episode_chunk=episode_chunk, episode_index=episode_index),
    )
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    actions = np.stack(df["action"].values)[:, :ACTION_DIM].astype(np.float64)
    return actions


def apply_chunk_noise(
    actions: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    middle_count: int = DEFAULT_NOISE_MIDDLE_COUNT,
    noise_range: float = DEFAULT_NOISE_RANGE,
    excluded_indices: tuple[int, ...] = GRIPPER_ACTION_INDICES,
    noise_reduce_after_step: int | None = None,
    reduced_noise_scale: float = DEFAULT_REDUCED_NOISE_SCALE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add uniform noise to the middle actions of each chunk without mutating input."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if middle_count < 0:
        raise ValueError("middle_count must be non-negative")
    if noise_range < 0:
        raise ValueError("noise_range must be non-negative")
    if noise_reduce_after_step is not None and noise_reduce_after_step < 0:
        raise ValueError("noise_reduce_after_step must be non-negative")
    if reduced_noise_scale < 0:
        raise ValueError("reduced_noise_scale must be non-negative")

    rng = rng or np.random.default_rng()
    noisy_actions = actions.copy()

    for chunk_start in range(0, len(noisy_actions) - chunk_size + 1, chunk_size):
        active_count = min(middle_count, chunk_size)
        active_start = chunk_start + (chunk_size - active_count) // 2
        active_end = active_start + active_count

        if active_count == 0 or noise_range == 0:
            continue
        noise = rng.uniform(-noise_range, noise_range, size=noisy_actions[active_start:active_end].shape)
        if noise_reduce_after_step is not None:
            reduced_rows = np.arange(active_start, active_end) >= noise_reduce_after_step
            noise[reduced_rows] *= reduced_noise_scale
        noise[:, excluded_indices] = 0.0
        noisy_actions[active_start:active_end] += noise

    return noisy_actions


def apply_gripper_compensation(
    actions: np.ndarray,
    gripper_indices: tuple[int, ...] = GRIPPER_ACTION_INDICES,
    threshold: float = DEFAULT_GRIPPER_THRESHOLD,
    compensation: float = DEFAULT_GRIPPER_COMPENSATION,
) -> np.ndarray:
    """Reduce gripper commands below the threshold without mutating input."""
    compensated_actions = actions.copy()
    for gripper_idx in gripper_indices:
        mask = compensated_actions[:, gripper_idx] < threshold
        compensated_actions[mask, gripper_idx] -= compensation
    return compensated_actions


def replay_actions(
    controller: PiperController,
    actions: np.ndarray,
    hz: float,
    speed_pct: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_pause_ms: float = DEFAULT_CHUNK_PAUSE_MS,
    camera_recorder: "CameraRecorder | None" = None,
    start_frame: int = 0,
) -> None:
    period = 1.0 / hz
    next_tick = time.monotonic()
    total = len(actions)
    for i, action in enumerate(actions):
        controller.move_arm(
            action[:7],
            action[7:],
            left_speed_pct=speed_pct,
            right_speed_pct=speed_pct,
        )
        if camera_recorder is not None:
            camera_recorder.save_frame(start_frame + i)
        print(f"  step {i + 1}/{total}", end="\r")
        next_tick += period
        sleep_time = next_tick - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)
        if chunk_size > 0 and (i + 1) % chunk_size == 0 and i + 1 < total:
            time.sleep(chunk_pause_ms / 1000.0)
            next_tick = time.monotonic()


def main():
    parser = argparse.ArgumentParser(description="Replay LeRobot dataset episode on Piper robot")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Path to LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="Replay frequency in Hz (default: 30)")
    parser.add_argument("--speed-pct", type=int, default=50, help="Joint speed percentage (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Actions per chunk (default: 12)")
    parser.add_argument(
        "--chunk-pause-ms",
        type=float,
        default=DEFAULT_CHUNK_PAUSE_MS,
        help="Pause after each full action chunk in milliseconds (default: 1100)",
    )
    parser.add_argument(
        "--noise-range",
        type=float,
        default=DEFAULT_NOISE_RANGE,
        help="Uniform noise range for middle actions, applied as [-range, range] (default: 0.01)",
    )
    parser.add_argument("--noise-seed", type=int, default=None, help="Optional RNG seed for reproducible dry-runs")
    parser.add_argument(
        "--noise-reduce-after-step",
        type=int,
        default=None,
        help="After this many steps, scale noise range to 25%% of the original",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving robot")
    args = parser.parse_args()

    actions = load_actions(args.dataset_path, args.episode)
    rng = np.random.default_rng(args.noise_seed)
    actions = apply_chunk_noise(
        actions,
        chunk_size=args.chunk_size,
        middle_count=DEFAULT_NOISE_MIDDLE_COUNT,
        noise_range=args.noise_range,
        noise_reduce_after_step=args.noise_reduce_after_step,
        rng=rng,
    )
    actions = apply_gripper_compensation(actions)
    print(f"Loaded {len(actions)} steps from episode {args.episode} in {args.dataset_path}")
    print(f"Replay frequency: {args.hz:.2f} Hz")
    print(
        f"Chunking: {args.chunk_size} steps, pause {args.chunk_pause_ms:.0f} ms, "
        f"middle {DEFAULT_NOISE_MIDDLE_COUNT} actions noise +/-{args.noise_range:g}"
    )
    if args.noise_reduce_after_step is not None:
        print(
            f"Noise reduction: after step {args.noise_reduce_after_step}, "
            f"range scale {DEFAULT_REDUCED_NOISE_SCALE:g}"
        )

    if args.dry_run:
        for i, a in enumerate(actions):
            print(f"  step {i:3d}: left={a[:7]}, right={a[7:]}")
            if args.chunk_size > 0 and (i + 1) % args.chunk_size == 0 and i + 1 < len(actions):
                print(f"  pause {args.chunk_pause_ms:.0f} ms")
        return 0

    ns_left = os.environ.get("PIPER_NS_LEFT", "/puppet_left")
    ns_right = os.environ.get("PIPER_NS_RIGHT", "/puppet_right")

    trajectory_save_root = prepare_trajectory_save_root(SAVE_ROOT, args.dataset_path)
    print(f"Saving this trajectory to {trajectory_save_root}")

    controller = PiperController(
        ns_left=ns_left,
        ns_right=ns_right,
        use_robot_base=False,
        joint_speed_pct=args.speed_pct,
    )

    camera_recorder = CameraRecorder(trajectory_save_root)

    print("Waiting for robot...")
    start = time.time()
    if not controller.wait_for_robot(timeout=30.0, poll_interval=0.5):
        print("ERROR: Robot not ready after 30s")
        print("Please check roscore, roslaunch, namespaces, and ROS_MASTER_URI.")
        return 1
    print(f"Robot ready ({time.time() - start:.1f}s)\n")

    print(f"Replaying {len(actions)} steps at {args.hz:.2f} Hz, saving images to {trajectory_save_root}...")
    try:
        replay_actions(
            controller,
            actions,
            hz=args.hz,
            speed_pct=args.speed_pct,
            chunk_size=args.chunk_size,
            chunk_pause_ms=args.chunk_pause_ms,
            camera_recorder=camera_recorder,
            start_frame=0,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1

    print(f"\nReplay complete ({len(actions)} steps).")
    return 0


if __name__ == "__main__":
    exit(main())
