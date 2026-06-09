import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def load_replay_module():
    controller_module = types.ModuleType("rlinf.envs.realworld.piper.piper_controller")
    controller_module.PiperController = object
    sys.modules["rlinf.envs.realworld.piper.piper_controller"] = controller_module

    script_path = Path(__file__).with_name("replay_episode_lerobot.py")
    spec = importlib.util.spec_from_file_location("replay_episode_lerobot", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_chunk_noise_skips_gripper_dimensions():
    module = load_replay_module()
    actions = np.ones((12, module.ACTION_DIM), dtype=np.float64)

    noisy_actions = module.apply_chunk_noise(
        actions,
        chunk_size=12,
        middle_count=8,
        noise_range=0.01,
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(noisy_actions[:, module.GRIPPER_ACTION_INDICES], 1.0)
    assert np.any(noisy_actions[2:10, :6] != actions[2:10, :6])


def test_apply_chunk_noise_reduces_noise_range_after_step():
    module = load_replay_module()
    actions = np.zeros((24, module.ACTION_DIM), dtype=np.float64)

    noisy_actions = module.apply_chunk_noise(
        actions,
        chunk_size=12,
        middle_count=8,
        noise_range=0.01,
        noise_reduce_after_step=12,
        rng=np.random.default_rng(0),
    )

    first_chunk_noise = np.abs(noisy_actions[2:10, :6])
    second_chunk_noise = np.abs(noisy_actions[14:22, :6])
    assert np.max(first_chunk_noise) > 0.0025
    assert np.max(first_chunk_noise) <= 0.01
    assert np.max(second_chunk_noise) <= 0.0025


def test_apply_gripper_compensation_reduces_small_gripper_actions_only():
    module = load_replay_module()
    actions = np.ones((2, module.ACTION_DIM), dtype=np.float64)
    actions[0, module.GRIPPER_ACTION_INDICES] = [0.2, 0.29]
    actions[1, module.GRIPPER_ACTION_INDICES] = [0.3, 0.4]

    compensated_actions = module.apply_gripper_compensation(actions)

    np.testing.assert_allclose(compensated_actions[0, module.GRIPPER_ACTION_INDICES], [0.1, 0.19])
    np.testing.assert_allclose(compensated_actions[1, module.GRIPPER_ACTION_INDICES], [0.3, 0.4])
    np.testing.assert_array_equal(compensated_actions[:, :6], actions[:, :6])
    np.testing.assert_array_equal(compensated_actions[:, 7:13], actions[:, 7:13])
    np.testing.assert_array_equal(actions[0, module.GRIPPER_ACTION_INDICES], [0.2, 0.29])


def test_prepare_trajectory_save_root_appends_next_dataset_task_trajectory(tmp_path):
    module = load_replay_module()
    dataset_path = "/home/ubuntu/users/angen.ye/gwp/repaly/dianyuan/260524155410_8c85"
    task_root = tmp_path / "dianyuan"
    (task_root / "001" / "cam_high").mkdir(parents=True)
    (task_root / "bad_name").mkdir()

    save_root = module.prepare_trajectory_save_root(str(tmp_path), dataset_path)

    assert Path(save_root) == task_root / "002"
    for cam_name in module.CAMERA_TOPICS:
        assert (task_root / "002" / cam_name).is_dir()
