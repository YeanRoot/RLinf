#!/usr/bin/env python3
"""
Convert a LeRobot v2.x Piper/GigaWorld dataset into RLinf TrajectoryReplayBuffer format.

This script is intended for actor BC warmup:

  LeRobot parquet + mp4 videos
      -> frozen GigaWA backbone features: visual_latent / robot_state / ref_action
      -> expert actions converted from executable absolute qpos to normalized-delta model actions
      -> TrajectoryReplayBuffer directory with metadata.json + trajectory_index.json

Output layout:

  <output-root>/
    rank_0/
      metadata.json
      trajectory_index.json
      trajectory_0_<model_weights_id>.pt
      ...

Use <output-root> as algorithm.demo_buffer.load_path in offline BC/RL configs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Dependency / project loading helpers
# -----------------------------------------------------------------------------


def _ensure_project_on_path(project_root: str | None) -> None:
    if project_root:
        root = Path(project_root).expanduser().resolve()
    else:
        # When this script is placed at examples/embodiment/, parents[2] is RLinf root.
        here = Path(__file__).resolve()
        root = here.parents[2] if len(here.parents) >= 3 else Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _torch_dtype_from_string(name: str | None) -> torch.dtype:
    name = (name or "bf16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {name}")


@contextmanager
def _hydra_compose_context(config_path: str):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(config_path).expanduser().resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(f"config_path is not a directory: {config_dir}")

    # Allow this converter to be called after other Hydra scripts in the same process.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        yield compose


def _load_actor_model_cfg(config_path: str, config_name: str):
    from omegaconf import OmegaConf

    with _hydra_compose_context(config_path) as compose:
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    if "actor" not in cfg or "model" not in cfg.actor:
        raise KeyError(
            "Hydra config must contain actor.model. "
            f"Got top-level keys: {list(cfg.keys())}"
        )
    return cfg, cfg.actor.model


def _build_policy(model_cfg, device: str, disable_progress: bool = True):
    from rlinf.models.embodiment.giga_world_policy.giga_world_policy import GigaWorldPolicy

    dtype = _torch_dtype_from_string(str(model_cfg.get("precision", "bf16")))
    policy = GigaWorldPolicy(model_cfg, torch_dtype=dtype)
    policy.eval()
    policy.to(torch.device(device))

    # Disable diffusers/tqdm progress bars if the runtime exposes this API.
    if disable_progress:
        try:
            policy.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
    return policy


# -----------------------------------------------------------------------------
# LeRobot loading helpers
# -----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _read_parquet(path: Path):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "This converter needs pandas plus a parquet engine such as pyarrow. "
            "Install them in your RLinf environment, e.g. `pip install pandas pyarrow`."
        ) from e

    try:
        return pd.read_parquet(path)
    except ImportError as e:
        raise ImportError(
            "pandas found no parquet engine. Please install pyarrow or fastparquet, "
            "e.g. `pip install pyarrow`."
        ) from e


def _stack_array_column(df, column: str, dim: int | None = None) -> np.ndarray:
    if column in df.columns:
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in df[column].to_list()], axis=0)
    else:
        # Fallback for flattened parquet columns: observation.state.0, action.0, ...
        prefix = f"{column}."
        cols = [c for c in df.columns if str(c).startswith(prefix)]
        if not cols:
            raise KeyError(f"Column {column!r} not found. Available columns: {list(df.columns)}")
        cols = sorted(cols, key=lambda c: int(str(c).split(".")[-1]))
        arr = df[cols].to_numpy(dtype=np.float32)
    if dim is not None:
        if arr.shape[-1] < dim:
            raise ValueError(f"Column {column} has dim {arr.shape[-1]}, expected at least {dim}")
        arr = arr[..., :dim]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return int(episode_index) // int(chunks_size)


def _format_data_path(info: dict[str, Any], episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size", 1000))
    return str(info["data_path"]).format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
    )


def _format_video_path(info: dict[str, Any], episode_index: int, video_key: str) -> str:
    chunks_size = int(info.get("chunks_size", 1000))
    return str(info["video_path"]).format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
        video_key=video_key,
    )


class EpisodeVideoReader:
    """Read LeRobot mp4 frames.

    OpenCV often fails on AV1-encoded LeRobot videos in minimal Docker images.
    This reader therefore supports a ffmpeg software-decoding fallback. In
    `auto` mode it first tries OpenCV; when OpenCV cannot read the requested
    frame, it decodes only the selected frame indices with ffmpeg and caches
    them for the current episode/video.
    """

    VIDEO_KEYS = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    def __init__(
        self,
        dataset_root: Path,
        info: dict[str, Any],
        episode_index: int,
        video_backend: str = "auto",
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
    ):
        self.dataset_root = dataset_root
        self.info = info
        self.episode_index = int(episode_index)
        self.video_backend = str(video_backend).lower()
        if self.video_backend not in {"auto", "opencv", "ffmpeg"}:
            raise ValueError(f"Unsupported video_backend={video_backend!r}; use auto/opencv/ffmpeg")
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.frame_cache: dict[str, dict[int, np.ndarray]] = {}
        self.key_backend: dict[str, str] = {}

    def _video_path(self, video_key: str) -> Path:
        rel = _format_video_path(self.info, self.episode_index, video_key)
        path = self.dataset_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing video file for {video_key}: {path}")
        return path

    def _cap(self, video_key: str) -> cv2.VideoCapture:
        if video_key not in self.caps:
            path = self._video_path(video_key)
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video with OpenCV: {path}")
            self.caps[video_key] = cap
        return self.caps[video_key]

    def _read_rgb_opencv(self, video_key: str, frame_idx: int) -> np.ndarray:
        cap = self._cap(video_key)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"OpenCV failed to read frame {frame_idx} from episode {self.episode_index}, key {video_key}"
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _probe_video_size(self, path: Path) -> tuple[int, int]:
        cmd = [
            self.ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(path),
        ]
        try:
            res = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            data = json.loads(res.stdout)
            stream = data["streams"][0]
            w, h = int(stream["width"]), int(stream["height"])
            if w <= 0 or h <= 0:
                raise ValueError(data)
            return w, h
        except Exception as e:
            raise RuntimeError(
                f"ffprobe failed for {path}. Please check ffmpeg/ffprobe installation. Error: {e}"
            ) from e

    def _ffmpeg_cmds(self, path: Path) -> list[list[str]]:
        # Try default software decoding first, then explicit AV1 decoders when available.
        base_tail = ["-i", str(path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return [
            [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-hwaccel", "none", *base_tail],
            [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-c:v", "libdav1d", *base_tail],
            [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-c:v", "libaom-av1", *base_tail],
        ]

    def _decode_selected_frames_ffmpeg(self, video_key: str, indices: Iterable[int]) -> None:
        requested = sorted(set(int(i) for i in indices))
        if not requested:
            return
        cache = self.frame_cache.setdefault(video_key, {})
        missing = [i for i in requested if i not in cache]
        if not missing:
            return

        path = self._video_path(video_key)
        width, height = self._probe_video_size(path)
        frame_size = width * height * 3
        max_needed = max(missing)
        last_error = ""

        for cmd in self._ffmpeg_cmds(path):
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdout is not None
            frame_idx = 0
            try:
                while frame_idx <= max_needed:
                    buf = proc.stdout.read(frame_size)
                    if len(buf) == 0:
                        break
                    if len(buf) != frame_size:
                        last_error = f"short rawvideo frame at frame={frame_idx}, got={len(buf)}, expected={frame_size}"
                        break
                    if frame_idx in missing:
                        arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()
                        cache[frame_idx] = arr
                    frame_idx += 1
                # Close stdout so ffmpeg can terminate if we stopped early.
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                ret = proc.wait(timeout=10)
                still_missing = [i for i in missing if i not in cache]
                if not still_missing:
                    self.key_backend[video_key] = "ffmpeg"
                    return
                last_error = stderr.strip() or last_error or f"missing frames {still_missing[:10]} from {path}"
            except Exception as e:
                try:
                    proc.kill()
                except Exception:
                    pass
                last_error = f"{type(e).__name__}: {e}"
                continue

        raise RuntimeError(
            f"ffmpeg failed to decode requested frames for episode {self.episode_index}, "
            f"key {video_key}, path={path}. Last error: {last_error}\n"
            "If this is an AV1 video, install an ffmpeg build with libdav1d/libaom-av1 "
            "or pre-convert videos to H.264."
        )

    def prepare_rgb_indices(self, indices: Iterable[int]) -> None:
        indices = sorted(set(int(i) for i in indices))
        if not indices:
            return
        test_idx = indices[0]
        for key in self.VIDEO_KEYS:
            if self.video_backend == "ffmpeg":
                self._decode_selected_frames_ffmpeg(key, indices)
                continue
            if self.video_backend == "opencv":
                # Do not pre-cache. read_rgb() will use OpenCV directly.
                continue
            # auto: test OpenCV once. If it fails, decode selected frames with ffmpeg.
            try:
                _ = self._read_rgb_opencv(key, test_idx)
                self.key_backend[key] = "opencv"
            except Exception:
                self._decode_selected_frames_ffmpeg(key, indices)

    def read_rgb(self, video_key: str, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        cached = self.frame_cache.get(video_key, {})
        if frame_idx in cached:
            return cached[frame_idx]

        backend = self.video_backend if self.video_backend != "auto" else self.key_backend.get(video_key, "opencv")
        if backend == "ffmpeg":
            self._decode_selected_frames_ffmpeg(video_key, [frame_idx])
            return self.frame_cache[video_key][frame_idx]

        try:
            return self._read_rgb_opencv(video_key, frame_idx)
        except Exception as e:
            if self.video_backend == "opencv":
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from episode {self.episode_index}, key {video_key}"
                ) from e
            # auto fallback for late failures.
            self._decode_selected_frames_ffmpeg(video_key, [frame_idx])
            return self.frame_cache[video_key][frame_idx]

    def close(self) -> None:
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()
        self.frame_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# -----------------------------------------------------------------------------
# Feature/action conversion
# -----------------------------------------------------------------------------


def _load_episode_prompt_embeds(policy, dataset_root: Path, ep_meta: dict[str, Any]) -> None:
    rel = ep_meta.get("t5_embedding_path", None)
    if not rel:
        return
    path = dataset_root / str(rel)
    if not path.is_file():
        raise FileNotFoundError(f"t5_embedding_path not found: {path}")
    prompt_embeds = policy._load_fixed_prompt_embeds(  # intentionally use the policy helper
        str(path),
        max_length=int(policy.prompt_embeds_max_length),
    )
    policy.fixed_prompt_embeds = prompt_embeds.to(policy.device_ref)


def _build_env_obs_from_indices(
    *,
    reader: EpisodeVideoReader,
    states: np.ndarray,
    indices: list[int],
    task_description: str,
) -> dict[str, Any]:
    main_images = []
    wrist_images = []
    robot_states = []
    for idx in indices:
        idx = int(idx)
        high = reader.read_rgb("observation.images.cam_high", idx)
        left = reader.read_rgb("observation.images.cam_left_wrist", idx)
        right = reader.read_rgb("observation.images.cam_right_wrist", idx)
        main_images.append(high)
        wrist_images.append(np.stack([left, right], axis=0))
        robot_states.append(states[idx])

    return {
        "main_images": torch.from_numpy(np.stack(main_images, axis=0)).to(torch.uint8),
        "wrist_images": torch.from_numpy(np.stack(wrist_images, axis=0)).to(torch.uint8),
        "states": torch.from_numpy(np.stack(robot_states, axis=0)).float(),
        "task_descriptions": [task_description for _ in indices],
    }


@torch.no_grad()
def _extract_feature_map_for_indices(
    *,
    policy,
    reader: EpisodeVideoReader,
    states: np.ndarray,
    indices: list[int],
    task_description: str,
    batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    feature_map: dict[int, dict[str, torch.Tensor]] = {}
    unique_indices = sorted(set(int(i) for i in indices))
    # Pre-cache selected frames in one pass when OpenCV cannot decode the videos.
    if hasattr(reader, "prepare_rgb_indices"):
        reader.prepare_rgb_indices(unique_indices)
    for start in range(0, len(unique_indices), batch_size):
        sub_indices = unique_indices[start : start + batch_size]
        env_obs = _build_env_obs_from_indices(
            reader=reader,
            states=states,
            indices=sub_indices,
            task_description=task_description,
        )
        backbone = policy.extract_frozen_backbone_batch(env_obs)
        visual_latent = backbone["visual_latent"].detach().cpu().float()
        robot_state = backbone["robot_state"].detach().cpu().float()
        ref_action = backbone["ref_action"].detach().cpu().float()
        # Store flattened ref_action to match existing online pt_debug format: [C*A].
        ref_action = ref_action.reshape(ref_action.shape[0], -1).contiguous()

        for local_i, frame_idx in enumerate(sub_indices):
            feature_map[int(frame_idx)] = {
                "visual_latent": visual_latent[local_i].contiguous(),
                "robot_state": robot_state[local_i].contiguous(),
                "ref_action": ref_action[local_i].contiguous(),
            }
    return feature_map


def _extract_zero_feature_map_for_indices(
    *,
    policy,
    states: np.ndarray,
    indices: list[int],
    zero_visual_shape: tuple[int, int, int, int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Debug fallback only. Real BC warmup should use the actual WA backbone."""
    feature_map = {}
    for idx in sorted(set(int(i) for i in indices)):
        state = torch.as_tensor(states[idx], dtype=torch.float32)
        norm_state, _ = policy._normalize_state(state)
        feature_map[idx] = {
            "visual_latent": torch.zeros(zero_visual_shape, dtype=torch.float32),
            "robot_state": norm_state[0].detach().cpu().float(),
            "ref_action": torch.zeros(policy.action_chunk * policy.model_action_dim, dtype=torch.float32),
        }
    return feature_map


def _convert_exec_chunk_to_model_action(
    *,
    policy,
    exec_chunk: np.ndarray,
    raw_state: np.ndarray,
) -> torch.Tensor:
    exec_tensor = torch.as_tensor(exec_chunk, dtype=torch.float32, device=policy.device_ref).unsqueeze(0)
    raw_state_tensor = torch.as_tensor(raw_state, dtype=torch.float32, device=policy.device_ref).unsqueeze(0)
    model_action = policy.exec_action_to_model_action(exec_tensor, raw_state_tensor)[0]
    return model_action.detach().cpu().float().reshape(-1).contiguous()


def _make_padded_exec_chunk(
    actions_exec: np.ndarray,
    start_idx: int,
    chunk_size: int,
) -> tuple[np.ndarray, int]:
    n = int(actions_exec.shape[0])
    valid_len = max(0, min(chunk_size, n - int(start_idx)))
    if valid_len <= 0:
        raise ValueError(f"Invalid start_idx={start_idx} for episode length={n}")
    chunk = np.zeros((chunk_size, actions_exec.shape[-1]), dtype=np.float32)
    chunk[:valid_len] = actions_exec[start_idx : start_idx + valid_len]
    # Repeat the last valid action for padded slots. They are masked out but this
    # keeps values finite and easier to inspect.
    if valid_len < chunk_size:
        chunk[valid_len:] = chunk[valid_len - 1]
    return chunk, valid_len


def _build_trajectory_for_episode(
    *,
    policy,
    dataset_root: Path,
    info: dict[str, Any],
    ep_meta: dict[str, Any],
    chunk_size: int,
    stride: int,
    backbone_batch_size: int,
    mark_expert_as_intervention: bool,
    use_episode_t5: bool,
    skip_backbone: bool,
    zero_visual_shape: tuple[int, int, int, int],
    max_episode_length: int,
    video_backend: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> Any:
    from rlinf.data.embodied_io_struct import Trajectory

    episode_index = int(ep_meta["episode_index"])
    data_path = dataset_root / _format_data_path(info, episode_index)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing parquet file: {data_path}")

    df = _read_parquet(data_path)
    states = _stack_array_column(df, "observation.state", dim=policy.env_action_dim)
    actions_exec_all = _stack_array_column(df, "action", dim=policy.env_action_dim)
    n_frames = min(int(states.shape[0]), int(actions_exec_all.shape[0]), int(ep_meta.get("length", states.shape[0])))
    states = states[:n_frames]
    actions_exec_all = actions_exec_all[:n_frames]
    if n_frames < 2:
        raise ValueError(f"Episode {episode_index} too short: {n_frames} frames")

    if use_episode_t5:
        _load_episode_prompt_embeds(policy, dataset_root, ep_meta)

    starts = list(range(0, n_frames, stride))
    # Keep the final partial chunk so that final reward=1 has a place to land.
    starts = [s for s in starts if s < n_frames]
    if not starts:
        raise ValueError(f"Episode {episode_index} produced no chunks")

    valid_lens = [min(chunk_size, n_frames - s) for s in starts]
    next_indices = [min(s + valid_len, n_frames - 1) for s, valid_len in zip(starts, valid_lens)]
    feature_indices = starts + next_indices

    task_description = ""
    if ep_meta.get("tasks"):
        task_description = str(ep_meta["tasks"][0])

    if skip_backbone:
        feature_map = _extract_zero_feature_map_for_indices(
            policy=policy,
            states=states,
            indices=feature_indices,
            zero_visual_shape=zero_visual_shape,
        )
    else:
        with EpisodeVideoReader(
            dataset_root,
            info,
            episode_index,
            video_backend=video_backend,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        ) as reader:
            feature_map = _extract_feature_map_for_indices(
                policy=policy,
                reader=reader,
                states=states,
                indices=feature_indices,
                task_description=task_description,
                batch_size=max(1, int(backbone_batch_size)),
            )

    T = len(starts)
    actions_model = []
    actions_exec = []
    rewards = torch.zeros(T, 1, chunk_size, dtype=torch.float32)
    dones = torch.zeros(T, 1, chunk_size, dtype=torch.bool)
    terminations = torch.zeros(T, 1, chunk_size, dtype=torch.bool)
    truncations = torch.zeros(T, 1, chunk_size, dtype=torch.bool)
    action_valid_mask = torch.zeros(T, 1, chunk_size, dtype=torch.bool)
    intervene_flags = torch.zeros(T, 1, chunk_size, dtype=torch.bool)

    curr_visual, curr_robot, curr_ref = [], [], []
    next_visual, next_robot, next_ref = [], [], []

    for i, (s, valid_len, next_idx) in enumerate(zip(starts, valid_lens, next_indices)):
        exec_chunk, valid_len = _make_padded_exec_chunk(actions_exec_all, s, chunk_size)
        model_action_flat = _convert_exec_chunk_to_model_action(
            policy=policy,
            exec_chunk=exec_chunk,
            raw_state=states[s],
        )
        actions_model.append(model_action_flat)
        actions_exec.append(torch.as_tensor(exec_chunk, dtype=torch.float32).reshape(-1))

        action_valid_mask[i, 0, :valid_len] = True
        if mark_expert_as_intervention:
            intervene_flags[i, 0, :valid_len] = True

        cur = feature_map[int(s)]
        nxt = feature_map[int(next_idx)]
        curr_visual.append(cur["visual_latent"])
        curr_robot.append(cur["robot_state"])
        curr_ref.append(cur["ref_action"])
        next_visual.append(nxt["visual_latent"])
        next_robot.append(nxt["robot_state"])
        next_ref.append(nxt["ref_action"])

    # All LeRobot demonstrations are assumed successful: final valid primitive step gets reward=1.
    last_i = T - 1
    last_step = int(valid_lens[-1]) - 1
    rewards[last_i, 0, last_step] = 1.0
    dones[last_i, 0, last_step] = True
    terminations[last_i, 0, last_step] = True

    actions_model_tensor = torch.stack(actions_model, dim=0).view(T, 1, -1).contiguous()
    actions_exec_tensor = torch.stack(actions_exec, dim=0).view(T, 1, -1).contiguous()

    curr_obs = {
        "visual_latent": torch.stack(curr_visual, dim=0).unsqueeze(1).contiguous(),
        "robot_state": torch.stack(curr_robot, dim=0).unsqueeze(1).contiguous(),
        "ref_action": torch.stack(curr_ref, dim=0).unsqueeze(1).contiguous(),
    }
    next_obs = {
        "visual_latent": torch.stack(next_visual, dim=0).unsqueeze(1).contiguous(),
        "robot_state": torch.stack(next_robot, dim=0).unsqueeze(1).contiguous(),
        "ref_action": torch.stack(next_ref, dim=0).unsqueeze(1).contiguous(),
    }

    # forward_inputs is not needed by offline BC forward_actor(), but keeping it
    # makes the saved files easier to inspect and compatible with debug utilities.
    action_source = torch.ones(T, 1, chunk_size, dtype=torch.long)  # 1 = expert/human demo
    action_source[~action_valid_mask] = 3  # 3 = padding
    forward_inputs = {
        "action": actions_exec_tensor.clone(),
        "action_exec": actions_exec_tensor.clone(),
        "model_action": actions_model_tensor.clone(),
        "action_valid_mask": action_valid_mask.clone(),
        "intervene_flags": intervene_flags.clone(),
        "action_source": action_source,
    }

    trajectory = Trajectory(
        max_episode_length=int(max_episode_length),
        model_weights_id="lerobot_piper_bc",
        actions=actions_model_tensor,
        intervene_flags=intervene_flags,
        action_valid_mask=action_valid_mask,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        forward_inputs=forward_inputs,
        curr_obs=curr_obs,
        next_obs=next_obs,
        sample_infos=[
            {
                "episode_index": episode_index,
                "source": "lerobot",
                "num_frames": n_frames,
                "chunk_size": chunk_size,
                "stride": stride,
            }
        ],
        metadata={
            "source": "lerobot",
            "episode_index": episode_index,
            "num_frames": n_frames,
            "num_chunks": T,
            "valid_steps": int(action_valid_mask.sum().item()),
            "reward_sum": float(rewards.sum().item()),
            "reward_max": float(rewards.max().item()),
            "is_success": True,
            "episode_outcome": "success",
            "chunk_size": chunk_size,
            "stride": stride,
        },
    )
    return trajectory.contiguous_()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _parse_episode_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _parse_shape(value: str) -> tuple[int, int, int, int]:
    nums = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(nums) != 4:
        raise ValueError("--zero-visual-shape must have 4 comma-separated ints, e.g. 48,1,12,48")
    return tuple(nums)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None, help="RLinf repo root. Default: infer from script location.")
    parser.add_argument("--dataset-root", required=True, help="LeRobot dataset root, e.g. .../260423160824_7c5a")
    parser.add_argument("--output-root", required=True, help="Output buffer root. This script creates output_root/rank_0")
    parser.add_argument("--config-path", default="./config", help="Hydra config dir, e.g. ./config")
    parser.add_argument("--config-name", default="online_rl_piper_gigawa", help="Config name containing actor.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12, help="Frame stride between chunks. Use 12 for non-overlap, 1 for sliding windows.")
    parser.add_argument("--max-episode-length", type=int, default=960)
    parser.add_argument("--backbone-batch-size", type=int, default=1, help="Number of frame observations per feature extraction batch.")
    parser.add_argument("--episode-ids", default=None, help="Comma/range list, e.g. 0,3,8-10. Default: all found episodes.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--end-episode", type=int, default=None, help="Inclusive episode index upper bound.")
    parser.add_argument("--use-episode-t5", action="store_true", help="Use each episode's t5_embedding_path instead of the fixed prompt_embeds_path in config.")
    parser.add_argument("--no-mark-expert-as-intervention", action="store_true", help="Set intervene_flags=false instead of true for expert valid steps.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip episodes whose parquet/videos are missing.")
    parser.add_argument("--skip-backbone", action="store_true", help="Debug only: save zero visual/ref features instead of running WA backbone.")
    parser.add_argument("--zero-visual-shape", default="48,1,12,48", help="Used only with --skip-backbone.")
    parser.add_argument("--video-backend", choices=["auto", "opencv", "ffmpeg"], default="auto", help="Use ffmpeg when OpenCV cannot decode AV1 videos.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--disable-progress", action="store_true", default=True)
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    _ensure_project_on_path(args.project_root)
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    cfg, model_cfg = _load_actor_model_cfg(args.config_path, args.config_name)
    policy = _build_policy(model_cfg, device=args.device, disable_progress=args.disable_progress)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_rank_dir = output_root / "rank_0"
    output_rank_dir.mkdir(parents=True, exist_ok=True)

    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot episodes.jsonl: {episodes_path}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonl(episodes_path)
    selected = _parse_episode_ids(args.episode_ids)

    filtered = []
    for ep in episodes:
        eid = int(ep["episode_index"])
        if selected is not None and eid not in selected:
            continue
        if args.start_episode is not None and eid < args.start_episode:
            continue
        if args.end_episode is not None and eid > args.end_episode:
            continue
        filtered.append(ep)
    if args.max_episodes is not None:
        filtered = filtered[: int(args.max_episodes)]
    if not filtered:
        raise RuntimeError("No episodes selected.")

    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=False,
        cache_size=1,
        sample_window_size=0,
        auto_save=True,
        auto_save_path=str(output_rank_dir),
        trajectory_format="pt",
    )

    zero_visual_shape = _parse_shape(args.zero_visual_shape)
    ok_count = 0
    skipped = []
    for ep_i, ep_meta in enumerate(filtered, start=1):
        eid = int(ep_meta["episode_index"])
        print(f"[convert] episode {ep_i}/{len(filtered)} | episode_index={eid}", flush=True)
        try:
            traj = _build_trajectory_for_episode(
                policy=policy,
                dataset_root=dataset_root,
                info=info,
                ep_meta=ep_meta,
                chunk_size=int(args.chunk_size),
                stride=int(args.stride),
                backbone_batch_size=int(args.backbone_batch_size),
                mark_expert_as_intervention=not bool(args.no_mark_expert_as_intervention),
                use_episode_t5=bool(args.use_episode_t5),
                skip_backbone=bool(args.skip_backbone),
                zero_visual_shape=zero_visual_shape,
                max_episode_length=int(args.max_episode_length),
                video_backend=str(args.video_backend),
                ffmpeg_bin=str(args.ffmpeg_bin),
                ffprobe_bin=str(args.ffprobe_bin),
            )
        except Exception as e:
            if args.skip_missing:
                print(f"[convert][skip] episode_index={eid}: {type(e).__name__}: {e}", flush=True)
                skipped.append({"episode_index": eid, "error": f"{type(e).__name__}: {e}"})
                continue
            raise

        buffer.add_trajectories([traj])
        ok_count += 1
        print(
            f"[convert] added episode_index={eid} | chunks={traj.actions.shape[0]} | "
            f"valid_steps={int(traj.action_valid_mask.sum().item())} | reward_sum={float(traj.rewards.sum().item()):.1f}",
            flush=True,
        )

    buffer.close(wait=True)

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "config_path": str(Path(args.config_path).expanduser().resolve()),
        "config_name": args.config_name,
        "num_converted": ok_count,
        "num_skipped": len(skipped),
        "chunk_size": int(args.chunk_size),
        "stride": int(args.stride),
        "use_episode_t5": bool(args.use_episode_t5),
        "mark_expert_as_intervention": not bool(args.no_mark_expert_as_intervention),
        "video_backend": str(args.video_backend),
        "ffmpeg_bin": str(args.ffmpeg_bin),
        "ffprobe_bin": str(args.ffprobe_bin),
        "skipped": skipped,
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[convert] done")
    print(f"[convert] converted episodes: {ok_count}")
    print(f"[convert] skipped episodes:   {len(skipped)}")
    print(f"[convert] demo_buffer.load_path should be:\n  {output_root}")


if __name__ == "__main__":
    main()
