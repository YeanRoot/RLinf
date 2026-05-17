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

import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render).
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        # A lightweight companion video containing only observations used for
        # policy/WA inference.  For chunked rollout this is the initial reset
        # observation and the last observation of every executed chunk, i.e. the
        # observation that will be sent to the next rollout call.
        self.inference_images: list[np.ndarray] = []
        self.video_cnt = 0
        self.inference_video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

        self._source_rank = int(getattr(env, "seed_offset", 0))
        self._episode_counters: Optional[np.ndarray] = None
        self._current_chunk_indices: Optional[np.ndarray] = None
        self._reward_chunk_indices: Optional[np.ndarray] = None
        self._done_chunk_indices: Optional[np.ndarray] = None
        self._done_latched: Optional[np.ndarray] = None

    def _ensure_episode_tracking(self) -> None:
        if self._episode_counters is not None:
            return
        self._episode_counters = np.full((self._num_envs,), -1, dtype=np.int64)
        self._current_chunk_indices = np.zeros((self._num_envs,), dtype=np.int64)
        self._reward_chunk_indices = np.full((self._num_envs,), -1, dtype=np.int64)
        self._done_chunk_indices = np.full((self._num_envs,), -1, dtype=np.int64)
        self._done_latched = np.zeros((self._num_envs,), dtype=bool)

    def _start_new_episode(self, env_mask: Optional[np.ndarray] = None) -> None:
        self._ensure_episode_tracking()
        if env_mask is None:
            env_mask = np.ones((self._num_envs,), dtype=bool)
        env_mask = np.asarray(env_mask, dtype=bool)
        self._episode_counters[env_mask] += 1
        self._current_chunk_indices[env_mask] = 0
        self._reward_chunk_indices[env_mask] = -1
        self._done_chunk_indices[env_mask] = -1
        self._done_latched[env_mask] = False

    def _active_video_path(self, video_sub_dir: Optional[str] = None) -> str:
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.env.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        return os.path.join(output_dir, f"{self.video_cnt}.mp4")

    def get_current_episode_infos(self, video_sub_dir: Optional[str] = None) -> list[dict[str, Any]]:
        self._ensure_episode_tracking()
        video_path = self._active_video_path(video_sub_dir)
        infos: list[dict[str, Any]] = []
        for env_id in range(self._num_envs):
            infos.append(
                {
                    "source_rank": int(self._source_rank),
                    "source_env_local_index": int(env_id),
                    "source_episode_index": int(self._episode_counters[env_id]),
                    "source_episode_name": f"rank{self._source_rank}_{int(self._episode_counters[env_id])}",
                    "source_video_path": video_path,
                    "source_video_env_local_index": int(env_id),
                }
            )
        return infos

    def _get_overlay_info(self, env_id: int) -> dict[str, Any]:
        self._ensure_episode_tracking()
        return {
            "trajectory_name": f"rank{self._source_rank}_{int(self._episode_counters[env_id])}",
            "current_chunk": int(self._current_chunk_indices[env_id]),
            "reward_chunk": int(self._reward_chunk_indices[env_id]),
            "done_chunk": int(self._done_chunk_indices[env_id]),
            "video_env_idx": int(env_id),
        }

    def _merge_overlay_info(self, info_item: dict[str, Any], env_id: int) -> dict[str, Any]:
        merged = self._get_overlay_info(env_id)
        merged.update(info_item)
        return merged

    def _to_env_bool_array(self, value: Any) -> np.ndarray:
        if value is None:
            return np.zeros((self._num_envs,), dtype=bool)
        value = self._to_numpy(value)
        if value.ndim == 0:
            return np.full((self._num_envs,), bool(value.item()), dtype=bool)
        if value.ndim == 1:
            if value.shape[0] == self._num_envs:
                return value.astype(bool)
            flat = value.reshape(-1)
            out = np.zeros((self._num_envs,), dtype=bool)
            out[: min(self._num_envs, flat.shape[0])] = flat[: self._num_envs].astype(bool)
            return out
        if value.ndim >= 2:
            if value.shape[0] == self._num_envs:
                return value.any(axis=tuple(range(1, value.ndim))).astype(bool)
            if value.shape[-1] == self._num_envs:
                return value.any(axis=tuple(range(0, value.ndim - 1))).astype(bool)
        flat = value.reshape(-1)
        out = np.zeros((self._num_envs,), dtype=bool)
        out[: min(self._num_envs, flat.shape[0])] = flat[: self._num_envs].astype(bool)
        return out

    def _update_chunk_tracking(self, rewards: Optional[Any], terminations: Optional[Any]) -> None:
        self._ensure_episode_tracking()
        reward_mask = self._to_env_bool_array(rewards)
        done_mask = self._to_env_bool_array(terminations)
        new_reward_mask = (~self._done_latched) & reward_mask & (self._reward_chunk_indices < 0)
        self._reward_chunk_indices[new_reward_mask] = self._current_chunk_indices[new_reward_mask]
        new_done_mask = (~self._done_latched) & done_mask & (self._done_chunk_indices < 0)
        self._done_chunk_indices[new_done_mask] = self._current_chunk_indices[new_done_mask]
        self._done_latched[new_done_mask] = True

    def _advance_chunk_indices(self, done_mask: Optional[np.ndarray] = None) -> None:
        self._ensure_episode_tracking()
        if done_mask is None:
            done_mask = np.zeros((self._num_envs,), dtype=bool)
        done_mask = np.asarray(done_mask, dtype=bool)
        advance_mask = (~done_mask) & (~self._done_latched)
        self._current_chunk_indices[advance_mask] += 1

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick/build the best video image field from an observation dict.

        Real-world GigaWA observations carry one main camera plus two wrist
        cameras.  For debugging, it is much more useful to save the same
        multi-view image that the WA policy sees, so when ``wrist_images`` is
        available we concatenate

            [main | left_wrist | right_wrist]

        along the width dimension for every vector-env element.  If wrist views
        are not present, fall back to the legacy single-image keys.
        """
        if obs.get("main_images", None) is not None:
            main = self._to_numpy(obs["main_images"])
            if main.ndim == 3:
                main_b = main[None]
            elif main.ndim == 4:
                main_b = main
            else:
                main_b = None

            wrist = obs.get("wrist_images", None)
            if main_b is not None and wrist is not None:
                wrist_np = self._to_numpy(wrist)
                # Expected layouts: [B, V, H, W, C] for real-world envs, or
                # [V, H, W, C] for a single unbatched observation.
                if wrist_np.ndim == 4:
                    wrist_np = wrist_np[None]
                if wrist_np.ndim == 5 and wrist_np.shape[0] == main_b.shape[0]:
                    per_env = []
                    for env_id in range(main_b.shape[0]):
                        views = [main_b[env_id]]
                        views.extend([wrist_np[env_id, view_id] for view_id in range(wrist_np.shape[1])])
                        try:
                            per_env.append(np.concatenate(views, axis=1))
                        except Exception:
                            # If resolutions unexpectedly differ, keep the main
                            # view rather than failing video recording.
                            per_env.append(main_b[env_id])
                    merged = np.stack(per_env, axis=0)
                    return merged if main.ndim == 4 else merged[0]
            return obs["main_images"]

        for key in ("images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            image_src = self._get_image_from_dict(obs)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    image_src = self._get_image_from_dict(item)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
        *,
        inference_frame: bool = False,
        inference_only: bool = False,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        if self.video_cfg.get("info_on_video", True):
            overlaid_images = []
            for env_id, img in enumerate(images):
                info = self._merge_overlay_info(
                    self._build_info_item(infos, rewards, terminations, env_id, time_idx),
                    env_id,
                )
                if inference_frame:
                    info["INFER_FRAME"] = 1
                overlaid_images.append(put_info_on_image(img, info))
            images = overlaid_images
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
        else:
            full_image = images[0]

        if inference_frame and self.video_cfg.get("save_inference_video", False):
            self.inference_images.append(full_image)
        if not inference_only:
            self.render_images.append(full_image)

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
        *,
        inference_frame_indices: Optional[set[int]] = None,
        inference_only: bool = False,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        inference_frame_indices = inference_frame_indices or set()
        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                step_info = infos[time_idx] if len(infos) > time_idx else None
                self._append_frame(
                    images,
                    step_info,
                    rewards,
                    terminations,
                    time_idx,
                    inference_frame=time_idx in inference_frame_indices,
                    inference_only=inference_only,
                )
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(
                images,
                infos,
                rewards,
                terminations,
                time_idx,
                inference_frame=time_idx in inference_frame_indices,
                inference_only=inference_only,
            )

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self._start_new_episode()
        # The reset observation is the first policy/WA inference observation.
        self.add_new_frames(obs, info, inference_frame_indices={0})
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self._update_chunk_tracking(reward, terminations)
        self.add_new_frames(obs, info, reward, terminations)
        done_mask = self._to_env_bool_array(terminations)
        self._advance_chunk_indices(done_mask)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]
            self._update_chunk_tracking(rewards, terminations)
            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            done_mask = self._to_env_bool_array(terminations)
            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                # The last observation before reset is the terminal frame; the
                # reset observation is the first inference frame for the next
                # episode.
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self._start_new_episode(done_mask)
                self.add_new_frames(reset_obs, None, inference_frame_indices={0})
            else:
                inference_indices = set()
                if self.video_cfg.get("mark_inference_frames", True) and isinstance(obs_list, (list, tuple)) and len(obs_list) > 0:
                    # The last observation of a chunk is the one that will be
                    # sent to rollout for the next chunk. Mark it in the full
                    # video and also mirror it to the inference-only video.
                    inference_indices.add(len(obs_list) - 1)
                self.add_new_frames(
                    obs_list,
                    infos_list,
                    rewards,
                    terminations,
                    inference_frame_indices=inference_indices,
                )
                self._advance_chunk_indices(done_mask)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to MP4 files (async)."""
        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        os.makedirs(output_dir, exist_ok=True)

        if self.render_images:
            mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
            frames = list(self.render_images)
            self.render_images = []
            self.video_cnt += 1
            self._submit_save(frames, mp4_path)

        if self.video_cfg.get("save_inference_video", False) and self.inference_images:
            subdir = str(self.video_cfg.get("inference_video_subdir", "inference_frames"))
            infer_dir = os.path.join(output_dir, subdir)
            os.makedirs(infer_dir, exist_ok=True)
            infer_path = os.path.join(infer_dir, f"{self.inference_video_cnt}.mp4")
            infer_frames = list(self.inference_images)
            self.inference_images = []
            self.inference_video_cnt += 1
            self._submit_save(infer_frames, infer_path)

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Submit a background job to save the video."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        try:
            video_writer = imageio.get_writer(mp4_path, fps=self._fps)
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
