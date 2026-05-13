#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
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

"""Convert a LeRobot v2.1 real-robot dataset into a GigaWA demo buffer.

The generated directory is directly loadable by ``TrajectoryReplayBuffer`` and
can be used as ``algorithm.demo_buffer.load_path`` in offline BC / critic / RL
training configs.

This converter intentionally stores demonstration actions in the same normalized
WA model-space used by the current actor/critic heads:

    executable absolute action -> delta/action normalization inverse of WA postprocess

It also stores the original executable action chunk in ``forward_inputs['action']``
for debugging and later real-robot action-space checks.

Typical usage from the RLinf repo root:

    python toolkits/replay_buffer/convert_lerobot_to_gigawa_demo_buffer.py \
      --lerobot-root /shared_disk/users/angen.ye/data/260423160824_7c5a \
      --output-path /shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/real_demo_buffer \
      --config-path examples/embodiment/config \
      --config-name offline_rl_pretrain \
      --prompt "put the cup into the cup holder"

For a one-episode smoke test, add:

    --max-episodes 1 --overwrite

Notes:
  1. LeRobot action can be 16-D in your real dataset. This converter crops it to
     the current GigaWA runtime action_dim, usually 14, by default.
  2. Your collected real demonstrations are assumed successful. Therefore the
     converter gives reward=1 at the last valid primitive step of each episode,
     and done/termination=True from that substep to the padded tail.
  3. ``curr_obs.ref_action`` remains the WA reference action by default. This is
     the correct semantic choice for RL: actor observes WA reference and learns
     to improve it toward ``actions``. If you have not patched the actor BC loss
     yet and only want a quick BC smoke test, pass
     ``--overwrite-ref-action-with-demo`` so existing BC code that targets
     curr_obs.ref_action will fit the human actions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Make this script runnable from either repo root or its own directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.giga_world_policy.giga_world_policy import GigaWorldPolicy


DEFAULT_MAIN_IMAGE_KEY = "observation.images.cam_high"
DEFAULT_LEFT_WRIST_IMAGE_KEY = "observation.images.cam_left_wrist"
DEFAULT_RIGHT_WRIST_IMAGE_KEY = "observation.images.cam_right_wrist"


@dataclass
class EpisodeRecord:
    episode_index: int
    parquet_path: Path
    chunk_index: int
    rows: list[dict[str, Any]]
    episode_meta: dict[str, Any]
    task: str


class VideoFrameReader:
    """Random-access-ish mp4 reader with robust AV1 fallback.

    The first backend is OpenCV because it is fast when the local FFmpeg build
    supports the video codec. Some LeRobot v2.1 exports are AV1-encoded mp4s,
    and many OpenCV wheels cannot decode AV1 even when the system `ffmpeg`
    command can. In that case we automatically fall back to a streaming ffmpeg
    rawvideo pipe.

    The converter asks frames in increasing order, so the ffmpeg fallback only
    decodes forward and caches recently requested frames. If an older frame is
    requested, the pipe is restarted from frame 0.
    """

    def __init__(self, path: Path, cache_size: int = 256, backend: str = "auto"):
        self.path = Path(path)
        self.cache_size = int(max(cache_size, 1))
        self.backend = str(backend or "auto").lower()
        if self.backend not in {"auto", "cv2", "ffmpeg"}:
            raise ValueError(f"Unsupported video backend: {backend}. Use auto/cv2/ffmpeg.")

        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._cap = None
        self._cv2_failed = False

        self._ffmpeg_proc: subprocess.Popen | None = None
        self._ffmpeg_next_idx = 0
        self._video_shape: tuple[int, int] | None = None  # (height, width)

        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")

    def _remember(self, frame_idx: int, frame_rgb: np.ndarray) -> np.ndarray:
        frame_idx = int(frame_idx)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        self._cache[frame_idx] = frame_rgb
        self._cache_order.append(frame_idx)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return frame_rgb

    def _require_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: opencv-python. Install cv2 or run with --video-backend ffmpeg."
            ) from exc
        return cv2

    def _open_cv2(self):
        if self._cap is not None:
            return
        cv2 = self._require_cv2()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            self._cv2_failed = True
            self._release_cv2()
            raise RuntimeError(f"OpenCV failed to open video: {self.path}")

    def _release_cv2(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_cv2(self, frame_idx: int) -> np.ndarray:
        self._open_cv2()
        cv2 = self._require_cv2()
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            self._cv2_failed = True
            self._release_cv2()
            raise RuntimeError(f"OpenCV failed to read frame {frame_idx} from {self.path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _probe_video_shape(self) -> tuple[int, int]:
        if self._video_shape is not None:
            return self._video_shape
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(self.path),
        ]
        try:
            out = subprocess.check_output(cmd, text=True)
            data = json.loads(out)
            stream = data["streams"][0]
            width = int(stream["width"])
            height = int(stream["height"])
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`ffprobe` was not found. Install ffmpeg, or transcode AV1 videos to H.264 first."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to probe video shape for {self.path}: {exc}") from exc
        self._video_shape = (height, width)
        return self._video_shape

    def _open_ffmpeg(self, restart: bool = False) -> None:
        if restart:
            self._close_ffmpeg()
        if self._ffmpeg_proc is not None:
            return
        height, width = self._probe_video_shape()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=height * width * 3 * 4,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "`ffmpeg` was not found. Install ffmpeg, or transcode AV1 videos to H.264 first."
            ) from exc
        self._ffmpeg_next_idx = 0

    def _close_ffmpeg(self) -> None:
        proc = self._ffmpeg_proc
        self._ffmpeg_proc = None
        self._ffmpeg_next_idx = 0
        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _read_exact_ffmpeg_frame(self) -> np.ndarray:
        height, width = self._probe_video_shape()
        frame_bytes = height * width * 3
        self._open_ffmpeg()
        assert self._ffmpeg_proc is not None and self._ffmpeg_proc.stdout is not None
        raw = self._ffmpeg_proc.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            stderr = ""
            try:
                if self._ffmpeg_proc.stderr:
                    stderr = self._ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                stderr = ""
            raise RuntimeError(
                f"ffmpeg failed to decode a full frame from {self.path}. "
                f"Got {len(raw)} bytes, expected {frame_bytes}. {stderr[:500]}"
            )
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()

    def _read_ffmpeg_sequential(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if self._ffmpeg_proc is None or frame_idx < self._ffmpeg_next_idx:
            self._open_ffmpeg(restart=True)
        while self._ffmpeg_next_idx <= frame_idx:
            frame = self._read_exact_ffmpeg_frame()
            current = self._ffmpeg_next_idx
            self._ffmpeg_next_idx += 1
            if current == frame_idx:
                return frame
            # Keep skipped frames only when cache is large enough to make it useful.
            if self.cache_size > 512:
                self._remember(current, frame)
        raise RuntimeError(f"Internal ffmpeg reader error for frame {frame_idx} in {self.path}")

    def read(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx in self._cache:
            return self._cache[frame_idx]

        if self.backend in {"auto", "cv2"} and not self._cv2_failed:
            try:
                return self._remember(frame_idx, self._read_cv2(frame_idx))
            except Exception:
                if self.backend == "cv2":
                    raise
                # AV1 videos often land here. Fall through to the ffmpeg CLI backend.
                print(f"[video] OpenCV cannot decode {self.path.name}; falling back to ffmpeg CLI.")

        if self.backend in {"auto", "ffmpeg"}:
            return self._remember(frame_idx, self._read_ffmpeg_sequential(frame_idx))

        raise RuntimeError(f"Failed to read frame {frame_idx} from {self.path}")

    def close(self) -> None:
        self._release_cv2()
        self._close_ffmpeg()
        self._cache.clear()
        self._cache_order.clear()


class EpisodeImageProvider:
    """Fetch high/left/right RGB images for one LeRobot episode.

    Supports the common LeRobot v2.1 video layout and a fallback for parquet rows
    that contain image bytes.
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        info_json: dict[str, Any],
        episode_index: int,
        rows: list[dict[str, Any]],
        main_image_key: str,
        left_wrist_image_key: str,
        right_wrist_image_key: str,
        video_cache_size: int = 256,
        video_backend: str = "auto",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.info_json = info_json
        self.episode_index = int(episode_index)
        self.rows = rows
        self.main_image_key = main_image_key
        self.left_wrist_image_key = left_wrist_image_key
        self.right_wrist_image_key = right_wrist_image_key
        self.video_cache_size = int(video_cache_size)
        self.video_backend = str(video_backend or "auto")
        self._readers: dict[str, VideoFrameReader] = {}

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    @staticmethod
    def _is_image_struct(value: Any) -> bool:
        return isinstance(value, dict) and ("bytes" in value or "path" in value)

    @staticmethod
    def _decode_image_bytes(raw_bytes: Any) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Missing dependency: Pillow is required for image-byte LeRobot rows.") from exc
        import io

        if isinstance(raw_bytes, memoryview):
            raw_bytes = raw_bytes.tobytes()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return np.asarray(image.convert("RGB"))

    def _episode_chunk(self) -> int:
        chunks_size = int(self.info_json.get("chunks_size", 1000))
        return self.episode_index // max(chunks_size, 1)

    def _video_path_for_key(self, video_key: str) -> Path:
        pattern = self.info_json.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
        rel = pattern.format(
            episode_chunk=self._episode_chunk(),
            video_key=video_key,
            episode_index=self.episode_index,
        )
        return self.dataset_root / rel

    def _get_video_reader(self, video_key: str) -> VideoFrameReader:
        if video_key not in self._readers:
            self._readers[video_key] = VideoFrameReader(
                self._video_path_for_key(video_key),
                cache_size=self.video_cache_size,
                backend=self.video_backend,
            )
        return self._readers[video_key]

    def _read_one_image(self, frame_idx: int, key: str) -> np.ndarray:
        row = self.rows[int(frame_idx)]
        value = row.get(key, None)
        if self._is_image_struct(value):
            if value.get("bytes"):
                return self._decode_image_bytes(value["bytes"])
            path_value = value.get("path")
            if path_value:
                path = Path(path_value)
                if not path.is_absolute():
                    path = self.dataset_root / path
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise RuntimeError("Missing dependency: Pillow is required for image-path LeRobot rows.") from exc
                with Image.open(path) as image:
                    return np.asarray(image.convert("RGB"))

        return self._get_video_reader(key).read(frame_idx)

    def build_env_obs(self, frame_idx: int, state: torch.Tensor, task_description: str) -> dict[str, Any]:
        main = self._read_one_image(frame_idx, self.main_image_key)
        left = self._read_one_image(frame_idx, self.left_wrist_image_key)
        right = self._read_one_image(frame_idx, self.right_wrist_image_key)

        return {
            "main_images": torch.from_numpy(np.ascontiguousarray(main)).unsqueeze(0),
            "wrist_images": torch.from_numpy(
                np.ascontiguousarray(np.stack([left, right], axis=0))
            ).unsqueeze(0),
            "states": state.float().view(1, -1),
            "task_descriptions": [task_description],
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2.1 data to a GigaWA TrajectoryReplayBuffer checkpoint."
    )
    parser.add_argument("--lerobot-root", type=str, required=True, help="LeRobot dataset root.")
    parser.add_argument("--output-path", type=str, required=True, help="Output replay/demo buffer directory.")
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/embodiment/config",
        help="Hydra config directory containing offline_rl_pretrain.yaml.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="offline_rl_pretrain",
        help="Hydra config name used to instantiate actor.model.giga_world_policy.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. actor.model.giga_world_policy.num_inference_steps=4",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Override WA text prompt for every episode.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes for smoke test.")
    parser.add_argument("--episode-indices", type=str, default=None, help="Comma/range list, e.g. '0,3,5-8'.")
    parser.add_argument("--num-action-chunks", type=int, default=None, help="Override actor.model.num_action_chunks.")
    parser.add_argument("--action-dim", type=int, default=None, help="Override actor.model.action_dim.")
    parser.add_argument("--reward-value", type=float, default=1.0, help="Terminal reward for a successful demo episode.")
    parser.add_argument(
        "--all-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat every LeRobot episode as successful and put terminal reward at its last valid step.",
    )
    parser.add_argument(
        "--mark-intervene-flags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set intervene_flags to all True because real demonstrations are expert data.",
    )
    parser.add_argument(
        "--overwrite-ref-action-with-demo",
        action="store_true",
        help=(
            "Store demo model-space action in curr_obs/next_obs.ref_action instead of WA ref_action. "
            "Only use this for quick compatibility with old BC code that targets curr_obs.ref_action."
        ),
    )
    parser.add_argument(
        "--state-key", type=str, default="observation.state", help="LeRobot state column name."
    )
    parser.add_argument("--action-key", type=str, default="action", help="LeRobot action column name.")
    parser.add_argument("--main-image-key", type=str, default=DEFAULT_MAIN_IMAGE_KEY)
    parser.add_argument("--left-wrist-image-key", type=str, default=DEFAULT_LEFT_WRIST_IMAGE_KEY)
    parser.add_argument("--right-wrist-image-key", type=str, default=DEFAULT_RIGHT_WRIST_IMAGE_KEY)
    parser.add_argument("--video-cache-size", type=int, default=256)
    parser.add_argument(
        "--video-backend",
        type=str,
        default="auto",
        choices=["auto", "cv2", "ffmpeg"],
        help="Video decoder backend. Use ffmpeg for AV1 LeRobot mp4s when OpenCV cannot decode them.",
    )
    parser.add_argument("--trajectory-format", type=str, default="pt", choices=["pt", "pkl"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Delete output dir if it already exists.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to an existing output directory is not supported; this only skips deleting unrelated files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read metadata/parquet and print planned episodes without loading GigaWA or writing output."
    )
    return parser


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the training environment, e.g. `pip install pyarrow`."
        ) from exc
    return pq


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_episode_indices(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def infer_episode_index(parquet_path: Path, rows: list[dict[str, Any]]) -> int:
    if rows and "episode_index" in rows[0]:
        value = rows[0]["episode_index"]
        if hasattr(value, "as_py"):
            value = value.as_py()
        return int(value)
    match = re.search(r"episode_(\d+)\.parquet$", parquet_path.name)
    if not match:
        raise ValueError(f"Unable to infer episode_index from {parquet_path}")
    return int(match.group(1))


def resolve_parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if data_dir.exists():
        files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    else:
        files = sorted(dataset_root.glob("**/episode_*.parquet"))
    return [p for p in files if not p.name.startswith("._")]


def episode_chunk_from_index(episode_index: int, info_json: dict[str, Any]) -> int:
    chunks_size = int(info_json.get("chunks_size", 1000))
    return int(episode_index) // max(chunks_size, 1)


def to_1d_float_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().flatten()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float().flatten()
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32).flatten()
    # pyarrow can produce dict-like fixed-size-list scalars in some versions.
    try:
        return torch.tensor(list(value), dtype=torch.float32).flatten()
    except Exception as exc:
        raise TypeError(f"Cannot convert {name}={type(value)} to float tensor.") from exc


def get_row_scalar(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_hydra_cfg(config_path: str, config_name: str, overrides: list[str]) -> DictConfig:
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise SystemExit("Missing dependency: hydra-core is required to load RLinf configs.") from exc

    config_dir = Path(config_path).expanduser()
    if not config_dir.is_absolute():
        config_dir = (REPO_ROOT / config_dir).resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Config path not found: {config_dir}")

    config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        return compose(config_name=config_name, overrides=overrides)


def dtype_from_precision(precision: str | None) -> torch.dtype:
    p = str(precision or "bf16").lower()
    if p in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if p in {"fp16", "float16", "half"}:
        return torch.float16
    if p in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def maybe_override_cfg(cfg: DictConfig, args: argparse.Namespace) -> None:
    if args.num_action_chunks is not None:
        cfg.actor.model.num_action_chunks = int(args.num_action_chunks)
    if args.action_dim is not None:
        cfg.actor.model.action_dim = int(args.action_dim)
    if args.prompt is not None:
        cfg.actor.model.giga_world_policy.prompt = str(args.prompt)


def build_policy(cfg: DictConfig) -> GigaWorldPolicy:
    model_cfg = cfg.actor.model
    dtype = dtype_from_precision(model_cfg.get("precision", "bf16"))
    policy = GigaWorldPolicy(model_cfg, torch_dtype=dtype)

    # Important: GigaWorldPolicy.__init__ moves the WA pipe to device_ref, but
    # several normalization statistics are registered as buffers *after* that.
    # In the training worker the whole policy is later moved with .to(device),
    # but this standalone converter has to do that explicitly; otherwise
    # _normalize_state/_postprocess_pred_delta can mix cuda tensors with CPU
    # buffers and raise "Expected all tensors to be on the same device".
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    return policy


def executable_to_model_action(
    *,
    action_exec: torch.Tensor,
    state: torch.Tensor,
    policy: GigaWorldPolicy,
) -> torch.Tensor:
    """Invert GigaWA _postprocess_pred_delta for one primitive action.

    _postprocess_pred_delta:
        pred_delta = action_model * delta_std + delta_mean
        pred_action = pred_delta
        pred_action[delta_mask] += state[delta_mask]

    Therefore:
        raw_delta[delta_mask] = action_exec[delta_mask] - state[delta_mask]
        action_model = (raw_delta - delta_mean) / delta_std
    """

    model_action_dim = int(policy.model_action_dim)
    env_action_dim = int(policy.env_action_dim)

    action_exec = action_exec.float().flatten()
    if action_exec.numel() < env_action_dim:
        raise ValueError(
            f"LeRobot action has only {action_exec.numel()} dims, but policy.env_action_dim={env_action_dim}."
        )
    action_exec = action_exec[:env_action_dim]

    if action_exec.numel() < model_action_dim:
        pad = torch.zeros(model_action_dim - action_exec.numel(), dtype=torch.float32)
        raw_delta = torch.cat([action_exec, pad], dim=0)
    else:
        raw_delta = action_exec[:model_action_dim].clone()

    state = state.float().flatten()
    if state.numel() < model_action_dim:
        state = torch.cat([state, torch.zeros(model_action_dim - state.numel())], dim=0)
    else:
        state = state[:model_action_dim]

    delta_mask = policy.delta_mask.detach().cpu().to(torch.bool)[:model_action_dim]
    raw_delta[delta_mask] -= state[delta_mask]

    delta_mean = policy.delta_mean.detach().cpu().float()[:model_action_dim]
    delta_std = policy.delta_std.detach().cpu().float()[:model_action_dim]
    delta_std = torch.where(delta_std.abs() < 1e-8, torch.ones_like(delta_std), delta_std)
    return ((raw_delta - delta_mean) / delta_std).float()


def pad_action_chunk(x: torch.Tensor, start_idx: int, valid_len: int, chunk: int) -> torch.Tensor:
    # x: [primitive_steps, action_dim]
    y = x[start_idx : start_idx + valid_len]
    if y.shape[0] <= 0:
        raise ValueError("valid_len must be positive")
    if valid_len < chunk:
        pad = y[-1:].expand(chunk - valid_len, -1)
        y = torch.cat([y, pad], dim=0)
    return y.reshape(1, chunk * y.shape[-1]).contiguous()


def build_scalar_chunk(
    *,
    start_idx: int,
    valid_len: int,
    chunk: int,
    dtype: torch.dtype,
    fill_value: float | bool,
) -> torch.Tensor:
    values = torch.full((chunk,), fill_value=fill_value, dtype=dtype)
    if valid_len < chunk:
        # Caller may overwrite terminal tail below. This function just creates the base.
        pass
    return values.view(1, chunk).contiguous()


def infer_task_for_episode(
    *,
    episode_meta: dict[str, Any],
    first_row: dict[str, Any],
    task_map: dict[int, str],
    prompt_override: str | None,
) -> str:
    if prompt_override is not None:
        return prompt_override
    tasks = episode_meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    task_index = get_row_scalar(first_row, "task_index", None)
    if task_index is not None and int(task_index) in task_map:
        return str(task_map[int(task_index)])
    return ""


def read_episode_record(
    *,
    parquet_path: Path,
    dataset_root: Path,
    info_json: dict[str, Any],
    episode_meta_map: dict[int, dict[str, Any]],
    task_map: dict[int, str],
    prompt_override: str | None,
) -> EpisodeRecord | None:
    pq = require_pyarrow()
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if not rows:
        return None
    episode_index = infer_episode_index(parquet_path, rows)
    episode_meta = episode_meta_map.get(episode_index, {})
    task = infer_task_for_episode(
        episode_meta=episode_meta,
        first_row=rows[0],
        task_map=task_map,
        prompt_override=prompt_override,
    )
    return EpisodeRecord(
        episode_index=episode_index,
        parquet_path=parquet_path,
        chunk_index=episode_chunk_from_index(episode_index, info_json),
        rows=rows,
        episode_meta=episode_meta,
        task=task,
    )


def collect_states_actions(
    rows: list[dict[str, Any]],
    *,
    state_key: str,
    action_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    actions = []
    for i, row in enumerate(rows):
        if state_key not in row:
            raise KeyError(f"Missing state key '{state_key}' in row {i}")
        if action_key not in row:
            raise KeyError(f"Missing action key '{action_key}' in row {i}")
        states.append(to_1d_float_tensor(row[state_key], name=f"{state_key}[{i}]"))
        actions.append(to_1d_float_tensor(row[action_key], name=f"{action_key}[{i}]"))
    return torch.stack(states, dim=0), torch.stack(actions, dim=0)


def extract_features_for_needed_frames(
    *,
    policy: GigaWorldPolicy,
    provider: EpisodeImageProvider,
    states: torch.Tensor,
    task: str,
    needed_frame_indices: Iterable[int],
) -> dict[int, dict[str, torch.Tensor]]:
    cache: dict[int, dict[str, torch.Tensor]] = {}
    for frame_idx in tqdm(
        sorted(set(int(i) for i in needed_frame_indices)),
        desc=f"extract WA features ep{provider.episode_index:06d}",
        leave=False,
    ):
        obs = provider.build_env_obs(frame_idx, states[frame_idx], task)
        with torch.no_grad():
            feat = policy.extract_frozen_backbone_batch(obs)
        cache[frame_idx] = {
            "visual_latent": feat["visual_latent"].detach().cpu().float().contiguous(),
            "robot_state": feat["robot_state"].detach().cpu().float().contiguous(),
            "ref_action": feat["ref_action"].detach().cpu().float().contiguous(),
            "ref_action_exec": feat.get("ref_action_exec", torch.empty(0)).detach().cpu().float().contiguous(),
        }
    return cache


def build_gigawa_trajectory_from_episode(
    *,
    episode: EpisodeRecord,
    dataset_root: Path,
    info_json: dict[str, Any],
    policy: GigaWorldPolicy,
    args: argparse.Namespace,
) -> Trajectory:
    rows = episode.rows
    states, raw_actions = collect_states_actions(rows, state_key=args.state_key, action_key=args.action_key)

    if states.shape[0] < 2:
        raise ValueError(f"Episode {episode.episode_index} has fewer than 2 frames.")

    action_chunk = int(policy.action_chunk)
    action_dim = int(policy.model_action_dim)
    env_action_dim = int(policy.env_action_dim)

    # A LeRobot frame sequence has observations at [0..N-1].  We use action[t]
    # for transition obs[t] -> obs[t+1], so there are N-1 valid primitive steps.
    primitive_steps = min(int(raw_actions.shape[0]), int(states.shape[0]) - 1)
    if primitive_steps <= 0:
        raise ValueError(f"Episode {episode.episode_index} has no valid primitive transitions.")

    raw_actions = raw_actions[:primitive_steps]
    states_for_actions = states[:primitive_steps]

    model_actions = []
    exec_actions = []
    for t in range(primitive_steps):
        action_exec = raw_actions[t, :env_action_dim].float().contiguous()
        action_model = executable_to_model_action(
            action_exec=action_exec,
            state=states_for_actions[t],
            policy=policy,
        )
        model_actions.append(action_model)
        exec_actions.append(action_exec)
    model_actions_t = torch.stack(model_actions, dim=0).float().contiguous()  # [P, A_model]
    exec_actions_t = torch.stack(exec_actions, dim=0).float().contiguous()    # [P, A_env]

    num_chunks = int(math.ceil(primitive_steps / action_chunk))
    start_indices = [i * action_chunk for i in range(num_chunks)]

    needed_frames: list[int] = []
    valid_lens: list[int] = []
    next_indices: list[int] = []
    for start_idx in start_indices:
        valid_len = min(action_chunk, primitive_steps - start_idx)
        next_idx = min(start_idx + valid_len, int(states.shape[0]) - 1)
        valid_lens.append(valid_len)
        next_indices.append(next_idx)
        needed_frames.append(start_idx)
        needed_frames.append(next_idx)

    provider = EpisodeImageProvider(
        dataset_root=dataset_root,
        info_json=info_json,
        episode_index=episode.episode_index,
        rows=rows,
        main_image_key=args.main_image_key,
        left_wrist_image_key=args.left_wrist_image_key,
        right_wrist_image_key=args.right_wrist_image_key,
        video_cache_size=args.video_cache_size,
        video_backend=args.video_backend,
    )
    try:
        feature_cache = extract_features_for_needed_frames(
            policy=policy,
            provider=provider,
            states=states,
            task=episode.task,
            needed_frame_indices=needed_frames,
        )
    finally:
        provider.close()

    actions_chunks = []
    exec_action_chunks = []
    rewards_chunks = []
    terminations_chunks = []
    truncations_chunks = []
    dones_chunks = []
    intervene_chunks = []
    versions_chunks = []
    curr_visual_latents = []
    curr_robot_states = []
    curr_ref_actions = []
    next_visual_latents = []
    next_robot_states = []
    next_ref_actions = []

    for chunk_idx, start_idx in enumerate(start_indices):
        valid_len = valid_lens[chunk_idx]
        next_idx = next_indices[chunk_idx]

        model_chunk = pad_action_chunk(model_actions_t, start_idx, valid_len, action_chunk)  # [1, C*A]
        exec_chunk = pad_action_chunk(exec_actions_t, start_idx, valid_len, action_chunk)    # [1, C*A_env]
        actions_chunks.append(model_chunk)
        exec_action_chunks.append(exec_chunk)
        intervene_chunks.append(torch.full_like(model_chunk, bool(args.mark_intervene_flags), dtype=torch.bool))
        versions_chunks.append(torch.zeros((1, action_chunk), dtype=torch.float32))

        rewards = torch.zeros((1, action_chunk), dtype=torch.float32)
        terminations = torch.zeros((1, action_chunk), dtype=torch.bool)
        truncations = torch.zeros((1, action_chunk), dtype=torch.bool)
        dones = torch.zeros((1, action_chunk), dtype=torch.bool)

        is_last_chunk = chunk_idx == len(start_indices) - 1
        if args.all_success and is_last_chunk:
            terminal_substep = max(0, valid_len - 1)
            rewards[0, terminal_substep] = float(args.reward_value)
            terminations[0, terminal_substep:] = True
            dones[0, terminal_substep:] = True
            # Padded tail is terminal continuation, not a timeout.
            truncations[0, terminal_substep:] = False

        rewards_chunks.append(rewards)
        terminations_chunks.append(terminations)
        truncations_chunks.append(truncations)
        dones_chunks.append(dones)

        curr_feat = feature_cache[start_idx]
        next_feat = feature_cache[next_idx]
        curr_visual_latents.append(curr_feat["visual_latent"])
        curr_robot_states.append(curr_feat["robot_state"])
        next_visual_latents.append(next_feat["visual_latent"])
        next_robot_states.append(next_feat["robot_state"])

        if args.overwrite_ref_action_with_demo:
            curr_ref_actions.append(model_chunk.view(1, action_chunk, action_dim))
            # For next_obs.ref_action, use the next chunk of demo actions if available;
            # otherwise repeat the current terminal chunk. This keeps shapes valid for
            # TD target computation during smoke tests.
            if chunk_idx + 1 < len(start_indices):
                next_start = start_indices[chunk_idx + 1]
                next_valid = valid_lens[chunk_idx + 1]
                next_demo_chunk = pad_action_chunk(model_actions_t, next_start, next_valid, action_chunk)
                next_ref_actions.append(next_demo_chunk.view(1, action_chunk, action_dim))
            else:
                next_ref_actions.append(model_chunk.view(1, action_chunk, action_dim))
        else:
            curr_ref_actions.append(curr_feat["ref_action"])
            next_ref_actions.append(next_feat["ref_action"])

    actions = torch.stack(actions_chunks, dim=0).float().contiguous()                  # [T,1,C*A]
    exec_actions_flat = torch.stack(exec_action_chunks, dim=0).float().contiguous()    # [T,1,C*A_env]
    rewards = torch.stack(rewards_chunks, dim=0).float().contiguous()                  # [T,1,C]
    terminations = torch.stack(terminations_chunks, dim=0).bool().contiguous()
    truncations = torch.stack(truncations_chunks, dim=0).bool().contiguous()
    dones = torch.stack(dones_chunks, dim=0).bool().contiguous()
    intervene_flags = torch.stack(intervene_chunks, dim=0).bool().contiguous()
    versions = torch.stack(versions_chunks, dim=0).float().contiguous()

    model_weights_id = get_model_weights_id(versions)

    sample_info = {
        "source_dataset": str(dataset_root),
        "source_format": "lerobot_v2.1",
        "source_episode_index": int(episode.episode_index),
        "source_parquet_path": str(episode.parquet_path),
        "source_episode_length": int(len(rows)),
        "primitive_steps": int(primitive_steps),
        "task": episode.task,
        "all_success_assumed": bool(args.all_success),
        "action_space": "giga_model_normalized_delta",
        "exec_action_space": "real_robot_executable_absolute",
        "cropped_lerobot_action_dim": int(env_action_dim),
        "lerobot_raw_action_dim": int(raw_actions.shape[-1]),
        "overwrite_ref_action_with_demo": bool(args.overwrite_ref_action_with_demo),
    }

    metadata = dict(sample_info)
    metadata.update(
        {
            "sliding_offset": 0,
            "is_sliding_window_augmented": False,
            "success": bool(args.all_success),
            "reward_max": float(rewards.max().item()) if rewards.numel() else 0.0,
            "reward_sum": float(rewards.sum().item()) if rewards.numel() else 0.0,
        }
    )

    return Trajectory(
        max_episode_length=int(primitive_steps),
        model_weights_id=model_weights_id,
        actions=actions,
        intervene_flags=intervene_flags,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        prev_logprobs=None,
        prev_values=None,
        versions=versions,
        forward_inputs={
            "action": exec_actions_flat,
            "model_action": actions,
        },
        curr_obs={
            "visual_latent": torch.stack(curr_visual_latents, dim=0).float().contiguous(),
            "robot_state": torch.stack(curr_robot_states, dim=0).float().contiguous(),
            "ref_action": torch.stack(curr_ref_actions, dim=0).float().contiguous(),
        },
        next_obs={
            "visual_latent": torch.stack(next_visual_latents, dim=0).float().contiguous(),
            "robot_state": torch.stack(next_robot_states, dim=0).float().contiguous(),
            "ref_action": torch.stack(next_ref_actions, dim=0).float().contiguous(),
        },
        sample_infos=[sample_info],
        metadata=metadata,
    ).contiguous_()


def validate_output_buffer(output_path: Path, num_chunks: int = 4) -> None:
    buffer = TrajectoryReplayBuffer(
        seed=1234,
        enable_cache=True,
        cache_size=4,
        sample_window_size=16,
        auto_save=False,
        trajectory_format="pt",
    )
    buffer.load_checkpoint(str(output_path), is_distributed=False)
    batch = buffer.sample(num_chunks=num_chunks)
    print("\n[validate] loaded buffer")
    print(f"  num_trajectories={len(buffer)} total_samples={buffer.total_samples}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  batch.{key}: {tuple(value.shape)} {value.dtype}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  batch.{key}.{sub_key}: {tuple(sub_value.shape)} {sub_value.dtype}")


def prepare_output_dir(output_path: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        elif any(output_path.iterdir()):
            raise FileExistsError(
                f"Output path already exists and is not empty: {output_path}\n"
                "Pass --overwrite to replace it."
            )
    output_path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = Path(args.lerobot_root).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"LeRobot root not found: {dataset_root}")

    meta_dir = dataset_root / "meta"
    info_json = load_json(meta_dir / "info.json")
    episode_meta_map = {
        int(row["episode_index"]): row
        for row in load_jsonl(meta_dir / "episodes.jsonl")
        if "episode_index" in row
    }
    task_map = {
        int(row["task_index"]): str(row["task"])
        for row in load_jsonl(meta_dir / "tasks.jsonl")
        if "task_index" in row and "task" in row
    }

    parquet_files = resolve_parquet_files(dataset_root)
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under: {dataset_root}")

    selected_indices = parse_episode_indices(args.episode_indices)

    planned_records: list[EpisodeRecord] = []
    for parquet_path in parquet_files:
        episode = read_episode_record(
            parquet_path=parquet_path,
            dataset_root=dataset_root,
            info_json=info_json,
            episode_meta_map=episode_meta_map,
            task_map=task_map,
            prompt_override=args.prompt,
        )
        if episode is None:
            continue
        if selected_indices is not None and episode.episode_index not in selected_indices:
            continue
        planned_records.append(episode)
        if args.max_episodes is not None and len(planned_records) >= int(args.max_episodes):
            break

    if not planned_records:
        raise RuntimeError("No episodes selected for conversion.")

    print("[plan]")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_path:  {output_path}")
    print(f"  episodes:     {len(planned_records)}")
    for episode in planned_records[:10]:
        print(
            f"  - ep{episode.episode_index:06d}: rows={len(episode.rows)} "
            f"task={episode.task!r} parquet={episode.parquet_path.name}"
        )
    if len(planned_records) > 10:
        print(f"  ... {len(planned_records) - 10} more")

    if args.dry_run:
        print("[dry-run] stop before loading GigaWA / writing output.")
        return

    prepare_output_dir(output_path, overwrite=args.overwrite, dry_run=args.dry_run)

    cfg = load_hydra_cfg(args.config_path, args.config_name, args.overrides)
    maybe_override_cfg(cfg, args)

    print("\n[policy]")
    print(f"  model_path:       {cfg.actor.model.model_path}")
    print(f"  norm_json:        {cfg.actor.model.giga_world_policy.norm_json}")
    print(f"  wa_root:          {cfg.actor.model.giga_world_policy.wa_root}")
    print(f"  base_model_dir:   {cfg.actor.model.giga_world_policy.base_model_dir}")
    print(f"  num_chunks:       {cfg.actor.model.num_action_chunks}")
    print(f"  action_dim:       {cfg.actor.model.action_dim}")
    print(f"  prompt override:  {cfg.actor.model.giga_world_policy.get('prompt', None)!r}")

    policy = build_policy(cfg)

    buffer = TrajectoryReplayBuffer(
        seed=int(args.seed),
        enable_cache=False,
        auto_save=True,
        auto_save_path=str(output_path),
        trajectory_format=args.trajectory_format,
    )

    converted = 0
    failed: list[tuple[int, str]] = []
    for episode in tqdm(planned_records, desc="convert episodes"):
        try:
            traj = build_gigawa_trajectory_from_episode(
                episode=episode,
                dataset_root=dataset_root,
                info_json=info_json,
                policy=policy,
                args=args,
            )
            buffer.add_trajectories([traj])
            converted += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            failed.append((episode.episode_index, repr(exc)))
            print(f"[warn] failed ep{episode.episode_index:06d}: {exc}")

    buffer.close(wait=True)

    print("\n[done]")
    print(f"  converted episodes: {converted}")
    print(f"  failed episodes:    {len(failed)}")
    print(f"  output_path:        {output_path}")
    if failed:
        fail_path = output_path / "failed_episodes.json"
        fail_path.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  failed list:        {fail_path}")

    if converted > 0:
        validate_output_buffer(output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
