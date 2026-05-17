# Copyright 2026 The RLinf Authors.

"""Episode-level real-world GigaWA collector.

This collector is designed for real-robot data collection with the smallest
practical GPU/RAM footprint:

- launch only rollout + env workers; do not launch actor/training workers;
- force one env/rollout interaction to contain one 12-step chunk;
- receive one small chunk trajectory at a time;
- accumulate chunks in the main process until c/a/timeout ends the episode;
- save one slim .pt and one inference-frame mp4 per episode.

Why this exists:
The normal rollout-level collector can return a trajectory with tens of chunks
and raw camera tensors. Passing that object through a Ray ChannelWorker can use
many GB of RAM. Here the channel only carries one chunk at a time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

try:
    from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
except Exception:  # pragma: no cover - collector can still run without keyboard polling.
    KeyboardListener = None

mp.set_start_method("spawn", force=True)

_IMAGE_KEYS = {
    "main_images",
    "wrist_images",
    "extra_view_images",
    "_chunk_step_main_images_seq",
    "_chunk_step_wrist_images_seq",
    "_chunk_step_extra_view_images_seq",
}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clone_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {k: _clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(v) for v in value)
    return value


def _drop_image_keys(obj: Any) -> Any:
    """Remove raw image tensors recursively from dictionaries.

    This keeps .pt files small. Inference-frame videos are written separately.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in _IMAGE_KEYS or (isinstance(k, str) and k.endswith("images")):
                continue
            out[k] = _drop_image_keys(v)
        return out
    if isinstance(obj, list):
        return [_drop_image_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_drop_image_keys(v) for v in obj)
    return obj


def _slim_trajectory(traj: Trajectory) -> Trajectory:
    """Clone a chunk trajectory and strip raw images before episode buffering."""
    slim = Trajectory(max_episode_length=traj.max_episode_length)
    for field_name in Trajectory.__dataclass_fields__.keys():
        value = getattr(traj, field_name, None)
        if field_name in {"curr_obs", "next_obs", "forward_inputs"}:
            value = _drop_image_keys(value)
        setattr(slim, field_name, _clone_cpu(value))
    _clean_invalid_padding_(slim)
    return slim


def _last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the last valid primitive-step index for each [T, B] row.

    mask shape is [num_chunks, num_envs, chunk_size].  Rows with no valid step
    return 0; such rows should not normally appear in real collection.
    """
    m = mask.to(torch.bool)
    step_idx = torch.arange(m.shape[-1], device=m.device).view(*([1] * (m.ndim - 1)), -1)
    idx = torch.where(m, step_idx, torch.zeros_like(step_idx)).amax(dim=-1)
    return idx.to(torch.long)


def _repair_terminal_flags_before_padding_cleanup_(traj: Trajectory) -> None:
    """Keep success/failure terminal events before clearing padded entries.

    RealWorldEnv may collapse a terminal flag to the end of a padded chunk when
    auto_reset=True.  If a failure key ``a`` has reward 0, blindly clearing
    invalid padded positions makes the collector unable to distinguish it from
    a normal nonterminal chunk.  Before masking invalid positions, move any
    terminal/truncation/done flag that sits in padding back to the last valid
    primitive step of the same chunk/env.

    Success is also made explicit: every valid reward>0 position is marked as
    done/termination, so saved .pt files have consistent reward/done semantics.
    """
    mask = getattr(traj, "action_valid_mask", None)
    if not torch.is_tensor(mask):
        return
    mask = mask.to(torch.bool)
    if mask.ndim < 3:
        return
    last_idx = _last_valid_indices(mask)  # [T, B]

    rewards = getattr(traj, "rewards", None)
    term = getattr(traj, "terminations", None)
    trunc = getattr(traj, "truncations", None)
    dones = getattr(traj, "dones", None)

    if torch.is_tensor(term) and term.shape == mask.shape:
        term = term.clone().to(torch.bool)
    else:
        term = torch.zeros_like(mask, dtype=torch.bool)
    if torch.is_tensor(trunc) and trunc.shape == mask.shape:
        trunc = trunc.clone().to(torch.bool)
    else:
        trunc = torch.zeros_like(mask, dtype=torch.bool)
    if torch.is_tensor(dones) and dones.shape == mask.shape:
        dones = dones.clone().to(torch.bool)
    else:
        dones = term | trunc

    terminal_any = (term | trunc | dones).any(dim=-1)  # [T, B]
    if terminal_any.any():
        flat_terminal = terminal_any.nonzero(as_tuple=False)
        for tb in flat_terminal.tolist():
            t, b = int(tb[0]), int(tb[1])
            j = int(last_idx[t, b].item())
            # Preserve whether it was truncation vs termination when possible.
            row_term_any = bool(term[t, b].any().item())
            row_trunc_any = bool(trunc[t, b].any().item())
            if row_term_any or not row_trunc_any:
                term[t, b, j] = True
            if row_trunc_any and not row_term_any:
                trunc[t, b, j] = True
            dones[t, b, j] = True

    if torch.is_tensor(rewards) and rewards.shape == mask.shape:
        success_pos = (rewards.float() > 0) & mask
        if success_pos.any():
            term[success_pos] = True
            dones[success_pos] = True

    setattr(traj, "terminations", term.cpu().contiguous())
    setattr(traj, "truncations", trunc.cpu().contiguous())
    setattr(traj, "dones", dones.cpu().contiguous())


def _clean_invalid_padding_(traj: Trajectory) -> None:
    """Make invalid/padded primitive steps semantically clean.

    For positions with action_valid_mask=False, reward/done/intervene must be 0.
    Terminal flags are first moved to the last valid primitive step so failure
    episodes with reward=0 are still saved.
    """
    _repair_terminal_flags_before_padding_cleanup_(traj)
    mask = getattr(traj, "action_valid_mask", None)
    if not torch.is_tensor(mask):
        return
    invalid = ~mask.to(torch.bool)
    for name in ("rewards",):
        value = getattr(traj, name, None)
        if torch.is_tensor(value) and value.shape == mask.shape:
            value = value.clone()
            value[invalid] = 0
            setattr(traj, name, value.cpu().contiguous())
    for name in ("terminations", "truncations", "dones", "intervene_flags"):
        value = getattr(traj, name, None)
        if torch.is_tensor(value) and value.shape == mask.shape:
            value = value.clone().to(torch.bool)
            value[invalid] = False
            setattr(traj, name, value.cpu().contiguous())
    if isinstance(traj.forward_inputs, dict):
        fi = dict(traj.forward_inputs)
        for name in ("action_valid_mask", "intervene_flags"):
            value = fi.get(name, None)
            if torch.is_tensor(value) and value.shape == mask.shape:
                value = value.clone().to(torch.bool)
                if name == "intervene_flags":
                    value[invalid] = False
                fi[name] = value.cpu().contiguous()
        traj.forward_inputs = fi


def _tensor_cat(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.cpu().contiguous() for v in values], dim=0).contiguous()


def _cat_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}
    keys = set(dicts[0].keys())
    for d in dicts[1:]:
        keys &= set(d.keys())
    out: dict[str, Any] = {}
    for key in sorted(keys):
        values = [d[key] for d in dicts]
        if all(torch.is_tensor(v) for v in values):
            out[key] = _tensor_cat(values)
        elif all(isinstance(v, dict) for v in values):
            out[key] = _cat_dicts(values)
        else:
            # Keep only non-image small python metadata if it is identical-ish.
            out[key] = values[-1]
    return out


def _concat_episode(parts: list[Trajectory], metadata: dict[str, Any]) -> Trajectory:
    assert parts, "Cannot concatenate an empty episode."
    ep = Trajectory(max_episode_length=parts[0].max_episode_length)
    ep.model_weights_id = parts[-1].model_weights_id
    ep.sample_infos = parts[0].sample_infos
    ep.metadata = dict(metadata)

    for field_name in (
        "actions",
        "intervene_flags",
        "action_valid_mask",
        "rewards",
        "terminations",
        "truncations",
        "dones",
        "prev_logprobs",
        "prev_values",
        "versions",
    ):
        tensors = [getattr(p, field_name, None) for p in parts]
        if all(torch.is_tensor(t) for t in tensors):
            setattr(ep, field_name, _tensor_cat(tensors))

    dict_values = [p.forward_inputs for p in parts if isinstance(p.forward_inputs, dict) and p.forward_inputs]
    ep.forward_inputs = _cat_dicts(dict_values) if dict_values else {}

    curr_values = [p.curr_obs for p in parts if isinstance(p.curr_obs, dict) and p.curr_obs]
    ep.curr_obs = _cat_dicts(curr_values) if curr_values else {}

    next_values = [p.next_obs for p in parts if isinstance(p.next_obs, dict) and p.next_obs]
    ep.next_obs = _cat_dicts(next_values) if next_values else {}

    _clean_invalid_padding_(ep)
    return ep.contiguous_()




def _last_valid_step_in_last_chunk(mask: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(chunk_idx, env_idx, step_idx)`` for the last valid step.

    Real-world collection uses one env, but this stays generic enough for the
    common ``[T, B, C]`` mask shape.
    """
    if not torch.is_tensor(mask) or mask.ndim < 3:
        raise RuntimeError("action_valid_mask with shape [T, B, C] is required")
    m = mask.to(torch.bool)
    valid_rows = torch.nonzero(m.any(dim=-1), as_tuple=False)
    if valid_rows.numel() == 0:
        return 0, 0, 0
    t, b = valid_rows[-1].tolist()
    valid_steps = torch.nonzero(m[int(t), int(b)], as_tuple=False).flatten()
    if valid_steps.numel() == 0:
        return int(t), int(b), 0
    return int(t), int(b), int(valid_steps[-1].item())


def _apply_manual_terminal_to_last_valid_step_(traj: Trajectory, key: str) -> None:
    """Apply a chunk-between manual c/a event to the just-finished chunk.

    If the operator presses c/a after a chunk has finished but before the next
    env.step(), the env-side keyboard wrapper cannot attach the event to the
    correct primitive step.  Starting a new chunk just to consume that key puts
    terminal at ``next_chunk[0]``.  Instead, the collector patches the terminal
    marker back onto the last valid primitive step of the current episode.

    c: success, reward=1, done/termination=True.
    a: failure, reward=0, done/termination=True.
    """
    if key not in {"a", "c"}:
        return
    mask = getattr(traj, "action_valid_mask", None)
    if not torch.is_tensor(mask):
        raise RuntimeError("Cannot apply manual terminal without action_valid_mask")
    t, b, j = _last_valid_step_in_last_chunk(mask)

    def _ensure_bool(name: str) -> torch.Tensor:
        value = getattr(traj, name, None)
        if torch.is_tensor(value) and value.shape == mask.shape:
            return value.clone().to(torch.bool)
        return torch.zeros_like(mask, dtype=torch.bool)

    def _ensure_reward() -> torch.Tensor:
        value = getattr(traj, "rewards", None)
        if torch.is_tensor(value) and value.shape == mask.shape:
            return value.clone().float()
        return torch.zeros_like(mask, dtype=torch.float32)

    rewards = _ensure_reward()
    terminations = _ensure_bool("terminations")
    truncations = _ensure_bool("truncations")
    dones = _ensure_bool("dones")

    # Keep the terminal step valid, and clear anything after it in the same chunk.
    mask = mask.clone().to(torch.bool)
    if j + 1 < mask.shape[-1]:
        mask[t, b, j + 1 :] = False
    setattr(traj, "action_valid_mask", mask.cpu().contiguous())

    rewards[t, b, j] = 1.0 if key == "c" else 0.0
    terminations[t, b, j] = True
    truncations[t, b, j] = False
    dones[t, b, j] = True

    setattr(traj, "rewards", rewards.cpu().contiguous())
    setattr(traj, "terminations", terminations.cpu().contiguous())
    setattr(traj, "truncations", truncations.cpu().contiguous())
    setattr(traj, "dones", dones.cpu().contiguous())

    metadata = dict(getattr(traj, "metadata", None) or {})
    metadata.update(
        {
            "manual_key": key,
            "manual_terminal_source": "collector_between_chunks",
            "manual_terminal_chunk": int(t),
            "manual_terminal_env": int(b),
            "manual_terminal_step": int(j),
            "episode_outcome": "success" if key == "c" else "failure",
            "terminal_reason": "keyboard_c_between_chunks" if key == "c" else "keyboard_a_between_chunks",
        }
    )
    traj.metadata = metadata
    _clean_invalid_padding_(traj)


def _drain_terminal_key(listener: Any) -> str | None:
    if listener is None:
        return None
    if hasattr(listener, "consume_first_press"):
        return listener.consume_first_press(("a", "c"))
    for key in ("a", "c"):
        try:
            if listener.consume_press(key):
                return key
        except Exception:
            return None
    return None


def _clear_terminal_keys(listener: Any) -> None:
    if listener is None:
        return
    try:
        if hasattr(listener, "clear_presses"):
            listener.clear_presses(("a", "c"))
        else:
            listener.consume_press("a")
            listener.consume_press("c")
    except Exception:
        pass

def _trajectory_to_dict(traj: Trajectory) -> dict[str, Any]:
    out = {}
    for field_name in Trajectory.__dataclass_fields__.keys():
        value = getattr(traj, field_name, None)
        if value is not None:
            out[field_name] = _clone_cpu(value)
    return out


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _quality(traj: Trajectory, threshold: float) -> dict[str, Any]:
    rewards = traj.rewards.float() if torch.is_tensor(traj.rewards) else torch.zeros(1)
    mask = traj.action_valid_mask.bool() if torch.is_tensor(traj.action_valid_mask) else None
    if mask is not None and mask.shape == rewards.shape:
        rewards_for_quality = rewards.masked_fill(~mask, 0.0)
    else:
        rewards_for_quality = rewards
    dones = traj.dones.bool() if torch.is_tensor(traj.dones) else torch.zeros_like(rewards_for_quality, dtype=torch.bool)
    terms = traj.terminations.bool() if torch.is_tensor(traj.terminations) else torch.zeros_like(dones)
    truncs = traj.truncations.bool() if torch.is_tensor(traj.truncations) else torch.zeros_like(dones)
    if mask is not None and mask.shape == dones.shape:
        dones = dones & mask
        terms = terms & mask
        truncs = truncs & mask
    reward_sum = float(rewards_for_quality.sum().item())
    reward_max = float(rewards_for_quality.max().item()) if rewards_for_quality.numel() else 0.0
    success = bool(reward_max >= threshold)
    done_any = bool((dones | terms | truncs).any().item())
    trunc_any = bool(truncs.any().item())
    term_any = bool(terms.any().item())
    if success:
        outcome = "success"
    elif trunc_any:
        outcome = "timeout"
    elif done_any or term_any:
        outcome = "failure"
    else:
        outcome = "unfinished"
    intervene_steps = int(traj.intervene_flags.bool().sum().item()) if torch.is_tensor(traj.intervene_flags) else 0
    valid_steps = int(mask.sum().item()) if mask is not None else int(rewards.numel())
    return {
        "num_samples": int(traj.actions.shape[0]) if torch.is_tensor(traj.actions) else 0,
        "reward_sum": reward_sum,
        "reward_max": reward_max,
        "is_success": success,
        "done_any": done_any,
        "terminal_any": term_any,
        "truncation_any": trunc_any,
        "episode_outcome": outcome,
        "intervene_steps": intervene_steps,
        "valid_steps": valid_steps,
    }


def _chunk_has_terminal(traj: Trajectory, threshold: float) -> bool:
    q = _quality(traj, threshold)
    return bool(q["done_any"] or q["reward_max"] >= threshold)


def _to_numpy_image(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    import cv2

    width = max(1, int(round(frame.shape[1] * (height / float(frame.shape[0])))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _make_inference_frame(curr_obs: dict[str, Any], t: int = 0, b: int = 0) -> np.ndarray | None:
    if not isinstance(curr_obs, dict):
        return None
    views = []
    main = curr_obs.get("main_images", None)
    if isinstance(main, torch.Tensor) and main.dim() >= 5:
        t0 = min(t, main.shape[0] - 1)
        b0 = min(b, main.shape[1] - 1)
        views.append(_to_numpy_image(main[t0, b0]))
    wrist = curr_obs.get("wrist_images", None)
    if isinstance(wrist, torch.Tensor) and wrist.dim() >= 6:
        t0 = min(t, wrist.shape[0] - 1)
        b0 = min(b, wrist.shape[1] - 1)
        for view_idx in range(int(wrist.shape[2])):
            views.append(_to_numpy_image(wrist[t0, b0, view_idx]))
    elif isinstance(wrist, torch.Tensor) and wrist.dim() >= 5:
        t0 = min(t, wrist.shape[0] - 1)
        b0 = min(b, wrist.shape[1] - 1)
        views.append(_to_numpy_image(wrist[t0, b0]))
    if not views:
        return None
    height = min(frame.shape[0] for frame in views)
    views = [_resize_to_height(frame, height) for frame in views]
    return np.concatenate(views, axis=1)


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> bool:
    if not frames:
        return False
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        return False
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return True


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_episode(
    episode: Trajectory,
    frames: list[np.ndarray],
    *,
    rank_dir: Path,
    episode_idx: int,
    success_threshold: float,
    video_fps: int,
    save_video: bool,
) -> dict[str, Any]:
    q = _quality(episode, success_threshold)
    split = "success" if q["is_success"] else "failure"
    metadata = dict(episode.metadata or {})
    metadata.update({"collection_index": episode_idx, **q})
    episode.metadata = metadata

    file_name = f"episode_{episode_idx:06d}.pt"
    all_path = rank_dir / "all" / file_name
    split_path = rank_dir / split / file_name
    payload = _trajectory_to_dict(episode)
    _atomic_torch_save(payload, all_path)
    _atomic_torch_save(payload, split_path)
    all_size_mb = all_path.stat().st_size / (1024.0 * 1024.0)
    split_size_mb = split_path.stat().st_size / (1024.0 * 1024.0)
    print(
        f"[collect-episode][SAVE_PT] episode={episode_idx:06d} split={split} "
        f"outcome={metadata.get('episode_outcome')} all={all_path} ({all_size_mb:.2f} MB) "
        f"split_path={split_path} ({split_size_mb:.2f} MB)",
        flush=True,
    )

    video_path = None
    if save_video:
        video_path = rank_dir / "inference_frame_videos" / f"episode_{episode_idx:06d}.mp4"
        if _write_video(frames, video_path, fps=video_fps):
            metadata["inference_frame_video"] = str(video_path)
            video_size_mb = video_path.stat().st_size / (1024.0 * 1024.0)
            print(
                f"[collect-episode][SAVE_VIDEO] episode={episode_idx:06d} "
                f"path={video_path} ({video_size_mb:.2f} MB) frames={len(frames)}",
                flush=True,
            )
        else:
            print(
                f"[collect-episode][SAVE_VIDEO][WARN] episode={episode_idx:06d} "
                f"failed_to_write path={video_path} frames={len(frames)}",
                flush=True,
            )
            video_path = None

    row = {
        "episode_index": episode_idx,
        "path": str(all_path),
        "split": split,
        "split_path": str(split_path),
        **metadata,
    }
    if video_path is not None:
        row["inference_frame_video"] = str(video_path)
    _append_jsonl(rank_dir / "trajectory_summaries.jsonl", row)
    return row


def _write_metadata(rank_dir: Path, num_episodes: int, total_samples: int, success: int, failure: int) -> None:
    rank_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trajectory_format": "pt",
        "save_unit": "episode",
        "num_episodes": int(num_episodes),
        "total_samples": int(total_samples),
        "success_episodes": int(success),
        "failure_episodes": int(failure),
    }
    with open(rank_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="collect_piper_gigawa_episode",
)
def main(cfg) -> None:
    # Force one interaction to contain exactly one GigaWA action chunk.  This is
    # the key memory fix: Ray channels never carry 80 chunks of raw images.
    with open_dict(cfg):
        chunk = int(cfg.actor.model.num_action_chunks)
        cfg.env.train.max_steps_per_rollout_epoch = chunk
        cfg.algorithm.rollout_epoch = 1
        cfg.rollout.collect_transitions = True
        cfg.runner.only_eval = False
        cfg.env.train.auto_reset = True

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=getattr(cfg.runner, "per_worker_log_path", None),
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    rollout_placement = component_placement.get_strategy("rollout")
    env_placement = component_placement.get_strategy("env")

    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    env_channel = Channel.create("EnvEpisodeCollect")
    rollout_channel = Channel.create("RolloutEpisodeCollect")
    trajectory_channel = Channel.create("TrajectoryEpisodeCollect")

    rollout_group.init_worker().wait()
    env_group.init_worker().wait()

    collect_cfg = cfg.algorithm.offline_collection
    output_dir = Path(str(collect_cfg.get("output_dir", Path(cfg.runner.logger.log_path) / "offline_collection")))
    rank_dir = output_dir / "rank_0"
    rank_dir.mkdir(parents=True, exist_ok=True)

    target_num_episodes = _as_int(collect_cfg.get("target_num_trajectories", 0), 0)
    max_collection_steps = _as_int(collect_cfg.get("max_collection_steps", 10**9), 10**9)
    log_interval = max(1, _as_int(collect_cfg.get("log_interval", 1), 1))
    success_threshold = _as_float(collect_cfg.get("success_threshold", 0.5), 0.5)
    save_video = bool(collect_cfg.get("inference_frame_video", {}).get("enable", True))
    video_fps = _as_int(collect_cfg.get("inference_frame_video", {}).get("fps", 5), 5)
    save_partial_on_exit = bool(collect_cfg.get("save_partial_on_exit", False))

    manual_key_listener = None
    if bool(collect_cfg.get("poll_keyboard_between_chunks", True)) and KeyboardListener is not None:
        try:
            manual_key_listener = KeyboardListener()
            print(
                "[collect-episode] enabled main-process keyboard polling for chunk-between a/c events.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[collect-episode][WARN] failed to enable main-process keyboard polling: {exc}",
                flush=True,
            )
            manual_key_listener = None

    episode_parts: list[Trajectory] = []
    episode_frames: list[np.ndarray] = []
    collected = 0
    success = 0
    failure = 0
    total_samples = 0
    start_time = time.time()

    try:
        for step in range(1, max_collection_steps + 1):
            env_handle = env_group.interact(
                input_channel=env_channel,
                output_channel=rollout_channel,
                actor_channel=trajectory_channel,
            )
            rollout_handle = rollout_group.generate(
                input_channel=rollout_channel,
                output_channel=env_channel,
            )
            env_metrics_list = env_handle.wait()
            rollout_handle.wait()

            # One chunk-level trajectory because max_steps_per_rollout_epoch is forced to chunk_size.
            raw_chunk: Trajectory = trajectory_channel.get()
            if hasattr(raw_chunk, "contiguous_"):
                raw_chunk.contiguous_()

            frame = _make_inference_frame(getattr(raw_chunk, "curr_obs", None), 0, 0)
            if frame is not None:
                episode_frames.append(frame)

            chunk = _slim_trajectory(raw_chunk)
            episode_parts.append(chunk)

            terminal_in_chunk = _chunk_has_terminal(chunk, success_threshold)
            between_chunk_key = None if terminal_in_chunk else _drain_terminal_key(manual_key_listener)
            save_reason = "terminal"
            if between_chunk_key in {"a", "c"}:
                # Do not start a new rollout chunk just to consume c/a.  Attach the
                # event to the last valid primitive step of the episode collected
                # so far, then reset env before the next episode.
                _apply_manual_terminal_to_last_valid_step_(episode_parts[-1], between_chunk_key)
                save_reason = f"between_chunk_key_{between_chunk_key}"
                print(
                    f"[collect-episode][MANUAL_KEY_BETWEEN_CHUNKS] key={between_chunk_key} "
                    f"applied_to_current_episode step={step}",
                    flush=True,
                )

            if terminal_in_chunk or between_chunk_key in {"a", "c"}:
                episode_idx = collected + 1
                metadata = {
                    "collection_step": step,
                    "num_chunks": len(episode_parts),
                    "save_reason": save_reason,
                }
                episode = _concat_episode(episode_parts, metadata)
                row = _save_episode(
                    episode,
                    episode_frames,
                    rank_dir=rank_dir,
                    episode_idx=episode_idx,
                    success_threshold=success_threshold,
                    video_fps=video_fps,
                    save_video=save_video,
                )
                collected += 1
                total_samples += int(row.get("num_samples", 0))
                if row.get("is_success", False):
                    success += 1
                else:
                    failure += 1
                episode_parts = []
                episode_frames = []
                _write_metadata(rank_dir, collected, total_samples, success, failure)
                print(
                    f"[collect-episode] saved episode={episode_idx} split={row['split']} "
                    f"outcome={row.get('episode_outcome')} chunks={row.get('num_chunks')} "
                    f"samples={row.get('num_samples')} reward_sum={row.get('reward_sum'):.3f} "
                    f"done_any={row.get('done_any')} valid_steps={row.get('valid_steps')} "
                    f"reason={save_reason} path={row['path']}",
                    flush=True,
                )
                _clear_terminal_keys(manual_key_listener)

                if between_chunk_key in {"a", "c"}:
                    # The env did not see this terminal via env.step(), so auto_reset
                    # did not happen inside EnvWorker.  Force reset before the next
                    # rollout chunk; otherwise the new episode would continue from the
                    # old scene/robot state.
                    try:
                        reset_work = env_group.reset_train_envs(reason=save_reason)
                        if hasattr(reset_work, "wait"):
                            reset_work.wait()
                        print(
                            f"[collect-episode][RESET_AFTER_BETWEEN_CHUNK_KEY] key={between_chunk_key}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(
                            f"[collect-episode][WARN] reset_train_envs failed after key={between_chunk_key}: {exc}",
                            flush=True,
                        )

            if step == 1 or step % log_interval == 0:
                env_metrics = compute_evaluate_metrics(env_metrics_list) if env_metrics_list else {}
                metric_msg = []
                for key in ["success_once", "success_at_end", "episode_reward"]:
                    if key in env_metrics:
                        value = env_metrics[key]
                        try:
                            value = float(value.float().mean().item()) if hasattr(value, "float") else float(value)
                            metric_msg.append(f"{key}={value:.4f}")
                        except Exception:
                            pass
                elapsed = time.time() - start_time
                print(
                    f"[collect-episode] step={step} | elapsed={elapsed:.1f}s "
                    f"| episodes={collected} | success={success} | failure={failure} "
                    f"| open_chunks={len(episode_parts)}"
                    + (" | " + " | ".join(metric_msg) if metric_msg else "")
                )

            if target_num_episodes > 0 and collected >= target_num_episodes:
                break
    finally:
        if save_partial_on_exit and episode_parts:
            episode_idx = collected + 1
            episode = _concat_episode(
                episode_parts,
                {
                    "collection_step": -1,
                    "num_chunks": len(episode_parts),
                    "save_reason": "partial_on_exit",
                },
            )
            row = _save_episode(
                episode,
                episode_frames,
                rank_dir=rank_dir,
                episode_idx=episode_idx,
                success_threshold=success_threshold,
                video_fps=video_fps,
                save_video=save_video,
            )
            collected += 1
            total_samples += int(row.get("num_samples", 0))
            if row.get("is_success", False):
                success += 1
            else:
                failure += 1
        _write_metadata(rank_dir, collected, total_samples, success, failure)

    final = {
        "output_dir": str(output_dir),
        "rank_dir": str(rank_dir),
        "num_episodes": collected,
        "success_episodes": success,
        "failure_episodes": failure,
        "total_samples": total_samples,
    }
    print("[collect-episode] finalized:")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
