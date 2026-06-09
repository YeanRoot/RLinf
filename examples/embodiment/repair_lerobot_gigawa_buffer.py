#!/usr/bin/env python3
"""Repair LeRobot->GigaWA pt buffers to match real-world collection schema.

This keeps expensive saved backbone features from an existing converted buffer,
then adds raw qpos fields and real-collection-compatible forward_inputs keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def stack_array_column(df, column: str, dim: int) -> np.ndarray:
    if column in df.columns:
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in df[column].to_list()], axis=0)
    else:
        prefix = f"{column}."
        cols = sorted(
            [c for c in df.columns if str(c).startswith(prefix)],
            key=lambda c: int(str(c).split(".")[-1]),
        )
        if not cols:
            raise KeyError(f"Missing column {column}")
        arr = df[cols].to_numpy(dtype=np.float32)
    if arr.shape[-1] < dim:
        raise ValueError(f"{column} has dim={arr.shape[-1]}, expected >= {dim}")
    return np.ascontiguousarray(arr[..., :dim], dtype=np.float32)


def format_data_path(info: dict[str, Any], episode_index: int) -> str:
    chunks_size = int(info.get("chunks_size", 1000))
    return str(info["data_path"]).format(
        episode_chunk=int(episode_index) // chunks_size,
        episode_index=int(episode_index),
    )


def make_padded_chunk(arr: np.ndarray, start_idx: int, chunk_size: int) -> tuple[np.ndarray, int]:
    n = int(arr.shape[0])
    valid_len = max(0, min(chunk_size, n - int(start_idx)))
    if valid_len <= 0:
        raise ValueError(f"Invalid start_idx={start_idx} for len={n}")
    chunk = np.zeros((chunk_size, arr.shape[-1]), dtype=np.float32)
    chunk[:valid_len] = arr[start_idx : start_idx + valid_len]
    if valid_len < chunk_size:
        chunk[valid_len:] = chunk[valid_len - 1]
    return chunk, valid_len


def reshape_action(x: torch.Tensor, action_dim: int, chunk_size: int, name: str) -> torch.Tensor:
    x = x.float()
    if x.ndim == 4 and x.shape[-2:] == (chunk_size, action_dim):
        return x
    if x.ndim == 3 and x.shape[-1] == chunk_size * action_dim:
        return x.view(x.shape[0], x.shape[1], chunk_size, action_dim)
    if x.ndim == 2 and x.shape[-1] == chunk_size * action_dim:
        return x.view(x.shape[0], 1, chunk_size, action_dim)
    raise ValueError(f"{name} has unsupported shape {tuple(x.shape)}")


def load_norm(norm_json: Path, action_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    stats = json.loads(norm_json.read_text())["norm_stats"]["action"]
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)[:action_dim]
    std = torch.as_tensor(stats["std"], dtype=torch.float32)[:action_dim]
    mask = torch.tensor(
        [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
        dtype=torch.bool,
    )[:action_dim]
    return mean, std, mask


def model_to_exec(model_action: torch.Tensor, raw_state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # model_action: [T, 1, C, A], raw_state: [T, 1, A]
    raw = raw_state[:, :, None, :].expand_as(model_action)
    delta = model_action * std.view(1, 1, 1, -1).clamp_min(1e-8) + mean.view(1, 1, 1, -1)
    out = delta.clone()
    out[..., mask] += raw[..., mask]
    return out


def exec_to_model(exec_action: torch.Tensor, raw_state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = raw_state[:, :, None, :].expand_as(exec_action)
    delta = exec_action.clone()
    delta[..., mask] = exec_action[..., mask] - raw[..., mask]
    return (delta - mean.view(1, 1, 1, -1)) / std.view(1, 1, 1, -1).clamp_min(1e-8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--norm-json", default="/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    input_rank = input_root / "rank_0"
    output_root = Path(args.output_root).expanduser().resolve()
    output_rank = output_root / "rank_0"
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_rank.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset_root / "meta/info.json").read_text())
    episodes = {int(ep["episode_index"]): ep for ep in read_jsonl(dataset_root / "meta/episodes.jsonl")}
    mean, std, mask = load_norm(Path(args.norm_json), args.action_dim)

    files = sorted(input_rank.glob("*.pt"))
    if args.max_files is not None:
        files = files[: int(args.max_files)]
    if not files:
        raise RuntimeError(f"No .pt files under {input_rank}")

    max_exec_errs = []
    for idx, src in enumerate(files, start=1):
        traj = torch.load(src, map_location="cpu", weights_only=False)
        meta = dict(traj.get("metadata", {}) or {})
        episode_index = int(meta.get("episode_index", traj.get("sample_infos", [{}])[0].get("episode_index")))
        ep_meta = episodes[episode_index]
        data_path = dataset_root / format_data_path(info, episode_index)
        df = read_parquet(data_path)
        states = stack_array_column(df, "observation.state", args.action_dim)
        actions_exec_all = stack_array_column(df, "action", args.action_dim)
        n_frames = min(int(states.shape[0]), int(actions_exec_all.shape[0]), int(ep_meta.get("length", states.shape[0])))
        states = states[:n_frames]
        actions_exec_all = actions_exec_all[:n_frames]

        T = int(traj["actions"].shape[0])
        starts = list(range(0, n_frames, args.chunk_size))[:T]
        if len(starts) != T:
            raise RuntimeError(f"{src}: expected {T} starts, got {len(starts)} from n_frames={n_frames}")
        valid_lens = [min(args.chunk_size, n_frames - s) for s in starts]
        next_indices = [min(s + valid_len, n_frames - 1) for s, valid_len in zip(starts, valid_lens)]

        exec_chunks = []
        raw_chunks = []
        curr_raw = []
        next_raw = []
        for s, nxt in zip(starts, next_indices):
            exec_chunk, _ = make_padded_chunk(actions_exec_all, s, args.chunk_size)
            raw_chunk, _ = make_padded_chunk(states, s, args.chunk_size)
            exec_chunks.append(torch.as_tensor(exec_chunk))
            raw_chunks.append(torch.as_tensor(raw_chunk))
            curr_raw.append(torch.as_tensor(states[s], dtype=torch.float32))
            next_raw.append(torch.as_tensor(states[nxt], dtype=torch.float32))

        exec_tensor = torch.stack(exec_chunks, dim=0).view(T, 1, args.chunk_size, args.action_dim)
        raw_chunk_tensor = torch.stack(raw_chunks, dim=0).view(T, 1, args.chunk_size, args.action_dim)
        curr_raw_tensor = torch.stack(curr_raw, dim=0).view(T, 1, args.action_dim)
        next_raw_tensor = torch.stack(next_raw, dim=0).view(T, 1, args.action_dim)

        model_tensor = reshape_action(traj["actions"], args.action_dim, args.chunk_size, "actions")
        recon_exec = model_to_exec(model_tensor, curr_raw_tensor, mean, std, mask)
        exec_err = (recon_exec - exec_tensor).abs().max().item()
        max_exec_errs.append(exec_err)

        traj["actions"] = model_tensor.reshape(T, 1, -1).contiguous()
        traj["action_valid_mask"] = traj["action_valid_mask"].bool().contiguous()
        traj["curr_obs"]["raw_robot_state"] = curr_raw_tensor.clone().contiguous()
        traj["curr_obs"]["states"] = curr_raw_tensor.clone().contiguous()
        traj["next_obs"]["raw_robot_state"] = next_raw_tensor.clone().contiguous()
        traj["next_obs"]["states"] = next_raw_tensor.clone().contiguous()
        traj["curr_obs"]["ref_action"] = reshape_action(
            traj["curr_obs"]["ref_action"], args.action_dim, args.chunk_size, "curr_obs.ref_action"
        ).contiguous()
        traj["next_obs"]["ref_action"] = reshape_action(
            traj["next_obs"]["ref_action"], args.action_dim, args.chunk_size, "next_obs.ref_action"
        ).contiguous()

        fi = dict(traj.get("forward_inputs", {}) or {})
        fi["action"] = exec_tensor.reshape(T, 1, -1).contiguous()
        fi["action_exec"] = exec_tensor.reshape(T, 1, -1).contiguous()
        fi["model_action"] = traj["actions"].clone()
        fi["policy_action_model"] = traj["curr_obs"]["ref_action"].clone()
        fi["policy_action_exec"] = model_to_exec(fi["policy_action_model"], curr_raw_tensor, mean, std, mask).reshape(T, 1, -1).contiguous()
        fi["raw_robot_state"] = curr_raw_tensor.clone().contiguous()
        fi["raw_states_before_action"] = raw_chunk_tensor.clone().contiguous()
        fi["robot_state"] = traj["curr_obs"]["robot_state"].clone().contiguous()
        fi["visual_latent"] = traj["curr_obs"]["visual_latent"].clone().contiguous()
        fi["ref_action"] = traj["curr_obs"]["ref_action"].clone().contiguous()
        fi["action_valid_mask"] = traj["action_valid_mask"].clone().contiguous()
        fi["intervene_flags"] = traj["intervene_flags"].clone().contiguous()
        if "action_source" not in fi:
            action_source = torch.ones(T, 1, args.chunk_size, dtype=torch.long)
            action_source[~traj["action_valid_mask"].bool()] = 3
            fi["action_source"] = action_source
        traj["forward_inputs"] = fi

        meta["schema_repaired"] = True
        meta["schema_repair_source"] = str(input_root)
        meta["schema_repair_norm_json"] = str(Path(args.norm_json).expanduser().resolve())
        meta["max_model_to_exec_abs_err"] = float(exec_err)
        traj["metadata"] = meta

        dst = output_rank / src.name
        torch.save(traj, dst)
        if idx % 20 == 0 or idx == len(files):
            print(f"[repair] {idx}/{len(files)} | last={src.name} | max_exec_err={exec_err:.8g}", flush=True)

    for name in ("metadata.json", "trajectory_index.json"):
        src_meta = input_rank / name
        if src_meta.exists():
            shutil.copy2(src_meta, output_rank / name)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "norm_json": str(Path(args.norm_json).expanduser().resolve()),
        "num_files": len(files),
        "max_model_to_exec_abs_err": float(max(max_exec_errs)),
        "mean_model_to_exec_abs_err": float(np.mean(max_exec_errs)),
    }
    (output_root / "repair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[repair] done", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
