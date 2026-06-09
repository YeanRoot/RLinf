#!/usr/bin/env python3
"""Raw LeRobot open-loop backfeed for Piper GigaWA + actor.

This reads LeRobot parquet + videos, reruns WA from raw images, feeds WA
ref_action into the actor, converts WA/actor outputs to executable qpos, and
compares both against the LeRobot GT action chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _ensure_project_on_path(project_root: str | None) -> Path:
    root = Path(project_root).expanduser().resolve() if project_root else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _diff_stats(prefix: str, pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().float().cpu()
    gt = gt.detach().float().cpu()
    valid = valid.detach().bool().cpu()
    while valid.ndim < pred.ndim:
        valid = valid.unsqueeze(-1)
    mask = valid.expand_as(pred)
    d = pred[mask] - gt[mask]
    ad = d.abs()
    if ad.numel() == 0:
        return {
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mae": float(ad.mean().item()),
        f"{prefix}_rmse": float(torch.sqrt((d * d).mean()).item()),
        f"{prefix}_max": float(ad.max().item()),
    }


def _chunk_motion_stats(prefix: str, x: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    valid = valid.detach().bool().cpu()
    out: dict[str, float] = {}
    if x.numel() == 0:
        return out

    delta0 = (x - x[:, :1]).abs().amax(dim=-1)
    step = (x[:, 1:] - x[:, :-1]).abs().amax(dim=-1) if x.shape[1] > 1 else torch.zeros(x.shape[0], 0)
    valid_step = valid[:, 1:] & valid[:, :-1] if x.shape[1] > 1 else torch.zeros_like(step, dtype=torch.bool)

    if valid.any():
        vals = delta0[valid]
        out[f"{prefix}_delta_from_step0_mean"] = float(vals.mean().item())
        out[f"{prefix}_delta_from_step0_max"] = float(vals.max().item())
    if valid_step.numel() and valid_step.any():
        vals = step[valid_step]
        out[f"{prefix}_step_to_step_mean"] = float(vals.mean().item())
        out[f"{prefix}_step_to_step_max"] = float(vals.max().item())
    out[f"{prefix}_exec_abs_mean"] = float(x.abs().mean().item())
    out[f"{prefix}_exec_abs_max"] = float(x.abs().max().item())
    return out


def _make_plots(out_dir: Path, eid: int, gt: torch.Tensor, wa: torch.Tensor, actor: torch.Tensor, max_chunks: int) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    t_chunks = min(gt.shape[0], max_chunks)
    if t_chunks <= 0:
        return paths
    flat_gt = gt[:t_chunks].reshape(-1, gt.shape[-1])
    flat_wa = wa[:t_chunks].reshape(-1, wa.shape[-1])
    flat_actor = actor[:t_chunks].reshape(-1, actor.shape[-1])
    x_axis = np.arange(flat_gt.shape[0])

    for arm_name, dims in (("left", range(0, 6)), ("right", range(7, 13))):
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
        axes = axes.reshape(-1)
        for ax_i, dim in enumerate(dims):
            ax = axes[ax_i]
            ax.plot(x_axis, flat_gt[:, dim].numpy(), color="black", label="GT", linewidth=1.5)
            ax.plot(x_axis, flat_wa[:, dim].numpy(), color="#2468b2", label="WA", linewidth=1.1)
            ax.plot(x_axis, flat_actor[:, dim].numpy(), color="#d43d2a", label="Actor", linewidth=1.1)
            ax.set_title(f"{arm_name} joint {dim}")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="best")
        fig.tight_layout()
        path = out_dir / f"episode_{eid:06d}_{arm_name}_joints.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config-path", default="./config")
    parser.add_argument("--config-name", default="collect_piper_gigawa_realworld_actor_takeover_test")
    parser.add_argument("--actor-ckpt", required=True, help="Checkpoint dir or full_weights.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode-ids", default="0,1,2")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--backbone-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--video-backend", choices=["auto", "opencv", "ffmpeg"], default="auto")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--use-episode-t5", action="store_true")
    parser.add_argument("--plot-chunks", type=int, default=8)
    parser.add_argument("--max-chunks-per-episode", type=int, default=None)
    args = parser.parse_args()

    _ensure_project_on_path(args.project_root)
    from examples.embodiment.convert_lerobot_piper_to_gigawa_buffer import (
        EpisodeVideoReader,
        _build_env_obs_from_indices,
        _format_data_path,
        _load_actor_model_cfg,
        _load_episode_prompt_embeds,
        _make_padded_exec_chunk,
        _parse_episode_ids,
        _read_jsonl,
        _read_parquet,
        _stack_array_column,
    )
    from examples.embodiment.visualize_piper_openloop import find_full_weight_file

    _, model_cfg = _load_actor_model_cfg(args.config_path, args.config_name)
    model_cfg.giga_world_policy.enable_absolute_action_bound = False
    model_cfg.giga_world_policy.use_rl_head_for_rollout = True
    model_cfg.giga_world_policy.print_rollout_action_debug = False

    from rlinf.models.embodiment.giga_world_policy.giga_world_policy import get_model

    dtype_name = str(model_cfg.get("precision", "bf16")).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        dtype = torch.float32
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    policy = get_model(model_cfg, torch_dtype=dtype)
    weight_file = find_full_weight_file(args.actor_ckpt)
    sd = torch.load(weight_file, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"[raw_lerobot] loaded weights={weight_file} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[raw_lerobot] missing sample: {missing[:20]}", flush=True)
    if unexpected:
        print(f"[raw_lerobot] unexpected sample: {unexpected[:20]}", flush=True)
    policy.to(device)
    policy.eval()
    try:
        policy.pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    selected = _parse_episode_ids(args.episode_ids)
    filtered = [ep for ep in episodes if selected is None or int(ep["episode_index"]) in selected]
    if args.max_episodes is not None:
        filtered = filtered[: args.max_episodes]
    if not filtered:
        raise RuntimeError("No selected episodes")

    rows: list[dict[str, Any]] = []
    all_gt, all_wa, all_actor, all_valid = [], [], [], []

    for ep_meta in filtered:
        eid = int(ep_meta["episode_index"])
        print(f"[raw_lerobot] episode={eid}", flush=True)
        data_path = dataset_root / _format_data_path(info, eid)
        df = _read_parquet(data_path)
        states = _stack_array_column(df, "observation.state", dim=policy.env_action_dim)
        gt_all = _stack_array_column(df, "action", dim=policy.env_action_dim)
        n = min(int(states.shape[0]), int(gt_all.shape[0]), int(ep_meta.get("length", states.shape[0])))
        states = states[:n]
        gt_all = gt_all[:n]
        if args.use_episode_t5:
            _load_episode_prompt_embeds(policy, dataset_root, ep_meta)

        starts = [s for s in range(0, n, args.stride) if s < n]
        if args.max_chunks_per_episode is not None:
            starts = starts[: int(args.max_chunks_per_episode)]
        task_description = str(ep_meta.get("tasks", [""])[0]) if ep_meta.get("tasks") else ""
        gt_chunks, wa_chunks, actor_chunks, valid_chunks = [], [], [], []

        with EpisodeVideoReader(
            dataset_root,
            info,
            eid,
            video_backend=args.video_backend,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        ) as reader:
            for start_i in range(0, len(starts), args.backbone_batch_size):
                sub_starts = starts[start_i : start_i + args.backbone_batch_size]
                reader.prepare_rgb_indices(sub_starts)
                env_obs = _build_env_obs_from_indices(
                    reader=reader,
                    states=states,
                    indices=sub_starts,
                    task_description=task_description,
                )
                env_obs["states"] = env_obs["states"].to(device)
                with torch.no_grad():
                    backbone = policy.extract_frozen_backbone_batch(env_obs)
                    wa_exec = backbone["ref_action_exec"].detach().float().cpu()
                    visual_feat = policy.encode_visual(backbone["visual_latent"])
                    actor_model, _ = policy.actor_forward(
                        visual_feat=visual_feat,
                        robot_state=backbone["robot_state"],
                        ref_action=backbone["ref_action"],
                        ref_action_dropout_p=0.0,
                        use_target=False,
                    )
                    actor_state = backbone.get(
                        "raw_robot_state",
                        torch.as_tensor(states[sub_starts], device=device, dtype=torch.float32),
                    )
                    actor_exec = policy.model_action_to_exec_action(actor_model, actor_state).detach().float().cpu()

                for local_idx, start in enumerate(sub_starts):
                    gt_chunk_np, valid_len = _make_padded_exec_chunk(gt_all, int(start), args.chunk_size)
                    gt_chunks.append(torch.as_tensor(gt_chunk_np, dtype=torch.float32))
                    wa_chunks.append(wa_exec[local_idx])
                    actor_chunks.append(actor_exec[local_idx])
                    valid = torch.zeros(args.chunk_size, dtype=torch.bool)
                    valid[:valid_len] = True
                    valid_chunks.append(valid)

        gt = torch.stack(gt_chunks, dim=0)
        wa = torch.stack(wa_chunks, dim=0)
        actor = torch.stack(actor_chunks, dim=0)
        valid = torch.stack(valid_chunks, dim=0)

        row: dict[str, Any] = {"episode_index": eid, "num_frames": n, "num_chunks": int(gt.shape[0])}
        row.update(_diff_stats("wa_vs_gt", wa, gt, valid))
        row.update(_diff_stats("actor_vs_gt", actor, gt, valid))
        row.update(_diff_stats("actor_vs_wa", actor, wa, valid))
        row.update(_chunk_motion_stats("gt", gt, valid))
        row.update(_chunk_motion_stats("wa", wa, valid))
        row.update(_chunk_motion_stats("actor", actor, valid))
        row["plot_paths"] = _make_plots(out_dir, eid, gt, wa, actor, args.plot_chunks)
        rows.append(row)
        all_gt.append(gt)
        all_wa.append(wa)
        all_actor.append(actor)
        all_valid.append(valid)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    gt_all_t = torch.cat(all_gt, dim=0)
    wa_all_t = torch.cat(all_wa, dim=0)
    actor_all_t = torch.cat(all_actor, dim=0)
    valid_all_t = torch.cat(all_valid, dim=0)
    summary: dict[str, Any] = {
        "episodes": [int(ep["episode_index"]) for ep in filtered],
        "num_chunks": int(gt_all_t.shape[0]),
    }
    summary.update(_diff_stats("wa_vs_gt", wa_all_t, gt_all_t, valid_all_t))
    summary.update(_diff_stats("actor_vs_gt", actor_all_t, gt_all_t, valid_all_t))
    summary.update(_diff_stats("actor_vs_wa", actor_all_t, wa_all_t, valid_all_t))
    summary.update(_chunk_motion_stats("gt", gt_all_t, valid_all_t))
    summary.update(_chunk_motion_stats("wa", wa_all_t, valid_all_t))
    summary.update(_chunk_motion_stats("actor", actor_all_t, valid_all_t))

    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    keys = sorted({k for row in rows for k in row if k != "plot_paths"})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})
    print("[raw_lerobot] SUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[raw_lerobot] wrote {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
