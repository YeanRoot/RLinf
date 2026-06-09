#!/usr/bin/env python3
"""
Compare saved trajectory target actions, WA/ref actions, and trained actor outputs on .pt trajectories.

Run from RLinf/examples/embodiment, for example:

python compare_wa_actor_on_pt.py \
  --config-path ./config \
  --config-name offline_piper_actor_bc_warmup \
  --pt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/collect_piper_gigawa_intervention100/offline_collection/rank_0/success \
  --actor-ckpt /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/offline_piper_actor_bc/global_step_200 \
  --out /home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/action_compare_success \
  --batch-size 8

It expects each episode_*.pt to contain curr_obs.visual_latent, curr_obs.robot_state,
curr_obs.ref_action, actions, and action_valid_mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

try:
    import hydra
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Hydra is required. Run inside the RLinf environment.") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SOURCE_NAME = {
    0: "policy",
    1: "human",
    2: "replan",
    3: "padding",
}


def _as_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def load_cfg(config_path: str, config_name: str, overrides: List[str]):
    config_path = os.path.abspath(config_path)
    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    return cfg


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Accept plain state_dict or common checkpoint wrappers."""
    if isinstance(obj, dict):
        for key in ["model", "state_dict", "model_state_dict", "module"]:
            if key in obj and isinstance(obj[key], dict):
                return unwrap_state_dict(obj[key])
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise RuntimeError(f"Unsupported checkpoint object type: {type(obj)}")


def clean_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    prefixes = ["module.", "model.", "_orig_mod."]
    for k, v in sd.items():
        kk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    changed = True
        out[kk] = v
    return out


def find_full_weight_file(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f"actor checkpoint path does not exist: {p}")

    candidates = [
        p / "model_state_dict" / "full_weights.pt",
        p / "actor" / "model_state_dict" / "full_weights.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c

    matches = sorted(p.glob("**/model_state_dict/full_weights.pt"), key=lambda x: x.stat().st_mtime)
    if matches:
        return matches[-1]

    local_shards = sorted(p.glob("**/local_shard_checkpoint/checkpoint_rank_0.pt"), key=lambda x: x.stat().st_mtime)
    if local_shards:
        raise RuntimeError(
            "Only local shard checkpoint was found. For this standalone compare script, "
            "please pass the full checkpoint file, usually: <ckpt>/model_state_dict/full_weights.pt. "
            f"Found local shard: {local_shards[-1]}"
        )

    raise FileNotFoundError(
        f"Could not find full_weights.pt under {p}. Expected <ckpt>/model_state_dict/full_weights.pt"
    )


def load_actor_weights(policy: torch.nn.Module, ckpt_path: Optional[str], device: torch.device) -> None:
    weight_file = find_full_weight_file(ckpt_path)
    if weight_file is None:
        print("[compare] No --actor-ckpt provided. Actor output will be from the current initialized model.")
        return
    print(f"[compare] Loading actor/model weights from: {weight_file}")
    obj = torch.load(weight_file, map_location="cpu", weights_only=False)
    sd = clean_state_dict_keys(unwrap_state_dict(obj))
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"[compare] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[compare] missing keys sample:", list(missing)[:20])
    if unexpected:
        print("[compare] unexpected keys sample:", list(unexpected)[:20])
    policy.to(device)
    policy.eval()


def find_episode_files(pt: str) -> List[Path]:
    p = Path(pt)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"pt path does not exist: {p}")
    files = sorted(p.glob("episode_*.pt"))
    if not files:
        files = sorted(p.glob("trajectory_*.pt"))
    if not files:
        files = sorted(p.glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt trajectories found under {p}")
    return files


def ensure_episode_dict(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        # Trajectory dataclass-like object
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        else:
            obj = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    return obj


def squeeze_env_dim(x: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be tensor, got {type(x)}")
    if x.dim() >= 2 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def reshape_action(x: torch.Tensor, action_dim: int, num_chunks: int, name: str) -> torch.Tensor:
    x = squeeze_env_dim(x, name)
    if x.dim() == 3 and x.shape[-2:] == (num_chunks, action_dim):
        return x.float()
    if x.dim() == 2 and x.shape[-1] == num_chunks * action_dim:
        return x.view(x.shape[0], num_chunks, action_dim).float()
    if x.dim() == 3 and x.shape[-1] == num_chunks * action_dim:
        return x.squeeze(1).view(x.shape[0], num_chunks, action_dim).float()
    raise ValueError(f"Cannot reshape {name} with shape {tuple(x.shape)} into [T,{num_chunks},{action_dim}]")


def get_target_action(data: Dict[str, Any], action_dim: int, num_chunks: int) -> torch.Tensor:
    if "actions" not in data:
        raise KeyError("trajectory has no top-level 'actions' field")
    return reshape_action(data["actions"], action_dim, num_chunks, "actions")


def get_wa_action(data: Dict[str, Any], action_dim: int, num_chunks: int) -> Tuple[torch.Tensor, str]:
    curr_obs = data.get("curr_obs", {})
    if isinstance(curr_obs, dict) and "ref_action" in curr_obs:
        return reshape_action(curr_obs["ref_action"], action_dim, num_chunks, "curr_obs.ref_action"), "curr_obs.ref_action"
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict) and "policy_action_model" in fi:
        return reshape_action(fi["policy_action_model"], action_dim, num_chunks, "forward_inputs.policy_action_model"), "forward_inputs.policy_action_model"
    if isinstance(fi, dict) and "model_action" in fi:
        return reshape_action(fi["model_action"], action_dim, num_chunks, "forward_inputs.model_action"), "forward_inputs.model_action"
    raise KeyError("Cannot find WA/ref action. Expected curr_obs.ref_action or forward_inputs.policy_action_model")


def get_exec_actions(data: Dict[str, Any], action_dim: int, num_chunks: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (wa_exec, target_exec) in real radian space from saved .pt fields, or (None, None)."""
    fi = data.get("forward_inputs", {})
    if not isinstance(fi, dict):
        return None, None
    wa_exec, target_exec = None, None
    if "policy_action_exec" in fi and torch.is_tensor(fi["policy_action_exec"]):
        try:
            wa_exec = reshape_action(fi["policy_action_exec"], action_dim, num_chunks, "forward_inputs.policy_action_exec").float()
        except Exception:
            pass
    if "action_exec" in fi and torch.is_tensor(fi["action_exec"]):
        try:
            target_exec = reshape_action(fi["action_exec"], action_dim, num_chunks, "forward_inputs.action_exec").float()
        except Exception:
            pass
    return wa_exec, target_exec


def get_valid_mask(data: Dict[str, Any], T: int, C: int) -> torch.Tensor:
    if "action_valid_mask" in data and torch.is_tensor(data["action_valid_mask"]):
        m = squeeze_env_dim(data["action_valid_mask"], "action_valid_mask").bool()
        if m.shape == (T, C):
            return m
    return torch.ones(T, C, dtype=torch.bool)


def get_source(data: Dict[str, Any], T: int, C: int) -> torch.Tensor:
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict) and "action_source" in fi and torch.is_tensor(fi["action_source"]):
        s = squeeze_env_dim(fi["action_source"], "action_source").long()
        if s.shape == (T, C):
            return s
    return torch.zeros(T, C, dtype=torch.long)


def build_actor_inputs(data: Dict[str, Any], device: torch.device, dtype: torch.dtype, action_dim: int, num_chunks: int):
    curr_obs = data.get("curr_obs", {})
    required = ["visual_latent", "robot_state", "ref_action"]
    missing = [k for k in required if k not in curr_obs]
    if missing:
        raise KeyError(
            f"curr_obs missing {missing}. This trajectory cannot be used for actor comparison. "
            "Re-collect with visual_latent/robot_state/ref_action saved."
        )
    visual_latent = squeeze_env_dim(curr_obs["visual_latent"], "curr_obs.visual_latent").to(device=device, dtype=dtype)
    robot_state = squeeze_env_dim(curr_obs["robot_state"], "curr_obs.robot_state").to(device=device, dtype=dtype)
    ref_action = reshape_action(curr_obs["ref_action"], action_dim, num_chunks, "curr_obs.ref_action").to(device=device, dtype=dtype)
    return visual_latent, robot_state, ref_action


def get_raw_states(data: Dict[str, Any], T: int, num_chunks: int, action_dim: int) -> Optional[torch.Tensor]:
    """Return raw qpos for action post-processing, preferably chunk-start [T,A]."""
    curr_obs = data.get("curr_obs", {})
    if isinstance(curr_obs, dict):
        for key in ("raw_robot_state", "states"):
            s = curr_obs.get(key, None)
            if torch.is_tensor(s):
                s = squeeze_env_dim(s, f"curr_obs.{key}").float()
                if s.shape == (T, action_dim):
                    return s
    fi = data.get("forward_inputs", {})
    if isinstance(fi, dict) and "raw_states_before_action" in fi:
        s = fi["raw_states_before_action"]
        if torch.is_tensor(s):
            s = squeeze_env_dim(s, "raw_states_before_action").float()  # [T, C, A]
            if s.shape == (T, num_chunks, action_dim):
                return s
    if isinstance(fi, dict) and "raw_robot_state" in fi:
        s = fi["raw_robot_state"]
        if torch.is_tensor(s):
            s = squeeze_env_dim(s, "forward_inputs.raw_robot_state").float()
            if s.shape == (T, action_dim):
                return s
    return None


def actor_predict(policy, data: Dict[str, Any], device: torch.device, dtype: torch.dtype, action_dim: int, num_chunks: int, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Returns (actor_model_space [T,C,A], actor_exec_space [T,C,A] or None)."""
    visual_latent, robot_state, ref_action = build_actor_inputs(data, device, dtype, action_dim, num_chunks)
    raw_states = get_raw_states(data, visual_latent.shape[0], num_chunks, action_dim)
    outs = []
    N = visual_latent.shape[0]
    policy.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            vl = visual_latent[start:end]
            rs = robot_state[start:end]
            ra = ref_action[start:end]
            visual_feat = policy.encode_visual(vl)
            pi, _ = policy.actor_forward(
                visual_feat=visual_feat,
                robot_state=rs,
                ref_action=ra,
                ref_action_dropout_p=0.0,
                use_target=False,
            )
            outs.append(pi.detach().float().cpu())
    actor_model = torch.cat(outs, dim=0)
    actor_exec = None
    if raw_states is not None and hasattr(policy, "model_action_to_exec_action"):
        with torch.no_grad():
            actor_exec = policy.model_action_to_exec_action(actor_model.to(device), raw_states.to(device)).float().cpu()
    return actor_model, actor_exec


def masked_flat(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask.unsqueeze(-1).expand_as(x)]


def metric_pair(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    diff = a - b
    vals = masked_flat(diff, mask)
    if vals.numel() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan")}
    return {
        "mae": float(vals.abs().mean().item()),
        "rmse": float(torch.sqrt((vals.float() ** 2).mean()).item()),
        "max_abs": float(vals.abs().max().item()),
    }


def step_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=-1)


def summarize_by_source(prefix: str, a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor, source: torch.Tensor) -> Dict[str, float]:
    out = {}
    for sid in [0, 1, 2, 3]:
        m = valid & (source == sid)
        if m.any():
            mm = metric_pair(a, b, m)
            for k, v in mm.items():
                out[f"{prefix}_{SOURCE_NAME.get(sid, sid)}_{k}"] = v
            out[f"{prefix}_{SOURCE_NAME.get(sid, sid)}_steps"] = int(m.sum().item())
    return out


def save_plot(path: Path, target: torch.Tensor, wa: torch.Tensor, actor: torch.Tensor, valid: torch.Tensor, source: torch.Tensor, title: str):
    if plt is None:
        return
    valid_np = valid.reshape(-1).cpu().numpy().astype(bool)
    src_np = source.reshape(-1).cpu().numpy()
    tw = step_l2(target, wa).reshape(-1).cpu().numpy()
    ta = step_l2(target, actor).reshape(-1).cpu().numpy()
    aw = step_l2(actor, wa).reshape(-1).cpu().numpy()
    x = list(range(len(tw)))
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, tw, label="target vs WA/ref")
    ax.plot(x, ta, label="target vs actor")
    ax.plot(x, aw, label="actor vs WA/ref")
    for i, (ok, sid) in enumerate(zip(valid_np, src_np)):
        if not ok:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08)
        elif sid == 1:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.10)
        elif sid == 2:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.06)
    ax.set_title(title)
    ax.set_xlabel("global action step slot = chunk_id * C + step_id")
    ax.set_ylabel("L2 error over 14-dim action")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_detail_plot(path: Path, target: torch.Tensor, wa: torch.Tensor, actor: torch.Tensor, valid: torch.Tensor, title: str):
    """Per-dimension per-step absolute error heatmaps: target vs actor and target vs WA."""
    if plt is None:
        return
    import numpy as np

    # [T*C, action_dim]
    t = target.reshape(-1, target.shape[-1]).cpu().numpy()
    w = wa.reshape(-1, wa.shape[-1]).cpu().numpy()
    a = actor.reshape(-1, actor.shape[-1]).cpu().numpy()
    v = valid.reshape(-1).cpu().numpy().astype(bool)
    action_dim = t.shape[-1]

    err_ta = np.abs(t - a)  # [steps, action_dim]
    err_tw = np.abs(t - w)

    # mask invalid steps
    err_ta[~v] = np.nan
    err_tw[~v] = np.nan

    fig, axes = plt.subplots(3, 1, figsize=(max(14, t.shape[0] // 4), 10))

    for ax, err, label in [
        (axes[0], err_ta, "target vs actor  |abs error|"),
        (axes[1], err_tw, "target vs WA/ref |abs error|"),
    ]:
        vmax = np.nanpercentile(np.concatenate([err_ta, err_tw]), 95)
        im = ax.imshow(err.T, aspect="auto", origin="lower", cmap="hot_r", vmin=0, vmax=vmax)
        ax.set_title(f"{title} — {label}")
        ax.set_xlabel("global step")
        ax.set_ylabel("action dim")
        ax.set_yticks(range(action_dim))
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    # bottom panel: per-step mean abs error comparison (line chart)
    ax = axes[2]
    steps = np.arange(t.shape[0])
    mean_ta = np.nanmean(err_ta, axis=1)
    mean_tw = np.nanmean(err_tw, axis=1)
    ax.plot(steps, mean_ta, label="target vs actor (mean over dims)")
    ax.plot(steps, mean_tw, label="target vs WA/ref (mean over dims)")
    ax.fill_between(steps, mean_ta, alpha=0.15)
    ax.fill_between(steps, mean_tw, alpha=0.15)
    ax.set_title(f"{title} — per-step mean |abs error|")
    ax.set_xlabel("global step")
    ax.set_ylabel("mean |abs error|")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare_one(policy, pt_path: Path, out_dir: Path, device: torch.device, dtype: torch.dtype, action_dim: int, num_chunks: int, batch_size: int) -> Dict[str, Any]:
    data = ensure_episode_dict(pt_path)
    target = get_target_action(data, action_dim, num_chunks)
    wa, wa_source = get_wa_action(data, action_dim, num_chunks)
    actor_model, actor_exec = actor_predict(policy, data, device, dtype, action_dim, num_chunks, batch_size=batch_size)

    T = target.shape[0]
    valid = get_valid_mask(data, T, num_chunks)
    source = get_source(data, T, num_chunks)

    # Try to get exec-space (real radian) actions from .pt
    wa_exec_saved, target_exec_saved = get_exec_actions(data, action_dim, num_chunks)

    # Determine which space to use as primary comparison
    if actor_exec is not None and wa_exec_saved is not None and target_exec_saved is not None:
        target_cmp = target_exec_saved
        wa_cmp = wa_exec_saved
        actor_cmp = actor_exec
        cmp_space = "exec_rad"
    else:
        target_cmp = target
        wa_cmp = wa
        actor_cmp = actor_model
        cmp_space = "model_normalized"
        if actor_exec is None:
            print(f"[compare] WARNING: raw_states_before_action missing or model_action_to_exec_action unavailable — comparing in normalized model space")
        else:
            print(f"[compare] WARNING: policy_action_exec/action_exec not found in .pt — comparing in normalized model space")

    assert target_cmp.shape == wa_cmp.shape == actor_cmp.shape, (target_cmp.shape, wa_cmp.shape, actor_cmp.shape)

    summary = {
        "file": str(pt_path),
        "episode": pt_path.stem,
        "chunks": int(T),
        "valid_steps": int(valid.sum().item()),
        "wa_source": wa_source,
        "comparison_space": cmp_space,
    }
    for name, a, b in [
        ("target_vs_wa", target_cmp, wa_cmp),
        ("target_vs_actor", target_cmp, actor_cmp),
        ("actor_vs_wa", actor_cmp, wa_cmp),
    ]:
        mm = metric_pair(a, b, valid)
        for k, v in mm.items():
            summary[f"{name}_{k}"] = v
        summary.update(summarize_by_source(name, a, b, valid, source))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{pt_path.stem}_compare_steps.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "chunk", "step", "global_step", "valid", "source_id", "source",
            "target_vs_wa_l2", "target_vs_actor_l2", "actor_vs_wa_l2",
            "target_mean", "wa_mean", "actor_mean",
            "target_absmax", "wa_absmax", "actor_absmax",
        ]
        fieldnames += [f"target_a{i}" for i in range(action_dim)]
        fieldnames += [f"wa_a{i}" for i in range(action_dim)]
        fieldnames += [f"actor_a{i}" for i in range(action_dim)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        l2_tw = step_l2(target_cmp, wa_cmp)
        l2_ta = step_l2(target_cmp, actor_cmp)
        l2_aw = step_l2(actor_cmp, wa_cmp)
        for c in range(T):
            for s in range(num_chunks):
                row = {
                    "chunk": c,
                    "step": s,
                    "global_step": c * num_chunks + s,
                    "valid": int(valid[c, s].item()),
                    "source_id": int(source[c, s].item()),
                    "source": SOURCE_NAME.get(int(source[c, s].item()), str(int(source[c, s].item()))),
                    "target_vs_wa_l2": float(l2_tw[c, s].item()),
                    "target_vs_actor_l2": float(l2_ta[c, s].item()),
                    "actor_vs_wa_l2": float(l2_aw[c, s].item()),
                    "target_mean": float(target_cmp[c, s].mean().item()),
                    "wa_mean": float(wa_cmp[c, s].mean().item()),
                    "actor_mean": float(actor_cmp[c, s].mean().item()),
                    "target_absmax": float(target_cmp[c, s].abs().max().item()),
                    "wa_absmax": float(wa_cmp[c, s].abs().max().item()),
                    "actor_absmax": float(actor_cmp[c, s].abs().max().item()),
                }
                row.update({f"target_a{i}": float(target_cmp[c, s, i].item()) for i in range(action_dim)})
                row.update({f"wa_a{i}": float(wa_cmp[c, s, i].item()) for i in range(action_dim)})
                row.update({f"actor_a{i}": float(actor_cmp[c, s, i].item()) for i in range(action_dim)})
                writer.writerow(row)

    torch.save(
        {
            "target_action": target_cmp,
            "wa_action": wa_cmp,
            "actor_action": actor_cmp,
            "valid_mask": valid,
            "action_source": source,
            "comparison_space": cmp_space,
            "summary": summary,
        },
        out_dir / f"{pt_path.stem}_compare_tensors.pt",
    )
    save_plot(out_dir / f"{pt_path.stem}_l2.png", target_cmp, wa_cmp, actor_cmp, valid, source, title=f"{pt_path.stem} [{cmp_space}]")
    save_detail_plot(out_dir / f"{pt_path.stem}_detail.png", target_cmp, wa_cmp, actor_cmp, valid, title=f"{pt_path.stem} [{cmp_space}]")
    print(
        f"[compare] {pt_path.name} [{cmp_space}] | valid={summary['valid_steps']} | "
        f"target-WA MAE={summary['target_vs_wa_mae']:.5f} | "
        f"target-actor MAE={summary['target_vs_actor_mae']:.5f} | "
        f"actor-WA MAE={summary['actor_vs_wa_mae']:.5f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--pt", required=True, help="episode_*.pt file or directory")
    parser.add_argument("--actor-ckpt", default=None, help="checkpoint dir or model_state_dict/full_weights.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all")
    parser.add_argument("overrides", nargs="*", help="optional Hydra overrides, e.g. actor.model.giga_world_policy.ref_action_dropout_p=0")
    args = parser.parse_args()

    cfg = load_cfg(args.config_path, args.config_name, args.overrides)
    model_cfg = cfg.actor.model
    action_dim = int(model_cfg.action_dim)
    num_chunks = int(model_cfg.num_action_chunks)
    dtype = _torch_dtype(str(model_cfg.get("precision", "bf16")))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # Make comparison deterministic and disable dropout.
    if "giga_world_policy" in model_cfg:
        model_cfg.giga_world_policy.ref_action_dropout_p = 0.0
        model_cfg.giga_world_policy.use_rl_head_for_rollout = False

    print("[compare] device=", device, "dtype=", dtype)
    print("[compare] action_dim=", action_dim, "num_chunks=", num_chunks)

    from rlinf.models.embodiment.giga_world_policy.giga_world_policy import get_model

    policy = get_model(model_cfg, torch_dtype=dtype)
    policy.to(device)
    policy.eval()
    load_actor_weights(policy, args.actor_ckpt, device)

    files = find_episode_files(args.pt)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for p in files:
        summaries.append(compare_one(policy, p, out_dir, device, dtype, action_dim, num_chunks, args.batch_size))

    # Write summary CSV/JSON.
    summary_csv = out_dir / "summary.csv"
    keys = sorted({k for s in summaries for k in s.keys()})
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"[compare] wrote: {summary_csv}")
    print(f"[compare] wrote per-episode CSV/PT/PNG under: {out_dir}")


if __name__ == "__main__":
    main()
