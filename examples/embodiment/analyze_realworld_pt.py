#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {repr(e)}")


def is_tensor(x: Any) -> bool:
    return torch.is_tensor(x)


def tensor_summary(x: torch.Tensor) -> Dict[str, Any]:
    out = {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
    }
    if x.numel() == 0:
        out.update({"numel": 0})
        return out

    xf = x.detach().float()
    out.update(
        {
            "numel": int(x.numel()),
            "mean": float(xf.mean().item()),
            "std": float(xf.std().item()) if x.numel() > 1 else 0.0,
            "min": float(xf.min().item()),
            "max": float(xf.max().item()),
            "sum": float(xf.sum().item()),
        }
    )
    return out


def flatten_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict/list/tuple for easier inspection."""
    items = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.update(flatten_dict(v, key))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            items.update(flatten_dict(v, key))
    else:
        items[prefix] = obj

    return items


def find_key(flat: Dict[str, Any], candidates: List[str]):
    """Find first tensor-like key that ends with or contains candidate names."""
    for cand in candidates:
        for k, v in flat.items():
            if k.endswith(cand) and is_tensor(v):
                return k, v
        for k, v in flat.items():
            if cand in k and is_tensor(v):
                return k, v
    return None, None


def maybe_reshape_chunk_tensor(x: torch.Tensor, action_dim: int = 14):
    """
    Common realworld trajectory shape:
      actions: [num_samples, num_envs, chunk_size * action_dim]
    Convert to:
      [num_samples, num_envs, chunk_size, action_dim]
    """
    if x is None or not is_tensor(x):
        return None

    if x.ndim >= 3 and x.shape[-1] % action_dim == 0:
        chunk = x.shape[-1] // action_dim
        return x.reshape(*x.shape[:-1], chunk, action_dim)

    return x


def summarize_basic(traj: Dict[str, Any], action_dim: int = 14) -> Dict[str, Any]:
    flat = flatten_dict(traj)

    summary = {}

    keys_to_check = [
        "actions",
        "rewards",
        "terminations",
        "truncations",
        "dones",
        "intervene_flags",
        "action_valid_mask",
    ]

    for name in keys_to_check:
        k, v = find_key(flat, [name])
        if v is not None:
            summary[name] = {
                "key": k,
                **tensor_summary(v),
            }
        else:
            summary[name] = None

    # action chunk stats
    k_act, actions = find_key(flat, ["actions"])
    if actions is not None:
        a = maybe_reshape_chunk_tensor(actions, action_dim=action_dim)
        if a is not None and a.ndim >= 4:
            af = a.float()
            summary["actions_chunk"] = {
                "key": k_act,
                "shape_after_reshape": list(a.shape),
                "abs_mean": float(af.abs().mean().item()),
                "abs_max": float(af.abs().max().item()),
                "per_dim_mean": af.reshape(-1, action_dim).mean(dim=0).tolist(),
                "per_dim_std": af.reshape(-1, action_dim).std(dim=0).tolist(),
                "per_dim_min": af.reshape(-1, action_dim).min(dim=0).values.tolist(),
                "per_dim_max": af.reshape(-1, action_dim).max(dim=0).values.tolist(),
            }

    return summary


def analyze_masks_and_rewards(traj: Dict[str, Any]) -> Dict[str, Any]:
    flat = flatten_dict(traj)

    _, rewards = find_key(flat, ["rewards"])
    _, terms = find_key(flat, ["terminations"])
    _, truncs = find_key(flat, ["truncations"])
    _, dones = find_key(flat, ["dones"])
    _, intervene_flags = find_key(flat, ["intervene_flags"])
    _, valid_mask = find_key(flat, ["action_valid_mask"])

    result = {}

    if rewards is not None:
        rf = rewards.float()
        result["reward_sum"] = float(rf.sum().item())
        result["reward_positive_count"] = int((rf > 0).sum().item())
        result["reward_nonzero_count"] = int((rf != 0).sum().item())
        pos = torch.nonzero(rf > 0, as_tuple=False)
        result["reward_positive_indices_first20"] = pos[:20].tolist()

    if intervene_flags is not None:
        f = intervene_flags.bool()
        result["intervene_step_count"] = int(f.sum().item())
        result["intervene_chunk_count"] = int(f.any(dim=-1).sum().item()) if f.ndim >= 1 else int(f.any().item())
        pos = torch.nonzero(f, as_tuple=False)
        result["intervene_indices_first20"] = pos[:20].tolist()

    if valid_mask is not None:
        vm = valid_mask.bool()
        result["valid_step_count"] = int(vm.sum().item())
        result["invalid_step_count"] = int((~vm).sum().item())
        result["valid_ratio"] = float(vm.float().mean().item())

    # Check padding pollution: invalid positions should have reward=0, done=False, intervene=False
    if valid_mask is not None:
        invalid = ~valid_mask.bool()

        if rewards is not None and rewards.shape == valid_mask.shape:
            bad_reward = invalid & (rewards.float() != 0)
            result["bad_padding_reward_count"] = int(bad_reward.sum().item())
            result["bad_padding_reward_indices_first20"] = torch.nonzero(bad_reward, as_tuple=False)[:20].tolist()

        done_tensor = None
        if dones is not None and dones.shape == valid_mask.shape:
            done_tensor = dones.bool()
        elif terms is not None and terms.shape == valid_mask.shape:
            done_tensor = terms.bool()
            if truncs is not None and truncs.shape == valid_mask.shape:
                done_tensor = done_tensor | truncs.bool()

        if done_tensor is not None:
            bad_done = invalid & done_tensor
            result["bad_padding_done_count"] = int(bad_done.sum().item())
            result["bad_padding_done_indices_first20"] = torch.nonzero(bad_done, as_tuple=False)[:20].tolist()

        if intervene_flags is not None and intervene_flags.shape == valid_mask.shape:
            bad_intervene = invalid & intervene_flags.bool()
            result["bad_padding_intervene_count"] = int(bad_intervene.sum().item())
            result["bad_padding_intervene_indices_first20"] = torch.nonzero(bad_intervene, as_tuple=False)[:20].tolist()

    return result


def is_image_like_tensor(x: torch.Tensor) -> bool:
    if x.ndim < 3:
        return False

    shape = list(x.shape)

    # HWC / NHWC / BCHW / ... common cases
    if shape[-1] in (1, 3, 4) and shape[-2] >= 32 and shape[-3] >= 32:
        return True
    if shape[-3] in (1, 3, 4) and shape[-2] >= 32 and shape[-1] >= 32:
        return True

    return False


def to_hwc_first_image(x: torch.Tensor) -> torch.Tensor:
    """
    Take an arbitrary image-like tensor and return first image as HWC float tensor.
    Supports:
      [..., H, W, C]
      [..., C, H, W]
    """
    x = x.detach().cpu()

    if x.ndim == 3:
        img = x
    else:
        # Flatten leading dims, keep last 3 dims
        img = x.reshape(-1, *x.shape[-3:])[0]

    if img.shape[-1] in (1, 3, 4):
        # HWC
        return img.float()

    if img.shape[0] in (1, 3, 4):
        # CHW -> HWC
        return img.permute(1, 2, 0).float()

    return img.float()


def analyze_images(traj: Dict[str, Any]) -> Dict[str, Any]:
    flat = flatten_dict(traj)
    image_infos = {}

    for k, v in flat.items():
        if not is_tensor(v):
            continue
        if not is_image_like_tensor(v):
            continue

        try:
            img = to_hwc_first_image(v)
            h, w = img.shape[:2]

            info = {
                "shape": list(v.shape),
                "first_image_hwc": list(img.shape),
                "mean": float(img.mean().item()),
                "std": float(img.std().item()) if img.numel() > 1 else 0.0,
                "min": float(img.min().item()),
                "max": float(img.max().item()),
            }

            # If horizontally concatenated 3-view image, split into 3 panels.
            if w >= 3 * 32:
                panels = []
                for i in range(3):
                    crop = img[:, i * w // 3 : (i + 1) * w // 3]
                    panels.append(
                        {
                            "view_index": i,
                            "mean": float(crop.mean().item()),
                            "std": float(crop.std().item()) if crop.numel() > 1 else 0.0,
                            "min": float(crop.min().item()),
                            "max": float(crop.max().item()),
                            "is_black": bool(crop.float().abs().mean().item() < 1e-6 and crop.float().std().item() < 1e-6),
                        }
                    )
                info["split_3_horizontal_views"] = panels

            image_infos[k] = info

        except Exception as e:
            image_infos[k] = {"error": repr(e), "shape": list(v.shape)}

    return image_infos


def print_summary_for_pt(path: Path, traj: Dict[str, Any], action_dim: int):
    print("=" * 120)
    print(f"[PT] {path}")
    print("-" * 120)

    flat = flatten_dict(traj)
    print("[Top-level keys]")
    if isinstance(traj, dict):
        print(list(traj.keys()))
    else:
        print(type(traj))

    print("\n[Tensor keys]")
    for k, v in flat.items():
        if is_tensor(v):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    print("\n[Basic summary]")
    basic = summarize_basic(traj, action_dim=action_dim)
    print(json.dumps(basic, indent=2, ensure_ascii=False)[:10000])

    print("\n[Mask / reward analysis]")
    mr = analyze_masks_and_rewards(traj)
    print(json.dumps(mr, indent=2, ensure_ascii=False))

    print("\n[Image analysis]")
    img = analyze_images(traj)
    if not img:
        print("No image-like tensors found.")
    else:
        print(json.dumps(img, indent=2, ensure_ascii=False)[:12000])


def analyze_folder(root: Path, action_dim: int = 14, max_files: int = -1, save_json: bool = True):
    root = root.expanduser().resolve()

    if root.is_file() and root.suffix == ".pt":
        pt_files = [root]
        out_dir = root.parent
    else:
        pt_files = sorted(root.rglob("*.pt"))
        out_dir = root

    if max_files > 0:
        pt_files = pt_files[:max_files]

    print(f"[INFO] root={root}")
    print(f"[INFO] found {len(pt_files)} pt files")

    # Metadata files
    for name in ["metadata.json", "trajectory_index.json"]:
        for p in sorted(root.rglob(name)) if root.is_dir() else []:
            print("=" * 120)
            print(f"[JSON] {p}")
            try:
                data = json.loads(p.read_text())
                print(json.dumps(data, indent=2, ensure_ascii=False)[:12000])
            except Exception as e:
                print(f"Failed to read {p}: {repr(e)}")

    all_reports = []
    bad_files = []

    for i, pt in enumerate(pt_files):
        try:
            traj = safe_torch_load(pt)
            print_summary_for_pt(pt, traj, action_dim=action_dim)

            report = {
                "file": str(pt),
                "basic": summarize_basic(traj, action_dim=action_dim),
                "mask_reward": analyze_masks_and_rewards(traj),
                "images": analyze_images(traj),
            }
            all_reports.append(report)

        except Exception as e:
            print("=" * 120)
            print(f"[BAD PT] {pt}")
            print(repr(e))
            traceback.print_exc()
            bad_files.append({"file": str(pt), "error": repr(e)})

    print("=" * 120)
    print("[GLOBAL SUMMARY]")
    print(f"total_pt_files={len(pt_files)}")
    print(f"loaded_ok={len(all_reports)}")
    print(f"bad_files={len(bad_files)}")

    total_reward = 0.0
    total_success_steps = 0
    total_intervene_steps = 0
    total_valid_steps = 0

    for r in all_reports:
        mr = r["mask_reward"]
        total_reward += float(mr.get("reward_sum", 0.0))
        total_success_steps += int(mr.get("reward_positive_count", 0))
        total_intervene_steps += int(mr.get("intervene_step_count", 0))
        total_valid_steps += int(mr.get("valid_step_count", 0))

    print(f"total_reward_sum={total_reward}")
    print(f"total_success_steps/reward_positive_count={total_success_steps}")
    print(f"total_intervene_steps={total_intervene_steps}")
    print(f"total_valid_steps={total_valid_steps}")

    if bad_files:
        print("\n[BAD FILES]")
        print(json.dumps(bad_files, indent=2, ensure_ascii=False))

    if save_json:
        out_path = out_dir / "pt_analysis_report.json"
        payload = {
            "root": str(root),
            "total_pt_files": len(pt_files),
            "loaded_ok": len(all_reports),
            "bad_files": bad_files,
            "reports": all_reports,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[INFO] saved report to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Path to pt file or directory containing .pt files.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Robot action dimension. Piper/GigaWorld is usually 14.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="Only analyze first N pt files. Default: all.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save pt_analysis_report.json.",
    )
    args = parser.parse_args()

    analyze_folder(
        Path(args.path),
        action_dim=args.action_dim,
        max_files=args.max_files,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()