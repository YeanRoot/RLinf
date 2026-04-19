
#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


CHUNK_STEPS = 12


def load_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def find_rank_dirs(root: Path) -> List[Path]:
    if (root / "metadata.json").exists() and any(root.glob("trajectory_*.pt")):
        return [root]
    rank_dirs = sorted([p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"rank_\d+", p.name)])
    return rank_dirs


def tensor_first_true(x: torch.Tensor) -> Optional[Tuple[int, int]]:
    pos = torch.nonzero(x.to(torch.bool), as_tuple=False)
    if pos.numel() == 0:
        return None
    return int(pos[0, 0].item()), int(pos[0, 1].item())


def tensor_first_nonzero(x: torch.Tensor) -> Optional[Tuple[int, int]]:
    pos = torch.nonzero(x != 0, as_tuple=False)
    if pos.numel() == 0:
        return None
    return int(pos[0, 0].item()), int(pos[0, 1].item())


def choose_terminal_step(rewards: torch.Tensor, dones: torch.Tensor, terms: torch.Tensor) -> Tuple[Optional[Tuple[int, int]], str]:
    reward_pos = tensor_first_nonzero(rewards)
    done_pos = tensor_first_true(dones)
    term_pos = tensor_first_true(terms)

    if reward_pos is not None:
        return reward_pos, "reward"
    if term_pos is not None:
        return term_pos, "termination"
    if done_pos is not None:
        return done_pos, "done"
    return None, "none"


def narrow_time_dim(x: Any, end_t: int) -> Any:
    if isinstance(x, torch.Tensor):
        if x.ndim >= 1:
            return x[:end_t].clone()
        return x.clone()
    if isinstance(x, dict):
        return {k: narrow_time_dim(v, end_t) for k, v in x.items()}
    return x


def select_batch_sample(x: Any, batch_idx: int, batch_size: int) -> Any:
    if isinstance(x, torch.Tensor):
        if x.ndim >= 2 and x.shape[1] == batch_size:
            return x[:, batch_idx:batch_idx + 1].clone()
        return x.clone()
    if isinstance(x, dict):
        return {k: select_batch_sample(v, batch_idx, batch_size) for k, v in x.items()}
    return x


def pad_terminal_chunk_action_tensor(x: torch.Tensor, chunk_idx: int, step_idx: int) -> torch.Tensor:
    y = x.clone()
    if y.ndim < 2 or y.shape[0] <= chunk_idx:
        return y
    if y.shape[1] != 1:
        return y

    chunk = y[chunk_idx, 0]
    if chunk.ndim == 1:
        if chunk.numel() % CHUNK_STEPS != 0:
            return y
        per_step = chunk.numel() // CHUNK_STEPS
        view = chunk.view(CHUNK_STEPS, per_step).clone()
        if step_idx + 1 < CHUNK_STEPS:
            view[step_idx + 1 :] = view[step_idx].unsqueeze(0).expand(CHUNK_STEPS - step_idx - 1, per_step)
        y[chunk_idx, 0] = view.reshape_as(chunk)
        return y

    if chunk.ndim >= 2 and chunk.shape[0] == CHUNK_STEPS:
        if step_idx + 1 < CHUNK_STEPS:
            view = chunk.clone()
            view[step_idx + 1 :] = view[step_idx].unsqueeze(0).expand_as(view[step_idx + 1 :])
            y[chunk_idx, 0] = view
        return y

    return y


def recursively_pad_action_like(obj: Any, chunk_idx: int, step_idx: int, key_hint: str = "") -> Any:
    if isinstance(obj, torch.Tensor):
        return pad_terminal_chunk_action_tensor(obj, chunk_idx, step_idx)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {"action", "model_action", "ref_action"}:
                out[k] = recursively_pad_action_like(v, chunk_idx, step_idx, k)
            else:
                out[k] = recursively_pad_action_like(v, chunk_idx, step_idx, k)
        return out
    return obj


def repair_single_sample(traj: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    batch_size = int(traj["actions"].shape[1])
    sample = {k: select_batch_sample(v, batch_idx, batch_size) for k, v in traj.items()}

    rewards = sample["rewards"][:, 0].clone()
    dones = sample["dones"][:, 0].to(torch.bool).clone()
    terms = sample["terminations"][:, 0].to(torch.bool).clone()
    truncs = sample["truncations"][:, 0].to(torch.bool).clone()

    terminal_pos, terminal_source = choose_terminal_step(rewards, dones, terms)

    info = {
        "batch_idx": batch_idx,
        "terminal_source": terminal_source,
        "original_len_chunks": int(sample["actions"].shape[0]),
        "reward_sum_before": float(rewards.sum().item()),
        "had_done_before": bool(dones.any().item()),
        "had_term_before": bool(terms.any().item()),
    }

    if terminal_pos is None:
        info.update({
            "terminal_chunk": None,
            "terminal_step": None,
            "repaired_len_chunks": int(sample["actions"].shape[0]),
            "reward_sum_after": float(rewards.sum().item()),
            "had_done_after": bool(dones.any().item()),
            "had_term_after": bool(terms.any().item()),
            "status": "failure_unchanged",
        })
        return sample, info

    tc, ts = terminal_pos
    end_t = tc + 1

    repaired = {k: narrow_time_dim(v, end_t) for k, v in sample.items()}

    # Pad action-like tensors in terminal chunk.
    repaired["actions"] = pad_terminal_chunk_action_tensor(repaired["actions"], tc, ts)
    if "forward_inputs" in repaired and isinstance(repaired["forward_inputs"], dict):
        for k in list(repaired["forward_inputs"].keys()):
            if k in {"action", "model_action", "ref_action"}:
                repaired["forward_inputs"][k] = pad_terminal_chunk_action_tensor(repaired["forward_inputs"][k], tc, ts)
    if "curr_obs" in repaired and isinstance(repaired["curr_obs"], dict) and "ref_action" in repaired["curr_obs"]:
        repaired["curr_obs"]["ref_action"] = pad_terminal_chunk_action_tensor(repaired["curr_obs"]["ref_action"], tc, ts)
    if "next_obs" in repaired and isinstance(repaired["next_obs"], dict) and "ref_action" in repaired["next_obs"]:
        repaired["next_obs"]["ref_action"] = pad_terminal_chunk_action_tensor(repaired["next_obs"]["ref_action"], tc, ts)

    # Rebuild terminal semantics on the kept terminal chunk.
    r = repaired["rewards"].clone()
    d = repaired["dones"].to(torch.bool).clone()
    t = repaired["terminations"].to(torch.bool).clone()
    tr = repaired["truncations"].to(torch.bool).clone()

    if ts + 1 < CHUNK_STEPS:
        r[tc, 0, ts + 1 :] = 0
    d[tc, 0, ts:] = True
    t[tc, 0, ts:] = True
    tr[tc, 0, ts:] = False

    # Force chunks after terminal to not exist by truncation above.
    repaired["rewards"] = r
    repaired["dones"] = d
    repaired["terminations"] = t
    repaired["truncations"] = tr

    info.update({
        "terminal_chunk": int(tc),
        "terminal_step": int(ts),
        "repaired_len_chunks": int(end_t),
        "reward_sum_after": float(repaired["rewards"].sum().item()),
        "had_done_after": bool(repaired["dones"].any().item()),
        "had_term_after": bool(repaired["terminations"].any().item()),
        "status": "success_repaired" if terminal_source in {"reward", "termination", "done"} else "unknown",
    })
    return repaired, info


def count_total_samples(rank_dir: Path) -> int:
    total = 0
    for pt in rank_dir.glob("trajectory_*.pt"):
        try:
            d = torch.load(pt, map_location="cpu")
            total += int(d["actions"].shape[0])
        except Exception:
            pass
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rank_dirs = find_rank_dirs(input_root)
    if not rank_dirs:
        raise RuntimeError(f"No rank directories or trajectory files found under: {input_root}")

    global_counter = 0
    all_infos: List[Dict[str, Any]] = []

    for rank_dir in rank_dirs:
        rank_name = rank_dir.name if re.fullmatch(r"rank_\d+", rank_dir.name) else "rank_0"
        out_rank = output_root / rank_name
        out_rank.mkdir(parents=True, exist_ok=True)

        files = sorted(rank_dir.glob("trajectory_*.pt"))
        local_counter = 0

        for pt_path in files:
            traj = torch.load(pt_path, map_location="cpu")
            batch_size = int(traj["actions"].shape[1])

            base_name = pt_path.stem
            for b in range(batch_size):
                repaired, info = repair_single_sample(traj, b)
                info["source_file"] = pt_path.name
                info["output_rank"] = rank_name
                info["output_index"] = local_counter
                all_infos.append(info)

                out_name = f"trajectory_{local_counter}_{base_name}_b{b}.pt"
                torch.save(repaired, out_rank / out_name)

                local_counter += 1
                global_counter += 1

        rank_meta = {
            "trajectory_format": "pt",
            "size": local_counter,
            "total_samples": count_total_samples(out_rank),
            "trajectory_counter": local_counter,
            "seed": load_metadata(rank_dir / "metadata.json").get("seed", 1234),
        }
        with open(out_rank / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(rank_meta, f, indent=2)

    root_meta = {
        "trajectory_format": "pt",
        "size": global_counter,
        "total_samples": sum(count_total_samples(p) for p in output_root.iterdir() if p.is_dir()),
        "trajectory_counter": global_counter,
        "seed": 1234,
    }
    with open(output_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(root_meta, f, indent=2)

    summary = {
        "total_output_trajectories": global_counter,
        "status_counts": {},
        "terminal_source_counts": {},
    }
    for info in all_infos:
        summary["status_counts"][info["status"]] = summary["status_counts"].get(info["status"], 0) + 1
        summary["terminal_source_counts"][info["terminal_source"]] = summary["terminal_source_counts"].get(info["terminal_source"], 0) + 1

    with open(output_root / "repair_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": all_infos[:2000]}, f, indent=2)

    print(json.dumps(root_meta, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
