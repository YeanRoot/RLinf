#!/usr/bin/env python3
"""Watch BC checkpoints and run open-loop visualization every checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/ubuntu/users/angen.ye/gwp/RLinf")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--pt", default="examples/results/lerobot_actor_warmup_buffer_fixed/rank_0")
    parser.add_argument("--norm-json", default="/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json")
    parser.add_argument("--out-root", default="examples/results/lerobot_actor_warmup_fixed_training_viz")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--stop-step", type=int, default=200)
    args = parser.parse_args()

    repo = Path(args.repo)
    exp = Path(args.experiment_dir)
    done: set[int] = set()
    print(f"[watch] watching {exp}", flush=True)
    while True:
        ckpt_root = exp / "checkpoints"
        candidates = []
        if ckpt_root.is_dir():
            for path in ckpt_root.glob("global_step_*"):
                try:
                    step = int(path.name.split("global_step_")[-1])
                except Exception:
                    continue
                weights = path / "actor/model_state_dict/full_weights.pt"
                if weights.is_file() and step not in done:
                    candidates.append((step, weights))
        for step, weights in sorted(candidates):
            out = Path(args.out_root) / f"global_step_{step}"
            cmd = [
                "python",
                "examples/embodiment/visualize_piper_openloop.py",
                "--pt",
                args.pt,
                "--actor-ckpt",
                str(weights),
                "--norm-json",
                args.norm_json,
                "--out",
                str(out),
                "--max-files",
                str(args.max_files),
                "--wa-source",
                "saved",
                "--batch-size",
                "8",
                "--device",
                "cuda",
                "--dtype",
                "bf16",
            ]
            print(f"[watch] visualizing step={step} -> {out}", flush=True)
            try:
                subprocess.run(cmd, cwd=str(repo), check=True)
                done.add(step)
                print(f"[watch] done step={step}", flush=True)
            except subprocess.CalledProcessError as exc:
                print(f"[watch][error] step={step} failed rc={exc.returncode}", flush=True)
            if step >= args.stop_step:
                print("[watch] reached stop-step", flush=True)
                return 0
        if args.stop_step in done:
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
