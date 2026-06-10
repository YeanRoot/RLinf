#!/usr/bin/env python3
"""Watch actor checkpoints and run Piper open-loop visualization.

This is intentionally small: it monitors checkpoints/global_step_* under one
experiment directory, then runs success/failure visualizations sequentially for
each new checkpoint so it does not spawn competing CUDA jobs.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.name.split("global_step_")[-1])
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/ubuntu/users/angen.ye/gwp/RLinf")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--success-pt", required=True)
    parser.add_argument("--failure-pt", required=True)
    parser.add_argument("--norm-json", default="/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json")
    parser.add_argument("--urdf", default="/home/ubuntu/users/angen.ye/gwp/RLinf/examples/results/piper_local_assets_tmp/piper.urdf")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--stop-step", type=int, default=600)
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--step-multiple", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--skip-q", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    exp = Path(args.experiment_dir)
    out_root = Path(args.out_root)
    done: set[int] = set()
    print(f"[watch-openloop] watching {exp}", flush=True)

    while True:
        candidates: list[tuple[int, Path]] = []
        ckpt_root = exp / "checkpoints"
        if ckpt_root.is_dir():
            for path in ckpt_root.glob("global_step_*"):
                step = _checkpoint_step(path)
                if step is None or step in done or step < int(args.min_step):
                    continue
                if int(args.step_multiple) > 1 and step % int(args.step_multiple) != 0:
                    continue
                weights = path / "actor/model_state_dict/full_weights.pt"
                if weights.is_file():
                    candidates.append((step, weights))

        for step, weights in sorted(candidates):
            for split_name, pt_path in (
                ("success", args.success_pt),
                ("failure", args.failure_pt),
            ):
                out = out_root / f"global_step_{step}" / split_name
                cmd = [
                    "python",
                    "examples/embodiment/visualize_piper_openloop.py",
                    "--pt",
                    pt_path,
                    "--actor-ckpt",
                    str(weights),
                    "--norm-json",
                    args.norm_json,
                    "--urdf",
                    args.urdf,
                    "--out",
                    str(out),
                    "--max-files",
                    str(args.max_files),
                    "--wa-source",
                    "saved",
                    "--batch-size",
                    "8",
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                ]
                if args.skip_q:
                    cmd.append("--skip-q")
                print(
                    f"[watch-openloop] visualizing step={step} split={split_name} -> {out}",
                    flush=True,
                )
                subprocess.run(cmd, cwd=str(repo), check=True)
            done.add(step)
            print(f"[watch-openloop] done step={step}", flush=True)
            if step >= int(args.stop_step):
                print("[watch-openloop] reached stop-step", flush=True)
                return 0

        if int(args.stop_step) in done:
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
