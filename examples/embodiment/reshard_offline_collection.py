#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def make_buffer_for_load(
    seed: int = 1234,
    cache_size: int = 64,
    sample_window_size: int = 1,
) -> TrajectoryReplayBuffer:
    # auto_save=False requires cache to be enabled in TrajectoryReplayBuffer.
    return TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=True,
        cache_size=max(1, cache_size),
        sample_window_size=max(1, sample_window_size),
        auto_save=False,
        trajectory_format="pt",
    )


def make_buffer_for_save(path: str, seed: int, trajectory_format: str = "pt") -> TrajectoryReplayBuffer:
    Path(path).mkdir(parents=True, exist_ok=True)
    return TrajectoryReplayBuffer(
        seed=seed,
        enable_cache=False,
        cache_size=1,
        sample_window_size=1,
        auto_save=True,
        auto_save_path=path,
        trajectory_format=trajectory_format,
    )


def discover_source_shards(input_root: Path, bucket: str) -> List[Path]:
    shards = []
    for p in sorted(input_root.glob("rank_*")):
        shard = p / bucket
        if shard.is_dir() and (shard / "metadata.json").exists():
            shards.append(shard)
    if not shards:
        raise FileNotFoundError(
            f"No source shards found under {input_root} matching rank_*/{bucket}/metadata.json"
        )
    return shards


def open_source_buffers(shards: List[Path], seed: int, cache_size: int) -> Dict[str, TrajectoryReplayBuffer]:
    buffers: Dict[str, TrajectoryReplayBuffer] = {}
    for shard_idx, shard in enumerate(shards):
        buf = make_buffer_for_load(seed + shard_idx, cache_size=cache_size)
        buf.load_checkpoint(str(shard), is_distributed=False)
        buffers[str(shard)] = buf
    return buffers


def collect_trajectory_handles(
    source_buffers: Dict[str, TrajectoryReplayBuffer],
    shuffle: bool,
    seed: int,
) -> List[Dict]:
    handles: List[Dict] = []
    for shard_path, buf in source_buffers.items():
        traj_ids = list(buf._trajectory_id_list)
        traj_ids.sort()
        for tid in traj_ids:
            info = buf._trajectory_index[tid]
            handles.append(
                {
                    "shard_path": shard_path,
                    "trajectory_id": int(tid),
                    "model_weights_id": info["model_weights_id"],
                }
            )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(handles)
    return handles


def load_one_trajectory(
    source_buffers: Dict[str, TrajectoryReplayBuffer],
    shard_path: str,
    trajectory_id: int,
    model_weights_id: str,
):
    buf = source_buffers[shard_path]
    return buf._load_trajectory(trajectory_id, model_weights_id)


def close_buffers(buffers: Dict[str, TrajectoryReplayBuffer]) -> None:
    for buf in buffers.values():
        try:
            buf.close(wait=True)
        except Exception:
            pass


def write_manifest(output_root: Path, manifest: Dict) -> None:
    with open(output_root / "reshard_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Merge/re-shard offline_collection rank shards into a new world-size layout."
    )
    parser.add_argument("--input-root", required=True, help="Path like .../offline_collection")
    parser.add_argument(
        "--bucket",
        required=True,
        choices=["all", "success", "failure"],
        help="Which bucket under rank_i to reshard",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root. Script creates output_root/rank_0 ... output_root/rank_{N-1}",
    )
    parser.add_argument(
        "--target-world-size",
        required=True,
        type=int,
        help="How many target rank shards to produce for later training",
    )
    parser.add_argument(
        "--trajectory-format",
        default="pt",
        choices=["pt"],
        help="Trajectory format for output buffer",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle trajectories before repartitioning (recommended)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="How many trajectories to buffer in memory per output shard before flushing",
    )
    parser.add_argument(
        "--source-cache-size",
        type=int,
        default=64,
        help="CPU cache size for each opened source shard buffer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print summary, do not write output",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    target_world_size = int(args.target_world_size)
    if target_world_size <= 0:
        raise ValueError("--target-world-size must be > 0")

    source_shards = discover_source_shards(input_root, args.bucket)
    source_buffers = open_source_buffers(source_shards, seed=args.seed, cache_size=args.source_cache_size)
    handles = collect_trajectory_handles(source_buffers, shuffle=args.shuffle, seed=args.seed)
    total = len(handles)
    if total == 0:
        close_buffers(source_buffers)
        raise RuntimeError("No trajectories found to reshard")

    per_target = math.ceil(total / target_world_size)
    print(f"[reshard] input_root={input_root}")
    print(f"[reshard] bucket={args.bucket}")
    print(f"[reshard] source_shards={len(source_shards)}")
    print(f"[reshard] total_trajectories={total}")
    print(f"[reshard] target_world_size={target_world_size}")
    print(f"[reshard] approx_trajectories_per_target={per_target}")
    print(f"[reshard] output_root={output_root}")

    manifest = {
        "input_root": str(input_root),
        "bucket": args.bucket,
        "source_shards": [str(x) for x in source_shards],
        "target_world_size": target_world_size,
        "total_trajectories": total,
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "source_cache_size": int(args.source_cache_size),
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        close_buffers(source_buffers)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    target_buffers: List[TrajectoryReplayBuffer] = []
    pending = [[] for _ in range(target_world_size)]
    target_counts = [0 for _ in range(target_world_size)]

    for rank in range(target_world_size):
        rank_dir = output_root / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        target_buffers.append(
            make_buffer_for_save(
                str(rank_dir),
                seed=args.seed + rank,
                trajectory_format=args.trajectory_format,
            )
        )

    try:
        for idx, h in enumerate(handles):
            target_rank = idx % target_world_size
            traj = load_one_trajectory(
                source_buffers=source_buffers,
                shard_path=h["shard_path"],
                trajectory_id=h["trajectory_id"],
                model_weights_id=h["model_weights_id"],
            )
            pending[target_rank].append(traj)
            target_counts[target_rank] += 1

            if len(pending[target_rank]) >= args.chunk_size:
                target_buffers[target_rank].add_trajectories(pending[target_rank])
                pending[target_rank].clear()

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[reshard] processed {idx + 1}/{total} trajectories")

        for rank in range(target_world_size):
            if pending[rank]:
                target_buffers[rank].add_trajectories(pending[rank])
                pending[rank].clear()
            target_buffers[rank].close(wait=True)
    finally:
        close_buffers(source_buffers)

    manifest["target_counts"] = target_counts
    write_manifest(output_root, manifest)
    print("[reshard] done")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
