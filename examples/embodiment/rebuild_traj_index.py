import os
import re
import json
import glob
import torch

root = "/shared_disk/users/angen.ye/code/world_module_rollout/RLinf/examples/results/gigawa_offline_collect4_12chunk_fix/mergeall_repaired3"


def build_index_for_dir(target_dir: str):
    traj_paths = sorted(glob.glob(os.path.join(target_dir, "trajectory_*.pt")))
    assert len(traj_paths) > 0, f"No trajectory_*.pt found in {target_dir}"

    trajectory_index = {}
    trajectory_id_list = []
    total_samples = 0
    max_tid = -1

    for path in traj_paths:
        name = os.path.basename(path)
        m = re.match(r"trajectory_(\d+)_(.+)\.pt$", name)
        if m is None:
            print(f"[skip] invalid filename: {name}")
            continue

        tid = int(m.group(1))
        model_weights_id = m.group(2)

        data = torch.load(path, map_location="cpu")

        if "prev_logprobs" in data and data["prev_logprobs"] is not None:
            T, B = data["prev_logprobs"].shape[:2]
            shape = list(data["prev_logprobs"].shape)
        elif "rewards" in data and data["rewards"] is not None:
            T, B = data["rewards"].shape[:2]
            shape = list(data["rewards"].shape)
        elif "actions" in data and data["actions"] is not None:
            T, B = data["actions"].shape[:2]
            shape = list(data["actions"].shape)
        else:
            print(f"[skip] empty traj: {name}")
            continue

        num_samples = int(T * B)
        max_episode_length = int(data.get("max_episode_length", T))

        trajectory_index[tid] = {
            "num_samples": num_samples,
            "trajectory_id": tid,
            "max_episode_length": max_episode_length,
            "shape": shape,
            "model_weights_id": model_weights_id,
        }
        trajectory_id_list.append(tid)
        total_samples += num_samples
        max_tid = max(max_tid, tid)

    trajectory_id_list = sorted(trajectory_id_list)

    index_data = {
        "trajectory_index": {str(k): v for k, v in trajectory_index.items()},
        "trajectory_id_list": trajectory_id_list,
    }

    index_path = os.path.join(target_dir, "trajectory_index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f)

    metadata_path = os.path.join(target_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        metadata = {
            "trajectory_format": "pt",
            "size": len(trajectory_id_list),
            "total_samples": total_samples,
            "trajectory_counter": max_tid + 1,
            "seed": 1234,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"[create] metadata.json -> {metadata_path}")
    else:
        print(f"[keep] metadata.json already exists -> {metadata_path}")

    print(
        f"[done] {target_dir}: "
        f"{len(trajectory_id_list)} trajectories, total_samples={total_samples}, "
        f"saved={index_path}"
    )


def main():
    rank_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "rank_*")) if os.path.isdir(d)
    )

    if len(rank_dirs) > 0:
        print(f"Found {len(rank_dirs)} rank dirs under {root}")
        for d in rank_dirs:
            build_index_for_dir(d)
    else:
        print(f"No rank_* dirs found, fallback to root: {root}")
        build_index_for_dir(root)


if __name__ == "__main__":
    main()