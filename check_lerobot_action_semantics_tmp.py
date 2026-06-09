from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/users/angen.ye/gwp/RLinf")
DATASET = Path("/home/ubuntu/users/angen.ye/gwp/260423160824_7c5a")
BUFFER = ROOT / "examples/results/lerobot_actor_warmup_buffer/rank_0"


def stack_array_column(df, column):
    if column in df.columns:
        return np.stack([np.asarray(x, dtype=np.float32) for x in df[column].to_list()], axis=0)
    prefix = f"{column}."
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    cols = sorted(cols, key=lambda c: int(str(c).split(".")[-1]))
    return df[cols].to_numpy(dtype=np.float32)


info = json.loads((DATASET / "meta/info.json").read_text())
episodes = [json.loads(x) for x in (DATASET / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
ep = episodes[0]
eid = int(ep["episode_index"])
data_rel = info["data_path"].format(
    episode_chunk=eid // int(info.get("chunks_size", 1000)),
    episode_index=eid,
)
df = pd.read_parquet(DATASET / data_rel)
states = stack_array_column(df, "observation.state")[:, :14]
actions = stack_array_column(df, "action")[:, :14]
n = min(len(states), len(actions))
states, actions = states[:n], actions[:n]

print("episode", eid, "n", n)
for name, arr in [("states", states), ("actions", actions)]:
    print(name, arr.shape, "mean", arr.mean(), "std", arr.std(), "min", arr.min(), "max", arr.max())

for label, ref in [("action-state[t]", states), ("action-state[t+1]", np.r_[states[1:], states[-1:]])]:
    d = actions - ref
    print(label, "abs_mean", np.abs(d).mean(), "l2_mean", np.linalg.norm(d, axis=-1).mean(), "l2_max", np.linalg.norm(d, axis=-1).max())
    print("  first3 l2", np.linalg.norm(d, axis=-1)[:3], "last3", np.linalg.norm(d, axis=-1)[-3:])

pt_path = sorted(BUFFER.glob("trajectory_0_*.pt"))[0]
traj = torch.load(pt_path, map_location="cpu", weights_only=False)
exec_flat = traj["forward_inputs"]["action_exec"].float().view(-1, 12, 14)
model_flat = traj["actions"].float().view(-1, 12, 14)
print("pt", pt_path)
print("pt exec first chunk first action", exec_flat[0, 0].numpy())
print("raw lerobot action[0]", actions[0])
print("raw lerobot state[0]", states[0])
print("raw lerobot state[1]", states[1])
print("exec[0,0]-action[0] maxabs", np.abs(exec_flat[0, 0].numpy() - actions[0]).max())
print("exec[0,1]-action[1] maxabs", np.abs(exec_flat[0, 1].numpy() - actions[1]).max())
print("model stats", model_flat.mean().item(), model_flat.std().item(), model_flat.min().item(), model_flat.max().item())
