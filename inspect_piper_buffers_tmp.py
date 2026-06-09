import glob
import torch

paths = {
    "lerobot": sorted(glob.glob("examples/results/lerobot_actor_warmup_buffer/rank_0/*.pt"))[:1],
    "real": sorted(glob.glob("examples/results/collect_piper_gigawa_intervention100/offline_collection/rank_0/all/episode_*.pt"))[:1],
}

keys = [
    "actions",
    "action_valid_mask",
    "intervene_flags",
    "curr_obs.visual_latent",
    "curr_obs.robot_state",
    "curr_obs.raw_robot_state",
    "curr_obs.states",
    "curr_obs.ref_action",
    "next_obs.robot_state",
    "next_obs.raw_robot_state",
    "next_obs.states",
    "next_obs.ref_action",
    "forward_inputs.action",
    "forward_inputs.action_exec",
    "forward_inputs.model_action",
    "forward_inputs.policy_action_model",
    "forward_inputs.policy_action_exec",
    "forward_inputs.raw_robot_state",
    "forward_inputs.raw_states_before_action",
    "forward_inputs.ref_action",
    "forward_inputs.robot_state",
    "forward_inputs.visual_latent",
]


def get(d, key):
    cur = d
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def stat(x):
    xf = x.float()
    return (
        f"shape={tuple(x.shape)} mean={xf.mean().item():.6f} "
        f"std={xf.std().item():.6f} min={xf.min().item():.6f} max={xf.max().item():.6f}"
    )


for name, ps in paths.items():
    print(f"\n==== {name} ====")
    if not ps:
        print("NO FILES")
        continue
    p = ps[0]
    print("file", p)
    data = torch.load(p, map_location="cpu", weights_only=False)
    print("top", sorted([k for k in data.keys() if not str(k).startswith("_")]))
    for k in keys:
        v = get(data, k)
        if torch.is_tensor(v):
            print(k, stat(v))
        elif v is not None:
            print(k, type(v).__name__)
