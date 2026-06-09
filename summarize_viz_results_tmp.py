import csv
import glob

paths = [("old_gs300", "examples/results/lerobot_actor_warmup_buffer_fixed_viz_check/summary.csv")]
paths.append(("old_gs300_5", "examples/results/lerobot_actor_warmup_buffer_fixed_viz_check_5/summary.csv"))
paths.append(("old_gs600_5", "examples/results/lerobot_actor_warmup_buffer_fixed_viz_check_gs600_5/summary.csv"))
paths += [
    (p.split("/")[-2], p)
    for p in sorted(glob.glob("examples/results/lerobot_actor_warmup_fixed_training_viz/global_step_*/summary.csv"))
]

for name, path in paths:
    rows = list(csv.DictReader(open(path)))

    def avg(key):
        return sum(float(row[key]) for row in rows) / len(rows)

    print(
        name,
        "n",
        len(rows),
        "actor_joint_mae",
        f"{avg('gt_vs_actor_joint_mae'):.6f}",
        "actor_tcp_mean",
        f"{avg('gt_vs_actor_tcp_mean_m'):.6f}",
        "actor_tcp_max_avg",
        f"{avg('gt_vs_actor_tcp_max_m'):.6f}",
        "wa_joint_mae",
        f"{avg('gt_vs_wa_joint_mae'):.6f}",
    )
