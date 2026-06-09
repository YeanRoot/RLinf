import json
from pathlib import Path
import numpy as np

paths = [
    Path("/home/ubuntu/users/angen.ye/gwp/norm_stats_delta.json"),
    Path("/home/ubuntu/users/angen.ye/gwp/rollout1/norm_stats_delta.json"),
]

loaded = []
for path in paths:
    data = json.loads(path.read_text())
    stats = data["norm_stats"]
    loaded.append(stats)
    print(f"\n== {path} ==")
    for key, value in stats.items():
        print(key, sorted(value.keys()))
        for subkey, arr in value.items():
            a = np.asarray(arr, dtype=np.float64)
            print(
                f"  {subkey:>8s} shape={a.shape} "
                f"mean={a.mean():.6f} std={a.std():.6f} "
                f"min={a.min():.6f} max={a.max():.6f} first3={a[:3]} last3={a[-3:]}"
            )

print("\n== abs diffs ==")
for key in sorted(set(loaded[0]).intersection(loaded[1])):
    for subkey in sorted(set(loaded[0][key]).intersection(loaded[1][key])):
        a = np.asarray(loaded[0][key][subkey], dtype=np.float64)
        b = np.asarray(loaded[1][key][subkey], dtype=np.float64)
        n = min(a.size, b.size)
        d = np.abs(a.reshape(-1)[:n] - b.reshape(-1)[:n])
        print(
            f"{key}.{subkey}: shape0={a.shape} shape1={b.shape} "
            f"mean_abs={d.mean():.6f} max_abs={d.max():.6f}"
        )
