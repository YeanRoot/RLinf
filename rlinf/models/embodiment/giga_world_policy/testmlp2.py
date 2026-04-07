import os
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split

# =========================
# 0. 输出目录
# =========================
SAVE_DIR = "./logs_mlp_identity"
CURVE_DIR = os.path.join(SAVE_DIR, "curves")
os.makedirs(CURVE_DIR, exist_ok=True)

LOG_TXT = os.path.join(SAVE_DIR, "experiment_log.txt")
RESULT_CSV = os.path.join(SAVE_DIR, "results.csv")
SUMMARY_PNG = os.path.join(SAVE_DIR, "summary_best_test_mse.png")

# =========================
# 1. 固定随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. 简单日志函数
# =========================
def log_print(msg: str):
    print(msg)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# 启动时写一个头
with open(LOG_TXT, "a", encoding="utf-8") as f:
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"New run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n")

log_print(f"device = {device}")
log_print(f"SAVE_DIR = {os.path.abspath(SAVE_DIR)}")

# =========================
# 3. 你的 MLP
# =========================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or activate_final:
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# =========================
# 4. 数据集
# =========================
class SyntheticDataset(Dataset):
    """
    mode:
        - 'zero'   : x = [zeros(2062), y]
        - 'noise'  : x = [noise(2062), y]
        - 'remove' : x = y
    """
    def __init__(
        self,
        num_samples=2000,
        aux_dim=2048 + 14,
        target_dim=672,
        mode="zero",
        noise_std=1.0,
    ):
        super().__init__()
        assert mode in ["zero", "noise", "remove"]
        self.num_samples = num_samples
        self.aux_dim = aux_dim
        self.target_dim = target_dim
        self.mode = mode
        self.noise_std = noise_std

        self.y = torch.randn(num_samples, target_dim)

        if mode == "zero":
            aux = torch.zeros(num_samples, aux_dim)
            self.x = torch.cat([aux, self.y], dim=1)
        elif mode == "noise":
            aux = torch.randn(num_samples, aux_dim) * noise_std
            self.x = torch.cat([aux, self.y], dim=1)
        elif mode == "remove":
            self.x = self.y.clone()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# =========================
# 5. 画单组曲线
# =========================
def plot_curves(train_losses, test_losses, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_mse")
    plt.plot(test_losses, label="test_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =========================
# 6. 单次训练
# =========================
def train_one_model(
    mode="zero",
    num_samples=2000,
    batch_size=128,
    epochs=50,
    lr=1e-3,
    noise_std=1.0,
    hidden_dims=[2048, 1024],
    weight_decay=0.0,
    layer_norm=False,
):
    exp_name = (
        f"mode={mode}"
        f"__n={num_samples}"
        f"__hd={str(hidden_dims).replace(' ', '')}"
        f"__wd={weight_decay}"
        f"__noise={noise_std}"
        f"__ln={layer_norm}"
    )

    log_print("\n" + "=" * 40)
    log_print(exp_name)
    log_print("=" * 40)

    dataset = SyntheticDataset(
        num_samples=num_samples,
        aux_dim=2048 + 14,
        target_dim=672,
        mode=mode,
        noise_std=noise_std,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    if mode in ["zero", "noise"]:
        input_dim = 2048 + 14 + 672
    else:
        input_dim = 672

    model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=672,
        activate_final=False,
        layer_norm=layer_norm,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_test_loss = float("inf")
    best_epoch = -1

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / train_count

        model.eval()
        test_loss_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                bs = x.size(0)
                test_loss_sum += loss.item() * bs
                test_count += bs

        test_loss = test_loss_sum / test_count

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            log_print(
                f"epoch {epoch:03d} | train_mse = {train_loss:.8f} | "
                f"test_mse = {test_loss:.8f} | best_test = {best_test_loss:.8f} @ {best_epoch}"
            )

    # 随机看几个样本误差
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        sample_mse = ((pred - y) ** 2).mean(dim=1)[:10].detach().cpu().numpy().tolist()

    log_print(f"sample mse (first 10): {sample_mse}")

    curve_path = os.path.join(CURVE_DIR, exp_name + ".png")
    plot_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        save_path=curve_path,
        title=exp_name,
    )
    log_print(f"curve saved to: {curve_path}")

    result = {
        "mode": mode,
        "num_samples": num_samples,
        "hidden_dims": json.dumps(hidden_dims),
        "weight_decay": weight_decay,
        "noise_std": noise_std,
        "layer_norm": layer_norm,
        "best_test_mse": best_test_loss,
        "best_epoch": best_epoch,
        "final_train_mse": train_losses[-1],
        "final_test_mse": test_losses[-1],
        "curve_path": curve_path,
    }

    return result

# =========================
# 7. 保存 CSV
# =========================
def save_results_csv(results, csv_path):
    if len(results) == 0:
        return
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

# =========================
# 8. 汇总图
# =========================
def plot_summary(results, save_path):
    if len(results) == 0:
        return

    labels = []
    values = []

    # 只取前 20 个最好结果画图，避免太挤
    sorted_results = sorted(results, key=lambda x: x["best_test_mse"])[:20]

    for r in sorted_results:
        label = (
            f"{r['mode']}\n"
            f"n={r['num_samples']}\n"
            f"hd={r['hidden_dims']}\n"
            f"wd={r['weight_decay']}"
        )
        labels.append(label)
        values.append(r["best_test_mse"])

    plt.figure(figsize=(14, 8))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("best_test_mse")
    plt.title("Top-20 Best Test MSE Results")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =========================
# 9. 主函数
# =========================
if __name__ == "__main__":
    results = []

    # 你可以先缩小搜索范围，避免一次跑太久
    num_samples_list = [2000, 20000, 100000]
    hidden_dims_list = [
        [2048, 1024],
        [512, 256],
        [256],
        [],
    ]
    weight_decay_list = [0.0, 1e-4]
    modes = ["remove", "zero", "noise"]
    noise_std_list = [0.1, 1.0]

    for mode in modes:
        for num_samples in num_samples_list:
            for hidden_dims in hidden_dims_list:
                for weight_decay in weight_decay_list:
                    if mode == "noise":
                        for noise_std in noise_std_list:
                            result = train_one_model(
                                mode=mode,
                                num_samples=num_samples,
                                batch_size=256,
                                epochs=50,
                                lr=1e-3,
                                noise_std=noise_std,
                                hidden_dims=hidden_dims,
                                weight_decay=weight_decay,
                                layer_norm=False,
                            )
                            results.append(result)
                            save_results_csv(results, RESULT_CSV)
                    else:
                        result = train_one_model(
                            mode=mode,
                            num_samples=num_samples,
                            batch_size=256,
                            epochs=50,
                            lr=1e-3,
                            noise_std=1.0,
                            hidden_dims=hidden_dims,
                            weight_decay=weight_decay,
                            layer_norm=False,
                        )
                        results.append(result)
                        save_results_csv(results, RESULT_CSV)

    results = sorted(results, key=lambda x: (x["mode"], x["best_test_mse"]))
    save_results_csv(results, RESULT_CSV)
    plot_summary(results, SUMMARY_PNG)

    log_print("\n" + "=" * 80)
    log_print("FINAL RESULTS")
    log_print("=" * 80)
    for r in results:
        log_print(str(r))

    log_print(f"\nresults csv saved to: {RESULT_CSV}")
    log_print(f"summary plot saved to: {SUMMARY_PNG}")