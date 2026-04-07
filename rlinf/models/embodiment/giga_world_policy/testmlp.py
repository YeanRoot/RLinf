import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

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
print("device =", device)

# =========================
# 2. 你的 MLP
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
# 3. 数据集
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
        target_dim=168,
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

        # 真值 y: [N, 168]
        self.y = torch.randn(num_samples, target_dim)

        if mode == "zero":
            aux = torch.zeros(num_samples, aux_dim)
            self.x = torch.cat([aux, self.y], dim=1)   # [N, 2062+168]
        elif mode == "noise":
            aux = torch.randn(num_samples, aux_dim) * noise_std
            self.x = torch.cat([aux, self.y], dim=1)   # [N, 2062+168]
        elif mode == "remove":
            self.x = self.y.clone()                    # [N, 168]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# =========================
# 4. 训练与评估
# =========================
def train_one_model(
    mode="zero",
    num_samples=2000,
    batch_size=128,
    epochs=100,
    lr=1e-3,
    noise_std=1.0,
):
    print(f"\n==================== mode = {mode} ====================")

    dataset = SyntheticDataset(
        num_samples=num_samples,
        aux_dim=2048 + 14,
        target_dim=168,
        mode=mode,
        noise_std=noise_std,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if mode in ["zero", "noise"]:
        input_dim = 2048 + 14 + 168
    else:
        input_dim = 168

    model = MLP(
        input_dim=input_dim,
        hidden_dims=[2048, 1024],
        output_dim=168,
        activate_final=False,
        layer_norm=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d} | train_mse = {train_loss:.8f} | test_mse = {test_loss:.8f}")

    # 随机看几个样本误差
    model.eval()
    with torch.no_grad():
        x, y = next(iter(test_loader))
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        sample_mse = ((pred - y) ** 2).mean(dim=1)
        print("sample mse (first 10):", sample_mse[:10].detach().cpu().numpy())

    return model

# =========================
# 5. 主函数：分别测试三种模式
# =========================
if __name__ == "__main__":
    # 情况1：前 2048+14 维全 0
    model_zero = train_one_model(
        mode="zero",
        num_samples=2000,
        batch_size=128,
        epochs=100,
        lr=1e-3,
    )

    # 情况2：前 2048+14 维是噪声
    model_noise = train_one_model(
        mode="noise",
        num_samples=20000,
        batch_size=128,
        epochs=100,
        lr=1e-3,
        noise_std=1.0,
    )

    # 情况3：直接删掉前 2048+14 维，只输入 168 维真值
    model_remove = train_one_model(
        mode="remove",
        num_samples=200000,
        batch_size=128,
        epochs=100,
        lr=1e-3,
    )