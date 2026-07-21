"""
主模型 v2 使用示例：构造数据加载器、训练、评估流程
可直接运行以验证 model_v2.py 是否能正常前向/反向传播。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model_v2 import MainSOHModelV2


def make_dummy_data(n_samples=500, in_channels=3, seq_len=32):
    """生成随机伪数据用于快速验证模型。"""
    X = torch.randn(n_samples, in_channels, seq_len)
    y = torch.sigmoid(X[:, 0, :].mean(dim=1) * 0.5 + X[:, 1, :].std(dim=1) * 0.3)
    y = y.unsqueeze(1)
    return X, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_metrics(y_true, y_pred):
    """计算 MAE, MAPE, RMSE, R²。"""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mae = np.mean(np.abs(y_true - y_pred))
    # MAPE: 避免除零，SOH > 0.01 正常情况不会出现
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        all_preds.append(pred)
        all_targets.append(yb)
    return compute_metrics(torch.cat(all_targets), torch.cat(all_preds))


def main():
    in_channels = 3
    seq_len = 32
    hidden_dim = 64
    batch_size = 16
    epochs = 20
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 构造伪数据集
    X_train, y_train = make_dummy_data(n_samples=500, seq_len=seq_len)
    X_val, y_val = make_dummy_data(n_samples=100, seq_len=seq_len)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    # 模型
    model = MainSOHModelV2(
        in_channels=in_channels,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=4,
        cnn_layers=3,
        gru_layers=1,
        gru_dropout=0.1,
        se_reduction=4,
        dropout=0.2,
        output_dim=1,
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch [{epoch:03d}/{epochs}]  Train MAE: {train_loss:.6f}")

    # 最终评估
    metrics = evaluate(model, val_loader, device)
    print(f"\n验证集指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if "R2" not in k else f"  {k}: {v:.4f}")

    torch.save(model.state_dict(), "main_soh_model_v2.pth")
    print("\n模型已保存为 main_soh_model_v2.pth")


if __name__ == "__main__":
    main()
