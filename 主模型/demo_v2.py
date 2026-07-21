"""
主模型 v2 使用示例：构造数据加载器、训练、评估流程
可直接运行以验证 model_v2.py 是否能正常前向/反向传播。
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch [{epoch:03d}/{epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), "main_soh_model_v2.pth")
    print("模型已保存为 main_soh_model_v2.pth")


if __name__ == "__main__":
    main()
