"""
贝叶斯超参数搜索 v2：基于 Optuna 的 TPE 采样器
自动搜索 MainSOHModelV2 的最优超参数组合，目标是最小化验证 MAE。

注意：当前使用伪数据演示流程。接入真实电池数据时，
      请把 make_data() 替换为你的数据加载器，并确保验证集按电池单元划分，
      避免同一电池样本同时出现在训练/验证集中造成数据泄漏。
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from model_v2 import MainSOHModelV2


# ═══════════════════════════════════════════════════════════════
# 1. 数据准备（伪数据，仅作流程验证）
# ═══════════════════════════════════════════════════════════════
def make_data(n_samples=600, in_channels=3, seq_len=32, seed=42):
    torch.manual_seed(seed)
    X = torch.randn(n_samples, in_channels, seq_len)
    y = torch.sigmoid(
        0.3 * X[:, 0, :].mean(dim=1)
        + 0.2 * X[:, 1, :].std(dim=1)
        - 0.1 * X[:, 2, :].max(dim=1)[0]
    ).unsqueeze(1)
    return X, y


# ═══════════════════════════════════════════════════════════════
# 2. 超参数采样 + 模型构造
# ═══════════════════════════════════════════════════════════════
def suggest_hyperparams(trial):
    """定义贝叶斯搜索空间，返回一个参数字典。"""

    # ----- 模型结构超参数 -----
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])

    # num_heads 必须整除 hidden_dim
    head_candidates = [h for h in [2, 4, 8] if hidden_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", head_candidates)

    cnn_layers = trial.suggest_int("cnn_layers", 1, 3)
    gru_layers = trial.suggest_int("gru_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    gru_dropout = trial.suggest_float("gru_dropout", 0.0, 0.3)

    # ----- 训练超参数 -----
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    return {
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "cnn_layers": cnn_layers,
        "gru_layers": gru_layers,
        "dropout": dropout,
        "gru_dropout": gru_dropout,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
    }


def build_model(params, in_channels=3, seq_len=32, device="cpu"):
    """根据已采样的参数字典构造模型、优化器和训练配置。"""

    model = MainSOHModelV2(
        in_channels=in_channels,
        seq_len=seq_len,
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        cnn_layers=params["cnn_layers"],
        gru_layers=params["gru_layers"],
        gru_dropout=params["gru_dropout"],
        se_reduction=4,
        dropout=params["dropout"],
        output_dim=1,
    ).to(device)

    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    return model, criterion, optimizer


# ═══════════════════════════════════════════════════════════════
# 3. 训练与验证辅助函数
# ═══════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, device):
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


# ═══════════════════════════════════════════════════════════════
# 4. Optuna 目标函数
# ═══════════════════════════════════════════════════════════════
def objective(trial, in_channels=3, seq_len=32, max_epochs=100, device="cpu"):
    params = suggest_hyperparams(trial)

    X, y = make_data(n_samples=600, in_channels=in_channels, seq_len=seq_len)
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
    )

    model, criterion, optimizer = build_model(params, in_channels, seq_len, device)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 20

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # 按 epoch 用 MAE 做剪枝
        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_mae_sum += nn.L1Loss()(model(xb), yb).item() * xb.size(0)
        val_mae = val_mae_sum / len(val_loader.dataset)

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


# ═══════════════════════════════════════════════════════════════
# 5. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    func = lambda trial: objective(trial, in_channels=3, seq_len=32, max_epochs=100, device=device)

    n_trials = 30
    study.optimize(func, n_trials=n_trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("贝叶斯搜索完成")
    print(f"最佳验证 MAE: {study.best_value:.6f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    with open("best_hyperparams_v2.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)
    print("最佳参数已保存到 best_hyperparams_v2.json")


if __name__ == "__main__":
    main()
