"""
贝叶斯超参数搜索 v2：基于 Optuna TPE 采样器 + k-fold 交叉验证。

每个 trial 跑全部 fold，取 fold 平均 MAE 作为优化目标，
搜索结束后用最佳参数跑最终评估（全部指标）。

用法:
    # 默认：Batch-1, seq_len=32, 30 trials
    python 主模型/bayesian_search_v2.py

    # 指定数据集和 seq_len
    python 主模型/bayesian_search_v2.py --dataset calce --seq_len 64 --trials 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v2 import MainSOHModelV2
from data_loader import DATASET_REGISTRY, load_fold, get_fold_count


# ═══════════════════════════════════════════════════════════════
# 1. 超参数搜索空间
# ═══════════════════════════════════════════════════════════════
def suggest_hyperparams(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    head_candidates = [h for h in [2, 4, 8] if hidden_dim % h == 0]
    num_heads = trial.suggest_categorical("num_heads", head_candidates)

    cnn_layers = trial.suggest_int("cnn_layers", 1, 3)
    gru_layers = trial.suggest_int("gru_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    gru_dropout = trial.suggest_float("gru_dropout", 0.0, 0.3)

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


# ═══════════════════════════════════════════════════════════════
# 2. 单 fold 训练
# ═══════════════════════════════════════════════════════════════
def train_one_fold(model, train_loader, val_loader, params, max_epochs, device, patience=20):
    """训练单个 fold，早停后返回最佳验证 MAE。"""
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)

    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.view(-1, 1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_mae_sum += nn.L1Loss()(model(xb), yb.view(-1, 1)).item() * xb.size(0)
        val_mae = val_mae_sum / len(val_loader.dataset)

        scheduler.step(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_mae


# ═══════════════════════════════════════════════════════════════
# 3. 评估指标
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}


def evaluate_fold(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        all_preds.append(model(xb))
        all_targets.append(yb.view(-1, 1))
    return compute_metrics(torch.cat(all_targets), torch.cat(all_preds))


# ═══════════════════════════════════════════════════════════════
# 4. Optuna 目标函数 — k-fold 交叉验证
# ═══════════════════════════════════════════════════════════════
def objective(trial, dataset, seq_len, max_epochs, device):
    params = suggest_hyperparams(trial)
    n_folds = get_fold_count(dataset, seq_len)

    fold_maes = []

    for fold_idx in range(n_folds):
        # 加载数据：test 是一个完整电池，train 是其余电池
        X_train_all, y_train_all, X_test, y_test = load_fold(dataset, seq_len, fold_idx)

        # 从训练集切 20% 做验证（早停用），测试集完全不参与训练
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=42
        )
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=params["batch_size"], shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=64, shuffle=False,
        )

        # 构造模型
        model = MainSOHModelV2(
            in_channels=3, seq_len=seq_len,
            hidden_dim=params["hidden_dim"],
            num_heads=params["num_heads"],
            cnn_layers=params["cnn_layers"],
            gru_layers=params["gru_layers"],
            gru_dropout=params["gru_dropout"],
            se_reduction=4,
            dropout=params["dropout"],
            output_dim=1,
        ).to(device)

        # 训练（用 val 做早停）
        train_one_fold(model, train_loader, val_loader, params, max_epochs, device)

        # 最终评估：用测试集（不参与训练的完整电池）
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=64, shuffle=False,
        )
        test_metrics = evaluate_fold(model, test_loader, device)
        fold_maes.append(test_metrics["MAE"])

        # 向 Optuna 报告中间结果（运行中 fold 的平均 MAE，用于剪枝）
        trial.report(np.mean(fold_maes), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_maes)


# ═══════════════════════════════════════════════════════════════
# 5. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="贝叶斯超参数搜索")
    parser.add_argument("--dataset", type=str, default="xjtu_batch1",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--seq_len", type=str, default="32",
                        help="序列长度: 32, 64, 128, 或 all（依次跑三个）")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trial 数")
    parser.add_argument("--epochs", type=int, default=60, help="搜索阶段每 fold 最大 epoch（少一点，快筛）")
    parser.add_argument("--final_epochs", type=int, default=200, help="最佳参数重跑时的 epoch（充足，出最终结果）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # 解析 seq_len
    seq_lens = [32, 64, 128] if args.seq_len == "all" else [int(args.seq_len)]

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for sl in seq_lens:
        args.seq_len = sl
        print(f"\n{'#' * 50}")
        print(f"#  seq_len = {sl}")
        print(f"{'#' * 50}")

        n_folds = get_fold_count(args.dataset, args.seq_len)
        print(f"数据集: {args.dataset}  |  seq_len: {args.seq_len}  |  folds: {n_folds}")
        print(f"设备: {device}  |  trials: {args.trials}  |  每 fold 最大 epochs: {args.epochs}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # ── 搜索（SQLite 持久化，所有 trial 写入磁盘） ──
        os.makedirs("results", exist_ok=True)
    db_path = f"results/optuna_{args.dataset}_n{args.seq_len}.db"
    study = optuna.create_study(
        study_name=f"{args.dataset}_n{args.seq_len}",
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    func = lambda trial: objective(trial, args.dataset, args.seq_len, args.epochs, device)

    t_start = time.time()
    study.optimize(func, n_trials=args.trials, show_progress_bar=True)
    elapsed = time.time() - t_start

    print("\n" + "=" * 60)
    print(f"搜索完成  |  耗时: {elapsed/60:.1f} min")

    # ── 导出所有 trial 数据（论文用） ──
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"results/all_trials_{args.dataset}_n{args.seq_len}.csv", index=False)
    print(f"全部 {len(study.trials)} 个 trial 已保存到 results/all_trials_{args.dataset}_n{args.seq_len}.csv")

    # 超参重要性
    try:
        importance = optuna.importance.get_param_importances(study)
        with open(f"results/param_importance_{args.dataset}_n{args.seq_len}.json", "w") as f:
            json.dump({k: float(v) for k, v in importance.items()}, f, indent=2)
        print("超参数重要性：")
        for k, v in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")
    except Exception:
        pass
    print(f"最佳 {n_folds}-fold 平均 MAE: {study.best_value:.6f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最佳参数
    out_path = f"results/best_params_{args.dataset}_n{args.seq_len}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"best_mae": study.best_value, **study.best_params}, f, indent=2, ensure_ascii=False)
    print(f"\n参数已保存: {out_path}")

    # ── 最终评估：用最佳参数跑全部 fold，输出四个指标 ──
    print("\n" + "=" * 60)
    print("用最佳参数跑全部 fold 最终评估...")
    best_params = study.best_params

    all_metrics = {"MAE": [], "MAPE": [], "RMSE": [], "R2": []}
    for fold_idx in range(n_folds):
        X_train_all, y_train_all, X_test, y_test = load_fold(args.dataset, args.seq_len, fold_idx)

        # 切 20% 验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all, y_train_all, test_size=0.2, random_state=42
        )
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params["batch_size"], shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
        test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

        model = MainSOHModelV2(
            in_channels=3, seq_len=args.seq_len,
            hidden_dim=best_params["hidden_dim"],
            num_heads=best_params["num_heads"],
            cnn_layers=best_params["cnn_layers"],
            gru_layers=best_params["gru_layers"],
            gru_dropout=best_params["gru_dropout"],
            se_reduction=4, dropout=best_params["dropout"], output_dim=1,
        ).to(device)

        train_one_fold(model, train_loader, val_loader, best_params, args.final_epochs, device)
        metrics = evaluate_fold(model, test_loader, device)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])
        print(f"  fold {fold_idx}: MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")

    print(f"\n  ── {n_folds}-fold 平均 ──")
    for k, vs in all_metrics.items():
        arr = np.array(vs)
        print(f"  {k}: {np.mean(arr):.4f} ± {np.std(arr):.4f}")


if __name__ == "__main__":
    main()
