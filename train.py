"""
SOH 预测模型训练脚本 — 使用预处理数据 + k-fold 交叉验证。

用法:
    # 单个数据集，单个 seq_len
    python train.py --dataset calce --seq_len 32

    # 指定超参数
    python train.py --dataset xjtu_batch1 --seq_len 64 --hidden_dim 128 --epochs 300 --batch_size 64

    # 跑所有数据集和所有 seq_len（全量评估）
    python train.py --dataset all

    # 在 AutoDL 上后台运行
    nohup python train.py --dataset all > train.log 2>&1 &
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ═══════════════════════════════════════════════════════════════
# 随机种子
# ═══════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from model_v2 import MainSOHModelV2
from data_loader import (
    DATASET_REGISTRY,
    load_fold,
    get_fold_count,
    get_n_samples,
)


# ═══════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_metrics(model, loader, device):
    """计算 RMSE, MAE, R²。"""
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).cpu()
        preds.append(pred)
        targets.append(yb.cpu())

    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8)))
    r2 = r2_score(targets, preds)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}


# ═══════════════════════════════════════════════════════════════
# 单个 fold 训练
# ═══════════════════════════════════════════════════════════════
def train_one_fold(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    weight_decay,
    device,
    patience=30,
    verbose=True,
):
    """
    在单个 fold 上训练，早停后返回最佳验证损失和 epoch。
    返回: (best_val_loss, best_epoch, best_state_dict)
    """
    criterion = nn.L1Loss()  # MAE — 对 SOH 预测更鲁棒
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb.view(-1, 1))
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    早停于 epoch {epoch}，最佳 epoch={best_epoch}")
                break

        if verbose and epoch % 50 == 0:
            print(f"    Epoch {epoch:4d}  Train MAE: {train_loss:.6f}  Val MAE: {val_loss:.6f}")

    return best_loss, best_epoch, best_state


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="训练 SOH 预测模型 (MainSOHModelV2)")
    parser.add_argument("--dataset", type=str, default="calce",
                        choices=["calce", "xjtu_batch1", "xjtu_batch3", "all"],
                        help="数据集 (default: calce)")
    parser.add_argument("--seq_len", type=int, default=32,
                        choices=[32, 64, 128],
                        help="时间窗口长度 (default: 32)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="隐藏维度 (default: 64)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="交叉注意力头数 (default: 4)")
    parser.add_argument("--cnn_layers", type=int, default=3,
                        help="CNN 分支层数 (default: 3)")
    parser.add_argument("--gru_layers", type=int, default=1,
                        help="BiGRU 层数 (default: 1)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="最大训练 epoch (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减 (default: 1e-5)")
    parser.add_argument("--patience", type=int, default=30,
                        help="早停耐心值 (default: 30)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="设备 (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果输出目录 (default: ./results)")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="数据根目录 (default: 当前目录)")
    args = parser.parse_args()

    # ── 固定随机种子 ──
    set_seed(args.seed)
    print(f"随机种子: {args.seed}")

    # ── 设备 ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ── 确定要跑的数据集 ──
    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        datasets = [args.dataset]

    # ── 产出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 结果汇总 ──
    all_results = {}

    for dataset in datasets:
        print(f"\n{'=' * 60}")
        print(f"数据集: {dataset}  |  seq_len: {args.seq_len}")
        print(f"{'=' * 60}")

        n_folds = get_fold_count(dataset, args.seq_len, base_dir=args.base_dir)
        n_total = get_n_samples(dataset, args.seq_len, base_dir=args.base_dir)
        print(f"总样本: {n_total}  |  Fold 数: {n_folds}")

        if n_folds == 0:
            print(f"  ⚠ 没有找到 folds，跳过 {dataset}")
            continue

        fold_results = {"RMSE": [], "MAE": [], "R²": []}
        best_fold = None
        best_fold_mae = float("inf")

        for fold_idx in range(n_folds):
            print(f"\n  ── Fold {fold_idx}/{n_folds-1} ──")

            # 加载数据
            X_train, y_train, X_test, y_test = load_fold(
                dataset, args.seq_len, fold_idx, base_dir=args.base_dir
            )
            train_loader = DataLoader(
                TensorDataset(X_train, y_train),
                batch_size=args.batch_size, shuffle=True
            )
            test_loader = DataLoader(
                TensorDataset(X_test, y_test),
                batch_size=args.batch_size, shuffle=False
            )

            # 构造模型
            model = MainSOHModelV2(
                in_channels=3,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                cnn_layers=args.cnn_layers,
                gru_layers=args.gru_layers,
                gru_dropout=0.1,
                se_reduction=4,
                dropout=0.2,
                output_dim=1,
            ).to(device)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    参数量: {n_params:,}")

            # 训练
            t0 = time.time()
            best_val_loss, best_epoch, best_state = train_one_fold(
                model, train_loader, test_loader,
                epochs=args.epochs, lr=args.lr,
                weight_decay=args.weight_decay,
                device=device, patience=args.patience, verbose=True,
            )
            elapsed = time.time() - t0

            # 加载最佳权重 → 测试集评估
            model.load_state_dict(best_state)
            metrics = compute_metrics(model, test_loader, device)

            for k in fold_results:
                fold_results[k].append(metrics[k])

            print(f"    ⏱ {elapsed:.0f}s  |  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R²']:.4f}")

            # 保存 fold 模型
            fold_model_path = os.path.join(
                args.output_dir, f"{dataset}_n{args.seq_len}_fold{fold_idx}.pth"
            )
            torch.save(best_state, fold_model_path)

            # 追踪最佳 fold
            if metrics["MAE"] < best_fold_mae:
                best_fold_mae = metrics["MAE"]
                best_fold = fold_idx
                best_model_path = os.path.join(
                    args.output_dir, f"{dataset}_n{args.seq_len}_best.pth"
                )
                torch.save(best_state, best_model_path)

        # ── Fold 汇总 ──
        print(f"\n  ── {dataset} n={args.seq_len} 汇总 ──")
        for metric, values in fold_results.items():
            vals = np.array(values)
            print(f"    {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (min={np.min(vals):.4f}, max={np.max(vals):.4f})")

        all_results[f"{dataset}_n{args.seq_len}"] = {
            "folds": {k: [float(v) for v in vs] for k, vs in fold_results.items()},
            "best_fold": best_fold,
            "n_params": n_params,
            "n_samples": n_total,
            "n_folds": n_folds,
            "args": vars(args),
        }

    # ── 保存结果 JSON ──
    result_path = os.path.join(args.output_dir, "results_summary.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {result_path}")


if __name__ == "__main__":
    main()
