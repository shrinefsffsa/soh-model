"""
电池 SOH 数据加载器：读取预处理好的 .npy fold 数据。

数据格式：
  X: [n_samples, 3, seq_len]  — 3 通道（IC 曲线分段 / 电压 / IC 值），seq_len 个时间步
  y: [n_samples]              — SOH 标签

目录结构（数据仅以 fold 形式存在，无顶层合并文件）：
  {dataset}数据/folds/{prefix}_n{seq}/
    fold_0/
      X_train.npy  y_train.npy
      X_test.npy   y_test.npy
    fold_1/ ...
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ── 数据集注册表 ──────────────────────────────────────────────
DATASET_REGISTRY = {
    "calce":       {"window_dir": "calce数据", "prefix": "CALCE"},
    "xjtu_batch1": {"window_dir": "xjtu数据",  "prefix": "Batch-1"},
    "xjtu_batch3": {"window_dir": "xjtu数据",  "prefix": "Batch-3"},
}


def _resolve(dataset: str, seq_len: int):
    if dataset not in DATASET_REGISTRY:
        raise ValueError(f"未知数据集: {dataset}，可选: {list(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[dataset]
    return info["window_dir"], f"{info['prefix']}_n{seq_len}"


def load_fold(dataset: str, seq_len: int, fold_idx: int, base_dir: str = "."):
    """
    加载单个 fold 的训练/测试集。

    返回:
        (X_train, y_train, X_test, y_test): 四个 torch.Tensor
    """
    window_dir, prefix = _resolve(dataset, seq_len)
    fold_dir = os.path.join(base_dir, window_dir, "folds", prefix, f"fold_{fold_idx}")

    X_train = torch.from_numpy(np.load(os.path.join(fold_dir, "X_train.npy"))).float()
    y_train = torch.from_numpy(np.load(os.path.join(fold_dir, "y_train.npy"))).float()
    X_test  = torch.from_numpy(np.load(os.path.join(fold_dir, "X_test.npy"))).float()
    y_test  = torch.from_numpy(np.load(os.path.join(fold_dir, "y_test.npy"))).float()
    return X_train, y_train, X_test, y_test


def get_fold_count(dataset: str, seq_len: int, base_dir: str = "."):
    """返回数据集在指定 seq_len 下的 fold 数。"""
    window_dir, prefix = _resolve(dataset, seq_len)
    fold_base = os.path.join(base_dir, window_dir, "folds", prefix)
    if not os.path.isdir(fold_base):
        return 0
    return len([d for d in os.listdir(fold_base) if d.startswith("fold_")])


def get_n_samples(dataset: str, seq_len: int, base_dir: str = "."):
    """返回数据集总样本数（汇总所有 fold）。"""
    n_total = 0
    n_folds = get_fold_count(dataset, seq_len, base_dir)
    for i in range(n_folds):
        Xtr, _, Xte, _ = load_fold(dataset, seq_len, i, base_dir)
        n_total += Xtr.shape[0] + Xte.shape[0]
    return n_total


def create_dataloader(X, y, batch_size=32, shuffle=True):
    """从 Tensor 创建 DataLoader。"""
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)


# ── 快速测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    for ds in DATASET_REGISTRY:
        for sl in [32, 64, 128]:
            n_folds = get_fold_count(ds, sl)
            n_total = get_n_samples(ds, sl)
            print(f"{ds:15s}  seq={sl:3d}  samples={n_total:5d}  folds={n_folds}")

            if n_folds > 0:
                Xtr, ytr, Xte, yte = load_fold(ds, sl, fold_idx=0)
                print(f"         fold_0:  train={list(Xtr.shape)} {list(ytr.shape)}  test={list(Xte.shape)} {list(yte.shape)}")
