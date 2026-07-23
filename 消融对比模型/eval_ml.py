"""
XGBoost / SVR 基准线评估：贝叶斯搜索 + k-fold 交叉验证

用法:
    python 消融对比模型/eval_ml.py --dataset xjtu_batch1 --seq_len 32 --trials 20
    python 消融对比模型/eval_ml.py --dataset calce --seq_len 32 --model svr --trials 15
"""

import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
from data_loader import load_fold, get_fold_count


def suggest_xgb(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def suggest_svr(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 100, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.001, 1, log=True),
    }


def objective(trial, dataset, seq_len, model_name, n_folds):
    if model_name == "xgb":
        from xgboost import XGBRegressor
        params = suggest_xgb(trial)
        model = XGBRegressor(**params, verbosity=0, random_state=42)
    else:
        from sklearn.svm import SVR
        params = suggest_svr(trial)
        model = SVR(kernel="rbf", **params)

    fold_maes = []
    for fold_idx in range(n_folds):
        X_train, y_train, X_test, y_test = load_fold(dataset, seq_len, fold_idx)
        X_train = X_train.numpy().reshape(X_train.shape[0], -1)
        X_test  = X_test.numpy().reshape(X_test.shape[0], -1)

        model.fit(X_train, y_train.numpy())
        preds = model.predict(X_test)
        fold_maes.append(mean_absolute_error(y_test.numpy(), preds))

        trial.report(np.mean(fold_maes), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_maes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xjtu_batch1")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "svr"])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_folds = get_fold_count(args.dataset, args.seq_len)
    print(f"模型: {args.model}  |  数据集: {args.dataset}  |  folds: {n_folds}  |  trials: {args.trials}")

    study = optuna.create_study(
        study_name=f"{args.model}_{args.dataset}_n{args.seq_len}",
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        storage=f"sqlite:///results/optuna_{args.model}_{args.dataset}_n{args.seq_len}.db",
        load_if_exists=True,
    )

    func = lambda trial: objective(trial, args.dataset, args.seq_len, args.model, n_folds)
    study.optimize(func, n_trials=args.trials, show_progress_bar=True)

    print(f"\n最佳 {n_folds}-fold 平均 MAE: {study.best_value:.6f}")
    print(f"最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 最终评估
    print(f"\n用最佳参数跑全部 fold 最终评估...")
    all_metrics = {"MAE": [], "MAPE": [], "RMSE": [], "R2": []}
    for fold_idx in range(n_folds):
        X_train, y_train, X_test, y_test = load_fold(args.dataset, args.seq_len, fold_idx)
        X_train = X_train.numpy().reshape(X_train.shape[0], -1)
        X_test  = X_test.numpy().reshape(X_test.shape[0], -1)

        if args.model == "xgb":
            from xgboost import XGBRegressor
            model = XGBRegressor(**study.best_params, verbosity=0, random_state=42)
        else:
            from sklearn.svm import SVR
            model = SVR(kernel="rbf", **study.best_params)
        model.fit(X_train, y_train.numpy())
        preds = model.predict(X_test)
        yt = y_test.numpy()

        mae  = mean_absolute_error(yt, preds)
        mape = np.mean(np.abs((yt - preds) / (yt + 1e-8)))
        rmse = np.sqrt(mean_squared_error(yt, preds))
        r2   = r2_score(yt, preds)

        for k, v in zip(all_metrics.keys(), [mae, mape, rmse, r2]):
            all_metrics[k].append(v)
        print(f"  fold {fold_idx}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    print(f"\n  ── {n_folds}-fold 平均 ({args.model}) ──")
    for k, vs in all_metrics.items():
        arr = np.array(vs)
        print(f"  {k}: {np.mean(arr):.4f} ± {np.std(arr):.4f}")

    with open(f"results/best_params_{args.model}_{args.dataset}_n{args.seq_len}.json", "w") as f:
        json.dump({"best_mae": study.best_value, **study.best_params}, f, indent=2)


if __name__ == "__main__":
    main()
