import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.config import N_SPLITS, RESULTS_DIR

logger = logging.getLogger(__name__)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)

def compute_all_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE":    rmse(y_true, y_pred),
        "MAE":     mae(y_true, y_pred),
        "MAPE":    mape(y_true, y_pred),
        "R2":      r2(y_true, y_pred),
        "Dir_Acc": directional_accuracy(y_true, y_pred),
        "Samples": len(y_true),
    }


def evaluate_model(model_cls, X: np.ndarray, y: np.ndarray,
                   model_name: str, **model_kwargs) -> dict:
    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_metrics = []
    all_true, all_pred = [], []

    logger.info(f"\n{'='*55}")
    logger.info(f"  Evaluating: {model_name}  ({N_SPLITS}-fold TimeSeriesCV)")
    logger.info(f"{'='*55}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model  = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_true = y_test[len(y_test) - len(y_pred):]

        metrics = compute_all_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        logger.info(
            f"  Fold {fold}  RMSE={metrics['RMSE']:.2f}  "
            f"R²={metrics['R2']:.4f}  Dir={metrics['Dir_Acc']:.1f}%"
        )

    agg = {
        k: {
            "mean": float(np.mean([m[k] for m in fold_metrics])),
            "std":  float(np.std( [m[k] for m in fold_metrics])),
        }
        for k in fold_metrics[0]
    }
    oof_metrics = compute_all_metrics(np.array(all_true), np.array(all_pred))

    logger.info(
        f"\n  OOF summary → "
        f"RMSE={oof_metrics['RMSE']:.2f}  "
        f"R²={oof_metrics['R2']:.4f}  "
        f"Dir={oof_metrics['Dir_Acc']:.1f}%"
    )

    result = {
        "model_name":   model_name,
        "fold_metrics": fold_metrics,
        "aggregate":    agg,
        "oof_metrics":  oof_metrics,
        "y_true":       all_true,
        "y_pred":       all_pred,
    }

    _save_results(result, model_name)
    return result


def _save_results(result: dict, model_name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
    path      = os.path.join(RESULTS_DIR, f"{safe_name}.json")

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(result, f, default=_convert, indent=2)
    logger.info(f"  Saved → {path}")
