import json

import numpy as np
from lstm_clv.libs import Config

# https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9


def r2value(true: np.ndarray, pred: np.ndarray) -> float:
    """Calculate R2 value for two arrays"""
    return np.corrcoef(true, pred)[0, 1] ** 2


def mse(true: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(np.square(true - pred)) # type: ignore


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(np.abs(true - pred))  # type: ignore


def hit_ratio_top_customers(true: np.ndarray, pred: np.ndarray, top: float) -> float:
    true_top = (true > np.quantile(true, top)).astype(int)  # 0/1
    pred_top = (pred > np.quantile(pred, top)).astype(int)
    return sum(true_top * pred_top) / sum(true_top)


class ModelPerformance:
    def __init__(self, true: np.ndarray, pred: np.ndarray) -> None:
        self._performance = {
            "r2": round(r2value(true, pred), 4),
            "mse": round(mse(true, pred), 4),
            "mae": round(mae(true, pred), 4),
            "hit_ratio_top_50": round(hit_ratio_top_customers(true, pred, 0.50), 4),
            "hit_ratio_top_25": round(hit_ratio_top_customers(true, pred, 0.75), 4),
            "hit_ratio_top_10": round(hit_ratio_top_customers(true, pred, 0.90), 4),
            "avg_clv_true": round(true.sum() / true.shape[0], 5),
            "avg_clv_pred": round(pred.sum() / pred.shape[0], 5),
            "total_true": round(true.sum(), 4),
            "total_pred": round(pred.sum(), 4),
        }

    def save(self, path: str) -> None:
        Config.create(path).set_values(self._performance).save()

    def __repr__(self) -> str:
        return f"Model performance: \n{json.dumps(self._performance, indent=4)}"
