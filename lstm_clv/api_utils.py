from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from tensorflow.keras.callbacks import History

from lstm_clv.libs import Config


def make_config(
    experiment_name: str,
    raw_data_path: str = "raw_data",
    aggregate_events: bool = False,
    dummify_events: bool = False,
    validation_ratio: float = 0.05,
    test_weeks: int = 10,
    min_nr_actions: int = 20,
    keep_user_ids: Optional[List[str]] = None,
    scaling_transformation: Literal[None, "log", "log+norm", "boxcox"] = "log+norm",
    scaling_outlier_threshold: int = 3,
    lstm_window: int = 12,
    drop_weeks_before_first_action: bool = True,
    churn_after_n_inactive_weeks: int = 12,
    model_type: int = 3,
    lstm_units: int = 50,
    learning_rate: float = 0.01,
    epochs: float = 100,
    batch_size: int = 1000,
    shuffle: bool = True,
    early_stop_after_epochs: int = 5,
    verbose: int = 1,
    seed: int = 7996,
) -> Config:
    """
    Create a new config for an experiment. Parameters: see README file.
    """
    config = {
        "experiment": experiment_name,
        "prepare_data": {
            "raw_data_path": raw_data_path,
            "aggregate_events": aggregate_events,
            "dummify_events": dummify_events,
            "validation_ratio": validation_ratio,
            "test_weeks": test_weeks,
            "min_nr_actions": min_nr_actions,
            "keep_user_ids": keep_user_ids,
            "seed": seed,
        },
        "scaling": {
            "transformation": scaling_transformation,
            "outlier_threshold": scaling_outlier_threshold,
        },
        "lstm": {
            "window": lstm_window,
            "drop_weeks_before_first_action": drop_weeks_before_first_action,
            "churn_after_n_inactive_weeks": churn_after_n_inactive_weeks,
            "model_type": model_type,
            "lstm_units": lstm_units,
        },
        "training": {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "early_stop_after_epochs": early_stop_after_epochs,
            "verbose": verbose,
        },
    }
    return Config(values=config, path="config.json")


def plot_training_history(history: History, title: str) -> None:

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(history.history["loss"], label="MSE (training data)")
    plt.plot(history.history["val_loss"], label="MSE (validation data)")
    plt.ylabel("MSE value")
    plt.xlabel("No. epoch")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(history.history["mean_absolute_error"], label="MAE (training data)")
    plt.plot(history.history["val_mean_absolute_error"], label="MAE (validation data)")
    plt.ylabel("MAE value")
    plt.xlabel("No. epoch")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


def plot_distribution(
    true_values: pd.Series,
    pnbd_values: pd.Series,
    lstm_values_aggr: pd.Series,
    lstm_values_prod: Optional[pd.Series],
    bin_width: int,
    right_limit: int,
    upper_limit: int,
    title: str,
    colors: List[str] = ["black", "darkgoldenrod", "crimson", "skyblue"],
    alpha: List[float] = [1.0, 0.7, 0.6, 0.6],
):
    bins = list(range(0, right_limit, bin_width))
    plt.figure(figsize=(10, 6), dpi=150)
    plt.hist(
        true_values, bins=bins, label="True values", alpha=alpha[0], color=colors[0]
    )
    plt.hist(
        pnbd_values,
        bins=bins,
        label="Predicted values - extended Pareto/NBD",
        alpha=alpha[1],
        color=colors[1],
    )
    plt.hist(
        lstm_values_aggr,
        bins=bins,
        label="Predicted values - aggregated LSTM-CLV",
        alpha=alpha[2],
        color=colors[2],
    )
    if lstm_values_prod is not None:
        plt.hist(
            lstm_values_prod,
            bins=bins,
            label="Predicted values - product-level LSTM-CLV",
            alpha=alpha[3],
            color=colors[3],
        )
    plt.axis((-right_limit * 0.05, right_limit, plt.axis()[2], upper_limit))
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


def describe_values(arr: np.ndarray, axis: int, names: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "min": arr.min(axis=axis),
            "mean": arr.mean(axis=axis),
            "max": arr.max(axis=axis),
            "std": arr.std(axis=axis),
            "skew": np.apply_along_axis(skew, axis=axis, arr=arr),
            "kurtosis": np.apply_along_axis(kurtosis, axis=axis, arr=arr),
            "nonzero": (arr > 0).mean(axis=axis),  # type: ignore
        },
        index=names,
    )
