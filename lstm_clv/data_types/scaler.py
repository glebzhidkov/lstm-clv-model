from __future__ import annotations

from typing import Literal, TypeVar, overload

import numpy as np
import pandas as pd
import scipy.special as scipy_special
from lstm_clv.data_types.user_attributes import UserAttributes
from lstm_clv.data_types.user_events import UserEvents
from scipy.stats import boxcox as fit_boxcox

T = TypeVar("T")
ScalerTransformationKind = Literal["log+norm", "log", "boxcox", None]


class ScalingMethods:
    """Methods to scale and inverse transformations"""

    @staticmethod
    def scale_for_lstm(
        values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray
    ) -> np.ndarray:
        """Input array can be 1d or 2d (no batches accepted)"""
        assert len(values.shape) < 3
        res: np.ndarray = 2 * (values - min_values) / (max_values - min_values) - 1  # type: ignore
        return res.clip(-1, 1)

    @staticmethod
    def inverse_scale_for_lstm(
        values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray
    ) -> np.ndarray:
        """Inverse normalization from [-1,1] to original scale"""
        assert len(values.shape) < 3
        return (max_values - min_values) * (values + 1) / 2 + min_values  # type: ignore

    @staticmethod
    def log_transform(values: T) -> T:
        return np.log(values + 1)  # type: ignore

    @staticmethod
    def inverse_log_transform(values: T) -> T:
        return np.exp(values) - 1  # type: ignore

    @staticmethod
    def boxcox_transform(values: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        assert len(values.shape) < 3
        assert len(lambdas.shape) == 1

        # values: (ts, event)
        for event_idx in range(values.shape[1]):
            values[:, event_idx] = scipy_special.boxcox(
                values[:, event_idx] + 1, lambdas[event_idx]
            )
        return values

    @staticmethod
    def inverse_boxcox_transform(values: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
        assert len(values.shape) < 3
        assert len(lambdas.shape) == 1
        for event_idx in range(values.shape[1]):
            values[:, event_idx] = (
                scipy_special.inv_boxcox(values[:, event_idx], lambdas[event_idx]) - 1
            )
        values[np.isnan(values)] = 0  # in certain cases, value may be not computable
        return values

    @staticmethod
    def normalize_transform(
        values: np.ndarray, means: np.ndarray, sd: np.ndarray
    ) -> np.ndarray:
        return (values - means) / sd  # type: ignore

    @staticmethod
    def inverse_normalize_transform(
        values: np.ndarray, means: np.ndarray, sd: np.ndarray
    ) -> np.ndarray:
        return values * sd + means


class Scaler:
    """Scale input data

    ```
    X <- raw_values
    X = log(X + 1)                             # log-transformation
    X = (X - X_mean) / X_std                   # normalization
    X = 2 * (X - X_min) / (X_max - X_min) - 1  # scaling to [-1, 1] for LSTM
    ```
    """

    _transformation: ScalerTransformationKind
    _outlier_threshold: float
    _features: np.ndarray
    _min_values: np.ndarray
    _max_values: np.ndarray
    _boxcox_values: np.ndarray
    _mean_values: np.ndarray
    _std_values: np.ndarray

    def __init__(
        self,
        transformation: ScalerTransformationKind,
        outlier_threshold: float,
        features: np.ndarray,
        min_values: np.ndarray,
        mean_values: np.ndarray,
        max_values: np.ndarray,
        std_values: np.ndarray,
        boxcox_values: np.ndarray,
    ) -> None:
        if transformation and transformation not in ("log", "log+norm", "boxcox"):
            raise ValueError(f"Invalid transformation type: {transformation}")

        self._transformation = transformation
        self._outlier_threshold = outlier_threshold
        self._features = features
        self._min_values = min_values
        self._mean_values = mean_values
        self._max_values = max_values
        self._std_values = std_values
        self._boxcox_values = boxcox_values

    @classmethod
    def fit_events_data(
        cls,
        data: UserEvents,
        transformation: ScalerTransformationKind,
        outlier_threshold: float,
    ) -> Scaler:

        if transformation in ("log+norm", "log"):
            data.histories = ScalingMethods.log_transform(data.histories)

        boxcox_values = np.zeros((data.nr_events))
        if transformation == "boxcox":
            for i in range(data.nr_events):
                _, boxcox_lambda = fit_boxcox(data.histories[:, :, i].reshape(-1) + 1)  # type: ignore
                boxcox_values[i] = boxcox_lambda

        scaler = cls(
            transformation=transformation,
            outlier_threshold=outlier_threshold,
            features=data.events,
            min_values=np.min(data.histories, axis=(0, 1)),
            mean_values=np.mean(data.histories, axis=(0, 1)),
            max_values=np.max(data.histories, axis=(0, 1)),
            std_values=np.std(data.histories, axis=(0, 1)),
            boxcox_values=boxcox_values,
        )
        assert scaler._min_values.sum() == 0
        return scaler

    @classmethod
    def fit_user_data(cls, data: UserAttributes, outlier_threshold: float) -> Scaler:

        df = data.df.drop("user_id", axis=1)
        return cls(
            transformation=None,
            outlier_threshold=outlier_threshold,
            features=np.array(df.columns.to_numpy()),
            min_values=df.min().to_numpy(),
            mean_values=df.mean().to_numpy(),
            max_values=df.max().to_numpy(),
            std_values=df.std().to_numpy(),
            boxcox_values=np.zeros((df.shape[1])),
        )

    @overload  # used for type hinting
    def transform(self, data: UserEvents) -> UserEvents:
        ...

    @overload  # used for type hinting
    def transform(self, data: UserAttributes) -> UserAttributes:
        ...

    def transform(self, data):
        if isinstance(data, UserEvents):
            array = data.histories.copy()
            for row_idx in range(array.shape[0]):
                array[row_idx] = self._transform(array[row_idx])
            return UserEvents(
                weeks=data.weeks, events=data.events, users=data.users, histories=array
            )
        elif isinstance(data, UserAttributes):
            if data.is_empty:
                return data
            data_scaled = UserAttributes()
            data_scaled.df = data.df
            for user_id, values in data.data.items():
                data_scaled.data[user_id] = self._transform(values)
            return data_scaled
        else:
            raise TypeError(type(data))

    def _transform(self, array: np.ndarray) -> np.ndarray:
        """Input array can be 1d or 2d (no batches accepted)"""
        array = array.copy()
        _min = self._min_values
        _max = self._max_values_rm_outliers
        _mean = self._mean_values
        _std = self._std_values

        if self._transformation == "boxcox":
            array = ScalingMethods.boxcox_transform(array, self._boxcox_values)

        elif self._transformation == "log":
            array = ScalingMethods.log_transform(array)

        elif self._transformation == "log+norm":
            array = ScalingMethods.log_transform(array)
            array = ScalingMethods.normalize_transform(array, _mean, _std)
            # normalize range for scaling as well
            _min = ScalingMethods.normalize_transform(_min, _mean, _std)
            _max = ScalingMethods.normalize_transform(_max, _mean, _std)

        # lstm scaling always applied
        return ScalingMethods.scale_for_lstm(array, _min, _max)

    def inverse_transform(self, data: UserEvents) -> UserEvents:
        if isinstance(data, UserEvents):
            array = data.histories.copy()
            for row_idx in range(array.shape[0]):
                array[row_idx] = self._inverse_transform(array[row_idx])
            return UserEvents(
                weeks=data.weeks, events=data.events, users=data.users, histories=array
            )
        else:
            raise TypeError(type(data))

    def _inverse_transform(self, array: np.ndarray) -> np.ndarray:
        """Inverse normalization from [-1,1] to original scale"""
        array = array.copy()
        _min = self._min_values
        _max = self._max_values_rm_outliers
        _mean = self._mean_values
        _std = self._std_values

        if self._transformation == "log+norm":
            # adjust scaling range according to normalization
            _min = ScalingMethods.normalize_transform(_min, _mean, _std)
            _max = ScalingMethods.normalize_transform(_max, _mean, _std)

        # lstm scaling always reversed
        array = ScalingMethods.inverse_scale_for_lstm(array, _min, _max)

        if self._transformation == "boxcox":
            array = ScalingMethods.inverse_boxcox_transform(array, self._boxcox_values)

        elif self._transformation == "log":
            array = ScalingMethods.inverse_log_transform(array)

        elif self._transformation == "log+norm":
            array = ScalingMethods.inverse_normalize_transform(array, _mean, _std)
            array = ScalingMethods.inverse_log_transform(array)

        return array

    @property
    def _max_values_rm_outliers(self) -> np.ndarray:
        """`mean + std * outlier_threshold`"""
        cap_values = self._mean_values + self._std_values * self._outlier_threshold
        return np.where(cap_values > self._max_values, self._max_values, cap_values)  # type: ignore

    @property
    def fitted_parms(self) -> pd.DataFrame:
        load = {
            "features": self._features,
            "min": self._min_values,
            "mean": self._mean_values,
            "max": self._max_values,
            "max-adj": self._max_values_rm_outliers,
            "std": self._std_values,
            "boxcox": self._boxcox_values,
            "transformation": [self._transformation] * len(self._features),
            "outlier_threshold": [self._outlier_threshold] * len(self._features),
        }
        return pd.DataFrame(load)

    def save(self, path: str) -> None:
        self.fitted_parms.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str) -> Scaler:
        data = pd.read_csv(path)
        return cls(
            transformation=data["transformation"].replace({np.nan: None}).to_numpy()[0],
            outlier_threshold=data["outlier_threshold"].to_numpy()[0],
            features=data["features"].to_numpy(),
            min_values=data["min"].to_numpy(),
            mean_values=data["mean"].to_numpy(),
            max_values=data["max"].to_numpy(),
            std_values=data["std"].to_numpy(),
            boxcox_values=data["boxcox"].to_numpy(),
        )

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:\n{self.fitted_parms}"
