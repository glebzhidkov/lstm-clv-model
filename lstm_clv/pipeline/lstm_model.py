from typing import Optional

import numpy as np
import pandas as pd
from lstm_clv.data_types import DataForLstm, Margins, Scaler, UserAttributes, UserEvents
from lstm_clv.libs import ExperimentPaths, ModelPerformance, build_model, get_logger
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

logger = get_logger("lstm-model")


class ClvPredictions:
    """
    Wrapper class for predictions returned by the LSTM-CLV model
    """

    def __init__(
        self, true_events: UserEvents, pred_events: UserEvents, margins: Margins
    ):
        self.true_values = true_events.histories
        self.pred_values = pred_events.histories
        self.users = true_events.get_user_ids()
        self.events = true_events.get_events()
        self.margins = margins.as_array(events=self.events)

        assert self.true_values.shape == self.pred_values.shape

        self.df = self.__calc_clv()
        self.performance = ModelPerformance(
            true=self.df["true_clv"].values,  # type: ignore
            pred=self.df["pred_clv"].values,  # type: ignore
        )

    def save(self, path_preds: str, path_performance: str):
        self.df.to_csv(path_preds, index=False)
        self.performance.save(path_performance)

    def __repr__(self) -> str:
        return (
            f"{self.performance}\n"
            f"Distribution of true values:\n{self.__describe(self.true_values)}\n"
            f"Distribution of predicted values:\n{self.__describe(self.pred_values)}\n"
        )

    def __calc_clv(self) -> pd.DataFrame:
        """doc"""

        def _calc_profit(values: np.ndarray) -> float:
            return round((values.sum(axis=0) * self.margins).sum(), 4)

        entries = []
        for idx, user_id in enumerate(self.users):
            user_entries = {
                "user_id": user_id,
                "true_clv": _calc_profit(self.true_values[idx]),
                "pred_clv": _calc_profit(self.pred_values[idx]),
            }
            entries.append(user_entries)

        return pd.DataFrame(entries)

    def __describe(self, values: np.ndarray) -> pd.DataFrame:
        min_values = values.min(axis=0).min(axis=0)
        avg_values = values.mean(axis=0).mean(axis=0)
        max_values = values.max(axis=0).max(axis=0)
        std_values = values.std(axis=0).std(axis=0)  # ?
        df = pd.DataFrame(
            [min_values, avg_values, max_values, std_values],
            index=["min", "mean", "max", "std"],
            columns=self.events,
        ).T
        return df


class LstmClvModel:
    """
    Wrapper class for the LSTM-CLV model
    """

    def __init__(
        self, paths: ExperimentPaths, data: Optional[DataForLstm] = None
    ) -> None:
        self.paths = paths
        self.data = data or DataForLstm.load(self.paths.data_for_lstm)

    def train(
        self,
        model_type: int,
        lstm_units: int,
        learning_rate: float,
        validation_ratio: float,
        batch_size: int,
        nr_epochs: int,
        early_stop_after_epochs: int,
        verbose: int,
        shuffle: bool,
        _save_model: bool = True,
    ) -> History:

        model = build_model(
            model_type=model_type,
            nr_events=self.data.nr_events,
            nr_user_attributes=self.data.nr_user_attributes,
            window=self.data.window,
            lstm_units=lstm_units,
        )

        optimizer = Adam(learning_rate=learning_rate)
        metrics = [MeanAbsoluteError()]
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=early_stop_after_epochs)
        ]

        model.compile(optimizer=optimizer, metrics=metrics, loss="mse")
        history = model.fit(
            x=self.data.X,
            y=self.data.y,
            validation_split=validation_ratio,
            batch_size=batch_size,
            epochs=nr_epochs,
            verbose=verbose,
            shuffle=shuffle,
            callbacks=callbacks,
        )
        if _save_model:
            model.save(self.paths.model_checkpoint)
        return history

    def predict_and_evaluate(self, nr_weeks: int) -> ClvPredictions:
        model: keras.Model = keras.models.load_model(self.paths.model_checkpoint)  # type: ignore
        window = self.data.window
        margins = Margins.load(self.paths.margins)

        user_events_true = UserEvents.load(self.paths.transactions_matrix)
        user_events = UserEvents.load(self.paths.transactions_matrix)
        user_attrs = UserAttributes.load(self.paths.user_attrs)

        user_events_scaler = Scaler.load(self.paths.transactions_scaler)
        user_attrs_scaler = Scaler.load(self.paths.user_attrs_scaler)

        user_events = user_events_scaler.transform(user_events)
        user_attrs = user_attrs_scaler.transform(user_attrs)

        user_ids = user_events.get_user_ids()
        user_attrs_matrix = user_attrs.construct_matrix(user_ids=user_ids)

        first_week_predict = user_events.nr_weeks - nr_weeks

        logger.info(
            f"Predicting values for weeks {first_week_predict}--{user_events.nr_weeks}"
        )

        for week in range(first_week_predict, user_events.nr_weeks):

            window_matrix = user_events.histories[:, week - window : week, :]
            history_vector = user_events.histories[:, :week, :].mean(axis=1)

            predictions = model((window_matrix, history_vector, user_attrs_matrix))
            # overwrite true values for now
            user_events.histories[:, week, :] = predictions

        user_events_pred = user_events_scaler.inverse_transform(user_events)

        self.predictions = ClvPredictions(
            true_events=user_events_true.filter(week_from=first_week_predict),
            pred_events=user_events_pred.filter(week_from=first_week_predict),
            margins=margins,
        )
        self.predictions.save(
            path_preds=self.paths.lstm_preds,
            path_performance=self.paths.lstm_performance,
        )
        return self.predictions
