from __future__ import annotations

import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from lstm_clv.data_types.events_metadata import EventsMetadata
from lstm_clv.data_types.profit_margins import Margins
from lstm_clv.libs import ArrayArchive, get_logger, numpy_utils, pandas_utils
from lstm_clv.libs.list_indexer import ListIndexer
from tqdm import tqdm

logger = get_logger("data")


class UserEvents:
    def __init__(
        self,
        weeks: np.ndarray,
        events: np.ndarray,
        users: np.ndarray,
        histories: np.ndarray,
    ) -> None:
        self.weeks = weeks
        self.events = events
        self.users = users
        self.histories = histories  # shape: [user, ts, event,]

        self._users = list(self.users)  # type: ignore
        self._user_indices = ListIndexer(self._users)

    def filter(
        self,
        week_from: Optional[int] = None,
        week_to: Optional[int] = None,
        user_indices: Optional[List[int]] = None,
    ) -> UserEvents:
        """
        week_from (including)
        week_to (excluding)
        """
        weeks = self.weeks.copy()
        events = self.events.copy()
        users = self.users.copy()
        histories = self.histories.copy()

        if week_from:
            weeks = weeks[week_from:]
            histories = histories[:, week_from:, :]
        if week_to:
            weeks = weeks[:week_to]
            histories = histories[:, :week_to, :]
        if user_indices:
            users = users[user_indices]
            histories = histories[user_indices, :, :]

        return UserEvents(weeks=weeks, events=events, users=users, histories=histories)

    def save(self, path: str) -> None:
        load = {
            "weeks": self.weeks,
            "events": self.events,
            "users": self.users,
            "histories": self.histories,
        }
        ArrayArchive(load).save(path)

    @classmethod
    def load(cls, path: str) -> UserEvents:
        data = ArrayArchive().load(path)
        return cls(
            weeks=data["weeks"],
            events=data["events"],
            users=data["users"],
            histories=data["histories"],
        )

    def get_user_ids(self) -> List[str]:
        """Returns a list of user ids"""
        return self._users

    def get_events(self) -> List[str]:
        """Returns a list of event types (same order as in the numpy array)"""
        return list(self.events)  # type: ignore

    def get_user_history(self, user_id: str) -> np.ndarray:
        """Returns a 2-dimensional array containing history of events for this user

        [ts, event]
        """
        idx = self._user_indices[user_id]
        return self.histories[idx, :, :]

    @property
    def nr_users(self) -> int:
        """Number of users for which event data is recorded"""
        return self.histories.shape[0]

    @property
    def nr_events(self) -> int:
        """Number of event types"""
        return len(self.events)

    @property
    def nr_weeks(self) -> int:
        """Number of weeks for which event data is recorded"""
        return len(self.weeks)

    @property
    def first_week(self) -> datetime:
        """First week for which event data is recorded"""
        return numpy_utils.numpy_time_to_datetime(self.weeks[0])

    @property
    def last_week(self) -> datetime:
        """Last week for which event data is recorded"""
        return numpy_utils.numpy_time_to_datetime(self.weeks[-1])

    @property
    def nr_nonzero_events(self) -> int:
        """A number of non-zero events"""
        return (self.histories != 0).sum()  # type: ignore

    @property
    def nr_transactions_per_user(self) -> np.ndarray:
        return (self.histories > 0).sum(axis=1).sum(axis=1)  # type: ignore

    def __len__(self) -> int:
        """Total number of week-user-event combinations"""
        return self.histories.reshape(-1).shape[0]

    @classmethod
    def construct(
        cls,
        df: pd.DataFrame,
        aggregate_events: bool,
        dummify_events: bool,
        profit_margins: Margins,
    ) -> UserEvents:
        """doc"""

        pandas_utils.assert_events_df(df)

        if aggregate_events:
            df["value"] = pandas_utils.revenue_to_profit(df, profit_margins.as_dict())
            df = pandas_utils.aggregate_by_week(df)
            df["event"] = "ALL"

        # only keep positive value transactions
        df = df[df["value"] > 0]
        df = df[["user_id", "ts", "event", "value"]]

        meta = EventsMetadata(df)

        # format: user, ts, event
        user_histories = np.zeros((meta.nr_users, meta.nr_weeks, meta.nr_events))

        _get_user_index = lambda user_id: meta.user_id_indices[user_id]
        _get_event_index = lambda event: meta.events_indices[event]
        _get_week_index = (
            lambda week: pandas_utils.weeks_between(meta.first_week, week) - 1
        )

        # values = [user_id, ts, event, value]
        for values in tqdm(df.values, leave=False):  # type: ignore

            user_histories[
                _get_user_index(values[0]),
                _get_week_index(values[1]),
                _get_event_index(values[2]),
            ] = values[3]

        if dummify_events:
            user_histories: np.ndarray = np.where(user_histories > 0, 1, 0)  # type: ignore

        return cls(
            weeks=meta.range_weeks.to_numpy(),  # type: ignore
            events=np.array(meta.events),
            users=np.array(meta.user_ids),
            histories=user_histories,
        )

    def __repr__(self) -> str:
        return (
            f"UserEvents with {self.nr_users} users, {self.nr_weeks} weeks, "
            f"{self.nr_nonzero_events} nonzero events, avg. of "
            f"{self.nr_transactions_per_user.mean():.2f} per user"
        )

    def get_summary(self, profit_margins: Optional[Margins] = None) -> pd.DataFrame:
        """doc

        Returns a dataframe with following columns:
        * user_id
        * frequency
        * recency
        * monetary_value
        """
        histories: np.ndarray = self.histories.copy()

        if profit_margins:
            margins_vector = profit_margins.as_array(events=self.get_events())
            histories = histories * margins_vector[None, None, :]

        if self.nr_events > 1:
            logger.warning("events will be aggregated!")

        histories = histories.sum(axis=2)  # sum along event axis
        assert len(histories.shape) == 2, len(histories.shape)

        W = self.nr_weeks

        # frequency:    nr. of (repeated) purchases by a customer
        # age:          nr. of weeks since first purchase
        # recency:      nr. of weeks when last purchased
        # value:        avg. purchase volume (of positive purchases)

        user_txs_bool: np.ndarray = histories > 0  # type: ignore

        frequency = user_txs_bool.sum(axis=1)
        first_purchase = numpy_utils.first_true(user_txs_bool, axis=1)
        last_purchase = W - numpy_utils.first_true(
            np.flip(user_txs_bool, axis=1), axis=1
        )
        age = W - first_purchase
        recency = last_purchase - first_purchase

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            value = np.nanmean(np.where(user_txs_bool, histories, np.nan), axis=1)

        total = histories.sum(axis=1)

        assert frequency.max() <= self.nr_weeks  # type: ignore
        assert age.max() <= self.nr_weeks  # type: ignore

        summary = {
            "user_id": self.get_user_ids(),
            "frequency": frequency,
            "recency": recency,
            "age": age,
            "monetary_value_avg": value,
            "monetary_value_total": total,
        }

        return pd.DataFrame(summary)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.histories.shape[2] == 1:
            raise ValueError(
                "Benchmark can be only evaluated if aggregate_events=True"
                "was used in the pipeline config."
            )
        return pd.DataFrame(
            self.histories[:, :, 0], index=self.users, columns=self.weeks
        )
