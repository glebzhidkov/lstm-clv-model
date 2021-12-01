from __future__ import annotations

from typing import List, Tuple

import numpy as np
from lstm_clv.data_types.user_attributes import UserAttributes
from lstm_clv.data_types.user_events import UserEvents
from lstm_clv.libs import ArrayArchive, get_logger, numpy_utils
from tqdm import tqdm

logger = get_logger("data")


class DataForLstm:
    def __init__(self):
        self._entries: List[Tuple[np.ndarray, ...]] = []
        self._W = np.array([])
        self._H = np.array([])
        self._U = np.array([])
        self._y = np.array([])
        self._is_compiled = False

    def add_entry(
        self,
        window_matrix: np.ndarray,
        history_vector: np.ndarray,
        user_attributes: np.ndarray,
        to_be_predicted: np.ndarray,
    ) -> None:
        """doc"""
        self._entries.append(
            (window_matrix, history_vector, user_attributes, to_be_predicted)
        )

    def compile(self) -> DataForLstm:
        nr_entries = len(self._entries)
        self._W = np.zeros((nr_entries, self.window, self.nr_events))
        self._H = np.zeros((nr_entries, self.nr_events))
        self._U = np.zeros((nr_entries, self.nr_user_attributes))
        self._y = np.zeros((nr_entries, self.nr_events))

        for idx, entry in enumerate(self._entries):
            self._W[idx, :, :] = entry[0]
            self._H[idx, :] = entry[1]
            self._U[idx, :] = entry[2]
            self._y[idx, :] = entry[3]

        self._entries = []
        self._is_compiled = True
        return self

    @property
    def _has_entries(self) -> bool:
        return len(self._entries) > 0

    @property
    def X(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._is_compiled, "data is not compiled"
        return self._W, self._H, self._U

    @property
    def y(self) -> np.ndarray:
        assert self._is_compiled, "data is not compiled"
        return self._y

    @classmethod
    def load(cls, path: str) -> DataForLstm:
        arrays = ArrayArchive().load(path)
        data = cls()
        data._W = arrays["W"]
        data._H = arrays["H"]
        data._U = arrays["U"]
        data._y = arrays["y"]
        data._is_compiled = True
        return data

    def save(self, path: str) -> None:
        if not self._is_compiled:
            raise ValueError("Not compiled, cannot save")
        arrays = {"W": self._W, "H": self._H, "U": self._U, "y": self._y}
        ArrayArchive(arrays).save(path)

    @property
    def window(self) -> int:
        if self._is_compiled:
            return self._W.shape[1]
        elif self._has_entries:
            return self._entries[0][0].shape[0]
        else:
            raise ValueError("Not compiled and no entries present")

    @property
    def nr_events(self) -> int:
        if self._is_compiled:
            return self._W.shape[2]
        elif self._has_entries:
            return self._entries[0][0].shape[1]
        else:
            raise ValueError("Not compiled and no entries present")

    @property
    def nr_user_attributes(self) -> int:
        if self._is_compiled:
            return self._U.shape[1]
        elif self._has_entries:
            return self._entries[0][2].shape[0]
        else:
            raise ValueError("Not compiled and no entries present")

    def __len__(self) -> int:
        if self._is_compiled:
            return self._U.shape[0]
        elif self._has_entries:
            return len(self._entries)
        else:
            return 0

    @classmethod
    def construct(
        cls,
        transactions: UserEvents,
        user_attrs: UserAttributes,
        window: int = 10,
        drop_weeks_before_first_action: bool = True,
        churn_after_n_inactive_weeks: int = 10,
    ) -> DataForLstm:
        """doc"""

        users = transactions.get_user_ids()
        nr_events = transactions.nr_events
        nr_weeks = transactions.nr_weeks
        data = cls()
        _warnings = []

        for user_id in tqdm(users, desc="Constructing batches", leave=False):

            user_attributes = user_attrs.get_user_attributes(user_id)
            user_history = transactions.get_user_history(user_id)
            assert user_history.shape == (nr_weeks, nr_events), "invalid shape"

            try:
                active_weeks = numpy_utils.items_where(
                    user_history.mean(axis=1), "!=", -1
                )
                first_active_week = numpy_utils.first_true(active_weeks)
                last_active_week = numpy_utils.last_true(active_weeks)
            except numpy_utils.NoTruesException:
                _warnings.append(f"No active weeks recorded for {user_id}.")
                continue

            first_week = first_active_week if drop_weeks_before_first_action else 0
            last_week = min(nr_weeks, last_active_week + churn_after_n_inactive_weeks)

            target_week = first_week + window  # week for which we predict

            while target_week < last_week:

                data.add_entry(
                    window_matrix=user_history[target_week - window : target_week, :],
                    history_vector=user_history[:target_week, :].mean(axis=0),
                    user_attributes=user_attributes,
                    to_be_predicted=user_history[target_week, :],
                )
                target_week += 1

        data.compile()
        return data
