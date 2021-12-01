from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from lstm_clv.libs import get_logger, pandas_utils

logger = get_logger("data")


class UserAttributes:
    def __init__(self) -> None:
        """Initiate user attributes with no data"""
        self.df = pd.DataFrame({"user_id": ["NO_ATTRIBUTES"], "values": [1]})
        self.data = {"mode_values": np.array([1])}
        self.unknown_users = []
        self.attributes: List[str] = ["values"]

    @property
    def is_empty(self) -> bool:
        return "NO_ATTRIBUTES" in self.df["user_id"].values

    @classmethod
    def load(cls, path: str) -> UserAttributes:
        """Load user attributes from a local CSV file

        First column needs to be "user_id", the rest the attributes
        """
        data = cls()
        data.df = pandas_utils.read_user_attributes_df(path)
        data.data = data._df_to_dict(data.df)
        data.attributes = list(data.df.columns)
        data.attributes.remove("user_id")
        return data

    def save(self, path: str) -> None:
        """Save user attributes from a local CSV file"""
        self.df.to_csv(path, index=False)

    def filter(self, user_ids: List[str]) -> UserAttributes:
        if self.is_empty:
            return self
        self.df = self.df[self.df["user_id"].isin(user_ids)]
        self.data = self._df_to_dict(self.df)
        return self

    def get_user_attributes(self, user_id: str) -> np.ndarray:
        """Returns a 1-dimensional array containing attributes for this user"""
        try:
            return self.data[user_id]
        except KeyError:
            self.unknown_users.append(user_id)
            return self.data["mode_values"]

    def construct_matrix(self, user_ids: List[str]) -> np.ndarray:
        """doc"""
        matrix = np.zeros((len(user_ids), self.nr_attributes))
        for idx, user_id in enumerate(user_ids):
            matrix[idx, :] = self.get_user_attributes(user_id)
        return matrix

    @staticmethod
    def _df_to_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:

        mode_values: np.ndarray = df.mode().values[0, 1:]  # type: ignore
        records = {"mode_values": mode_values}

        for r in df.to_dict("records"):
            t = list(r.values())
            records[t[0]] = np.array(t[1:])

        return records  # type: ignore

    @property
    def nr_attributes(self) -> int:
        return self.data["mode_values"].shape[0]
