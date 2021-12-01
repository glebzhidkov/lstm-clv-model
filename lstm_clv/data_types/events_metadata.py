from typing import List, cast

import pandas as pd
from lstm_clv.libs.list_indexer import ListIndexer


class EventsMetadata:
    """Some important stats based for the dataframe with transactions"""

    def __init__(self, df: pd.DataFrame):
        self.first_week = cast(pd.Timestamp, df["ts"].min())
        self.last_week = cast(pd.Timestamp, df["ts"].max())
        self.range_weeks = pd.date_range(self.first_week, self.last_week, freq="W-MON")
        self.nr_weeks = len(self.range_weeks)
        self.user_ids: List[str] = df["user_id"].unique().tolist()
        self.nr_users = len(self.user_ids)
        self.events: List[str] = df["event"].unique().tolist()
        self.events.sort()  # alphabetically, inplace
        self.nr_events = len(self.events)
        self.nr_rows = df.shape[0]

        self.user_id_indices = ListIndexer(self.user_ids)
        self.events_indices = ListIndexer(self.events)

    def __repr__(self) -> str:
        return (
            f"Events data with {self.nr_rows} records for {self.nr_users} users, "
            f"{self.nr_weeks} weeks, {self.nr_events} events"
        )
