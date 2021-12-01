import glob
import os
from typing import Dict, List, Tuple

import pandas as pd


def assert_events_df(df: pd.DataFrame):
    CSV_HEADER_EVENTS = ["user_id", "ts", "event", "value"]
    assert (
        list(df.columns) == CSV_HEADER_EVENTS
    ), f"unexpected columns: {list(df.columns)}; should be {CSV_HEADER_EVENTS}."


def read_user_attributes_df(path: str) -> pd.DataFrame:
    """Read dataframe containing user attributes and assert its validity"""
    df = pd.read_csv(path)
    assert df.columns[0] == "user_id", f"1st col should be user_id, not {df.columns[0]}"
    assert len(df.columns) > 1, "df should have at least 1 attribute column"
    return df


def read_events_df(path: str) -> pd.DataFrame:
    """Read dataframe containing events and assert its validity"""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])  # type: ignore
    assert_events_df(df)
    return df


def read_events_dfs(folder_path: str) -> pd.DataFrame:
    """
    Read all CSV files from the directory containing event data, construct a
    single dataframe, and assert its validity
    """
    path = os.path.join(folder_path, "*.csv")
    return pd.concat(map(read_events_df, glob.glob(path)))  # type: ignore


def drop_rows_where_count_lt(
    df: pd.DataFrame, groupby: str, min_count: int
) -> pd.DataFrame:
    # https://stackoverflow.com/questions/49735683/python-removing-rows-on-count-condition
    counts = df[groupby].value_counts()
    to_drop = counts[counts <= min_count].index  # type: ignore
    return df[~df[groupby].isin(to_drop)]  # type: ignore


def split_df(
    df: pd.DataFrame, condition: pd.Series  # [bool]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataframe based on a condition

    Returns: df where condition is True, df where confition is False
    """
    assert condition.dtype == bool, "idx should be a series of booleans"

    return df.loc[condition], df.loc[~condition]


def weeks_between(ts1: pd.Timestamp, ts2: pd.Timestamp) -> int:
    """Number of weeks between two timestamps

    Raises ValueError if timestamps are not exactly N weeks apart
    """
    weeks = abs((ts1 - ts2).days / 7)  # type: ignore
    if int(weeks) != weeks:
        raise ValueError(f"{weeks=} is not exact")
    return int(weeks)


def add_weeks(ts: pd.Timestamp, weeks: int) -> pd.Timestamp:
    """Returns a timestamp that is n weeks away from the original timestamp"""
    return ts + pd.Timedelta(weeks, unit="w")  # type: ignore


def get_unique_values(var: pd.Series) -> List[str]:
    """Shortcut, used primarily for the type hint"""
    return var.unique().tolist()


def revenue_to_profit(df: pd.DataFrame, profit_margins: Dict[str, float]):
    """Transform revenue values to profit values"""
    profit_margins["ALL"] = 1.0
    return [
        value * profit_margins.get(event, 0.0)
        for value, event in zip(df["value"], df["event"])
    ]


def aggregate_by_week(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["user_id", "ts"]).sum("value").reset_index()  # type: ignore
