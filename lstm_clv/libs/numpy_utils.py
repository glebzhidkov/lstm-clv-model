import operator
from datetime import datetime
from typing import Any, List, Literal, Union

import numpy as np
import pandas as pd


class NoTruesException(Exception):
    pass


_OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "=": operator.eq,
    "!=": operator.ne,
}


def _get_operator(op: Literal[">", "<", ">=", "<=", "=", "!="]):
    """Returns a proxy function for Python comparison operator"""
    return _OPERATORS[op]


def items_where(
    arr: np.ndarray, operator: Literal[">", "<", ">=", "<=", "=", "!="], value: Any
) -> np.ndarray:
    """Returns an array of Trues/Falses based on the specified criteria

    ```
    values = np.array([0, 1, 1])
    items_where(values, "=", 1)  # np.array([False, True, True])
    ```"""
    return _get_operator(operator)(arr, value)


def first_true(arr: np.ndarray, axis=None) -> int:
    """Index of the first array entry that is True. If none, raises `NoTruesException`"""
    assert isinstance(arr, np.ndarray), f"invalid type: {type(arr)}"
    if arr.sum() == 0:
        raise NoTruesException(arr)
    return arr.argmax(axis=axis)  # type: ignore


def last_true(arr: np.ndarray) -> int:
    """Index of the last array entry that is True. If none, raises `NoTruesException`"""
    assert isinstance(arr, np.ndarray), f"invalid type: {type(arr)}"
    if arr.sum() == 0:
        raise NoTruesException(arr)
    return len(arr) - np.flip(arr).argmax() - 1


def values_have_improved(
    values: List[Union[float, int]], window: int, decreasing: bool = True
):
    compare = _get_operator(">") if decreasing else _get_operator("<")

    if window == 0:
        raise ValueError("window should be >0")
    if len(values) < (window + 1):
        return True

    first_value = values[-window - 1]
    values_ = values[-window:]
    improvements = [compare(first_value, v) for v in values_]
    return any(improvements)


def numpy_time_to_datetime(ts: np.datetime64) -> datetime:
    return pd.Timestamp(ts).to_pydatetime()
