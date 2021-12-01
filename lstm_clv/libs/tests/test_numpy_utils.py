import numpy as np
from libs.numpy_utils import (
    NoTruesException,
    first_true,
    items_where,
    last_true,
    values_have_improved,
)


def test_items_where():

    values = np.array([0, 1, 1])
    assert np.array_equal(items_where(values, "=", 1), np.array([False, True, True]))


def test_first_true():
    assert first_true(np.array([False, False, True, False])) == 3


def test_last_true():
    assert last_true(np.array([True, False, True, False])) == 3


def test_no_trues():

    try:
        first_true(np.array([False, False]))
    except NoTruesException:
        pass
    else:
        raise AssertionError


def test_values_have_improved():
    pass


def test_early_stopping():

    try:
        values_have_improved([], 0)
    except ValueError:
        pass
    else:
        raise AssertionError

    assert values_have_improved([1, 2, 3], 1, decreasing=True) is False
    assert values_have_improved([1, 2, 3], 1, decreasing=False) is True

    assert values_have_improved([2, 3, 1], 2, decreasing=True) is True
    assert values_have_improved([2, 3, 1], 2, decreasing=False) is True
