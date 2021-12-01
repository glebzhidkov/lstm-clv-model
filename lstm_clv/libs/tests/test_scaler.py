import numpy as np
from libs import Scaler


def test_2d_array_transform():

    scaler = Scaler().set_scaling_parms(
        features=["a", "b"],
        min_values=np.array([0, 1]),
        max_values=np.array([1, 2]),
    )

    array = np.array([[0, 1], [0.5, 1.5], [1, 2]])  # [min, min], [avg, avg], [max, max]

    assert np.array_equal(scaler.transform(array), np.array([[-1, -1], [0, 0], [1, 1]]))


def test_1d_array_tranform():

    scaler = Scaler().set_scaling_parms(
        features=["a", "b", "c"],
        min_values=np.array([0, 1, 2]),
        max_values=np.array([1, 2, 3]),
    )

    array = np.array([0, 1.5, 3])  # [min, avg, max]

    assert np.array_equal(scaler.transform(array), np.array([-1, 0, 1]))


def test_2d_array_inverse_transform():

    scaler = Scaler().set_scaling_parms(
        features=["a", "b"],
        min_values=np.array([0, 1]),
        max_values=np.array([1, 2]),
    )

    array = np.array([[0, 1], [0.5, 1.5], [1, 2]])
    inverse = scaler.inverse_transform(scaler.transform(array))
    assert np.array_equal(array, inverse)


def test_1d_array_inverse_tranform():

    scaler = Scaler().set_scaling_parms(
        features=["a", "b", "c"],
        min_values=np.array([0, 1, 2]),
        max_values=np.array([1, 2, 3]),
    )

    array = np.array([0, 1.5, 3])  # [min, avg, max]
    inverse = scaler.inverse_transform(scaler.transform(array))
    assert np.array_equal(array, inverse)
