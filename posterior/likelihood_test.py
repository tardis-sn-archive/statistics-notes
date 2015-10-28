from likelihood import *
from numpy import testing
import pytest

def test_log_pochhammer():
    args = ((13, 11), (11, 13), (1, 2))
    res  = (31.6195, 36.5023, 0.693147)
    for a, r in zip(args, res):
        testing.assert_almost_equal(log_pochhammer(*a), r, decimal=3)

def test_log_poisson_posterior_predictive():
    args = ((13, 11), (11, 13), (1, 2))
    res  = (-2.63578, -2.554, -1.50972)
    for a, r in zip(args, res):
        testing.assert_almost_equal(log_poisson_posterior_predictive(*a), r, decimal=3)

def test_binary_search():
    data = (1, 3, 6, 7, 9, 9, 9, 9, 10)
    keys = (0.1, 1, 2.2, 7, 9, 9.2, 11)
    imin = (0,   0, 1,   3, 4, 8)
    imax =      (0, 0,   3, 7, 7,   8)

    for k, i in zip(keys[:-1], imin):
        assert binary_search_min(data, k) == i

    # key larger than all elements in ``data``
    with pytest.raises(KeyError):
        binary_search_min(data, keys[-1])

    for k, i in zip(keys[1:], imax):
        assert binary_search_max(data, k) == i

    # key larger than all elements in ``data``
    with pytest.raises(KeyError):
        binary_search_max(data, keys[0])



