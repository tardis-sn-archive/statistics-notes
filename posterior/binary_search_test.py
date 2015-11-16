from binary_search import *

import pytest

def test_binary_search():
    data = (1, 3, 6, 7, 9, 9, 9, 9, 10)
    keys = (0.1, 1, 2.2, 7, 9, 9.2, 11, 500)
    imin = (0,   0, 1,   3, 4, 8,    9,   9)
    imax =      (0, 0,   3, 7, 7,   8)

    for k, i in zip(keys, imin):
        assert binary_search_min(data, k) == i

    # # key larger than all elements in ``data``
    # with pytest.raises(ValueError):
    #     print binary_search_min(data, keys[-1])

    for k, i in zip(keys[1:], imax):
        assert binary_search_max(data, k) == i

    # key larger than all elements in ``data``
    with pytest.raises(KeyError):
        binary_search_max(data, keys[0])



