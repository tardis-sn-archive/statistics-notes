'''Collect various binary search variants'''

import bisect

def binary_search_min(data, key, **kwargs):
    '''
    Find the index of the smallest element in ``data`` that is >= ``key``.
    '''
    return bisect.bisect_left(data, key, **kwargs)


def binary_search_max(data, key, **kwargs):
    '''
    Find the index of the first element in ``data`` that is > ``key``.
    '''
    i = bisect.bisect_right(data, key, **kwargs)
    if i != len(data):
        return i
    raise KeyError('Key ' + str(key) + ' exceeds all elements')

def binary_search_min_mine(data, key, low=None, high=None):
    '''
    Find the index of the smallest element in ``data`` that is >= ``key``.
    '''

    # update low and high until element found
    if low is None:
        low = 0
    if high is None:
        high = len(data) - 1

    assert low < high

    # first element is the right one
    if data[0] >= key:
        return low
    if data[-1] < key:
        raise KeyError('Key ' + str(key) + ' exceeds all elements')

    while low < (high - 1):
        mid = (low + high) // 2
        midval = data[mid]
        if midval >= key:
            high = mid
        if midval < key:
            low = mid

    return high


def binary_search_max_mine(data, key, low=None, high=None):
    '''
    Find the index of the first element in ``data`` that is > ``key``.
    '''

    # update low and high until element found
    if low is None:
        low = 0
    if high is None:
        high = len(data) - 1

    assert low < high

    # first element is the right one
    if data[-1] <= key:
        return high
    if data[0] > key:
        raise KeyError('Key ' + str(key) + ' less than all elements')

    while low < (high - 1):
        mid = (low + high) // 2
        midval = data[mid]
        if midval <= key:
            low = mid
        if midval > key:
            high = mid

    return low


def binary_search_min_max_mine(data, minkey, maxkey, low=None, high=None):
    '''
    Find the indices of the smallest element exceeding ``minkey`` and of the
    first element exceeding ``maxkey``.

    :param data:
    :param minkey:
    :param maxkey:
    :param low:
    :param high:
    :return:
    '''
    imin = binary_search_min(data, minkey, low, high)
    imax = binary_search_max(data, maxkey, imin, high)
    return (imin, imax)


