import numpy as np
from levin.common.utils import Decompress, Multidim_Size, arange, TSVD_Solve
from numpy.testing import assert_allclose


def test_decompress():
    p = np.array([[1, 2, 0], [3, 4, 5], [4, 5, 6]])
    s = [2, 1, 3]
    result = Decompress(p, s)
    expected_result = np.array([1, 2, 3, 4, 5, 6])
    assert_allclose(result, expected_result)


def test_decompress_list():
    p = [[1, 2, 0], [3, 4, 5], [4, 5, 6]]
    s = [2, 1, 3]
    result = Decompress(p, s)
    expected_result = np.array([1, 2, 3, 4, 5, 6])
    assert_allclose(result, expected_result)


def test_multidim_size():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    result = Multidim_Size(a)
    expected_result = 6
    assert result == expected_result


def test_multidim_size_single():
    a = 1
    result = Multidim_Size(a)
    expected_result = 1
    assert result == expected_result


def test_multidim_size_list():
    a = [1, [3, 4], [5, 6]]
    result = Multidim_Size(a)
    expected_result = 5
    assert result == expected_result


def test_arange():
    result = arange(1, 1)
    expected_result = np.array([1])
    assert_allclose(result, expected_result)

def test_TSVD_solve():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([1, 2, 3])
    result = TSVD_Solve(A, b, 1e-5)
    expected_result = np.array([ 0.  , -0.25,  0.25])
    assert_allclose(result, expected_result, rtol=1e-5)