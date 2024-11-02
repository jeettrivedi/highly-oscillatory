import numpy as np
from typing import List


def Decompress(p: np.ndarray | List, s):
    """
    Decompresses vector p with respect to confluency vector s
    p = vector to be "Decompressed". Can be of type
    s = confluency vector
    """
    size = np.sum(s)
    ret_arr = np.full(size, complex(0, 0))
    k = 0
    if isinstance(p, list):
        p = np.array(p)

    for i in range(0, len(p)):
        for j in range(0, s[i]):
            ret_arr[k] = p[i, j]
            k += 1

    return ret_arr


def Multidim_Size(a: np.ndarray) -> int:
    """
    Returns the length of a "folded" data vector
    Folded in the sense that each entry is a vector too
    [ [1,2],1,...]

    Args:
        a (np.ndarray): The array to be measured.

    Returns:
        size (int): The total number of elements in the array.
    """
    if isinstance(a, (int, float)):
        return 1

    size = 0
    for a_i in a:
        if isinstance(a_i, (list, np.ndarray)):
            size += len(a_i)
        else:
            size += 1
    return size


def arange(a: int, b: int) -> np.ndarray:
    """
    Modified arange function, it returns an array of size 1
    if a=b whereas the regular arange function would return
    an empty array.

    Args:
        a (int): lower limit
        b (int): upper limit

    Returns:
        x (np.ndarray): An array of integers
    """
    return np.full(1, a, dtype=int) if a == b else np.arange(a, b,dtype=int)


def range_inc(a: float, b: float) -> np.ndarray:
    """
    Modified range function that will include the right end point in the
    return array.

    It will also return an array of size 1 if a=b (as opposed
    to an empty array returned by regular arange)

    Args:
        a (float): lower limit
        b (float): upper limit

    Returns:
        x (np.ndarray): An array of integers
    """
    return np.full(1, a) if a == b else np.arange(a, b + 1)


def Cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    """
    Returns n Chebyshev nodes for the interval [a,b]

    Args:
        a (float): The lower bound of the interval.
        b (float): The upper bound of the interval.
        n (int): The number of Chebyshev nodes to generate.


    Returns:
        x (np.ndarray): An array of n Chebyshev nodes in the interval [a,b].
    """
    nodes = np.zeros(n)
    for i in range(0, n):
        nodes[n - i - 1] = ((a + b) + (b - a) *
                            np.cos(np.pi * (i) / (n - 1))) / 2
    return nodes


def TSVD_Solve(A: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """
    Computes the solution to the linear system Ax = b using Truncated Singular Value Decomposition (TSVD).

    This function performs a Singular Value Decomposition (SVD) on the matrix A, truncates small singular
    values based on the provided tolerance, and then reconstructs the solution.

    Args:
        A (np.ndarray): The coefficient matrix of the linear system.
        b (np.ndarray): The right-hand side vector of the linear system.
        tol (float): The tolerance for truncating small singular values. Singular values
                        smaller than this tolerance are set to zero.

    Returns:
        x (np.ndarray): The solution vector x that satisfies the linear system Ax = b.
    """
    U, S, V = np.linalg.svd(A)
    S_b = 1 / S
    S_b[np.abs(S_b) == np.inf] = 0
    S_b[np.abs(S_b) < tol] = 0
    S_d = np.zeros(A.shape)
    S_d[: len(S), : len(S)] = np.diag(S_b)
    return (V.T.conj().dot(S_d.T).dot(U.T.conj())).dot(b)
