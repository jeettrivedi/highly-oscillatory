import numpy as np


def Decompress(p, s):
    """
    Decompresses vector p with respect to confluency vector s
    p = vector to be "Decompressed". Can be of type
    s = confluency vector
    """
    size = np.sum(s)
    ret_arr = np.full(size, np.complex(0, 0))
    k = 0
    if isinstance(p, list):
        p = np.array(p)

    for i in range(0, len(p)):
        for j in range(0, s[i]):
            ret_arr[k] = p[i, j]
            k += 1

    return ret_arr


def Multidim_Size(a):
    """
    Returns the length of a "folded" data vector
    Folded in the sense that each entry is a vector too
    [ [1,2],1,...]

    a = Data vector
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


def arange(a, b):
    """
    Modified arange function, it returns an array of size 1
    if a=b whereas the regular arange function would return
    an empty array.
    """
    if a == b:
        return np.full(1, a).astype(int)
    return np.arange(a, b).astype(int)


def range_inc(a, b):
    """Modified range function that will include the right end point in the
    return array.

    It will also return an array of size 1 if a=b (as opposed
    to an empty array returned by regular arange)
    a = lower limit
    b = upper limit
    """
    if a == b:
        return np.full(1, a)
    return np.arange(a, b + 1)


def Cheb_nodes(a, b, n):
    """Returns n Chebyshev nodes for the interval [a,b]"""
    nodes = np.zeros(n)
    for i in range(0, n):
        nodes[n - i - 1] = ((a + b) + (b - a) * np.cos(np.pi * (i) / (n - 1))) / 2
    return nodes


def TSVD_Solve(A, b, tol):
    """Solve a linear system using Truncated Singular
    Value Decomposition
    """
    [U, S, V] = np.linalg.svd(A)
    S_b = 1 / S
    S_b[np.abs(S_b) == np.inf] = 0
    S_b[np.abs(S_b) < tol] = 0
    S_d = np.zeros(A.shape)
    S_d[: len(S), : len(S)] = np.diag(S_b)
    return (V.T.conj().dot(S_d.T).dot(U.T.conj())).dot(b)
