import numpy as np

from .utils import range_inc


def strainer_array(i, n):
    """
    Returns an array of size n with a 0 in the i^th position
    and 1's for all the other entries

    Args:
        i (int): The index of the 0
        n (int): The size of the array

    Returns:
        np.ndarray: An array of 1's with a 0 in the i^th
    """
    temp = np.full(n, 1)
    temp[i] = 0
    return temp


def Genbarywts(tau, s):
    """Generates Barycentric Weights given a set of nodes tau with
    confluency vector s

    Args:
        tau (np.ndarray): The nodes
        s (np.ndarray): The confluency vector

    Returns:
        np.ndarray: The differentiation matrix
        np.ndarray: The barycentric weights
    """

    n = len(tau)
    s_max = max(s)
    d = sum(s)

    delta_tau = np.zeros([n, n])
    beta = np.zeros(n)
    [u, v, w] = [np.zeros([n, s_max]), np.zeros(
        [n, s_max + 1]), np.zeros([n, s_max])]

    for i in range(0, n):
        v[i, 0] = 1
        for j in range(0, n):
            delta_tau[i, j] = tau[i] - tau[j]
    delta_tau = delta_tau + np.eye(n)
    delta_tau_recip = 1 / (delta_tau)

    for i in range(0, n):
        delta_tau_recip[:, i] = delta_tau_recip[:, i] ** s[i]

    for i in range_inc(0, n - 1):
        for m in range_inc(0, s_max - 1):
            u[i, m] = np.sum(strainer_array(i, n) * s *
                             delta_tau[:, i] ** (-m - 1))

        for m in range_inc(0, s_max - 1):
            v[i, m + 1] = np.sum(u[i, 0: m + 1] *
                                 v[i, 0: m + 1][::-1]) / (m + 1)

        beta[i] = np.prod(delta_tau_recip[i, :])

        for m in range_inc(1, s[i]):
            w[i, m - 1] = beta[i] * v[i, s[i] - m]

    # Hermitian differentiation matrix calculation below
    D = np.zeros([d, d])
    brks = np.cumsum([0] + s)
    irow = 0
    sum_range = np.arange(0, n)

    for k in range(0, n):
        # Trivial Rows
        for j in np.arange(0, s[k] - 1):
            D[irow, brks[k].astype(int) + j + 1] = j + 1
            irow += 1

        # Non-Trivial Rows
        for i in sum_range[sum_range != k]:
            for j in np.arange(0, s[i]):
                g = 0
                for mu in np.arange(j, s[i]):
                    g = g + w[i, mu] * (tau[k] - tau[i]) ** (j - 1 - mu)
                D[irow, brks[i].astype(int) + j] = g / w[k, s[k] - 1]

        D[irow, brks[k] + 1: brks[k] + s[k]] = - \
            w[k, 0: s[k] - 1] / w[k, s[k] - 1]
        D[irow, brks[k]] = -np.sum(D[irow, brks[0: len(brks) - 1]])
        D[irow, :] = D[irow, :] * s[k]
        irow += 1

    return [D, w]
