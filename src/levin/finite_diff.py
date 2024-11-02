import numpy as np
from sympy import (
    N,
    Symbol,
    diff,
)

from .common.utils import TSVD_Solve


def Compact_diff_matrix(tau):
    """
    Returns the differentiation matrix for the compact scheme
    
    Args:
        tau (np.ndarray): array of nodes
    """

    N = len(tau)
    [A, B] = [np.zeros([N, N]), np.zeros([N, N])]
    h = np.diff(tau)

    for i in range(1, N - 1):
        A[i, i - 1] = 1.0 / (h[i] + h[i - 1]) ** 2
        A[i, i] = 1.0 / h[i] ** 2
        A[i, i + 1] = h[i - 1] ** 2 / (h[i] + h[i - 1]) ** 2 / h[i] ** 2
        B[i, i - 1] = (4 * h[i - 1] + 2 * h[i]) / h[i - 1] / (h[i - 1] + h[i]) ** 3
        B[i, i] = 2 * (h[i - 1] - h[i]) / h[i - 1] / h[i] ** 3
        B[i, i + 1] = (
            -(4 * h[i] + 2 * h[i - 1])
            * h[i - 1] ** 2
            / (h[i] + h[i - 1]) ** 3
            / h[i] ** 3
        )

    A[0, 0] = 1 / (h[0] + h[1]) / (h[0] + h[1] + h[2])
    A[0, 1] = 1 / h[1] / (h[1] + h[2])
    B[0, 0] = (
        (
            4 * h[0] ** 2
            + 6 * h[0] * h[1]
            + 3 * h[0] * h[2]
            + 2 * h[1] ** 2
            + 2 * h[1] * h[2]
        )
        / (h[0] + h[1]) ** 2
        / (h[0] + h[1] + h[2]) ** 2
        / h[0]
    )
    B[0, 1] = (
        1
        / h[0]
        * ((-2 * h[1] + h[0]) * h[2] + 2 * h[1] * (-h[1] + h[0]))
        / h[1] ** 2
        / (h[1] + h[2]) ** 2
    )
    B[0, 2] = -h[0] ** 2 / (h[1] + h[0]) ** 2 / h[1] ** 2 / h[2]
    B[0, 3] = h[0] ** 2 / (h[2] + h[1] + h[0]) ** 2 / (h[2] + h[1]) ** 2 / h[2]

    A[-1, -2] = 1 / h[-2] / (h[-2] + h[-3])
    A[-1, -1] = 1 / (h[-1] + h[-2]) / (h[-1] + h[-2] + h[-3])
    B[-1, -4] = (
        -h[-1] ** 2 / (h[-3] + h[-2] + h[-1]) ** 2 / (h[-3] + h[-2]) ** 2 / h[-3]
    )
    B[-1, -3] = h[-1] ** 2 / (h[-2] + h[-1]) ** 2 / h[-2] ** 2 / h[-3]
    B[-1, -2] = (
        1
        / h[-1]
        * ((2 * h[-2] - h[-1]) * h[-3] + 2 * h[-2] * (h[-2] - h[-1]))
        / h[-2] ** 2
        / (h[-3] + h[-2]) ** 2
    )
    B[-1, -1] = -(
        (4 * h[-1] ** 2 + (6 * h[-2] + 3 * h[-3]) * h[-1] + 2 * h[-2] * (h[-3] + h[-2]))
        / h[-1]
        / (h[-2] + h[-1]) ** 2
        / (h[-3] + h[-2] + h[-1]) ** 2
    )

    return [-A, B]


def Levin_Int_Comp_diff(F, G, tau, omega_range, solver='TSVD'):
    x = Symbol("x")

    def f(x):
        return F(x)

    def g(x):
        return G(x)

    Levin = np.full(len(omega_range), complex(0, 0))
    Number_Of_Nodes = len(tau)

    p = np.full([Number_Of_Nodes], np.complex(0, 0))
    g_matrix = np.diag(np.ones(Number_Of_Nodes))

    for i in range(0, Number_Of_Nodes):
        g_matrix[i, i] = diff(g(x), x).subs(x, tau[i])
        p[i] = f(tau[i])

    [A, B] = Compact_diff_matrix(tau)

    i = 0
    for omega in omega_range:
        if solver == 'TSVD':
            P = TSVD_Solve(
                B + np.complex(0, omega) * np.dot(A, g_matrix), np.dot(A, p), 1e-13
            )
        else:
            P = np.linalg.solve(B+np.complex(0,omega)*np.dot(A,g_matrix),np.dot(A,p))

        Levin[i] = P[-1] * np.exp(np.complex(0, omega) * float(N(g(tau[-1])))) - P[
            0
        ] * np.exp(np.complex(0, omega) * float(N(g(tau[0]))))
        i += 1
    return Levin
