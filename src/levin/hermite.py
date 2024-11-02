import numpy as np
import sympy as sp
from scipy.special import comb, factorial
from sympy import diff, simplify

from .common.genbarywts import Genbarywts
from .common.utils import Decompress, range_inc


def levin_hermite_integral(F: callable, G: callable, tau: np.ndarray, s: np.ndarray, omega_range: np.ndarray) -> np.ndarray:
    """
    Levin-Hermite integration method

    Integrates the function $F(x)*\exp(i*omega*G(x))$ over the interval [a,b] using the Levin-Hermite method

    Args:
        F (callable): Function to be integrated
        G (callable): Function to be integrated
        tau (np.ndarray): Nodes
        s (np.ndarray): Derivative orders
        omega_range (np.ndarray): Omega range

    Returns:
        np.ndarray: Array of complex numbers representing the integral values at each tau point for each omega value
    """
    x = sp.Symbol("x")

    def f(x):
        return F(x)

    def g(x):
        return G(x)

    g_num = sp.lambdify(x, g(x))
    Levin = np.full(len(omega_range), complex(0, 0))

    Number_Of_Nodes = len(tau)
    d = sum(s)

    # Calculating the function and derivative values of f(x) at the nodes
    #
    # p = [[f(t_0),f'(t_0),...,f^(s_0-1)(t_0)],
    #       ...,[f(t_n),f'(t_n),...,f^(s_n-1)(t_n)]]
    p = np.full([Number_Of_Nodes, max(s)], complex(0, 0))
    p_rhs = np.full([Number_Of_Nodes, max(s)], complex(0, 0))
    for i in range(0, len(tau)):
        for j in range(0, s[i]):
            p[i, j] = simplify(sp.diff(f(x), x, j).subs(
                x, tau[i])) / factorial(j)
            p_rhs[i, j] = simplify(sp.diff(f(x), x, j).subs(x, tau[i]))

    p = Decompress(p, s)
    p_rhs = Decompress(p_rhs, s)

    [D, w] = Genbarywts(tau.tolist(), s.tolist())

    tau_decompressed = np.zeros(d)
    k = 0
    for i in range(0, len(tau)):
        for j in range(0, s[i]):
            tau_decompressed[k] = tau[i]
            k += 1
    u = g_num(tau_decompressed)

    Sum_s = np.cumsum([0] + s.tolist())[0:-1]

    i = 0
    for omega in omega_range:
        # Create a Coefficient matrix for the linear system
        # f = (D + iwg')*P
        # The non-trivial work here is calculating the product term g'P
        Coeff_matrix = np.full([d, d], complex(0, 0))
        k = 0
        for j in Sum_s:
            for s_i in range(0, s[k]):
                Coeff_matrix[j + s_i, :] = D[j + s_i, :] * factorial(s_i)
                for l in range_inc(1, s_i):
                    Coeff_matrix[j + s_i, :] = Coeff_matrix[j + s_i, :] + complex(
                        0, omega
                    ) * comb(s_i, l) * factorial(l - 1) * D[j + l - 1, :] * diff(
                        g(x), x, s_i - l + 1
                    ).subs(
                        x, tau[k]
                    )

                Coeff_matrix[j + s_i, :] = Coeff_matrix[j + s_i, :] + complex(
                    0, omega
                ) * np.eye(d)[j, :] * diff(g(x), x, s_i + 1).subs(x, tau[k])
            k += 1

        # Solving f = (D + iwg')*P
        P = np.linalg.solve(Coeff_matrix, p_rhs)

        Levin[i] = (P[-s[-1]]) * np.exp(complex(0, omega) * u[-1]) - P[0] * np.exp(
            complex(0, omega) * u[0]
        )
        i += 1

    return Levin
