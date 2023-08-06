import numpy as np
from scipy.special import comb
from scipy.integrate import quad
from sympy import (
    diff,
    limit,
    evalf,
    sqrt,
    sin,
    cos,
    root,
    log,
    exp,
    simplify,
    lambdify,
    Symbol,
)
from .helper_methods import Cheb_nodes

# from Hermite_Levin import Internal_Quad


def B(x, n, k, a=0, b=1):
    """Input:
    x = point at which Bernstein polynomial needs to be evaulated
    n = degree of the polynomial
    k = k^th bernstein polynomial of degree n
    Output:
    scalar or array of real/complex numbers
    """
    return comb(n, k, exact=True) * (x - a) ** k * (b - x) ** (n - k) / (b - a) ** n


def Coeff_to_Poly(coeff_list, x, a=0, b=1):
    """
    "Synthesis" Operator: Given a sequence of coefficients,
    this returns the value of the polynomial at a point x
    (or for an array of points xran)

        coeff_list = array of coefficients (real or complex numbers)
        x          = point at which polynomial needs to be evaluated
        a          = lower integration limit
        b          = upper integration limit
    """
    degree = len(coeff_list) - 1
    result = 0
    for j in range(0, degree + 1):
        result += coeff_list[j] * B(x, degree, j, a, b)
    return result


def Bern_Interpolate(tau, ftau, a=0, b=1):
    """
    Interpolates the set of points {(tau,f(tau))} in the Bernstein Basis
    (Using the obvious approach and not a fast algorithm for simplicity)
    """
    n = len(tau) - 1
    [a, b] = [tau[0], tau[-1]]
    c = np.full(n + 1, 0)
    A = np.zeros([n + 1, n + 1])
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            A[i, j] = B(tau[i], n, j, a, b)
    return np.linalg.solve(A, ftau)


def Bernstein_diff_matrix(n, a=0, b=1):
    """
    n = order of the family of Bernstein polynomials that
         we want the diff matrix for
    (n^th order family has n+1 elements)
    """
    D = np.zeros([n + 1, n + 1])
    D[0, 0] = -n
    D[0, 1] = n
    D[-1, -1] = n
    D[-1, -2] = -n
    for i in range(1, n):
        D[i, i] = 2 * i - n
        D[i, i - 1] = -i
        D[i, i + 1] = n - i
    return D / (b - a)


def Mul_Operator_Matrix(f):
    """
    This method creates the multiplication matrix for a function f
    """
    n = len(f) - 1
    w = np.zeros([2 * n + 1, n + 1])
    for i in range(0, 2 * n + 1):
        for j in range(max(0, i - n), min(n, i) + 1):
            w[i, j] = comb(n, j) * comb(n, i - j) / comb(2 * n, i) * f[i - j]
    return w


def Levin_Bern_int(F, G, tau, omega_range):
    """
    Levin-Bernstein integration method
    """
    n = len(tau) - 1
    [a, b] = [tau[0], tau[-1]]

    coef_f = Bern_Interpolate(
        Cheb_nodes(a, b, 2 * n + 1), F(Cheb_nodes(a, b, 2 * n + 1)), a, b
    )
    coef_one = np.ones(n + 1)
    coef_g = Bern_Interpolate(tau, G(tau), a, b)

    D = Bernstein_diff_matrix(n, a, b)
    coef_dg = np.dot(D, coef_g)

    Levin = np.full(len(omega_range), np.complex(0, 0))

    k = 0
    for w in omega_range:
        A = np.dot(Mul_Operator_Matrix(coef_one), D) + np.complex(
            0, w
        ) * Mul_Operator_Matrix(coef_dg)
        Augmented_A = np.column_stack((A, coef_f))
        V = np.linalg.svd(Augmented_A)[2].conj()

        if V[-1, -1] != 0:
            sol = -1 / V[-1, -1] * V[-1, :-1]

        Levin[k] = sol[-1] * np.exp(complex(0, w) * G(tau[-1])) - sol[0] * np.exp(
            complex(0, w) * G(tau[0])
        )
        k += 1
    return Levin
