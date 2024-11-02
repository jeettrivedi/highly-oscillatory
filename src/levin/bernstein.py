import numpy as np
from scipy.special import comb

from .common.utils import Cheb_nodes


def B(x: np.ndarray, n: int, k: int, a: float = 0, b: float = 1):
    """
    Evaluate the k-th Bernstein polynomial of degree n at point x.

    Parameters:
        x (float): Point at which the Bernstein polynomial is evaluated.
        n (int): Degree of the Bernstein polynomial.
        k (int): Index of the Bernstein polynomial.
        a (float, optional): Lower bound of the interval. Default is 0.
        b (float, optional): Upper bound of the interval. Default is 1.

    Returns:
        float or array-like: Value(s) of the k-th Bernstein polynomial of degree n at point x.
    """
    return comb(n, k, exact=True) * (x - a) ** k * (b - x) ** (n - k) / (b - a) ** n


def _Coeff_to_Poly(coeff_list, x, a=0, b=1):
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


def bernstein_basis_interpolate(tau: np.ndarray, ftau: np.ndarray, a: float = 0, b: float = 1):
    """
    Interpolates the set of points {(tau,f(tau))} in the Bernstein Basis
    (Using the obvious approach and not a fast algorithm for simplicity)

    Args:
        tau (np.ndarray): Nodes for interpolation
        ftau (np.ndarray): Function values at the nodes
        a (float, optional): Lower limit of the interval. Defaults to 0.
        b (float, optional): Upper limit of the interval. Defaults to 1.

    Returns:
        np.ndarray: Coefficients of the interpolating polynomial
    """
    n = len(tau) - 1
    [a, b] = [tau[0], tau[-1]]
    A = np.zeros([n + 1, n + 1])
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            A[i, j] = B(tau[i], n, j, a, b)
    return np.linalg.solve(A, ftau)


def Compute_bernstein_diff_matrix(n, a: float = 0, b: float = 1) -> np.ndarray:
    """
    Computes the differentiation matrix for the Bernstein basis of order n over [a,b]

    Args:
        n (int): order of the family of Bernstein polynomials that we want the diff matrix
                 for (n^th order family has n+1 elements)
        a (float, optional): Lower limit of the interval. Defaults to 0.
        b (float, optional): Upper limit of the interval. Defaults to 1.

    Returns:
        np.ndarray: Differentiation matrix of shape (n+1, n+1)
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

    Args:
        f (np.ndarray): Coefficients of the function

    Returns:
        np.ndarray: Matrix representing the multiplication operator
    """
    n = len(f) - 1
    w = np.zeros([2 * n + 1, n + 1])
    for i in range(0, 2 * n + 1):
        for j in range(max(0, i - n), min(n, i) + 1):
            w[i, j] = comb(n, j) * comb(n, i - j) / comb(2 * n, i) * f[i - j]
    return w


def levin_bernstein_integral(F: callable, G: callable, tau: np.ndarray, omega_range: np.ndarray):
    """Levin-Bernstein integration method

    Integrates the function $F(x)*\exp(i*omega*G(x))$ over the interval [a,b] using the
    nodes tau and the function G(x) to generate the Levin series

    Args:
        F (function): Function to be integrated
        G (function): Function in the exponential term
        tau (array): Nodes for interpolation
        omega_range (array): Range of omega values


    Returns:
        array: Array of complex numbers representing the integral values at each tau point for each omega value
    """
    n = len(tau) - 1
    [a, b] = [tau[0], tau[-1]]

    coef_f = bernstein_basis_interpolate(
        Cheb_nodes(a, b, 2 * n + 1), F(Cheb_nodes(a, b, 2 * n + 1)), a, b
    )
    coef_one = np.ones(n + 1)
    coef_g = bernstein_basis_interpolate(tau, G(tau), a, b)

    D = Compute_bernstein_diff_matrix(n, a, b)
    coef_dg = np.dot(D, coef_g)

    Levin = np.full(len(omega_range), complex(0, 0))

    k = 0
    for w in omega_range:
        A = np.dot(Mul_Operator_Matrix(coef_one), D) + complex(
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
