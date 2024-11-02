import numpy as np
import pytest
from numpy.testing import assert_allclose

from levin.common.utils import Cheb_nodes
from levin.hermite import levin_hermite_integral

TEST_CASES = [{
    'f': lambda x: 1 / (1 + 25 * x**2) - 1 / 26,
    'g': lambda x: x,
    'tau': Cheb_nodes(-1, 1, 5),
    's': np.array([2, 1, 1, 1, 2]),
    'omega': np.arange(10, 110, 10),
    'result': [-1.51527781e-02+8.67361738e-19j,  1.51353966e-03-2.16840434e-19j,
               -4.12395701e-04+5.42101086e-20j,  1.50386167e-04+4.33680869e-19j,
               -5.19387736e-05+2.71050543e-20j,  6.73896446e-06-2.03287907e-20j,
               1.44939623e-05+6.77626358e-21j, -2.24306306e-05-3.38813179e-21j,
               2.22843724e-05-4.23516474e-21j, -1.74189376e-05+3.38813179e-21j]
}]


@pytest.mark.parametrize('case', TEST_CASES)
def test_levin_hermite_integral(case):
    result = levin_hermite_integral(
        case['f'], case['g'], case['tau'], case['s'], case['omega'])
    assert_allclose(result, case['result'], rtol=1e-5)
