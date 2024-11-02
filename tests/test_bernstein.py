import numpy as np
import pytest
from numpy.testing import assert_allclose

from levin.bernstein import _Coeff_to_Poly, levin_bernstein_integral
from levin.common.utils import Cheb_nodes

TEST_CASES = [{
    'f': lambda x: 1 / (1 + 25 * x**2) - 1 / 26,
    'g': lambda x: x,
    'tau': Cheb_nodes(-1, 1, 5),
    'omega': np.arange(10, 110, 10),
    'result': [-2.28698713e+01-8.48210391e-14j, 9.71233096e+00+7.77156117e-15j,
               -4.71006185e+00+5.99520433e-15j, 1.63976191e+00+1.11022302e-15j,
               -3.82357407e-03-3.55271368e-15j, -3.08895765e-01+1.94289029e-15j,
               7.78868341e-02-3.71230824e-16j, -3.24059607e-02-3.79687601e-16j,
               1.55686509e-02+1.50920942e-16j, -5.14156782e-03-6.67868538e-17j]
}, {
    'f': lambda x: np.sin(x),
    'g': lambda x: x,
    'tau': Cheb_nodes(-1, 1, 5),
    'omega': np.arange(10, 110, 10),
    'result': [-2.08166817e-17+0.13681063j, 1.80411242e-16-0.03198582j,
               -2.08166817e-17-0.00980913j, 6.38378239e-16+0.02853619j,
               7.80625564e-18-0.03256554j, -1.16226473e-16+0.02660152j,
               1.07552856e-16-0.01504551j, -4.51028104e-17+0.00215663j,
               6.41847686e-17+0.00848575j, -2.16840434e-17-0.01454972j]
},
    {
    'f': lambda x: x * (1 - x) / np.sqrt(1 + x**2),
    'g': lambda x: x,
    'tau': Cheb_nodes(-1, 1, 5),
    'omega': np.arange(10, 110, 10),
    'result': [0.09362942+0.11701429j, -0.0660457-0.02768005j, 0.04586106-0.00748668j,
               -0.0253271+0.02326644j, 0.00663342-0.02677906j, 0.00760703+0.02196531j,
               -0.01572337-0.01248279j, 0.01743016+0.00186221j, -0.01380738+0.0069299j,
               0.0069306-0.011958j]
}, {
    'f': lambda x: x * (1 - x),
    'g': lambda x: x,
    'tau': Cheb_nodes(-1, 1, 5),
    'omega': np.arange(10, 110, 10),
    'result': [0.140191 + 0.15693388j, -0.09491887-0.03624348j, 0.06503684-0.01247906j,
               -0.03554174+0.03427829j, 0.00894265-0.03880854j, 0.01121295+0.03157776j,
               -0.02261913-0.01777896j, 0.02490844+0.00244909j, -0.01964042+0.01017793j,
               0.00978036-0.01734765j]
}, {
    'f': lambda x: x * (1 - x),
    'g': lambda x: -(x - 0.1)**2,
    'tau': Cheb_nodes(-1, 1, 5),
    'omega': np.arange(10, 110, 10),
    'result': [0.03955488-0.07974943j, 0.03678411-0.02779984j, 0.02969675-0.00569883j,
               0.02166772+0.00656395j, 0.01331977+0.0125625j, 0.00521582+0.01418957j,
               -0.00160388+0.01287578j, -0.00630552+0.00952361j, -0.00874035+0.00500936j,
               -0.00908914+0.00042966j]
}
]


@pytest.mark.parametrize('case', TEST_CASES)
def test_levin_bernstein(case):
    f, g, tau, omega, expected_result = case['f'], case['g'], case['tau'], case['omega'], case['result']
    result = levin_bernstein_integral(f, g, tau, omega)
    assert_allclose(result, expected_result, rtol=1e-7, atol=1e-7)


def test_coeff_to_poly():
    coef = np.array([1, 2])
    result = _Coeff_to_Poly(coef,np.array([0,1]))
    expected_result = np.array([1, 2])
    assert_allclose(result, expected_result)