
import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest


from jaxrk.rkhs import FiniteVec, inner
from jaxrk.kern import GenGaussKernel
import jaxrk.models.gp as gp

rng = np.random.RandomState(1)


kernel_setups = [
    GenGaussKernel.make_gauss(2.)
]


@pytest.mark.parametrize('D_X', [1, 5])
@pytest.mark.parametrize('D_Y', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_gp_init(D_X, D_Y, kernel, N):
    y = rng.randn(N, D_Y)
    v_X = FiniteVec(kernel, rng.randn(N, D_X) * 20, [])
    v_X_p = FiniteVec(kernel, rng.randn(N, D_X) * 20, [])
    chol, y_rescaled, prec_y, ymean, ystd, noise = gp.gp_init(
        v_X.inner() + np.eye(N) * 2, y, None, True)
    y_p_m, y_p_cov = gp.gp_predictive(
        v_X.inner(v_X_p), v_X_p.inner(), chol, prec_y, ymean, ystd)
    y_p_m, y_p_cov, lhood = gp.gp_predictive(v_X.inner(
        v_X_p), v_X_p.inner(), chol, prec_y, ymean, ystd, np.zeros((N, D_Y)))
    # print(lhood)


@pytest.mark.parametrize('D_X', [1, 5])
@pytest.mark.parametrize('D_Y', [1, 5])
@pytest.mark.parametrize('kernel', kernel_setups)
@pytest.mark.parametrize('N', [10])
def test_gp(D_X, D_Y, kernel, N):
    y = rng.randn(N, D_Y)
    v_X = FiniteVec(kernel, rng.randn(N, D_X) * 20, [])
    v_X_p = FiniteVec(kernel, rng.randn(N, D_X) * 20, [])
    g = gp.GP(v_X, y, 2., True)

    # print(g.marginal_loglhood())
