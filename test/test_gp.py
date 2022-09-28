import copy
import jax.numpy as np
import jax.scipy as sp
import jax.random as jr
from numpy.testing import assert_allclose
import pytest


from jaxrk.rkhs import FiniteVec, inner
from jaxrk.kern import GenGaussKernel
import jaxrk.models.gp as gp

rng = jr.PRNGKey(0)


kernel_setups = [GenGaussKernel.make_gauss(2.0)]


@pytest.mark.parametrize("D_X", [1, 5])
@pytest.mark.parametrize("D_Y", [1, 5])
@pytest.mark.parametrize("kernel", kernel_setups)
@pytest.mark.parametrize("N", [10])
def test_gp_init(D_X, D_Y, kernel, N):
    k1, k2, k3 = jr.split(rng, 3)
    y = jr.normal(k1, (N, D_Y))
    v_X, v_Xp = [FiniteVec(kernel, jr.normal(k, (N, D_X)) * 20, []) for k in (k2, k3)]
    chol, y_rescaled, prec_y, ymean, ystd, noise = gp.gp_init(
        v_X.inner() + np.eye(N) * 2, y, None, True
    )
    y_p_m, y_p_cov = gp.gp_predictive(
        v_X.inner(v_Xp), v_Xp.inner(), chol, prec_y, ymean, ystd
    )
    y_p_m, y_p_cov, lhood = gp.gp_predictive(
        v_X.inner(v_Xp), v_Xp.inner(), chol, prec_y, ymean, ystd, np.zeros((N, D_Y))
    )
    # print(lhood)


@pytest.mark.parametrize("D_X", [1, 5])
@pytest.mark.parametrize("D_Y", [1, 5])
@pytest.mark.parametrize("kernel", kernel_setups)
@pytest.mark.parametrize("N", [10])
def test_gp(D_X, D_Y, kernel, N):
    k1, k2, k3 = jr.split(rng, 3)
    y = jr.normal(k1, (N, D_Y))
    v_X, v_Xp = [FiniteVec(kernel, jr.normal(k, (N, D_X)) * 20, []) for k in (k2, k3)]
    g = gp.GP(v_X, y, 2.0, True)
    G_X = v_X.inner()
    G_Xp = v_Xp.inner()
    G_X_Xp = v_X.inner(v_Xp)

    Chol_X = sp.linalg.cholesky(G_X, lower=True)

    v = gp.gp_predictive_var(G_X_Xp, G_Xp, chol_gram_train=Chol_X)
    c = gp.gp_predictive_cov(G_X_Xp, G_Xp, chol_gram_train=Chol_X)
    assert np.allclose(v, np.diagonal(c).T)

    # print(g.marginal_loglhood())
