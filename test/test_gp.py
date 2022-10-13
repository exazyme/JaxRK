import copy
import jax.numpy as np
import jax.scipy as sp
import jax.random as jr
from numpy.testing import assert_allclose
import pytest
import jaxopt


from jaxrk.rkhs import FiniteVec, inner
from jaxrk.kern import GenGaussKernel
import jaxrk.models.gp as gp
import scipy.stats as stats

from jaxrk.flax.models import FlaxGP, OneToOneEncoder
from jaxrk.flax.factories import GenGaussKernelFactory

rng = jr.PRNGKey(0)


kernel_setups = []


@pytest.mark.parametrize("D_X", [1, 5])
@pytest.mark.parametrize("D_Y", [1, 5])
@pytest.mark.parametrize("kernel", [GenGaussKernel.make_gauss(2.0)])
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


@pytest.mark.parametrize("D_X", [1, 5])
@pytest.mark.parametrize("D_Y", [1, 5])
@pytest.mark.parametrize("kernel", [GenGaussKernel.make_gauss(2.0)])
@pytest.mark.parametrize("N", [10])
def test_gp(D_X, D_Y, kernel, N):
    k1, k2, k3, k4 = jr.split(rng, 4)
    y1 = jr.normal(k1, (N, D_Y))
    y2 = jr.normal(k4, (N, D_Y))
    v_X, v_Xp = [FiniteVec(kernel, jr.normal(k, (N, D_X)) * 20, []) for k in (k2, k3)]
    g = gp.GP(v_X, y1, 2.0, True)
    G_X = v_X.inner()
    Chol_X = np.linalg.cholesky(G_X)
    G_Xp = v_Xp.inner()
    G_X_Xp = v_X.inner(v_Xp)

    Chol_X = sp.linalg.cholesky(G_X, lower=True)

    v = gp.gp_predictive_var(G_X_Xp, G_Xp, chol_gram_train=Chol_X)
    c = gp.gp_predictive_cov(G_X_Xp, G_Xp, chol_gram_train=Chol_X)

    assert np.allclose(v, np.diagonal(c).T)

    assert np.allclose(
        stats.multivariate_normal(np.zeros(N), G_X).logpdf(y1.T),
        gp.gp_loglhood_mean0_univ(y1, Chol_X),
        atol=1e-4,
    )
    prec_y1 = sp.linalg.cho_solve((Chol_X, True), y1)
    lh1 = stats.multivariate_normal(np.zeros(N), G_X).logpdf(y1.T)
    assert np.allclose(lh1, gp.gp_loglhood_mean0_univ(y1, Chol_X, prec_y1), atol=1e-4)
    prec_y2 = sp.linalg.cho_solve((Chol_X, True), y2)
    lh2 = stats.multivariate_normal(np.zeros(N), G_X).logpdf(y2.T)
    assert np.allclose(lh2, gp.gp_loglhood_mean0_univ(y2, Chol_X, prec_y2), atol=1e-4)

    assert np.allclose(lh1.sum(), gp.gp_loglhood_mean0(y1, Chol_X, prec_y1), atol=1e-4)
    assert np.allclose(
        lh1.sum(), gp.gp_loglhood_mean0_old(y1, Chol_X, prec_y1), atol=1e-4
    )


def test_FlaxGp():
    x = np.linspace(0, 40, 400).reshape((-1, 1))
    rng_key = jr.PRNGKey(0)
    y = np.sin(x) + jr.normal(rng_key, (len(x), 1)) * 0.2
    gauss_noise = lambda key: jr.normal(key, (1,))
    enc = OneToOneEncoder(
        GenGaussKernelFactory.from_constrained(
            1.0, 1.0, gauss_noise, gauss_noise, 0.002, 0.002, 2.0
        )
    )
    gp_obj = FlaxGP(enc, x, y, None)
    l = jaxopt.LBFGS(
        lambda param: gp_obj.apply(param, method=gp_obj.neg_llhood), maxiter=1000
    )
    init = gp_obj.init(rng_key, method=FlaxGP.neg_llhood)
    # gp_obj.apply(init, method=gp_obj.neg_llhood)
    l.run(
        init,
    )
