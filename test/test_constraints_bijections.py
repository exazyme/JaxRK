import jax.numpy as np
from jaxrk.core.constraints import SoftBound, CholeskyBijection, SoftPlus
import pytest
import jaxrk.core.constraints as constraints
from functools import partial


lower_bound_values = (-0.4, 5.0)
upper_bound_values = (5.5, 6.0)


@pytest.mark.parametrize(
    "l",
    [
        partial(constraints.NonnegToLowerBd, bij=constraints.SoftPlus),
        partial(constraints.NonnegToLowerBd, bij=constraints.SquarePlus),
    ],
)
def _test_lower_bounds(l: constraints.Bijection, lb: float):
    atol = 1e-2
    x = np.linspace(lb - 50, 50, 1000)
    for lb in lower_bound_values:
        for bij in [constraints.SoftPlus, constraints.SquarePlus]:
            # test the lower bound bijections
            l = constraints.NonnegToLowerBd(lb, bij)
            lx = l(x)
            assert np.all(lx > lb)
            assert np.abs(lx[0] - lb) < 0.2
            assert np.abs(l.inv(lx) - x).mean() < atol


def test_simple_bijections(atol=1e-2):
    for lb in (-0.4, 5.0):
        for ub in (5.5, 6.0):
            assert lb < ub

            l = SoftBound(l=lb)
            u = SoftBound(u=ub)
            lu = SoftBound(lb, ub)

            x = np.linspace(lb - 50, ub + 50, 1000)
            lx = l(x)
            ux = u(x)
            lux = lu(x)
            assert np.all(lx > lb)
            assert np.abs(lx[0] - lb) < 0.2

            assert np.all(ux < ub)
            assert np.abs(ux[-1] - ub) < 0.2

            assert np.all(lux > lb) and np.all(lux < ub)
            assert np.abs(lux[0] - lb) < 0.2 and np.abs(lux[-1] - ub) < 0.2
            for i, n in [
                (l.inv(lx), "lower"),
                (lu.inv(lux), "lower & upper"),
                (u.inv(ux), "upper"),
            ]:
                assert (
                    np.abs(i - x).mean() < atol
                ), f"Inverse of {n} bound Bijection is inaccurate(max abs error: {np.abs(x-i).max()}, mean abs error: {np.abs(x-i).mean()}, min abs error: {np.abs(x-i).min()})"


def test_cholesky_bijection():
    cb = CholeskyBijection()
    psd = -np.ones((3, 3)) + 0.7 + np.eye(3)
    chol = np.linalg.cholesky(psd)
    param = cb.psd_to_param(psd)
    assert cb.is_chol(cb.param_to_chol(param))
    assert cb.is_symmetric(psd)
    assert cb.is_param(cb.chol_to_param(chol))
    assert np.allclose(cb.param_to_chol(param), chol)
    assert np.allclose(cb(param), psd)
