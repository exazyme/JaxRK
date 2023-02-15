from collections import namedtuple
from turtle import position
import jax.numpy as np
import jax.scipy as sp
import jax.random as jr
import pytest
import jaxopt
import optax as ot


from jaxrk.rkhs import FiniteVec, inner, StandardEncoder
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
    enc = StandardEncoder(kernel)
    X, Xp = [jr.normal(k, (N, D_X)) * 20 for k in (k2, k3)]
    v_X, v_Xp = [enc(X), enc(Xp)]

    g = gp.GP(enc, v_X, y1, 2.0, True)
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


def todo_test_FlaxGp_Optax():
    x = np.linspace(0, 40, 400).reshape((-1, 1))
    rng_key = jr.PRNGKey(0)
    y = np.sin(x) + jr.normal(rng_key, (len(x), 1)) * 0.2
    gauss_noise = lambda key: jr.normal(key, (1,))
    enc = OneToOneEncoder(
        GenGaussKernelFactory.from_constrained(
            5.0, 1.0, gauss_noise, gauss_noise, 0.002, 0.002, 2.0
        )
    )
    gp_obj = FlaxGP(enc, x, y, None)
    Res = namedtuple("Res", ["gp", "y_m", "y_sd"])
    UnTrRes = namedtuple("UnTrRes", ["x", "y", "untr", "tr"])
    if False:
        from eep.models.flax_train import train_cv

        title = "Optax"
        rval = {}
        for (opt_steps, name) in [(100, "trained"), (0, "untrained")]:
            train_state = train_cv(
                rng_key,
                gp_obj,
                len(x),
                optimizer=ot.sgd(0.05),
                loss=gp_obj.neg_llhood,
                ckpt_dir=".",
                opt_steps=opt_steps,
                restore_ckpt=False,
            )

            bound_model = gp_obj.bind({"params": train_state.params})
            bound_gp = bound_model.get_gp()
            y_m, y_v = bound_gp.predict(bound_model.encode_inp(x), diag=True)
            rval[name] = Res(bound_gp, y_m, np.sqrt(y_v))
        un_tr = UnTrRes(x, y, rval["untrained"], rval["trained"])
    else:
        title = "JaxOpt"
        l = jaxopt.LBFGS(
            lambda param: gp_obj.apply(param, None, method=gp_obj.neg_llhood),
            maxiter=1000,
        )

        init_vars = gp_obj.init(rng_key, None, method=FlaxGP.neg_llhood)
        params = init_vars["params"]
        state = l.init_state(params)

        l.update(
            init_vars,
            state,
        )
        opt_res = l.run(
            init_vars,
        )
        bound_model = gp_obj.bind(opt_res.params)
        bound_gp = bound_model.get_gp()
        y_m, y_v = bound_gp.predict(bound_model.encode_inp(x), diag=True)
        untr_bound = gp_obj.bind(init_vars)
        untr_gp = untr_bound.get_gp()
        uy_m, uy_v = untr_gp.predict(untr_bound.encode_inp(x), diag=True)

        un_tr = UnTrRes(
            x,
            y,
            Res(bound_gp, uy_m, np.sqrt(uy_v)),
            Res(untr_gp, y_m, np.sqrt(y_v)),
        )
    # un_tr = tgp.todo_test_FlaxGp()
    import pylab as pl

    pl.figure()
    pl.scatter(un_tr.x, un_tr.y)
    for ((_, y_m, y_sd), name, c) in [
        (un_tr.untr, "untrained", "blue"),
        (un_tr.tr, "trained", "red"),
    ]:
        pl.plot(un_tr.x.squeeze(), y_m, color=c, label=name)
        pl.fill_between(
            un_tr.x.squeeze(),
            np.squeeze(y_m - y_sd),
            np.squeeze(y_m + y_sd),
            alpha=0.5,
            color=c,
        )
    pl.legend(position="best")
    pl.title(title)
    pl.tight_layout()
    return un_tr


def todo_test_FlaxGp_Jaxopt():
    import pylab as pl

    x = np.linspace(0, 40, 400).reshape((-1, 1))
    rng_key = jr.PRNGKey(0)
    y = np.sin(x) + jr.normal(rng_key, (len(x), 1)) * 0.2
    gauss_noise = lambda key: jr.normal(key, (1,))
    enc = OneToOneEncoder(
        GenGaussKernelFactory.from_constrained(
            5.0, 1.0, gauss_noise, gauss_noise, 0.002, 0.002, 2.0
        )
    )
    gp_obj = FlaxGP(enc, x, y, None)
    Res = namedtuple("Res", ["gp", "y_m", "y_sd"])
    UnTrRes = namedtuple("UnTrRes", ["x", "y", "untr", "tr"])
    title = "JaxOpt"
    l = jaxopt.LBFGS(
        lambda param: gp_obj.apply(param, None, method=gp_obj.neg_llhood),
        maxiter=1000,
    )

    init_vars = gp_obj.init(rng_key, None, method=FlaxGP.neg_llhood)
    params = init_vars["params"]
    state = l.init_state(params)

    l.update(
        init_vars,
        state,
    )
    opt_res = l.run(
        init_vars,
    )
    bound_model = gp_obj.bind(opt_res.params)
    bound_gp = bound_model.get_gp()
    y_m, y_v = bound_gp.predict(bound_model.encode_inp(x), diag=True)
    untr_bound = gp_obj.bind(init_vars)
    untr_gp = untr_bound.get_gp()
    uy_m, uy_v = untr_gp.predict(untr_bound.encode_inp(x), diag=True)

    un_tr = UnTrRes(
        x,
        y,
        Res(bound_gp, uy_m, np.sqrt(uy_v)),
        Res(untr_gp, y_m, np.sqrt(y_v)),
    )
    # un_tr = tgp.todo_test_FlaxGp()

    pl.figure()
    pl.scatter(un_tr.x, un_tr.y)
    for ((_, y_m, y_sd), name, c) in [
        (un_tr.untr, "untrained", "blue"),
        (un_tr.tr, "trained", "red"),
    ]:
        pl.plot(un_tr.x.squeeze(), y_m, color=c, label=name)
        pl.fill_between(
            un_tr.x.squeeze(),
            np.squeeze(y_m - y_sd),
            np.squeeze(y_m + y_sd),
            alpha=0.5,
            color=c,
        )
    pl.legend(position="best")
    pl.title(title)
    pl.tight_layout()
    return un_tr
