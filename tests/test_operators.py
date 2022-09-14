import copy
import re
from jaxrk.reduce.base import TileView
from jaxrk.reduce.lincomb import LinearReduce
from jaxrk.reduce.base import BalancedRed
from jaxrk.rkhs.cov import Cov_regul
from jaxrk.models.conditional_operator import RidgeCmo
from operator import inv

import jax.numpy as np
from numpy.random import randn
import pytest
from numpy.testing import assert_allclose
from jax import random

from jaxrk.rkhs import Cov_regul, CovOp, FiniteOp, FiniteVec, inner, CombVec, Cov_solve 
from jaxrk.models.conditional_operator import Cdo, Cmo
from jaxrk.kern import (SplitDimsKernel, PeriodicKernel, GenGaussKernel)
from jaxrk.utilities.array_manipulation import all_combinations

from mixtures_tools import location_mixture_logpdf, mixt

rng = random.PRNGKey(1)



kernel_setups = [
    GenGaussKernel.make_gauss(1.)
]


def test_apply_matmul():
    x = np.linspace(-2.5, 15, 5)[:, np.newaxis].astype(np.float32)
    y = randn(x.size)[:, np.newaxis].astype(np.float32)
    
    gk_x = GenGaussKernel.make_gauss(0.1)

    
    x_e1 = FiniteVec.construct_RKHS_Elem(gk_x, x)
    x_e2 = FiniteVec.construct_RKHS_Elem(gk_x, y)
    stacked_factors = np.hstack([x_e1.reduce[0].linear_map] * 2).squeeze()
    diag_stacked_factors = np.diag(stacked_factors)
    x_fv = FiniteVec(gk_x, np.vstack([x,y]), reduce=[LinearReduce(diag_stacked_factors), BalancedRed(x.size)])

    oper_feat_vec  = FiniteVec(gk_x, x)

    oper = FiniteOp(oper_feat_vec, oper_feat_vec, np.eye(len(x)))
    res_e1 = oper @ x_e1
    res_e2 = (oper @ x_e2)
    res_v = (oper @ x_fv)
    assert np.allclose(res_e1.reduce[0].linear_map, (oper.matr @ oper.inp_feat.inner(x_e1)).flatten()), "Application of operator to RKHS element failed."
    assert np.allclose(res_v.insp_pts, res_e1.insp_pts ), "Application of operator to all vectors in RKHS vector failed at inspace points."
    assert np.allclose(res_v.reduce[0].linear_map, np.vstack([res_e1.reduce[0].linear_map, res_e2.reduce[0].linear_map])), "Application of operator to all vectors in RKHS vector failed."
    assert np.allclose((oper @ oper).matr, oper.inp_feat.inner(oper.outp_feat)), "Application of operator to operator failed."

def test_FiniteOp():
    gk_x = GenGaussKernel.make_gauss(0.1)
    x = np.linspace(-2.5, 15, 20)[:, np.newaxis].astype(np.float32)
    #x = np.random.randn(20, 1).astype(np.float)
    ref_fvec = FiniteVec(gk_x, x)
    ref_elem = FiniteVec.construct_RKHS_Elem(gk_x, x, np.ones(len(x)))

    C1 = FiniteOp(ref_fvec, ref_fvec, np.linalg.inv(inner(ref_fvec)))
    assert(np.allclose((C1 @ ref_elem).reduce[0].linear_map, 1.))

    C2 = FiniteOp(ref_fvec, ref_fvec, C1.matr@C1.matr)
    assert(np.allclose((C2 @ ref_elem).reduce[0].linear_map, np.sum(C1.matr, 0)))

    n_rvs = 50
    rv_fvec = FiniteVec(gk_x, random.normal(rng, (n_rvs, 1)) * 5)
    C3 = FiniteOp(rv_fvec, rv_fvec, np.eye(n_rvs))
    assert np.allclose((C3 @ C1).matr, gk_x(rv_fvec.insp_pts, ref_fvec.insp_pts) @ C1.matr, 0.001, 0.001)


def test_CovOp(plot = False):   
    from scipy.stats import multivariate_normal

    nsamps = 1000
    samps_unif = None
    regul_C_ref=0.0001
    D = 1
    import pylab as pl
    if samps_unif is None:
        samps_unif = nsamps
    gk_x = GenGaussKernel.make_gauss(0.2)

    targ = mixt(D, [multivariate_normal(3*np.ones(D), np.eye(D)*0.7**2), multivariate_normal(7*np.ones(D), np.eye(D)*1.5**2)], [0.5, 0.5])
    out_samps = targ.rvs(nsamps).reshape([nsamps, 1]).astype(float)
    out_fvec = FiniteVec(gk_x, out_samps, )
    out_meanemb = out_fvec.sum()
    

    x = np.linspace(-2.5, 15, samps_unif)[:, np.newaxis].astype(float)
    ref_fvec = FiniteVec(gk_x, x)
    ref_elem = ref_fvec.sum()

    C_ref = CovOp(ref_fvec) # CovOp_compl(out_fvec.k, out_fvec.inspace_points, regul=0.)

    inv_Gram_ref = np.linalg.inv(inner(ref_fvec))

    C_samps = CovOp(out_fvec)
    unif_obj = Cov_solve(C_samps, out_meanemb, regul=regul_C_ref).dens_proj()
    C_ref = CovOp(ref_fvec)
    dens_obj = Cov_solve(C_ref, out_meanemb, regul=regul_C_ref).dens_proj()
    


    targp = np.exp(targ.logpdf(ref_fvec.insp_pts.squeeze())).squeeze()
    estp = np.squeeze(inner(dens_obj, ref_fvec))
    estp2 = np.squeeze(inner(dens_obj, ref_fvec))
    est_sup = unif_obj(x).squeeze()
    assert (np.abs(targp.squeeze()-estp).mean() < 0.8), "Estimated density strongly deviates from true density"
    if plot:
        pl.plot(ref_fvec.insp_pts.squeeze(), estp/np.max(estp) * np.max(targp), "b--", label="scaled estimate")
        pl.plot(ref_fvec.insp_pts.squeeze(), estp2/np.max(estp2) * np.max(targp), "g-.", label="scaled estimate (uns)")
        pl.plot(ref_fvec.insp_pts.squeeze(), targp, label = "truth")
        pl.plot(x.squeeze(), est_sup.squeeze(), label = "support")
        
        #pl.plot(ref_fvec.inspace_points.squeeze(), np.squeeze(inner(unif_obj, ref_fvec)), label="unif")
        pl.legend(loc="best")
        pl.show()
    supp = unif_obj(x).squeeze()
    assert (np.std(supp) < 0.15), "Estimated support has high variance, in data points, while it should be almost constant."


def generate_donut(nmeans = 10, nsamps_per_mean = 50):
    from scipy.stats import multivariate_normal
    from numpy import exp

    def pol2cart(theta, rho):
        x = (rho * np.cos(theta)).reshape(-1,1)
        y = (rho * np.sin(theta)).reshape(-1,1)
        return np.concatenate([x, y], axis = 1)

    comp_distribution = multivariate_normal(np.zeros(2), np.eye(2)/100)
    means = pol2cart(np.linspace(0,2*3.141, nmeans + 1)[:-1], 1)

    rvs = comp_distribution.rvs(nmeans * nsamps_per_mean) + np.repeat(means, nsamps_per_mean, 0)
    true_dens = lambda samps: exp(location_mixture_logpdf(samps, means, np.ones(nmeans) / nmeans, comp_distribution))
    return rvs, means, true_dens


def test_Cdmo(plot = False):
    cent_vals = [True, False]
    site_vals = [0., 1.]

    x_vals = [np.zeros((1,1)) + i for i in site_vals]

    (rvs, means, true_dens) = generate_donut(500, 10)

    regul = CovOp.regul(1, len(rvs)) # we will look at 1 point inputs

    invec = FiniteVec(GenGaussKernel(0.3, 1.7), rvs[:, :1])
    outvec = FiniteVec(GenGaussKernel(0.3, 1.7), rvs[:, 1:])
    refervec = FiniteVec(outvec.k, np.linspace(-4, 4, 10000)[:, None])
    C_ref = CovOp(refervec)

    maps = {}
    for center in cent_vals:
        cm = Cmo(invec, outvec, regul, center = center)        
        maps[center] = {"emb":cm, "dens":C_ref.solve(cm)}
        print(np.abs(cm.const_cent_term - maps[center]["dens"].const_cent_term).max())

    ests = {map_type:
                    {cent: np.array([maps[cent][map_type](x).dens_proj()(refervec.insp_pts).squeeze()
                                        for x in x_vals])
                        for cent in cent_vals}
            for map_type in ["emb", "dens"]}

                                             
    t = np.array([true_dens(np.hstack([np.repeat(x, len(refervec.insp_pts), 0), refervec.insp_pts]))
                                for x in x_vals])
    if plot:
        import matplotlib.pyplot as plt

        (fig, ax) = plt.subplots(len(site_vals) + 1, 1, False, False)

        for i, site in enumerate(site_vals):
            ax[i].plot(refervec.insp_pts, t[i], linewidth=2, color="b", label = "true dens", alpha = 0.5)
            for map_type in ["emb", "dens"]:
                for cent in cent_vals:
                    if map_type == "emb":
                        color = "r"
                    else:
                        color = "g"
                    if cent == True:
                        style = ":"
                    else:
                        style= "--"
                    ax[i].plot(refervec.insp_pts, ests[map_type][cent][i], style, color = color, label = map_type+" "+("cent" if cent else "unc"), alpha = 0.5)

        ax[-1].scatter(*rvs.T)
        fig.legend()
        fig.show()

    for cent in cent_vals:
        assert(np.allclose(ests["dens"][cent],t, atol=0.5))

