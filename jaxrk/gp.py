from .rkhs import FiniteVec
from .core.typing import Array
from .utilities import cv
import jax.numpy as np, flax.linen as ln, jax.scipy as sp
from typing import Any
from jaxrk.rkhs import Cov_regul
import jax

__log2pi_half = np.log(2. * np.pi) / 2

"""def mvgauss_likelihood(y:Array, cov_chol:Array, prec_y:Array = None):
    if prec_y is None:
        prec_y = sp.linalg.cho_solve((cov_chol, True), y)
    mll = np.sum(- 0.5 * np.einsum("ik,ik->k", y, prec_y)
                    - np.log(np.diag(cov_chol)).sum()
                    - len(cov_chol) * __log2pi_half)
    return -mll"""

def mvgauss_loglhood_mean0(y:Array, cov_chol:Array, prec_y:Array = None):
    if prec_y is None:
        prec_y = sp.linalg.cho_solve((cov_chol, True), y)
    mll  = - 0.5 * np.einsum("ik,ik->k", y, prec_y)
    mll -= np.log(np.diag(cov_chol)).sum()
    mll -= len(cov_chol) * __log2pi_half
    return mll.sum()

def gp_predictive_distr(gram_train_test:Array, gram_test:Array, train_cov_chol:Array, train_prec_y:Array, outp_mean:Array = np.zeros(1), outp_std:Array = np.ones(1)):
    mu = outp_mean + outp_std * np.dot(gram_train_test.T, train_prec_y)
    v = sp.linalg.cho_solve((train_cov_chol, True), gram_train_test)
    cov = (gram_test - np.dot(gram_train_test.T, v)) * outp_std**2
    return mu, cov

@jax.jit
def gp_cv_likelihood_single(gram_full:Array, y_full:Array, train_sel, val_sel, chol_train, normalize_y:bool = True):
    gram_full_val = gram_full@val_sel.T
    train_y = train_sel@y_full
    if not normalize_y:
        y_mean, y_std = np.zeros(1), np.ones(1)
    else:
        y_mean, y_std = train_y.mean(0, keepdims=True), train_y.std(0, keepdims = True)
        train_y = (train_y - y_mean) / y_std
    
    gram_train_val = train_sel@gram_full_val
    print(gram_train_val.shape, len(train_y), len(val_sel))
    mu, cov = gp_predictive_distr(gram_train_val, val_sel@gram_full_val, chol_train, sp.linalg.cho_solve((chol_train, True), train_y))
    val_y_cent = (val_sel@y_full - y_mean) / y_std - mu
    return mvgauss_loglhood_mean0(val_y_cent, np.diag(np.sqrt(np.diagonal(cov))), sp.linalg.solve(cov, val_y_cent))

gp_cv_likelihood = jax.vmap(gp_cv_likelihood_single, (None, None, 0, 0, 0))

def gp_cv_mlhood(inp:FiniteVec, outp:Array, train_val_idcs:Array, regul:float = None,):
    if outp.ndim == 1:
        outp = outp[:, np.newaxis]
    train_idcs, val_idcs = train_val_idcs
    if regul is None:
        regul = Cov_regul(val_idcs.shape[1], train_idcs.shape[1])
    gram_inp = inp.inner()
    gram_regul = (gram_inp + regul * np.eye(len(inp)))
    cv_chol = cv.cholesky_submatr(gram_regul, train_idcs, zerofill=False)

    val_sel = cv.idcs_to_selection_matr(len(inp), val_idcs)
    rval = gp_cv_likelihood(gram_inp, outp,
                            cv.idcs_to_selection_matr(len(inp), train_idcs),
                            cv.idcs_to_selection_matr(len(inp), val_idcs),
                            cv_chol)
    return rval.sum()



class GP(object):
    def __init__(self, x:FiniteVec, y:Array, noise:float, normalize_y:bool = False):
        if noise is None:
            noise = Cov_regul(1, len(x))
        self.x, self.y, self.noise, self.normalize_y = (x, y, noise, normalize_y)
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        if not normalize_y:
            self.ymean = np.zeros(1)
            self.ystd = np.ones(1)
        else:
            self.ymean = np.mean(y, 0, keepdims=True)
            self.ystd = np.std(y, 0, keepdims=True)
            self.y = (self.y - self.ymean) / self.ystd

        train_cov = self.x.inner() + np.eye(len(self.x)) * self.noise 
        self.chol = sp.linalg.cholesky(train_cov, lower=True)

        #matrix product of precision matrix and y. Called alpha in sklearn implementation
        self.prec_y = sp.linalg.cho_solve((self.chol, True), self.y)

    
    def marginal_likelihood(self):
        ml = mvgauss_loglhood_mean0(self.y, self.chol, self.prec_y)
        ml += np.log(2. * np.pi) / 2 # lognormal prior
        return ml

    def predict(self, xtest:FiniteVec):
        return gp_predictive_distr(self.x.inner(xtest), xtest.inner(), self.chol, self.prec_y, self.ymean, self.ystd)
    
    def post_pred_likelihood(self, xtest:FiniteVec, ytest:Array):
        m, cov = self.predict(xtest)
        if self.normalize_y:
            ytest = (ytest - self.ymean) / self.ystd
        return m, cov, mvgauss_loglhood_mean0(ytest - m, sp.linalg.cholesky(cov, lower=True))