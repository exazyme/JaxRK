from functools import partial

from jaxrk.reduce.lincomb import LinearReduce
from .rkhs import FiniteVec
from .core.typing import Array
from .utilities import cv
import jax.numpy as np, jax.scipy as sp
from typing import Any
from jaxrk.rkhs import Cov_regul
import jax

__log2pi_half = np.log(2. * np.pi) / 2


def mvgauss_loglhood_mean0(y:Array, cov_chol:Array, prec_y:Array = None):
    if prec_y is None:
        prec_y = sp.linalg.cho_solve((cov_chol, True), y)
    mll  = - 0.5 * np.einsum("ik,ik->k", y, prec_y)
    mll -= np.log(np.diag(cov_chol)).sum()
    mll -= len(cov_chol) * __log2pi_half
    return mll.sum()

def gp_init(x_inner_x:Array, y:Array, noise:float, normalize_y:bool = False):
    self = lambda x: None
    #print("noise is ", noise)
    if noise is None:
        noise = Cov_regul(1, len(x_inner_x))
    self.y, self.noise, self.normalize_y = (y, noise, normalize_y)
    if self.y.ndim == 1:
        self.y = self.y[:, np.newaxis]
    
    y_standard_shape = (1, self.y.shape[1])
    self.ymean = np.zeros(y_standard_shape)
    self.ystd = np.ones(y_standard_shape)
    if normalize_y:
        self.ymean = np.mean(y, 0, keepdims=True)
        if len(x_inner_x) > 1:
            self.ystd = np.std(y, 0, keepdims=True)
        assert self.ymean.shape == y_standard_shape
        assert self.ystd.shape == y_standard_shape
        self.y = (self.y - self.ymean) / self.ystd
    train_cov = x_inner_x + np.eye(len(x_inner_x)) * self.noise
    self.chol = sp.linalg.cholesky(train_cov, lower=True)

    #matrix product of precision matrix and y. Called alpha in sklearn implementation
    self.prec_y = sp.linalg.cho_solve((self.chol, True), self.y)
    return self.chol, self.y, self.prec_y, self.ymean, self.ystd, noise

def gp_predictive_mean(gram_train_test:Array, train_prec_y:Array, outp_mean:Array = np.zeros(1), outp_std:Array = np.ones(1)):
    return outp_mean + outp_std * np.dot(gram_train_test.T, train_prec_y)

def gp_predictive_cov(gram_train_test:Array, gram_test:Array, inv_train_cov:Array = None, chol_train_cov:Array = None, outp_std:Array = np.ones(1)):
    if chol_train_cov is None:
        cov = (gram_test - gram_train_test.T @ inv_train_cov @ gram_train_test) * outp_std**2
    else:
        v = sp.linalg.cho_solve((chol_train_cov, True), gram_train_test)
        cov = (gram_test - np.dot(gram_train_test.T, v)) * outp_std**2
    return cov

def gp_predictive(gram_train_test:Array, gram_test:Array, chol_train_cov:Array, train_prec_y:Array, y_mean:Array = np.zeros(1), y_std:Array = np.ones(1), y_test:Array = None):
    m = gp_predictive_mean(gram_train_test, train_prec_y, y_mean, y_std)
    cov = gp_predictive_cov(gram_train_test, gram_test, chol_train_cov = chol_train_cov, outp_std = y_std)
    if y_test is None:
        return m, cov
    else:
        y_test = (y_test - y_mean) / y_std
        return m, cov, mvgauss_loglhood_mean0(y_test - m, sp.linalg.cholesky(cov, lower=True))

#@jax.jit
def gp_val_lhood(train_sel:Array, val_sel:Array, x_inner_x:Array, y:Array, noise:float):
    x_train = train_sel @ x_inner_x @ train_sel.T
    x_train_val = train_sel @ x_inner_x @ val_sel.T
    x_val = val_sel @ x_inner_x @ val_sel.T
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_train = train_sel @ y
    y_val = val_sel @ y

    chol, y, prec_y, ymean, ystd = gp_init(x_train, y_train, noise, True)
    return gp_predictive(x_train_val,
                         x_val,
                         chol,
                         prec_y,
                         y_mean = ymean,
                         y_std = ystd,
                         y_test = y_val)[-1]
    
vmap_gp_val_lhood = jax.jit(jax.vmap(gp_val_lhood, (0, 0, None, None, None)))

def gp_mlhood(train_sel:Array, val_sel:Array, x_inner_x:Array, y:Array, noise:float):
    x_train = train_sel @ x_inner_x @ train_sel.T
    x_train_val = train_sel @ x_inner_x @ val_sel.T
    x_val = val_sel @ x_inner_x @ val_sel.T
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_train = train_sel @ y
    y_val = val_sel @ y

    chol, y, prec_y, ymean, ystd = gp_init(x_train, y_train, noise, True)
    return gp_predictive(x_train_val,
                         x_val,
                         chol,
                         prec_y,
                         y_mean = ymean,
                         y_std = ystd,
                         y_test = y_val)[-1]
    
vmap_gp_mlhood = jax.jit(jax.vmap(gp_val_lhood, (0, 0, None, None, None)))

def gp_cv_val_lhood(train_val_idcs:Array, x_inner_x:Array, y:Array, regul:float = None):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    train_idcs, val_idcs = train_val_idcs

    #val_sel = cv.idcs_to_selection_matr(len(inp), val_idcs)
    rval = vmap_gp_val_lhood(cv.idcs_to_selection_matr(len(x_inner_x), train_idcs),
                             cv.idcs_to_selection_matr(len(x_inner_x), val_idcs),
                             x_inner_x, y, regul)
    return rval.sum()

class GP(object):
    def __init__(self, x:FiniteVec, y:Array, noise:float, normalize_y:bool = False):
        self.x = x
        self.x_inner_x = x.inner()
        self.chol, self.y, self.prec_y, self.ymean, self.ystd, self.noise = gp_init(self.x_inner_x, y, noise, normalize_y)
        
    def __str__(self,):
        return f"μ_Y = {self.ymean.squeeze()} ± σ_Y ={self.ystd.squeeze()}, σ_noise: {self.noise}, trace_chol: {self.chol.trace().squeeze()} trace_xx^t: {self.x_inner_x.trace().squeeze()}, {self.x_inner_x[0,:5]}, {self.x_inner_x.shape}"

    def marginal_loglhood(self):
        return mvgauss_loglhood_mean0(self.y, self.chol, self.prec_y)

    def predict(self, xtest:FiniteVec):
        return gp_predictive(self.x.inner(xtest), xtest.inner(), self.chol, self.prec_y, self.ymean, self.ystd)
    
    def post_pred_likelihood(self, xtest:FiniteVec, ytest:Array):
        return gp_predictive(self.x.inner(xtest), xtest.inner(), self.chol, self.prec_y, y_mean = self.ymean, y_std = self.ystd, y_test = ytest)
