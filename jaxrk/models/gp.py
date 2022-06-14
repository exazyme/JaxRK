from functools import partial

from jaxrk.reduce.lincomb import LinearReduce
from ..rkhs import FiniteVec
from ..core.typing import Array
from ..utilities import cv
import jax.numpy as np, jax.scipy as sp
from typing import Any, Callable
from jaxrk.rkhs import Cov_regul
import jax

__log2pi_half = np.log(2. * np.pi) / 2


#this is the loglikelihood for a univariate GP with zero mean.

def gp_loglhood_mean0_univ(y:Array, cov_chol:Array, prec_y:Array = None) -> float:
    """Log likelihood for a univariate GP with zero mean.
    (Which is the same as that of a multivariate gaussian distribution)

    Args:
        y (Array): The observed values.
        cov_chol (Array): A cholesky factor of the gram matrix/covariance matrix
        prec_y (Array, optional): The product of precision matrix and y. Defaults to None, in which case it is calculated from the other arguments.

    Returns:
        float: Log likelihood
    """
    if prec_y is None:
        prec_y = sp.linalg.cho_solve((cov_chol, True), y)
    mll  = - 0.5 * np.einsum("ik,ik->k", y, prec_y)
    mll -= np.log(np.diag(cov_chol)).sum()
    mll -= len(cov_chol) * __log2pi_half
    return mll.sum()

@jax.jit
def gp_loglhood_mean0(y:Array, cov_chol:Array, prec_y:Array = None) -> float:
    """Log likelihood for a multivariate GP with zero mean, for which the covariance matrix is the same for all dimensions.
    Args:
        y (Array): The observed values.
        cov_chol (Array): A cholesky factor of the gram matrix/covariance matrix
        prec_y (Array, optional): The product of precision matrix and y. Defaults to None, in which case it is calculated from the other arguments.

    Returns:
        float: Log likelihood
    """
    rval = 0.
    for y_dim in y.T:
        incr = gp_loglhood_mean0_univ(y_dim[:, np.newaxis], cov_chol, prec_y)
        rval += incr
    return rval

def gp_init(x_inner_x:Array, y:Array, noise:float, normalize_y:bool = False):
    self = lambda x: None
    #print("noise is ", noise)
    if noise is None:
        noise = Cov_regul(1, len(x_inner_x))
    y, noise, normalize_y = (y, noise, normalize_y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    
    y_standard_shape = (1, y.shape[1])
    ymean = np.zeros(y_standard_shape)
    ystd = np.ones(y_standard_shape)
    if normalize_y:
        ymean = np.mean(y, 0, keepdims=True)
        if len(x_inner_x) > 1:
            ystd = np.std(y, 0, keepdims=True)
        assert ymean.shape == y_standard_shape, f"Mean had shape {ymean.shape} instead of expected {y_standard_shape}"
        assert ystd.shape == y_standard_shape, f"Standard deviation had shape {ystd.shape} instead of expected {y_standard_shape}"
        y = (y - ymean) / ystd
    train_cov = x_inner_x + np.eye(len(x_inner_x)) * noise
    chol = sp.linalg.cholesky(train_cov, lower=True)

    #matrix product of precision matrix and y. Called alpha in sklearn implementation
    prec_y = sp.linalg.cho_solve((chol, True), y)
    return chol, y, prec_y, ymean, ystd, noise


def gp_predictive_mean_univ(gram_train_test:Array, train_prec_y:Array, outp_mean:Array = np.zeros(1), outp_std:Array = np.ones(1)):
    return outp_mean + outp_std * np.dot(gram_train_test.T, train_prec_y)
gp_predictive_mean = jax.vmap(gp_predictive_mean_univ, (None, 1, 1, 1), 1)

def gp_predictive_cov_univ_chol(gram_train_test:Array, gram_test:Array, chol_train_cov:Array):
    v = sp.linalg.cho_solve((chol_train_cov, True), gram_train_test)
    return (gram_test - np.dot(gram_train_test.T, v)) 

def gp_predictive_cov_univ_inv(gram_train_test:Array, gram_test:Array, inv_train_cov:Array):
    (gram_test - gram_train_test.T @ inv_train_cov @ gram_train_test)


def gp_predictive_cov_univ(gram_train_test:Array, gram_test:Array, inv_train_cov:Array = None, chol_train_cov:Array = None, outp_std:Array = np.ones(1)):
    if chol_train_cov is None:
        return gp_predictive_cov_univ_inv(gram_train_test, gram_test, inv_train_cov, outp_std)
    return gp_predictive_cov_univ_chol(gram_train_test, gram_test,  chol_train_cov, outp_std)

@partial(jax.vmap, in_axes = (None, 1), out_axes = 2)
def scale_dims(inp, scale_per_dim) -> np.ndarray:
    return inp * scale_per_dim

@partial(jax.vmap, in_axes = (None, -1), out_axes = 2)
def scale_dims_inv(inp, scale_per_dim) -> np.ndarray:
    return inp / scale_per_dim

#@partial(jax.vmap, in_axes = (None, 1, 1), out_axes = 1)
def scale_and_shift_dims(inp, shift_per_dim, scale_per_dim) -> np.ndarray:
    return inp * scale_per_dim + shift_per_dim

#@partial(jax.vmap, in_axes = (None, 1, 1), out_axes = 1)
def scale_and_shift_dims_inv(inp, shift_per_dim, scale_per_dim) -> np.ndarray:
    return (inp - shift_per_dim) / scale_per_dim

#gp_predictive_cov = jax.vmap(gp_predictive_cov_univ, (None, None, None, None, 1), 2)
    
# def gp_predictive_var_1(gram_train_test:Array, gram_test:Array, inv_train_cov:Array = None, chol_train_cov:Array = None, outp_std:Array = np.ones(1)):
#     vm = jax.vmap(partial(gp_predictive_cov, inv_train_cov = inv_train_cov, chol_train_cov=chol_train_cov, outp_std=outp_std), (1, 0))
#     return vm(gram_train_test, jax.numpy.diagonal(gram_test))


def gp_predictive(gram_train_test:Array, gram_test:Array, chol_train_cov:Array, train_prec_y:Array, y_mean:Array = np.zeros(1), y_std:Array = np.ones(1), y_test:Array = None):
    m = gp_predictive_mean_univ(gram_train_test, train_prec_y)
    cov = gp_predictive_cov_univ_chol(gram_train_test, gram_test, chol_train_cov)
    pred_m, pred_cov = scale_and_shift_dims(m, y_mean, y_std), scale_dims(cov, y_std**2)
    if y_test is None:
        return pred_m, pred_cov
    else:
        y_test2 = scale_and_shift_dims_inv(y_test, y_mean, y_std) - m
        cov_chol = sp.linalg.cholesky(cov, lower=True)
        return pred_m, pred_cov, gp_loglhood_mean0(y_test2, cov_chol)

def loglhood_loss(y_test:Array, pred_mean_y:Array, pred_cov_y:Array, loglhood_y:Array) -> float:
    return loglhood_y

def differences(a:np.ndarray):
    result = []
    for i, x1 in enumerate(a):
        for x2 in a[i+1:]:
            result.append(x1 - x2)
    return (np.array(result))

@jax.jit
def concordant_discordant(x:Array, y:Array):
    ox = np.sign(differences(x))
    oy = np.sign(differences(y))
    return np.sum(ox == oy), np.sum(ox != oy)

def distance_of_distances_loss(x:Array, y:Array):
    return -np.sum(np.abs(differences(x) - differences(y)))

def condis_loss(x, y):
    con, dis =  concordant_discordant(x, y)
    return np.sum(con - dis) #.astype(float)#/x.size

def spearman_rho_loss(x:Array, y:Array):
    rx = np.argsort(x).squeeze()
    ry = np.argsort(y).squeeze()
    c =  np.mean((rx - rx.mean()) * (ry - ry.mean()))
    sd = rx.std().squeeze() * ry.std().squeeze()
    print(rx, ry, sd)
    return c / sd

class UcbRankLoss(object):
    def __init__(self, loss, exploration_factor:float):
        self.loss = loss
        self.exploration_factor = exploration_factor

    def __call__(self, y_test:Array, pred_mean_y:Array, pred_cov_y:Array, loglhood_y:Array) -> float:
        
        pred = pred_mean_y.squeeze() + self.exploration_factor * np.diagonal(pred_cov_y).squeeze()
        #print(f"cov-shape {pred_cov_y.shape}, {np.diagonal(pred_cov_y).shape}, {pred.shape}")
        return self.loss(y_test, pred).astype(float)


#@jax.jit
def gp_val_loss(train_sel:Array, val_sel:Array, x_inner_x:Array, y:Array, noise:float, loss:Callable[[Array, Array, Array, Array], float] = loglhood_loss):
    x_train = train_sel @ x_inner_x @ train_sel.T
    x_train_val = train_sel @ x_inner_x @ val_sel.T
    x_val = val_sel @ x_inner_x @ val_sel.T
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_train = train_sel @ y
    y_val = val_sel @ y

    chol, y, prec_y, ymean, ystd, noise = gp_init(x_train, y_train, noise, True)

    pred = gp_predictive(x_train_val,
                         x_val,
                         chol,
                         prec_y,
                         y_mean = ymean,
                         y_std = ystd,
                         y_test = y_val)
    return loss(y_val, *pred)
    
vmap_gp_val_lhood = jax.jit(jax.vmap(partial(gp_val_loss, loss = loglhood_loss), (0, 0, None, None, None)))
#vmap_gp_val_kendall = jax.jit(jax.vmap(partial(gp_val_loss, loss = UcbRankLoss(stats.kendalls_tau, 0.)), (0, 0, None, None, None)))
vmap_gp_val_rank = jax.jit(jax.vmap(partial(gp_val_loss, loss = UcbRankLoss(condis_loss, 0.)), (0, 0, None, None, None)))
vmap_gp_val_rho = jax.jit(jax.vmap(partial(gp_val_loss, loss = UcbRankLoss(spearman_rho_loss, 0.)), (0, 0, None, None, None)))
vmap_gp_val_dists = jax.jit(jax.vmap(partial(gp_val_loss, loss = UcbRankLoss(distance_of_distances_loss, 0.)), (0, 0, None, None, None)))
#vmap_gp_val_spearman = jax.jit(jax.vmap(partial(gp_val_loss, loss = UcbRankLoss(stats.spearmanr, 0.)), (0, 0, None, None, None)))

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
    
vmap_gp_mlhood = jax.jit(jax.vmap(gp_val_loss, (0, 0, None, None, None)))

def gp_cv_val_lhood(train_val_idcs:Array, x_inner_x:Array, y:Array, regul:float = None, vmapped_loss:Callable[[Array, Array, Array, Array, float], Array] = vmap_gp_val_lhood):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    train_idcs, val_idcs = train_val_idcs

    #val_sel = cv.idcs_to_selection_matr(len(inp), val_idcs)
    rval = vmapped_loss(cv.idcs_to_selection_matr(len(x_inner_x), train_idcs),
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
        return gp_loglhood_mean0(self.y, self.chol, self.prec_y)

    def predict(self, xtest:FiniteVec, diag=True):
        pred_m, pred_cov = gp_predictive(self.x.inner(xtest), xtest.inner(), self.chol, self.prec_y, self.ymean, self.ystd)
        if diag:
            return pred_m, np.diagonal(pred_cov).T
        return pred_m, pred_cov
        
    
    def post_pred_likelihood(self, xtest:FiniteVec, ytest:Array):
        return gp_predictive(self.x.inner(xtest), xtest.inner(), self.chol, self.prec_y, y_mean = self.ymean, y_std = self.ystd, y_test = ytest)
