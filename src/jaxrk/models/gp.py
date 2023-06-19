from collections import namedtuple
from functools import partial

from ..rkhs import FiniteVec, RkhsVecEncoder
from ..core.typing import Array
from ..utilities import cv
import jax.numpy as np
import jax.scipy as sp
from typing import Any, Callable, Generic, TypeVar, Union
from jaxrk.rkhs import Cov_regul
import jax

__log2pi_half = np.log(2.0 * np.pi) / 2


# this is the loglikelihood for a univariate GP with zero mean.


@jax.jit
def gp_loglhood_mean0_univ(y: Array, cov_chol: Array, prec_y: Array = None) -> float:
    """Log likelihood for a univariate GP with zero mean.
    This is the same as that of a multivariate gaussian distribution.
    Assume that covariance structure is between rows only and that
    columns of y represent different output dimensions (all with the same covariance structure).

    Args:
        y (Array): The observed values.
        cov_chol (Array): A cholesky factor of the gram matrix/covariance matrix
        prec_y (Array, optional): The product of precision matrix and y. Defaults to None, in which case it is calculated from the other arguments.

    Returns:
        float: Log likelihood for each dimension (column of y) separately
    """
    if prec_y is None:
        prec_y = sp.linalg.cho_solve((cov_chol, True), y)
    mll = -0.5 * np.einsum("ik,ik->k", y, prec_y)
    mll -= np.log(np.diag(cov_chol)).sum()
    mll -= len(cov_chol) * __log2pi_half
    return mll  # .sum(axis=-1)


def gp_loglhood_mean0(y: Array, cov_chol: Array, prec_y: Array = None) -> float:
    """Log likelihood for a multivariate GP with zero mean, for which the covariance matrix is assumed to be between rows.
    Args:
        y (Array): The observed values. Covariance is among rows (i.e. len(y) == cov_chol.shape[0] == cov_chol.shape[1]). Different dimensions of observations in columns.
        cov_chol (Array): A cholesky factor of the gram matrix/covariance matrix
        prec_y (Array, optional): The product of precision matrix and y. Defaults to None, in which case it is calculated from the other arguments.

    Returns:
        float: Log likelihood
    """
    return gp_loglhood_mean0_univ(y, cov_chol, prec_y).sum()


GPInitParams = namedtuple(
    "GPInitParams", ["chol", "y", "prec_y", "ymean", "ystd", "noise"]
)


def gp_init(
    x_inner_x: Array, y: Array, noise: float, normalize_y: bool = False
) -> GPInitParams:
    """Compute the initial parameters for a GP regression model.

    Args:
        x_inner_x (Array): The inner product matrix of the input data.
        y (Array): The observed values.
        noise (float): The noise level/regularization parameter.
        normalize_y (bool, optional): Whether to normalize the output values. Defaults to False.
    """

    # print("noise is ", noise)
    if noise is None:
        noise = Cov_regul(1, len(x_inner_x))

    if y.ndim == 1:
        y = y[:, np.newaxis]

    y_standard_shape = (1, y.shape[1])
    ymean = np.zeros(y_standard_shape)
    ystd = np.ones(y_standard_shape)
    if normalize_y:
        ymean = np.mean(y, 0, keepdims=True)
        if len(x_inner_x) > 1:
            ystd = np.std(y, 0, keepdims=True)
        assert (
            ymean.shape == y_standard_shape
        ), f"Mean had shape {ymean.shape} instead of expected {y_standard_shape}"
        assert (
            ystd.shape == y_standard_shape
        ), f"Standard deviation had shape {ystd.shape} instead of expected {y_standard_shape}"
        y = (y - ymean) / ystd
    train_cov = x_inner_x + np.eye(len(x_inner_x)) * noise
    chol = sp.linalg.cholesky(train_cov, lower=True)

    # matrix product of precision matrix and y. Called alpha in sklearn
    # implementation
    prec_y = sp.linalg.cho_solve((chol, True), y)
    return GPInitParams(chol, y, prec_y, ymean, ystd, noise)


def gp_predictive_mean_univ(
    gram_train_test: Array,
    train_prec_y: Array,
    outp_mean: Array = np.zeros(1),
    outp_std: Array = np.ones(1),
) -> Array:
    """Predictive mean for a univariate GP.

    Args:
        gram_train_test (Array): The inner product matrix between the training and test data.
        train_prec_y (Array): The product of the precision matrix and the training data.
        outp_mean (Array, optional): The mean of the output of the univariate GP. Defaults to np.zeros(1).
        outp_std (Array, optional): The standard deviation of the output of the univariate GP. Defaults to np.ones(1).

    Returns:
        Array: The predictive mean for each univariate GP.
    """
    return outp_mean + outp_std * np.dot(gram_train_test.T, train_prec_y)


gp_predictive_mean = jax.vmap(gp_predictive_mean_univ, (None, 1, 1, 1), 1)


def gp_predictive_cov_univ_chol(
    gram_train_test: Array, gram_test: Array, chol_gram_train: Array
) -> Array:
    """Predictive covariance for a univariate GP using the cholesky factor of the gram matrix of the training data.

    Args:
        gram_train_test (Array): The inner product matrix between the training and test data.
        gram_test (Array): The inner product matrix of the test data.
        chol_gram_train (Array): The cholesky factor of the inner product matrix of the training data.

    Returns:
        Array: The predictive covariance for the univariate GP.
    """
    v = sp.linalg.cho_solve((chol_gram_train, True), gram_train_test)
    return gram_test - np.dot(gram_train_test.T, v)


def gp_predictive_cov_univ_inv(
    gram_train_test: Array, gram_test: Array, inv_gram_train: Array
):
    """Predictive covariance for a univariate GP using the inverse of the gram matrix of the training data.
    Args:
        gram_train_test (Array): The inner product matrix between the training and test data.
        gram_test (Array): The inner product matrix of the test data.
        chol_gram_train (Array): The cholesky factor of the inner product matrix of the training data.

    Returns:
        Array: The predictive covariance for the univariate GP.
    """
    return gram_test - gram_train_test.T @ inv_gram_train @ gram_train_test


def gp_predictive_cov_univ(
    gram_train_test: Array,
    gram_test: Array,
    inv_gram_train: Array = None,
    chol_gram_train: Array = None,
):
    """Predictive covariance for a univariate GP using either the inverse or cholesky of the gram matrix of the training data.

    Args:
        gram_train_test (Array): The inner product matrix between the training and test data.
        gram_test (Array): The inner product matrix of the test data.
        inv_gram_train (Array, optional): The inverse of the inner product matrix of the training data. Defaults to None, in which case the cholesky factor is used.
        chol_gram_train (Array): The cholesky factor of the inner product matrix of the training data. Defaults to None, in which case the inverse is used.

    Returns:
        Array: The predictive covariance for the univariate GP.
    """
    if chol_gram_train is None:
        return gp_predictive_cov_univ_inv(gram_train_test, gram_test, inv_gram_train)
    return gp_predictive_cov_univ_chol(gram_train_test, gram_test, chol_gram_train)


@partial(jax.vmap, in_axes=(None, 1), out_axes=2)
def scale_dims(inp: np.ndarray, scale_per_dim: np.ndarray) -> np.ndarray:
    """Scale the input data by a different scale for each dimension.

    Args:
        inp (np.ndarray): The input data.
        scale_per_dim (np.ndarray): The scale for each dimension.

    Returns:
        np.ndarray: The scaled input data.
    """
    return inp * scale_per_dim


@partial(jax.vmap, in_axes=(None, -1), out_axes=2)
def scale_dims_inv(scaled_inp, scale_per_dim) -> np.ndarray:
    """Inverse scaling of the input data given the scale for each dimension.

    Args:
        scaled_inp (np.ndarray): The scaled input data.
        scale_per_dim (np.ndarray): The scale for each dimension.

    Returns:
        np.ndarray: The input data without scaling.
    """
    return scaled_inp / scale_per_dim


# @partial(jax.vmap, in_axes = (None, 1, 1), out_axes = 1)


def scale_and_shift_dims(
    inp: np.ndarray, shift_per_dim: np.ndarray, scale_per_dim: np.ndarray
) -> np.ndarray:
    """Scale and shift the input data by a different scale and shift for each dimension.

    Args:
        inp (np.ndarray): The input data.
        shift_per_dim (np.ndarray): The shift for each dimension.
        scale_per_dim (np.ndarray): The scale for each dimension.

    Returns:
        np.ndarray: The scaled and shifted input data.
    """
    return inp * scale_per_dim + shift_per_dim


# @partial(jax.vmap, in_axes = (None, 1, 1), out_axes = 1)


def scale_and_shift_dims_inv(
    scaled_shifted_inp: np.ndarray, shift_per_dim: np.ndarray, scale_per_dim: np.ndarray
) -> np.ndarray:
    """Inverse scaling and shifting of the input data given the scale and shift for each dimension.

    Args:
        scaled_shifted_inp (np.ndarray): The scaled and shifted input data.
        shift_per_dim (np.ndarray): The shift for each dimension.
        scale_per_dim (np.ndarray): The scale for each dimension.

    Returns:
        np.ndarray: The input data without scaling and shifting.
    """
    return (scaled_shifted_inp - shift_per_dim) / scale_per_dim


# gp_predictive_cov = jax.vmap(gp_predictive_cov_univ, (None, None, None, None, 1), 2)

# def gp_predictive_var_1(gram_train_test:Array, gram_test:Array, inv_gram_train:Array = None, chol_gram_train:Array = None, outp_std:Array = np.ones(1)):
#     vm = jax.vmap(partial(gp_predictive_cov, inv_gram_train = inv_gram_train, chol_gram_train=chol_gram_train, outp_std=outp_std), (1, 0))
#     return vm(gram_train_test, jax.numpy.diagonal(gram_test))


def gp_predictive_mean(
    gram_train_test: Array,
    train_prec_y: Array,
    y_mean: Array = np.zeros(1),
    y_std: Array = np.ones((1, 1)),
) -> Array:
    """Predictive mean for univariate GPs all with the same kernel but different mean and variance per dimension.

    Args:
        gram_train_test (Array): The inner product matrix between the training and test data.
        train_prec_y (Array): The precision of the training data.
        y_mean (Array, optional): The mean of the training data output dimensions. Defaults to np.zeros(1).
        y_std (Array, optional): The standard deviation of the training data output dimensions. Defaults to np.ones((1, 1)).

    Returns:
        Array: The predictive means for the univariate GPs.
    """
    m = gp_predictive_mean_univ(gram_train_test, train_prec_y)
    return scale_and_shift_dims(m, y_mean, y_std)


def gp_predictive_cov(
    gram_train_test: Array,
    gram_test: Array,
    inv_gram_train: Array = None,
    chol_gram_train: Array = None,
    y_std: Array = np.ones((1, 1)),
) -> Array:
    """Compute the predictive covariance matrix for a GP.

    Args:
        gram_train_test (Array): Gram matrix train x test
        gram_test (Array): Gram matrix test x test
        inv_gram_train (Array, optional): Inverse of gram matrix train x train. Defaults to None.
        chol_gram_train (Array, optional): Cholesky of gram matrix train x train. Defaults to None.
        y_std (Array, optional): Standard deviations by which to scale in rows of a 2D array. Defaults to np.ones((1, 1)).

    Returns:
        Array: The predictive covariance matrix. Last dimension is scaling by y_std.
    """
    cov = gp_predictive_cov_univ(
        gram_train_test,
        gram_test,
        inv_gram_train=inv_gram_train,
        chol_gram_train=chol_gram_train,
    )
    return scale_dims(cov, y_std**2)


vm_cov = jax.vmap(gp_predictive_cov, (1, 0, None, None, None), 0)


def gp_predictive_var(
    gram_train_test: Array,
    gram_test: Array,
    inv_gram_train: Array = None,
    chol_gram_train: Array = None,
    y_std: Array = np.ones((1, 1)),
):
    """Compute the predictive variance for a GP.

    Args:
        gram_train_test (Array): Gram matrix train x test
        gram_test (Array): Gram matrix test x test
        inv_gram_train (Array, optional): Inverse of gram matrix train x train. Defaults to None.
        chol_gram_train (Array, optional): Cholesky of gram matrix train x train. Defaults to None.
        y_std (Array, optional): Standard deviations by which to scale in rows of a 2D array. Defaults to np.ones((1, 1)).

    Returns:
        Array: The predictive variance matrix. Last dimension is scaling by y_std.
    """
    vm_cov = jax.vmap(gp_predictive_cov, (1, 0, None, None, None), 0)

    return vm_cov(
        gram_train_test,
        np.diagonal(gram_test)[:, None, None],
        inv_gram_train,
        chol_gram_train,
        y_std,
    ).reshape((gram_test.shape[0], y_std.shape[1]))


def gp_predictive(
    gram_train_test: Array,
    gram_test: Array,
    chol_gram_train: Array,
    train_prec_y: Array,
    y_mean: Array = np.zeros(1),
    y_std: Array = np.ones(1),
    y_test: Array = None,
) -> Union[tuple[Array, Array], tuple[Array, Array, float]]:
    """Predictive mean and variance for a GP. if `y_test` is not None, the log marginal likelihood is also computed.

    Args:
        gram_train_test (Array): Gram matrix train x test
        gram_test (Array): Gram matrix test x test
        chol_gram_train (Array): Cholesky of gram matrix train x train
        train_prec_y (Array): Precision of training data
        y_mean (Array, optional): Mean of training data. Defaults to np.zeros(1).
        y_std (Array, optional): Standard deviation of training data. Defaults to np.ones(1).
        y_test (Array, optional): Test data. If not None, the log marginal likelihood is computed. Defaults to None.

    Returns:
        Union[tuple[Array, Array], tuple[Array, Array, float]]: Predictive mean and variance and potentially log marginal likelihood.
    """
    m = gp_predictive_mean_univ(gram_train_test, train_prec_y)
    cov = gp_predictive_cov_univ_chol(gram_train_test, gram_test, chol_gram_train)
    pred_m, pred_cov = scale_and_shift_dims(m, y_mean, y_std), scale_dims(cov, y_std**2)
    if y_test is None:
        return pred_m, pred_cov
    else:
        y_test2 = scale_and_shift_dims_inv(y_test, y_mean, y_std) - m
        cov_chol = sp.linalg.cholesky(cov, lower=True)
        return pred_m, pred_cov, gp_loglhood_mean0(y_test2, cov_chol)


def loglhood_loss(
    y_test: Array, pred_mean_y: Array, pred_cov_y: Array, loglhood_y: Array
) -> float:
    """Log likelihood loss.

    Args:
        y_test (Array): Test data (ground truth)
        pred_mean_y (Array): Predictive mean
        pred_cov_y (Array): Predictive covariance
        loglhood_y (Array): Log likelihood

    Returns:
        float: The log likelihood loss, simply returns loglhood_y.
    """
    return loglhood_y


# @jax.jit
def gp_val_loss(
    train_sel: Array,
    val_sel: Array,
    x_inner_x: Array,
    y: Array,
    noise: float,
    loss: Callable[[Array, Array, Array, Array], float] = loglhood_loss,
):
    x_train = train_sel @ x_inner_x @ train_sel.T
    x_train_val = train_sel @ x_inner_x @ val_sel.T
    x_val = val_sel @ x_inner_x @ val_sel.T
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_train = train_sel @ y
    y_val = val_sel @ y

    chol, y, prec_y, ymean, ystd, noise = gp_init(x_train, y_train, noise, True)

    pred = gp_predictive(
        x_train_val, x_val, chol, prec_y, y_mean=ymean, y_std=ystd, y_test=y_val
    )
    return loss(y_val, *pred)


vmap_gp_val_lhood = jax.jit(
    jax.vmap(partial(gp_val_loss, loss=loglhood_loss), (0, 0, None, None, None))
)


def gp_mlhood(train_sel: Array, val_sel: Array, x_inner_x: Array, y: Array, noise: float):
    x_train = train_sel @ x_inner_x @ train_sel.T
    x_train_val = train_sel @ x_inner_x @ val_sel.T
    x_val = val_sel @ x_inner_x @ val_sel.T
    if y.ndim == 1:
        y = y[:, np.newaxis]
    y_train = train_sel @ y
    y_val = val_sel @ y

    chol, y, prec_y, ymean, ystd = gp_init(x_train, y_train, noise, True)
    return gp_predictive(
        x_train_val, x_val, chol, prec_y, y_mean=ymean, y_std=ystd, y_test=y_val
    )[-1]


vmap_gp_mlhood = jax.jit(jax.vmap(gp_val_loss, (0, 0, None, None, None)))


def gp_cv_val_lhood(
    train_val_idcs: Array,
    x_inner_x: Array,
    y: Array,
    regul: float = None,
    vmapped_loss: Callable[
        [Array, Array, Array, Array, float], Array
    ] = vmap_gp_val_lhood,
):
    if y.ndim == 1:
        y = y[:, np.newaxis]
    train_idcs, val_idcs = train_val_idcs

    # val_sel = cv.idcs_to_selection_matr(len(inp), val_idcs)
    rval = vmapped_loss(
        cv.idcs_to_selection_matr(len(x_inner_x), train_idcs),
        cv.idcs_to_selection_matr(len(x_inner_x), val_idcs),
        x_inner_x,
        y,
        regul,
    )
    return rval.sum()


InputT = TypeVar("InputT")


class GP(Generic[InputT]):
    """Gaussian Process Regression class"""

    def __init__(
        self,
        enc: RkhsVecEncoder[InputT],
        x: InputT,
        y: Array,
        noise: float,
        normalize_y: bool = False,
    ):
        """Constructor for GP class

        Args:
            enc (RkhsVecEncoder): RkhsVecEncoder object that encodes the input data (e.g. x) into an RKHS vector
            x (InputT): Input data representing n input space points
            y (Array): Array of shape (n, 1) or (n,) containing the target values
            noise (float): Noise level/regularization parameter
            normalize_y (bool, optional): Whether to normalize the y values. Defaults to False.
        """
        self.enc = enc
        self.x_enc = enc(x)
        self.x_inner_x = self.x_enc.inner()
        self.chol, self.y, self.prec_y, self.ymean, self.ystd, self.noise = gp_init(
            self.x_inner_x, y, noise, normalize_y
        )

    def __str__(
        self,
    ) -> str:
        """Compute string representation

        Returns:
            str: String representation of GP object
        """
        return f"μ_Y = {self.ymean.squeeze()} ± σ_Y ={self.ystd.squeeze()}, σ_noise: {self.noise}, trace_chol: {self.chol.trace().squeeze()} trace_xx^t: {self.x_inner_x.trace().squeeze()}, {self.x_inner_x[0,:5]}, {self.x_inner_x.shape}"

    def marginal_loglhood(self) -> float:
        """Compute the marginal log-likelihood of the GP

        Returns:
            float: Marginal log-likelihood
        """
        return gp_loglhood_mean0(self.y, self.chol, self.prec_y)

    def predict(self, xtest: InputT, diag=True) -> tuple[Array, Array]:
        """Compute predictive mean and (co)variance of the GP

        Args:
            xtest (InputT): Input data representing m input space points
            diag (bool, optional): Whether to compute only the diagonal of the predictive covariance matrix. Defaults to True.

        Returns:
            tuple[Array, Array]: Predictive mean and variance or covariance matrix
        """
        xtest = self.enc(xtest)
        pred_m, pred_cov = gp_predictive(
            self.x_enc.inner(xtest),
            xtest.inner(),
            self.chol,
            self.prec_y,
            self.ymean,
            self.ystd,
        )
        if diag:
            return pred_m, np.diagonal(pred_cov).T
        return pred_m, pred_cov

    def post_pred_likelihood(self, xtest: FiniteVec, ytest: Array):
        """Compute the log-likelihood of the test data given the GP

        Args:
            xtest (FiniteVec): Test points given as a vector of RKHS points (i.e. a FiniteVec object)
            ytest (Array): Test targets

        Returns:
            tuple[Array, Array, float]: Predictive mean, variance and log marginal likelihood.
        """
        return gp_predictive(
            self.x_enc.inner(xtest),
            xtest.inner(),
            self.chol,
            self.prec_y,
            y_mean=self.ymean,
            y_std=self.ystd,
            y_test=ytest,
        )
