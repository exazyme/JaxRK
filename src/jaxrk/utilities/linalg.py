from ..core.typing import Array
from ..rkhs.base import Vec
from typing import TypeVar, Union
import jax.numpy as np


def outer(a: Union[Vec, Array], b: Union[Vec, Array] = None) -> Array:
    """Compute the outer product of two vectors, which can be either RKHS vectors or JAX numpy arrays. If both vectors are vectors of RKHS elements, the outer product is computed using the RKHS inner product of all pairs of elements. Otherwise, the outer product is computed using the JAX numpy outer product.

    Args:
        a (Union[Vec, Array]): The first vector
        b (Union[Vec, Array], optional): The second vector. Defaults to None, in which case the inner product is computed with X with itself.

    Returns:
        Array: The outer product
    """
    if isinstance(a, Vec):
        return a.inner(b)
    else:
        return a @ b.T


def block_matrix(UL: Array, UR: Array, LL: Array, LR: Array) -> Array:
    """Correctly stack a block matrix from its parts.

    Args:
        UL (Array): Upper left part (shape N x M) of the resulting matrix.
        UR (Array): Upper right part (shape N x L) of the resulting matrix.
        LL (Array): Upper left part (shape K x M)of the resulting matrix.
        LR (Array): Upper right part (shape K x L) of the resulting matrix.

    Returns:
        Array: Resulting matrix from stacking parts, shape (N+K) x (M+L).
    """
    return np.vstack([np.hstack([UL, UR]), np.hstack([LL, LR])])


def augment_gram(G_old: Array, G_old_new: Array, G_new: Array) -> Array:
    """Correctly stack a gram matrix from its parts of old and new kernel evaluations.

    Args:
        G_old (Array): Gram matrix involving only old data points (shape N x N).
        G_old_new (Array): Gram matrix involving old data and new data points (shape N x M).
        G_new (Array): Gram matrix involving only new data points (shape M x M).

    Returns:
        Array: Resulting gram matrix from stacking parts, shape (N+M) x (N+M).
    """
    return block_matrix(G_old, G_old_new, G_old_new.T, G_new)


def inv_blockmatr(P: Array, P_inv: Array, Q: Array, R: Array, S: Array) -> Array:
    """Given P and P^{-1}, compute the inverse of the block-partitioned matrix
    P Q
    R S
    and return it. Based on Woodbury, Sherman & Morrison formula.

    Args:
        P (Array): Upper left matrix.
        P_inv (Array): Inverse of upper left matrix.
        Q (Array): Upper right matrix.
        R (Array): Lower left matrix.
        S (Array): Lower right matrix.

    Returns:
        Array: Inverse of the block matrix [[P, Q], [R, S]]
    """
    S_ = np.linalg.inv(S - R @ P_inv @ Q)
    R_ = -S_ @ R @ P_inv
    Q_ = -P_inv @ Q @ S_
    P_ = P_inv + P_inv @ Q @ S_ @ R @ P_inv

    return np.vstack([np.hstack([P_, Q_]), np.hstack([R_, S_])])


# FIXME: implement Cholesky up/downdates: see
# https://en.wikipedia.org/wiki/Cholesky_decomposition#Adding_and_removing_rows_and_columns
# and
# https://math.stackexchange.com/questions/955874/cholesky-factor-when-adding-a-row-and-column-to-already-factorized-matrix
