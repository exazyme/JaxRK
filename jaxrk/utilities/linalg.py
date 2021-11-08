from ..core.typing import Array
import jax.numpy as np
import numpy.random as random

def inv_blockmatr(P:Array, P_inv:Array, Q:Array, R:Array, S:Array) -> Array:
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