import jax.numpy as np, jax.scipy as sp
from jax import vmap, jit
from functools import partial



vinvert = jit(vmap(np.linalg.inv))

def delete_symmetric(gram:np.ndarray, i):
    """Delete i'th row and column from a square matrix"""
    return np.delete(np.delete(gram, i, 0), i, 1)

def delete_rc(m:np.ndarray, i:int):
    """Returns tuple of matrices with row, col of index i deleted

    Args:
        m (np.ndarray): Input array, expected to have at least i+1 rows and cols
        i (int): The row & col to delete
    """
    return np.delete(m, i, 0), np.delete(m, i, 1)


def submatrices_loo(gram:np.ndarray):
    """Compute square submatrices from a square gram matrix for leave-one-out-scheme"""
    d = partial(delete_symmetric, gram)
    return np.array([d(i) for i in range(gram.shape[0])])

def invert_loo(gram:np.ndarray):
    """Invert square gram-submatrices for leave-one-out-scheme"""
    return sp.linalg.block_diag(*vinvert(submatrices_loo(gram)))

def loo_I(n:int):
    I = np.eye(n)
    loo_I = np.array([np.delete(I, x, 0) for x in range(I.shape[0])])
    rloo_I = np.vstack(loo_I)
    cloo_I = np.hstack(np.swapaxes(loo_I, 1, 2))
    return rloo_I, cloo_I