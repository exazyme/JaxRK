import jax.numpy as np, jax.scipy as sp
from jax import vmap, jit, pmap
from jax import random
from functools import partial
from jax.ops import index, index_update
from ..core.typing import PRNGKeyT



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

def zerofill_loo(inverted_subgrams:np.ndarray):
    """Fill row & column of inverted subgrams for leave-one-out scheme
    such that original size is retained

    Args:
        inverted_submatrices (np.ndarray): [description]
    """
    a = np.zeros([inverted_subgrams.shape[-1] + 1]*3)
    for i, gram in enumerate(inverted_subgrams):
        a = a.at[index[i, :i, :i]].set(gram[:i, :i])
        a = a.at[index[i, i + 1:, i + 1:]].set(gram[i:, i:])
        a = a.at[index[i, i + 1:, :i]].set(gram[i:, :i])
        a = a.at[index[i, :i, i + 1:]].set(gram[:i, i:])
    return a

def invert_loo(gram:np.ndarray, zerofill = True):
    """Invert square gram-submatrices for leave-one-out-scheme"""
    rval = vinvert(submatrices_loo(gram))
    if not zerofill:
        return rval
    return zerofill_loo(rval)

select_rows = jit(vmap(lambda sel, inp: sel@inp, (0, None)))
sym_matmul_fixed_inp = jit(vmap(lambda sel, inp: sel@inp@sel.T, (0, None)))
sym_matmul_variable_inp = jit(vmap(lambda sel, inp: sel@inp@sel.T,))

vmatmul_fixed_inp = jit(vmap(lambda sel, inp: sel@inp, (0, None)))
vmatmul_variable_inp = jit(vmap(lambda sel, inp: sel@inp,))


def loo_train_val(n_orig:int):
    val = np.arange(n_orig)
    train = np.array([np.delete(val, i) for i in val])
    return train, val.reshape(-1, 1)

def cv_train_val(n_orig:int, n_train:int, n_splits:int, rng:PRNGKeyT):
    p = vmap(random.permutation, (0, None))(random.split(rng, n_splits), n_orig)
    return p[:, :n_train], p[:, n_train:]


def idcs_to_selection_matr(n_orig:int, idcs:np.ndarray, idcs_sorted = False):
    if not idcs_sorted:
        idcs = np.sort(idcs, 1)
    rval = np.zeros((*idcs.shape, n_orig))
    for split, vi in enumerate(idcs):
        for r, c in enumerate(vi):
            rval = rval.at[index[split, r, c]].set(1.)
    return rval

def invert_submatr(gram:np.ndarray, train_idcs:np.ndarray, zerofill = True):
    train_sel_matr = idcs_to_selection_matr(gram.shape[0], train_idcs)
    rval = vinvert(sym_matmul_fixed_inp(train_sel_matr, gram))
    if zerofill:
        rval = sym_matmul_variable_inp(np.swapaxes(train_sel_matr, -1, -2), rval)
    return rval


def loo_I(n:int, stacked = False):
    I = np.eye(n)
    loo_I = np.array([np.delete(I, x, 0) for x in range(I.shape[0])])
    if stacked:
        return np.vstack(loo_I)
    else:
        return loo_I