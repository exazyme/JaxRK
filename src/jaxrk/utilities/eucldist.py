from typing import TypeVar
import jax.numpy as np
from jax import jit

# based on
# https://stackoverflow.com/questions/52030458/vectorized-spatial-distance-in-python-using-numpy

__all__ = ["eucldist"]


@jit
def sqeucldist_simple(a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """Compute squared Euclidean distance between two arrays. Simple version using einsum.

    Args:
        a (np.ndarray): First array with shape (n, d)
        b (np.ndarray, optional): Second array with shape (m, d), Defaults to None, in which case the distance is computed between the rows of a and itself.

    Returns:
        np.ndarray: Squared Euclidean distance between a and b with shape (n, m)
    """
    a_sumrows = np.einsum("ij,ij->i", a, a)
    if b is not None:
        b_sumrows = np.einsum("ij,ij->i", b, b)
    else:
        b = a
        b_sumrows = a_sumrows
    return a_sumrows[:, np.newaxis] + b_sumrows - 2 * a @ b.T


@jit
def sqeucldist_extension(a: np.ndarray, b: np.ndarray = None) -> np.ndarray:
    """Compute squared Euclidean distance between two arrays. Version using matrix extension.

    Args:
        a (np.ndarray): First array with shape (n, d)
        b (np.ndarray, optional): Second array with shape (m, d), Defaults to None, in which case the distance is computed between the rows of a and itself.

    Returns:
        np.ndarray: Squared Euclidean distance between a and b with shape (n, m)
    """
    A_sq = a**2

    if b is not None:
        B_sq = b**2
    else:
        b = a
        B_sq = A_sq

    nA, dim = a.shape
    nB = b.shape[0]

    A_ext = np.hstack([np.ones((nA, dim)), a, A_sq])
    B_ext = np.vstack([B_sq.T, -2.0 * b.T, np.ones((dim, nB))])
    return A_ext @ B_ext


def eucldist(
    a: np.ndarray, b: np.ndarray = None, power: float = 1.0, variant: str = "simple"
) -> np.ndarray:
    """Compute Euclidean distance between two arrays.

    Args:
        a (np.ndarray): First array with shape (n, d)
        b (np.ndarray, optional): Second array with shape (m, d), Defaults to None, in which case the distance is computed between the rows of a and itself.
        power (float, optional): Power of the euclidean distance to return. Defaults to 1.0.
        variant (str, optional): Variant of the euclidean distance to use. Defaults to "simple", options are "simple" and "extension".

    Returns:
        np.ndarray: Euclidean distance between a and b with shape (n, m)
    """
    if variant == "simple":
        sqdist = sqeucldist_simple(a, b)
    elif variant == "extension":
        sqdist = sqeucldist_extension(a, b)
    else:
        assert ()

    if power == 2:
        return sqdist
    else:
        return np.power(np.clip(sqdist, 0.0), power / 2.0)
