from jax import grad
import jax.numpy as np
import scipy as osp
from numpy.random import rand
from typing import Callable, Any

__all__ = [
    "rkhs_gram_cdist",
    "rkhs_gram_cdist_ignore_const",
    "choose_representer",
    "choose_representer_from_gram",
    "gram_projection",
]


def rkhs_gram_cdist(
    G_ab: np.ndarray, G_a: np.ndarray = None, G_b: np.ndarray = None, power: float = 2.0
) -> np.ndarray:
    """Compute RKHS distances between RKHS elements a and b given only by their Gram matrices

    Args:
        G_ab (np.ndarray): Cross Gram matrix between a and b
        G_a (np.ndarray, optional): Gram matrix of a. If None, it is assumed that a = b and G_a = G_b = G_ab. Defaults to None.
        G_b (np.ndarray, optional): Gram matrix of b. If None, it is assumed that a = b and G_a = G_b = G_ab. Defaults to None.
        power (float, optional): The power to raise the distance to. Defaults to 2.

    Returns:
        np.ndarray: RKHS distances between a and b
    """
    assert len(G_ab.shape) == 2
    if G_a is not None:
        assert (
            len(G_a.shape) == 2
            and G_a.shape[0] == G_a.shape[1]
            and G_ab.shape[0] == G_a.shape[0]
        )
        assert (
            G_ab.shape[0] % G_a.shape[1] == 0
        ), "Shapes of Gram matrices do not broadcast"
    if G_b is not None:
        assert G_b.shape[0] == G_b.shape[1] and G_ab.shape[1] == G_b.shape[1]
        assert (
            G_ab.shape[1] % G_b.shape[0] == 0
        ), "Shapes of Gram matrices do not broadcast"
    if G_a is None or G_b is None:
        assert (
            G_a is None and G_b is None
        ), "Either none or both of G_a, G_b should be None"
        assert np.all(G_ab == G_ab.T)
        # representer
        G_a = G_b = G_ab
    return rkhs_gram_cdist_unchecked(G_ab, G_a, G_b, power)


def rkhs_gram_cdist_unchecked(
    G_ab: np.ndarray, G_a: np.ndarray, G_b: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """Compute RKHS distances between RKHS elements a and b given only by their Gram matrices.
    This function does not perform any checks on the input.

    Args:
        G_ab (np.ndarray): Cross Gram matrix between a and b
        G_a (np.ndarray, optional): Gram matrix of a.
        G_b (np.ndarray, optional): Gram matrix of b.
        power (float, optional): The power to raise the distance to. Defaults to 2.

    Returns:
        np.ndarray: RKHS distances between a and b
    """
    sqdist = np.diagonal(G_a)[:, np.newaxis] + np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.0:
        return sqdist
    else:
        return np.power(sqdist, power / 2.0)


def rkhs_gram_cdist_ignore_const(
    G_ab: np.ndarray, G_b: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """Compute RKHS distances between RKHS elements a and b given only by their Gram matrices.
    This function does not perform any checks on the input and ignores the constant term in the distance related to the Gram matrix of a.

    Args:
        G_ab (np.ndarray): Cross Gram matrix between a and b
        G_b (np.ndarray, optional): Gram matrix of b.
        power (float, optional): The power to raise the distance to. Defaults to 2.

    Returns:
        np.ndarray: RKHS distances between a and b
    """
    sqdist = np.diagonal(G_b)[np.newaxis, :] - 2 * G_ab
    if power == 2.0:
        return sqdist
    else:
        return np.power(sqdist, power / 2.0)


def choose_representer(support_points: Any, factors: np.ndarray, kernel: Callable) -> int:
    """Choose a representer from a set of support points given their support points, a factors/weights per support point and a kernel

    Args:
        support_points (Any): Support points
        factors (np.ndarray): Factors/weights per support point
        kernel (Callable): Kernel function

    Returns:
        int: Index of the chosen representer
    """
    return choose_representer_from_gram(
        kernel(support_points).astype(np.float64), factors
    )


def choose_representer_from_gram(G: np.ndarray, factors: np.ndarray) -> int:
    """Choose a representer from a set of support points given by their Gram matrix and a set of factors/weights per support point

    Args:
        G (np.ndarray): Gram matrix of the support points
        factors (np.ndarray): Factors/weights per support points to be applied to the Gram matrix

    Returns:
        int: Index of the chosen representer
    """
    fG = np.dot(factors, G)
    rkhs_distances_sq = (np.dot(factors, fG).flatten() + np.diag(G) - 2 * fG).squeeze()
    rval = np.argmin(rkhs_distances_sq)
    assert rval < rkhs_distances_sq.size
    return rval


def __casted_output(function: callable) -> callable:
    """Cast the output of a function to np.float64

    Args:
        function (callable): Original function

    Returns:
        callable: Function with casted output
    """
    return lambda x: np.asarray(function(x), dtype=np.float64)


def gram_projection(
    G_orig_repr: np.ndarray,
    G_orig: np.ndarray = None,
    G_repr: np.ndarray = None,
    method: str = "representer",
) -> np.ndarray:
    """Project a set of support points onto a new set of support points, both given by their (cross) Gram matrices

    Args:
        G_orig_repr (np.ndarray): Cross Gram matrix between the original and the new support points
        G_orig (np.ndarray, optional): Gram matrix of the original support points. Defaults to None.
        G_repr (np.ndarray, optional): Gram matrix of the new support points. Defaults to None.
        method (str, optional): Choice between "representer" (best representation) and "pos_proj" (positive prefactors). Defaults to "representer".

    Returns:
        np.ndarray: Prefactors for the new support points. Positivity constraints are applied if method is "pos_proj".
    """
    if method == "representer":
        return np.argmin(rkhs_gram_cdist(G_orig_repr, G_repr, G_orig), 0)
    elif method == "pos_proj":
        assert G_repr is not None
        s = G_orig_repr.shape
        n_pref = np.prod(np.ndarray(s))

        def cost(M):
            """Cost function to be minimized for the positive projection based on RKHS distances"""
            M = M.reshape(s)
            return np.trace(
                rkhs_gram_cdist_ignore_const(G_orig_repr @ M.T, M @ G_repr @ M.T)
            )

        res = osp.optimize.minimize(
            __casted_output(cost),
            rand(n_pref) + 0.0001,
            jac=__casted_output(grad(cost)),
            bounds=[(0.0, None)] * n_pref,
        )
        return res["x"].reshape(s)
    else:
        assert False, "No valid method selected"
