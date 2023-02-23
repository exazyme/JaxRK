import jax.numpy as np
from typing import TypeVar
from ..core.typing import Array
from ..rkhs.base import Vec
from .gram import rkhs_gram_cdist
from .eucldist import eucldist
import jax

__all__ = ["dist", "median_heuristic", "outer"]


def median_heuristic(data: any, distance: callable, per_dimension: bool = True) -> float:
    """Compute the median heuristic for a given distance function, resulting in a kernel width

    Args:
        data (any): Data to compute the median heuristic for
        distance (callable): Distance function to use
        per_dimension (bool, optional): Whether to compute the median heuristic per dimension. Defaults to True.

    Returns:
        float: Kernel width for the given distance function
    """

    import numpy as onp
    from scipy.spatial.distance import pdist

    if isinstance(distance, str):

        def dist_fn(x):
            """Wrapper for pdist"""
            return pdist(x, distance)

    else:
        dist_fn = distance
    if per_dimension is False:
        return onp.median(dist_fn(data))
    else:

        def single_dim_heuristic(data_dim):
            """Compute the median heuristic for a single dimension"""
            return median_heuristic(data_dim[:, None], dist_fn, per_dimension=False)

        return onp.apply_along_axis(single_dim_heuristic, 0, data)


T = TypeVar("T", Vec, Array)


def rkhs_cdist(a: Vec, b: Vec = None, power: float = 2.0):
    """Compute RKHS distances between RKHS elements in vectors a and b

    Args:
        a (Vec): Vector to compute distances from.
        b (Vec, optional): Vector to compute distances to. Defaults to None, in which case the function returns the distances between all elements of a.
        power (float, optional): The power to raise the distance to. Defaults to 2.

    Returns:
        [type]: [description]
    """
    if b is None:  # or a == b
        return rkhs_gram_cdist(a.inner(), power=power)
    else:
        return rkhs_gram_cdist(a.inner(b), a.inner(), b.inner(), power=power)


def dist(a: T, b: T = None, power: float = 2.0) -> Array:
    """Compute distances between elements in vectors a and b, which can be either RKHS vectors or JAX numpy arrays.
    If a and b are RKHS vectors, the distances are computed using the RKHS inner product, otherwise the Euclidean distance is used.

    Args:
        a (T): Vector a to compute distances from.
        b (T, optional): Vector b to compute distances to. Defaults to None, in which case the function returns the distances between all elements of a.
        power (float, optional): Power to raise the distance to. Defaults to 2.

    Returns:
        Array: Distances between elements in a and b
    """
    if isinstance(a, Vec):
        dfunc = rkhs_cdist
    else:
        dfunc = eucldist
    return dfunc(a, b, power=power)


def outer(a: T, b: T = None) -> Array:
    """Compute the outer product of two vectors, which can be either RKHS vectors or JAX numpy arrays. If both vectors are vectors of RKHS elements, the outer product is computed using the RKHS inner product of all pairs of elements. Otherwise, the outer product is computed using the JAX numpy outer product.

    Args:
        a (T): The first vector
        b (T, optional): The second vector. Defaults to None, in which case the inner product is computed with X with itself.

    Returns:
        Array: The inner product
    """
    if isinstance(a, Vec):
        return a.inner(b)
    else:
        return a @ b.T
