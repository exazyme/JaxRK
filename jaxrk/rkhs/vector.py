from copy import copy
from dataclasses import field
from functools import partial

from numpy.core.fromnumeric import squeeze
from time import time
from typing import Generic, TypeVar, List, Callable, Union

import jax
import jax.numpy as np
import numpy as onp
import scipy as osp
from jax import grad
from jax.numpy import dot, exp, log
from jax.scipy.special import logsumexp
from numpy.random import rand

from ..reduce import Reduce, LinearReduce, Prefactors, Sum, Mean, Scale, Center
from ..kern import Kernel
from ..utilities.gram import (
    rkhs_gram_cdist,
    rkhs_gram_cdist_ignore_const,
    gram_projection,
)
from ..core.typing import Array


from .base import Vec

# from ..utilities.frank_wolfe import frank_wolfe_pos_proj


class FiniteVec(Vec):
    """
    RKHS feature vector using input space points. This is the simplest possible vector.
    """

    def __init__(self, k: Kernel, insp_pts: Union[Vec, Array], reduce: List[Reduce] = []):
        """Constructor for FiniteVec.

        Args:
            k (Kernel): The kernel.
            insp_pts (Union[Vec, Array]): The input space points. These might themselves be RKHS elements if given as a Vec.
            reduce (List[Reduce], optional): The list of reductions to apply to the vector. Defaults to [].
        """
        super().__init__()
        if not isinstance(insp_pts, Vec):
            assert len(insp_pts.shape) == 2
        if reduce is None:
            self.reduce = []
        else:
            self.reduce = reduce
        self.k = k

        self.insp_pts = insp_pts
        self.__len = Reduce.final_len(len(self.insp_pts), self.reduce)
        self._raw_gram_cache = None

    def __eq__(self, other: "FiniteVec") -> bool:
        """Equality test.

        Args:
            other (FiniteVec): The other vector.

        Returns:
            bool: True if the vectors are equal.
        """
        assert False, "not yet implemented checking equality of reduce"
        return (
            isinstance(other, self.__class__)
            and np.all(other.insp_pts == self.insp_pts)
            and other.k == self.k
        )

    def __len__(self) -> int:
        """The length of the vector.

        Returns:
            int: The length of the vector.
        """
        return self.__len

    def __neg__(self):
        """Negation of the vector."""
        self.extend_reduce([Scale(-1)])

    def inner(self, Y: "FiniteVec" = None) -> Array:
        """Compute the inner product of the vector with another vector.

        Args:
            Y (FiniteVec, optional): The other vector. Defaults to None, in which case `Y = self`.

        Returns:
            Array: The inner product.
        """

        if Y is not None:
            assert self.k == Y.k
        else:
            Y = self
        gram = self.k(self.insp_pts, Y.insp_pts).astype(float)
        r1 = self.reduce_gram(gram, axis=0)
        r2 = Y.reduce_gram(r1, axis=1)
        return r2

    def normalized(self) -> "FiniteVec":
        """Normalize the vector so that it is a convex combination of RKHS elements.

        Returns:
            FiniteVec: The normalized vector.
        """
        r = self.reduce
        if isinstance(r[-1], Prefactors):
            p = r[-1].prefactors / np.sum(r[-1].prefactors)
            return self.updated(p)
        elif isinstance(r[-1], LinearReduce):
            p = r[-1].linear_map / np.sum(r[-1].linear_map, 1, keepdims=True)
            return self.updated(p)
        else:
            assert False
            p = np.ones(len(self)) / len(self)

    def updated(self, prefactors: np.ndarray) -> "FiniteVec":
        """Update the vector with new prefactors.

        Args:
            prefactors (np.ndarray): The new prefactors.

        Returns:
            FiniteVec: The updated vector.
        """
        _r = copy(self.reduce)
        previous_reduction = None
        if len(_r) > 0 and (
            isinstance(_r[-1], Prefactors) or isinstance(_r[-1], LinearReduce)
        ):
            final_len = Reduce.final_len(len(self.insp_pts), _r)
            assert (final_len == prefactors.shape[0]) or (
                final_len == 1 and len(prefactors.shape) == 1
            )
            previous_reduction = _r[-1]
            _r = _r[:-1]

        if len(prefactors.shape) == 1:
            assert previous_reduction is None or isinstance(
                previous_reduction, Prefactors
            )
            _r.append(Prefactors(prefactors))
        elif len(prefactors.shape) == 2:
            assert previous_reduction is None or isinstance(
                previous_reduction, LinearReduce
            )
            assert len(self) == prefactors.shape[0]
            _r.append(LinearReduce(prefactors))
        return FiniteVec(self.k, self.insp_pts, _r)

    def centered(self) -> "FiniteVec":
        """Centered version of the vector.

        Returns:
            FiniteVec: Centered vector.
        """
        return self.extend_reduce([Center()])

    def extend_reduce(self, r: List[Reduce]) -> "FiniteVec":
        """Extend the list of reductions.

        Args:
            r (List[Reduce]): The list of reductions to extend with.

        Returns:
            FiniteVec: The extended version of the vector.
        """
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return FiniteVec(self.k, self.insp_pts, _r)

    def reduce_gram(self, gram: np.ndarray, axis: np.uint = 0) -> np.ndarray:
        """Reduce the gram matrix.

        Args:
            gram (np.ndarray): The gram matrix.
            axis (np.uint, optional): The axis to reduce along. Defaults to 0.

        Returns:
            np.ndarray: The reduced gram matrix.
        """
        return Reduce.apply(gram, self.reduce, axis)

    def nsamps(self, mean=False) -> np.ndarray:
        """Number of samples, either for each element of the vector or their mean.

        Args:
            mean (bool, optional): Whether to return the mean number of samples. Defaults to False.

        Returns:
            np.ndarray: The number of samples.
        """
        n = len(self.insp_pts)
        rval = Reduce.apply(np.ones(n)[:, None] * n, self.reduce, 0)
        return rval.mean() if mean else rval

    def get_mean_var(self, keepdims=False) -> np.ndarray:
        """Get the mean and variance of the vector.

        Args:
            keepdims (bool, optional): Whether to keep the dimensions. Defaults to False.

        Returns:
            np.ndarray: The mean and variance.
        """
        mean = self.reduce_gram(self.insp_pts, 0)
        variance_of_expectations = self.reduce_gram(self.insp_pts**2, 0) - mean**2
        var = self.k.var + variance_of_expectations

        if keepdims:
            return (mean, var)
        else:
            return (np.squeeze(mean), np.squeeze(var))

    def sum(self) -> "FiniteVec":
        """Sum the elements of the vector.

        Returns:
            FiniteVec: The sum of the elements of the vector, i.e. a vector of length 1.
        """
        return self.extend_reduce([Sum().linearize((len(self),))])

    def mean(
        self,
    ) -> "FiniteVec":
        """Mean of the elements of the vector.

        Returns:
            FiniteVec: The mean of the elements of the vector, i.e. a vector of length 1.
        """
        return self.extend_reduce([Mean().linearize((len(self),))])

    @classmethod
    def construct_RKHS_Elem(
        cls, kern: Kernel, inspace_points: np.ndarray, prefactors: np.ndarray = None
    ) -> "FiniteVec":
        """Construct a length 1 RKHS vector that is a linear combination of RKHS elements.

        Args:
            kern (Kernel): The kernel.
            inspace_points (np.ndarray): The points in the input space.
            prefactors (np.ndarray, optional): The prefactors. Defaults to None, in which case the prefactors are set to 1.

        Returns:
            FiniteVec: The RKHS element / length 1 RKHS vector.
        """
        if prefactors is None:
            prefactors = np.ones(len(inspace_points))
        else:
            assert len(prefactors.squeeze().shape) == 1
        return FiniteVec(
            kern, inspace_points, [LinearReduce(prefactors.squeeze()[None, :])]
        )

    @classmethod
    def construct_RKHS_Elem_from_estimate(
        cls,
        kern: Kernel,
        inspace_points: np.ndarray,
        estimate: str = "support",
        unsigned: bool = True,
        regul: float = 0.1,
    ) -> "FiniteVec":
        """Construct a length 1 RKHS vector by estimation from a set of distribution samples in the input space.

        Args:
            kern (Kernel): The kernel.
            inspace_points (np.ndarray): The samples in the input space.
            estimate (str, optional): The type of estimate. Defaults to "support", in which case the estimate is the support of the distribution.
            unsigned (bool, optional): Whether to use the unsigned version of the estimate. Defaults to True.
            regul (float, optional): The regularization parameter. Defaults to 0.1.

        Returns:
            FiniteVec: The RKHS element / length 1 RKHS vector representing the estimate.
        """

        prefactors = distr_estimate_optimization(kern, inspace_points, est=estimate)
        return FiniteVec(
            kern, inspace_points, prefactors, points_per_split=len(inspace_points)
        )

    @property
    def _raw_gram(self):
        """The raw gram matrix of this vector with itself, i.e. without any reductions applied."""
        if self._raw_gram_cache is None:
            self._raw_gram_cache = self.k(self.insp_pts).astype(np.float64)
        return self._raw_gram_cache

    def point_representant(
        self, method: str = "inspace_point", keepdims: bool = False
    ) -> np.ndarray:
        """Get a point representant of the vector.

        Args:
            method (str, optional): The method to use. If `mean`, return the mean of the distribution defined by the RKHS element under the assumption that a DensityKernel is used. Defaults to "inspace_point", in which case the point representant is the point in the input space whoms feature map is closest to the vector in RKHS norm.
            keepdims (bool, optional): Whether to keep the dimensions. Defaults to False.

        Returns:
            np.ndarray: The point representant.
        """
        if method == "inspace_point":
            assert isinstance(self.reduce[-1], Prefactors) or isinstance(
                self.reduce[-1], LinearReduce
            )
            G_orig_repr = Reduce.apply(self._raw_gram, self.reduce, 1)
            repr_idx = gram_projection(
                G_orig_repr,
                Reduce.apply(G_orig_repr, self.reduce, 0),
                self._raw_gram,
                method="representer",
            ).squeeze()
            rval = self.insp_pts[repr_idx, :]
        elif method == "mean":
            rval = self.get_mean_var(keepdims=keepdims)[0]
        else:
            assert False, "No known method selected for point_representant"
        if not keepdims:
            return rval.squeeze()
        else:
            return rval

    def pos_proj(self, nsamps: int = None) -> "FiniteVec":
        """Project to RKHS element with purely positive prefactors. Assumes `len(self) == 1`.

        Args:
            nsamps (int, optional): Number of input space points. Defaults to None, in which case the input space points of self are reused.

        Returns:
            FiniteVec: The result of the projection.
        """
        assert len(self) == 1
        if nsamps is None:
            lin_map = gram_projection(
                Reduce.apply(self._raw_gram, self.reduce, 0),
                G_repr=self._raw_gram,
                method="pos_proj",
            )
            return FiniteVec(self.k, self.insp_pts, [LinearReduce(lin_map)])
        else:
            assert False, "Frank-Wolfe needs attention."
            # the problem are circular imports.

            # return frank_wolfe_pos_proj(self,
            # self.updated(pos_proj(self.inspace_points, self.prefactors,
            # self.k)), nsamps - self.inspace_points.shape[0])

    def dens_proj(self, nsamps: int = None) -> "FiniteVec":
        """Project to an RKHS object that is also a density in the usual sense. In particular, a projection to positive prefactors and then normalization so prefactors sum to 1.

        Returns:
            FiniteVec: The result of the projection
        """
        return self.normalized().pos_proj(nsamps).normalized()

    def __call__(self, argument: np.ndarray) -> np.ndarray:
        """Evaluate the RKHS elements at points in the input space.

        Args:
            argument (np.ndarray): The points in the input space.

        Returns:
            np.ndarray: The values of the RKHS elements at the points in the input space.
        """
        return self.inner(FiniteVec(self.k, argument, []))


def distr_estimate_optimization(
    kernel: Kernel, sampled_pts: np.ndarray, est: str = "support"
):
    """Distribution estimate by optimization.

    Args:
        kernel (Kernel): Kernel to use.
        sampled_pts (np.ndarray): The input space points sampled from a distribution of interest.
        est (str, optional): The type of estimate. Defaults to "support", in which case the estimate is the support of the distribution. The other option is "density", in which case the estimate is the density of the distribution from which the samples were drawn.
    """

    def __casted_output(function):
        return lambda x: onp.asarray(function(x), dtype=np.float64)

    G = kernel(sampled_pts).astype(np.float64)

    if est == "support":
        # solution evaluated in support points should be positive constant
        def cost(f):
            return np.abs(dot(f, G) - 1).sum()

    elif est == "density":
        # minimum negative log likelihood of support_points under solution
        def cost(f):
            return -log(dot(f, G)).sum()

    bounds = [(0.0, None)] * len(sampled_pts)

    res = osp.optimize.minimize(
        __casted_output(cost),
        rand(len(sampled_pts)) + 0.0001,
        jac=__casted_output(grad(cost)),
        bounds=bounds,
    )

    return res["x"] / res["x"].sum()


VrightT = TypeVar("VrightT", bound=Vec)
VleftT = TypeVar("VleftT", bound=Vec)


class CombVec(Vec, Generic[VrightT, VleftT]):
    """Combination of two vectors by applying an operation like addition or multiplication to the gram matrices resulting from the two vectors."""

    def __init__(
        self, vR: VrightT, vL: VleftT, operation: Callable, reduce: List[Reduce] = []
    ):
        """Constructor.

        Args:
            vR (VrightT): The right vector.
            vL (VleftT): The left vector.
            operation (Callable): The operation to apply to the gram matrices.
            reduce (List[Reduce], optional): The reduction to apply to the gram matrix resulting from combination of the original gram matrices. Defaults to [].
        """
        super().__init__()
        assert len(vR) == len(vL)
        self.vR, self.vL = vR, vL
        self.reduce = reduce
        self.operation = operation
        self.__len = Reduce.final_len(len(self.vR), reduce)

    def reduce_gram(self, gram: np.ndarray, axis: np.uint = 0) -> np.ndarray:
        """Reduce the gram matrix.

        Args:
            gram (np.ndarray): The raw (unreduced) gram matrix.
            axis (np.uint, optional): The axis to reduce. Defaults to 0.

        Returns:
            np.ndarray: The reduced gram matrix.
        """
        return Reduce.apply(gram, self.reduce, axis)

    # @partial(jax.jit, static_argnums=(0, 1))
    def inner(
        self, Y: "CombVec[VrightT, VleftT]" = None, diag: bool = False
    ) -> np.ndarray:
        """Inner product of two vectors.

        Args:
            Y (CombVec[VrightT, VleftT], optional): The other vector. Defaults to None, in which case `Y = self`.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            np.ndarray: The inner product.
        """
        if Y is None:
            Y = self
        else:
            assert Y.operation == self.operation
        rval = self.reduce_gram(
            Y.reduce_gram(self.operation(self.vR.inner(Y.vR), self.vL.inner(Y.vL)), 1),
            0,
        )
        if diag:
            return np.diagonal(rval)

        return rval

    @partial(jax.jit, static_argnums=(0))
    def diag_inner(
        self,
    ) -> np.ndarray:
        """Diagonal of the inner product of this vector with itself.

        Returns:
            np.ndarray: Diagonal of the inner product.
        """
        rval = self.reduce_gram(
            self.reduce_gram(self.operation(self.vR.inner(), self.vL.inner()), 1), 0
        )
        return np.diagonal(rval)

    def extend_reduce(self, r: List[Reduce]) -> "CombVec":
        """Copy of this vector with additional reductions.

        Args:
            r (List[Reduce]): The additional reductions.

        Returns:
            CombVec: The copy with additional reductions.
        """
        if r is None or len(r) == 0:
            return self
        else:
            _r = copy(self.reduce)
            _r.extend(r)
            return CombVec(self.vR, self.vL, self.operation, _r)

    def centered(self) -> "CombVec":
        """Centered version of this vector.

        Returns:
            CombVec: The centered vector.
        """
        return self.extend_reduce([Center()])

    def __len__(self) -> int:
        """Length of this vector.

        Returns:
            int: The length.
        """
        if self.reduce is None:
            return len(self.vR)
        else:
            return self.__len

    def updated(self, prefactors) -> "CombVec":
        """Update the vector with new prefactors.

        Args:
            prefactors (np.ndarray): The new prefactors.

        Returns:
            CombVec: The updated vector.

        Raises:
            NotImplementedError: This method is not implemented for this class.
        """
        raise NotImplementedError()
