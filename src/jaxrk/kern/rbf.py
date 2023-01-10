from typing import Callable, Tuple, Union, Optional
from ..core.typing import PRNGKeyT, Shape, Dtype, Array, ConstOrInitFn
from functools import partial
from ..core.constraints import NonnegToLowerBd, Bijection, SoftBound, SquashingToBounded
from dataclasses import dataclass

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.random import PRNGKey

# from jax.experimental.checkify import check, checkify

from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

from ..utilities.eucldist import eucldist
from ..kern.base import DensityKernel, Kernel
from ..kern.util import ScaledPairwiseDistance, SimpleScaler


class GenGaussKernel(DensityKernel):  # this is the gennorm distribution from scipy
    """Kernel derived from the pdf of the generalized Gaussian distribution (https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1)."""

    def __init__(self, dist: ScaledPairwiseDistance):
        """Initialize GenGaussKernel.

        Args:
            dist (ScaledPairwiseDistance): Pairwise distance function.
        """
        super().__init__()
        self.dist = dist
        self.nconst = self.dist.power / (
            2 * self.dist._scale * np.exp(sp.special.gammaln(1.0 / self.dist.power))
        )

    def __str__(self) -> str:
        """Return a string representation of the kernel.

        Returns:
            str: String representation of the kernel.
        """
        return f"GenGaussKernel({self.dist})"

    @classmethod
    def make_unconstr(
        cls,
        unconstr_scale: Array,
        unconstr_shape: float,
        scale_bij: Bijection = NonnegToLowerBd(),
        shape_bij: Bijection = SquashingToBounded(0.0, 2.0),
    ) -> "GenGaussKernel":
        """Factory for constructing a GenGaussKernel from unconstrained parameters.
        The constraints for each parameters are then guaranteed by applying their accompanying bijections.

        Args:
            scale (Array): Scale parameter, unconstrained.
            shape (float): Shape parameter, unconstrained. Lower values result in pointier kernel functions.
            scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
            shape_bij (Bijection): Bijection mapping from unconstrained real numbers to half-open interval (0,2]. Defaults to Sigmoid(0., 2.).
        """
        return cls.make(scale_bij(unconstr_scale), shape_bij(unconstr_shape))

    @classmethod
    def init_from_constrained(
        cls,
        constr_scale: Array,
        constr_shape: float,
        scale_bij: Bijection = NonnegToLowerBd(),
        shape_bij: Bijection = SquashingToBounded(0.0, 2.0),
    ):
        """Return unconstrained init values from parameters by applying the inverse of the bijection that belongs to the parameter.

        Args:
            constr_scale (Array): _description_
            constr_shape (float): _description_
            scale_bij (Bijection, optional): _description_. Defaults to NonnegToLowerBd().
            shape_bij (Bijection, optional): _description_. Defaults to SquashingToBounded(0.0, 2.0).

        Returns:
            _type_: _description_
        """
        return scale_bij.inv(constr_scale), shape_bij.inv(constr_shape)

    @classmethod
    # @checkify
    def make(cls, length_scale: Array, shape: float) -> "GenGaussKernel":
        """Method for constructing a GenGaussKernel from scale and shape parameters.
        Args:
            scale (Array): Scale parameter, nonnegative.
            shape (float): Shape parameter, in half-open interval (0,2]. Lower values result in pointier kernel functions. Shape == 2 results in usual Gaussian kernel, shape == 1 results in Laplace kernel.
        """
        # check(shape > 0, "Shape parameter must be in (0,2].")
        # check(shape <= 2, "Shape parameter must be in (0,2].")
        dist = ScaledPairwiseDistance(
            scaler=SimpleScaler(1.0 / length_scale), power=shape
        )
        return GenGaussKernel(dist)

    @classmethod
    def make_laplace(cls, length_scale: Array) -> "GenGaussKernel":
        """Method for constructing a Laplace kernel from scale parameter.
        Args:
            scale (Array): Scale parameter, nonnegative.
        """
        return GenGaussKernel.make(length_scale, 1.0)

    @classmethod
    def make_gauss(cls, length_scale: Array) -> "GenGaussKernel":
        """Method for constructing a Laplace kernel from scale parameter.
        Args:
            scale (Array): Scale parameter, nonnegative.
        """
        # f = sp.special.gammaln(np.array([3, 1]) / 2)
        # f = np.exp((f[0] - f[1]) / 2)
        f = 0.70710695

        return GenGaussKernel.make(length_scale / f, 2.0)

    @property
    def std(self):
        """Standard deviation of the kernel."""
        return np.sqrt(self.var)

    @property
    def var(self):
        """Variance of the kernel."""
        f = sp.special.gammaln(np.array([3, 1]) / self.dist.power)
        return self.dist._scale**2 * np.exp(f[0] - f[1])

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        diag: bool = False,
    ) -> np.ndarray:
        """Evaluate the kernel function.

        Args:
            X (np.ndarray): First input array.
            Y (np.ndarray, optional): Second input array. Defaults to None, in which case Y = X.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            np.ndarray: Gram matrix or its diagonal.
        """
        return self.nconst * exp(-self.dist(X, Y, diag))


class PeriodicKernel(Kernel):
    """Periodic kernel class. A periodic kernel is defined by
    exp(-2 * (sin(dist(X, Y, diag)) / length_scale)**power)"""

    def __init__(self, period: Union[float, Array], length_scale: float):
        """Constructor for PeriodicKernel.

        Args:
            period (Union[float, Array]): Period, this is used as a scaling parameter inside the distance computation.
            length_scale (float): Length scale.
        """
        super().__init__()
        # check(period > 0 and length_scale > 0, "Period and length scale must be positive.")
        self.dist = ScaledPairwiseDistance(scaler=SimpleScaler(1.0 / period), power=1.0)
        self.ls = length_scale

    @classmethod
    def make_unconstr(
        cls,
        unconstr_period: float,
        unconstr_length_scale: float,
        period_bij: Bijection = NonnegToLowerBd(),
        length_scale_bij: Bijection = NonnegToLowerBd(),
    ) -> "PeriodicKernel":
        """Factory for constructing a PeriodicKernel from unconstrained parameters.
           The constraints for each parameters are then guaranteed by applying their accompanying bijections.
            Args:
                period (float): Scale parameter, unconstrained.
                length_scale (float): Lengscale parameter, unconstrained.
                period_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
                length_scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.

        Returns:
            PeriodicKernel: The constructed kernel.
        """
        return cls(period_bij(unconstr_period), length_scale_bij(unconstr_length_scale))

    @classmethod
    def init_from_constrained(
        cls,
        constr_period: float,
        constr_length_scale: float,
        period_bij: Bijection = NonnegToLowerBd(),
        length_scale_bij: Bijection = NonnegToLowerBd(),
    ) -> tuple[float, float]:
        """Return unconstrained init values from parameters by applying the inverse of the bijection
        that belongs to the parameter.

        Args:
            constr_period (float): Constrained period parameter.
            constr_length_scale (float): Constrained length scale parameter.
            period_bij (Bijection, optional): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
            length_scale_bij (Bijection, optional): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.

        Returns:
            tuple[float, float]: Unconstrained init values for period and length scale.
        """
        return period_bij.inv(constr_period), length_scale_bij.inv(constr_length_scale)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        diag: bool = False,
    ) -> np.ndarray:
        """Evaluate the kernel function.

        Args:
            X (np.ndarray): First input array.
            Y (np.ndarray, optional): Second input array. Defaults to None, in which case Y = X.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            np.ndarray: Gram matrix or its diagonal.
        """
        d = self.dist(X, Y, diag)
        return exp(-2 * (np.sin(np.pi * d) / self.ls) ** 2.0)


class ThreshSpikeKernel(Kernel):
    """Thresholded spike kernel class. A thresholded spike kernel is defined by
    spike if dist(X, Y) < threshold else non_spike"""

    def __init__(
        self,
        dist: ScaledPairwiseDistance,
        spike: float,
        non_spike: float,
        threshold: float,
    ):
        """Constructor for ThreshSpikeKernel. Takes value `spike` if distance is below threshold, otherwise `non_spike`.

        Args:
            spike (float): Kernel value when distance between input points is below threshold_distance, nonnegative.
            non_spike (float): Kernel value when distance between input points is above threshold_distance. Has to satisfy abs(non_spike) < spike.
            threshold_distance (float): Distance threshold.
        """
        super().__init__()
        # check(spike > 0, "Spike value must be positive.")
        # check(abs(non_spike) < spike, "Non-spike value must be smaller in absolute value than spike value.")
        # check(threshold >= 0, "Threshold must be nonnegative.")
        self.dist = dist
        self.spike, self.non_spike, self.threshold = spike, non_spike, threshold

    @classmethod
    def make_unconstr(
        cls,
        length_scale: Array,
        shape: float,
        spike: float,
        non_spike: float,
        threshold: float,
        scale_bij: Bijection = NonnegToLowerBd(),
        shape_bij: Bijection = NonnegToLowerBd(),
        spike_bij: Bijection = NonnegToLowerBd(),
        non_spike_bij: Bijection = SoftBound(u=1.0),
        threshold_bij: Bijection = SoftBound(l=0.0),
    ) -> "ThreshSpikeKernel":
        """Factory for constructing a PeriodicKernel from unconstrained parameters.
        Args:
            scale (Array): Scale parameter for distance computation.
            shape (float): Shape parameter for distance computation.
            spike (float): Kernel value when distance between input points is above threshold_distance.
            non_spike (float): Non-spike value, has to satisfy abs(non_spike) < spike
            threshold (float): Below theshold
            scale_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
            shape_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
            spike_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftPlus.
            non_spike_bij (Bijection): Bijection mapping from unconstrained real numbers to numbers smaller than 1. Defaults to SoftBd(upper_bound = 1.).
            threshold_bij (Bijection): Bijection mapping from unconstrained real numbers to non-negative numbers. Defaults to SoftBd(lower_bound = 0.).
        Returns:
            ThreshSpikeKernel: Spike kernel.
        """
        dist = ScaledPairwiseDistance(
            scaler=SimpleScaler(1.0 / scale_bij(length_scale)), power=shape_bij(shape)
        )
        return cls(
            dist, spike_bij(spike), non_spike_bij(non_spike), threshold_bij(threshold)
        )

    def __call__(
        self, X: np.ndarray, Y: np.ndarray = None, diag: bool = False
    ) -> np.ndarray:
        """Evaluate the kernel function.

        Args:
            X (np.ndarray): First input array.
            Y (np.ndarray, optional): Second input array. Defaults to None, in which case Y = X.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            np.ndarray: Gram matrix or its diagonal.
        """
        # check(len(np.shape(X)) == 2, "X must be a 2D array.")
        # check(not diag, "Diagonal only version not implemented for this kernel.")
        return np.where(
            self.dist(X, Y, diag) <= self.threshold, self.spike, self.non_spike
        )
