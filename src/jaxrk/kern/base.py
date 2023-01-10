"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable

from abc import abstractmethod, abstractproperty
import numpy as onp
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist


class Kernel:
    """A generic kernel type."""

    @abstractmethod
    def __call__(
        self, X: np.ndarray, Y: np.ndarray = None, diag: bool = False
    ) -> np.ndarray:
        """Compute the gram matrix, i.e. the kernel evaluated at every element of X paired with each element of Y (if not None, otherwise each element of X).

        Args:
            X (Array): Input space points, one per row.
            Y (Array, optional): Input space points, one per row. Defaults to None, in which case Y = X.
            diag (bool, optional): If `True`, compute only the diagonal elements of the gram matrix. Defaults to False.

        Returns:
            Array: Gram matrix or its diagonal."""

        raise NotImplementedError()


class DensityKernel(Kernel):
    """Positive definite kernel that is also a density."""

    @abstractproperty
    def std(self):
        """Standard deviation of the kernel."""
        return np.sqrt(self.var())

    @abstractproperty
    def var(self):
        """Variance of the kernel."""
        raise NotImplementedError()

    @abstractmethod
    def rvs(self, nsamps):
        """Draw samples from the distribution defined by the kernel.

        Args:
            nsamps (int): Number of samples to draw.
        """
        raise NotImplementedError()
