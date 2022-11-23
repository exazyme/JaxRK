from typing import Callable, List
from ..core.typing import Array, Any

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from ..utilities.eucldist import eucldist
from ..kern.base import DensityKernel, Kernel


class SplitDimsKernel(Kernel):
    """Apply different kernels to different dimensions of the input data."""

    def __init__(
        self,
        intervals: Array,
        kernels: list[Kernel],
        operation: Callable = lambda x: np.prod(x, 0),
        weights: Array = None,
    ):
        """Initialize SplitDimsKernel, giving it a list of input dimension intervals and a list of kernels to apply to each interval.

        Args:
            intervals (Array): Array of input dimension intervals.
            kernels (list[Kernel]): List of kernels to apply to each interval.
            operation (Callable, optional): Operation to apply to the gram matrices resulting from evaluating kernels. Defaults to lambda x: np.prod(x, 0).
            weights (Array, optional): Weights to apply to each kernel. Defaults to None.
        """
        super().__init__()
        self.intervals, self.kernels, self.operation, self.weights = (
            intervals,
            kernels,
            operation,
            weights,
        )
        assert len(self.intervals) - 1 == len(self.kernels)
        assert self.weights is None or self.weights.size == len(self.kernels)
        if self.weights is None:
            self.weights = np.ones(len(self.kernels))

    def __call__(self, X: Array, Y: Array = None, diag: bool = False) -> Array:
        """Evaluate the kernel on the given data.

        Args:
            X (Array): Input data.
            Y (Array, optional): Input data. Defaults to None.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            Array: Gram matrix.
        """
        split_X = [
            X[:, self.intervals[i] : self.intervals[i + 1]]
            for i in range(len(self.kernels))
        ]
        if Y is None:
            split_Y = [None] * len(self.kernels)
        else:
            split_Y = [
                Y[:, self.intervals[i] : self.intervals[i + 1]]
                for i in range(len(self.kernels))
            ]
        sub_grams = np.array(
            [
                self.kernels[i](split_X[i], split_Y[i], diag=diag) * self.weights[i]
                for i in range(len(self.kernels))
            ]
        )
        return self.operation(sub_grams)


class SKlKernel(Kernel):
    """Apply an sklearn kernel to the input data."""

    def __init__(self, sklearn_kernel: Any):
        """Initialize SKlKernel, giving it an sklearn kernel.

        Args:
            sklearn_kernel (Any): sklearn kernel.
        """
        super().__init__()
        self.sklearn_kernel = sklearn_kernel

    def __call__(self, X: Array, Y: Array = None, diag: bool = False) -> Array:
        """Evaluate the kernel on the given data.

        Args:
            X (Array): Input data.
            Y (Array, optional): Input data. Defaults to None.
            diag (bool, optional): Whether to return the diagonal of the gram matrix. Defaults to False.

        Returns:
            Array: Gram matrix.
        """
        if diag:
            assert Y is None
            rval = self.sklearn_kernel.diag(X)
        else:
            rval = self.sklearn_kernel(X, Y)
        return rval
