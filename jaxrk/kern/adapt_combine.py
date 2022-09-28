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
    def __init__(
        self,
        intervals: Array,
        kernels: List[Kernel],
        operation: Callable = lambda x: np.prod(x, 0),
        weights: Array = None,
    ):
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

    def __call__(self, X, Y=None, diag=False):

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
    def __init__(self, sklearn_kernel: Any):
        super().__init__()
        self.sklearn_kernel = sklearn_kernel

    def __call__(self, X, Y=None, diag=False):
        if diag:
            assert Y is None
            rval = self.sklearn_kernel.diag(X)
        else:
            rval = self.sklearn_kernel(X, Y)
        return rval
