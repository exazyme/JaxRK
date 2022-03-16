from pathlib import Path
from typing import Callable
from ..core.typing import PRNGKeyT, Shape, Dtype, Array
from functools import partial

import numpy as onp
import jax.numpy as np, jax.scipy as sp, jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

from ..kern.base import Kernel
from ..core.constraints import NonnegToLowerBd, Bijection, CholeskyBijection
from ..utilities.views import tile_view

class FeatMapKernel(Kernel):
    def __init__(self, feat_map:Callable[[Array], Array] = None):
        """A kernel that is defined by a feature map.
        
        Args:
            feat_map: A callable that computes the feature map, i.e. given input space points it returns real valued features, one per input space point."""

        assert feat_map is not None
        self.feat_map = feat_map

    def __call__(self, X, Y = None, diag = False):
        f_X = self.feat_map(X)
        if Y is None:
            f_Y = f_X
        else:
            f_Y = self.feat_map(Y)
        if diag:
            return np.sum(f_X * f_Y, 1)
        else:
            return f_X.dot(f_Y.T)


LinearKernel = partial(FeatMapKernel, feat_map = lambda x:x)

class DictKernel(Kernel):
    """Kernel for a fixed dictionary of input space values and accompanying gram values. Example:
        ```
            k = DictKernel(np.array([1,3]), np.array([(2, -1), (-1, 1.2)]))
            assert k(np.array(1), np.array(1)) == 2
            assert k(np.array(3), np.array(1)) == -1
            assert k(np.array(3), np.array(3)) == 1.2
            k(np.array(2), np.array(3)) #this will throw an exception, as 2 is not a valid inspace value
        ```
        Args:
            gram_values: A square, positive semidefinite matrix.
            cholesky_lower: A square lower cholesky factor.
    """

    def __init__(self, inspace_vals:Array, gram_values:Array = None, cholesky_lower:Array = None, drop_neg_gram = True):
        super().__init__()
        
        assert gram_values != cholesky_lower, "Exactly one of gram_values and cholesky_lower has to be defined."
            
        if gram_values is None:
            assert cholesky_lower is not None, "Exactly one of gram_values and cholesky_lower has to be defined."
            assert len(cholesky_lower.shape) == 2
            assert cholesky_lower.shape[0] == cholesky_lower.shape[1]
            assert np.all(np.diag(cholesky_lower) > 0)
            gram_values = cholesky_lower @ cholesky_lower.T
        else:
            assert cholesky_lower is None, "Exactly one of gram_values and cholesky_lower has to be defined."
            assert len(gram_values.shape) == 2
            assert gram_values.shape[0] == gram_values.shape[1]
            
        
        nonneg = onp.array(np.diag(gram_values) > 0)
        if not drop_neg_gram:
            assert nonneg.sum() == nonneg.size, "gram_values is not a PSD matrix and drop_neg_gram is False. Provide a PSD matrix or set drop_neg_gram to True."
        else:
            inspace_vals = inspace_vals[nonneg]
            gram_values = gram_values[nonneg, :][:, nonneg]
        self.gram_values = gram_values
        self.inspace_vals = inspace_vals

        

    @classmethod
    def make_unconstr(cls, cholesky_lower:Array, diag_bij:Bijection = NonnegToLowerBd(0.1)) -> "DictKernel":
        """Make a DictKernel from unconstrained parameters.

        Args:
            cholesky_lower (Array): Unconstrained parameter for lower cholesky factor.
            diag_bij (Bijection, optional): Bijection from real numbers to non-negative numbers. Defaults to SoftPlus(0.1).

        Returns:
            DictKernel: The constructed kernel.
        """
        chol_bij = CholeskyBijection(diag_bij = diag_bij)
        return cls(gram_values=chol_bij(cholesky_lower))
    

    def __call__(self, idx_X, idx_Y=None, diag = False):
        assert (len(np.shape(idx_X))==2) and (idx_Y is None or len(np.shape(idx_Y))==2)
        assert idx_X.shape[1] == 1 and (idx_Y is None or idx_X.shape[1] == idx_Y.shape[1])
        if idx_Y is None:
            idx_Y = idx_X
        if diag:
            return self.gram_values[idx_X, idx_Y]
        else:
            #FIXME: repeat_view
            #using https://stackoverflow.com/questions/5564098/repeat-numpy-array-without-replicating-data
            #and https://github.com/google/jax/issues/3171
            #as starting points
            return self.gram_values[np.repeat(idx_X, idx_Y.size).squeeze(), tile_view(idx_Y, idx_X.size).squeeze()].reshape((idx_X.size, idx_Y.size))