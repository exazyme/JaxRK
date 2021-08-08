from .rkhs import FiniteVec
from .core.typing import Array
import jax.numpy as np, flax.linen as ln, jax.scipy as scipy
from typing import Any

class GP(object):
    def __init__(self, x:FiniteVec, y:Array, noise:float, normalize_y:bool = False):
        self.x, self.y, self.noise, self.normalize_y = (x, y, noise, normalize_y)
        if self.y.ndim == 1:
            self.y = self.y[:, np.newaxis]
        if not normalize_y:
            self.ymean = np.zeros(1)
            self.ystd = np.ones(1)
        else:
            self.ymean = np.mean(y, 0, keepdims=True)
            self.ystd = np.std(y, 0, keepdims=True)
            self.y = (self.y - self.ymean) / self.ystd

        train_cov = self.x.inner() + np.eye(len(self.x)) * self.noise 
        self.chol = scipy.linalg.cholesky(train_cov, lower=True)
        self.alpha = scipy.linalg.cho_solve((self.chol, True), self.y)
        self.log2pi_half = np.log(2. * np.pi) / 2

    
    def marginal_likelihood(self):        
        ml = np.sum(- 0.5 * np.einsum("ik,ik->k", self.y, self.alpha)
                    - np.log(np.diag(self.chol)).sum()
                    - len(self.x) * self.log2pi_half)
        ml -= np.sum(-self.log2pi_half) # lognormal prior
        return -ml

    def predict(self, xtest:FiniteVec = None):
        cross_cov = self.x.inner(xtest)
        mu = self.ystd * np.dot(cross_cov.T, self.alpha) + self.ymean
        v = scipy.linalg.cho_solve((self.chol, True), cross_cov)
        cov = (xtest.inner() - np.dot(cross_cov.T, v)) * self.ystd**2
        return mu, cov
    
    def post_pred_likelihood(self, xtest:FiniteVec, ytest:Array):
        m, v = self.predict(xtest)
        if self.normalize_y:
            ytest = (ytest - self.ymean) / self.ystd
        return m, v, scipy.stats.multivariate_normal.logpdf(ytest.ravel(), m.ravel(), v)