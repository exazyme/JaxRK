from dataclasses import dataclass
import jax.numpy as np
import jax.scipy as sp
from jax.scipy.special import expit, logit
from typing import Callable, Union

from numpy.core.defchararray import upper
from ..core.typing import Bijection, ArrayOrFloatT, Array
import jax.lax as lax


class SoftPlus(Bijection):
    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a real number to a non-negative number using the softplus function defined as
        f(x) = log(1 + exp(x))

        Args:
            x (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        return np.log(1 + np.exp(x))

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a non-negative number to a real number using the inverse softplus function defined as
        f(y) = log(exp(y) - 1)

        Args:
            y (ArrayOrFloatT): A float or Array.
        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        return np.log(np.exp(y) - 1)


class SquarePlus(Bijection):
    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a real number to a non-negative number using the squareplus function.
        f(x) = (x + sqrt(x**2 + 4))/2

        Args:
            x (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
       """
        #(x + np.sqrt(x**2 + 4))/2
        return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.))))

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a non-negative number to a real number using the inverse squareplus function defined as
        f(y) = (y**2-1)/y

        Args:
            y (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        return lax.div(lax.sub(lax.square(y), 1.), y)


class SquareSquash(Bijection):
    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a real number to the intervall (0,1) using the square squash function defined as
        f(x) = (1 + x/sqrt(4 + x**2))/2

        Args:
            x (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        #(1 + x/np.sqrt(4 + x**2))/2
        return lax.mul(
            0.5,
            lax.add(
                lax.div(
                    x,
                    lax.sqrt(
                        lax.add(
                            lax.square(x),
                            4.))),
                1.))

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a number in the intervall (0,1) to a real number using the inverse square squash function.

        Args:
            y (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        pos_res = np.sqrt((-4. * y**2 + 4. * y - 1) / (y**2 - y))
        # 2. / np.sqrt(1./np.sqrt(2*y - 2) - 1)
        return np.where(y < 0.5, -pos_res, pos_res)


class SquashingToBounded(Bijection):
    def __init__(
            self,
            lower_bound: float,
            upper_bound: float,
            bij: Bijection = SquareSquash()):
        """SquashingToBounded is a bijection mapping the reals to the intervall [lower_bound, upper_bound] using a squashing function.

        Args:
            lower_bound (float): Lower bound of the intervall.
            upper_bound (float): Upper bound of the intervall.
            bij (Bijection, optional): Squashing function as a bijection. Defaults to SquareSquash().
        """
        assert lower_bound < upper_bound
        assert lower_bound is not None and upper_bound is not None
        self.lower_bound = lower_bound
        self.scale = upper_bound - lower_bound
        self.bij = bij

    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a real number to a number with lower_bound <= y <= upper_bound using the bijection.

        Args:
            x (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
       """
        return self.scale * np.clip(self.bij(x), 0., 1.) + self.lower_bound

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a number with lower_bound <= y <= upper_bound to a real number using the inverse bijection.

        Args:
            y (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        return self.bij.inv((y - self.lower_bound) / self.scale)


class NonnegToLowerBd(Bijection):
    def __init__(self, lower_bound: float = 0., bij: Bijection = SquarePlus()):
        assert lower_bound is not None
        self.lower_bound = lower_bound
        self.bij = bij

    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a real number with lower bound to a real number using a nonnegative bijection.

        Args:
            x (ArrayOrFloatT): A float or Array.

        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
       """
        return np.clip(self.bij(x), 0.) + self.lower_bound

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Map a non-negative number to a real number with lower bound using the inverse of a nonnegative bijection.

        Args:
            y (ArrayOrFloatT): A float or Array.
        Returns:
            ArrayOrFloatT: A float or Array (depending on the input).
        """
        return self.bij.inv(y - self.lower_bound)


class FlipLowerToUpperBound(Bijection):
    def __init__(self, upper_bound: float, lb_bij: Callable[..., Bijection]):
        assert upper_bound is not None
        self.lb = lb_bij(-upper_bound)

    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        return -self.lb.__call__(-x)

    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        return -self.lb.inv(-y)


def NonnegToUpperBd(upper_bound: float = 0.):
    return FlipLowerToUpperBound(upper_bound, NonnegToLowerBd)


def SoftBound(l: float = None, u: float = None) -> Bijection:
    if l is None:
        assert u is not None, "Requiring one bound."
        return NonnegToUpperBd(u)
    elif u is None:
        assert l is not None, "Requiring one bound."
        return NonnegToLowerBd(l)
    else:
        return SquashingToBounded(l, u)


@dataclass
class CholeskyBijection(Bijection):
    diag_bij: Bijection = NonnegToLowerBd(bij=SquarePlus())
    lower: bool = True

    def is_standard(self, inp):
        return len(inp.shape) == 2 and inp.shape[0] == inp.shape[1]

    def is_symmetric(self, inp):
        return self.is_standard(inp) and np.allclose(inp, inp.T)

    def is_param(self, inp):
        return self.is_standard(inp) and np.allclose(inp, np.tril(inp))

    def is_chol(self, inp):
        return self.is_param(inp) and np.all(np.diagonal(inp) > 0)

    def param_to_chol(self, param):
        return np.tril(param, -1) + \
            np.diagflat(self.diag_bij(np.diagonal(param)))

    def psd_to_param(self, psd_matr):
        L = sp.linalg.cholesky(psd_matr, lower=True)
        return self.chol_to_param(L)

    def chol_to_param(self, chol):
        return np.tril(chol, -1) + \
            np.diagflat(self.diag_bij.inv(np.diagonal(chol)))

    def __call__(self, x: Array) -> Array:
        """Map a parameter vector to a PSD matrix using the CholeskyBijecton.

        Args:
            x (Array): An array of shape (n, n) representing a symmetric positive definite matrix.

        Returns:
            Array: _description_
        Args:
            x (ArrayOrFloatT): An array representing unconstrained parameters from which a PSD matrix can be reconstructed.

        Returns:
            ArrayOrFloatT: An array of shape (n, n) representing a symmetric positive definite matrix.
       """
        c = self.param_to_chol(x)
        return c @ c.T

    def inv(self, y: Array) -> Array:
        """Map a PSD Matrix to a parameter vector using the inverse CholeskyBijection function.

        Args:
            y (ArrayOrFloatT): An array of shape (n, n) representing a symmetric positive definite matrix.
        Returns:
            ArrayOrFloatT: An array representing unconstrained parameters from which a PSD matrix can be reconstructed.
        """
        return self.chol_to_param(sp.linalg.cholesky(y, lower=True))
