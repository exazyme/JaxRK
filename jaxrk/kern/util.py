from abc import ABC, abstractmethod
import jax.numpy as np

from typing import Union

from ..core.constraints import NonnegToLowerBd
from ..core.typing import *
from ..utilities.distances import dist

# from jax.experimental.checkify import check


class Scaler(ABC):
    """Abstract base class for scalers."""

    @abstractmethod
    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Scale the input data.

        Args:
            inp (np.ndarray): Input data.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            np.ndarray: Scaled input data.
        """
        raise NotImplementedError()

    @abstractmethod
    def inv(self) -> Union[Array, float]:
        """Inverse of the scaling.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            Union[Array, float]: Inverse of the scaling.
        """
        raise NotImplementedError()

    @abstractmethod
    def scale(self) -> Union[Array, float]:
        """Scaling.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Returns:
            Union[Array, float]: Scaling.
        """
        raise NotImplementedError()


class NoScaler(Scaler):
    """A dummy scaler that does not scale the input data."""

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Identity function.

        Args:
            inp (np.ndarray): Input data.

        Returns:
            np.ndarray: `inp`.
        """
        return inp

    def inv(self) -> Union[Array, float]:
        """Inverse of the scaling. Constant 1."""
        return np.ones(1)

    def scale(self) -> Union[Array, float]:
        """Scaling. Constant 1."""
        return np.ones(1)


class SimpleScaler(Scaler):
    """A simple scaler that scales the input data by a float or an array of floats."""

    def __init__(self, scale: Union[Array, float]):
        """Scale input either by global scale parameter or by per-dimension scaling parameters

        Args:
            scale (Union[Array, float]): Scaling parameter(s).
        """
        super().__init__()
        if isinstance(scale, float):
            scale = np.array([[scale]])
        else:
            scale = np.atleast_1d(scale)
            if len(scale.shape) == 1:
                scale = scale[np.newaxis, :]
            else:
                # check(len(scale.shape) == 2 and scale.shape[0] == 1, "Scale must be a vector or a scalar")
                pass
        # check(np.all(scale > 0.0), "Scale must be positive")
        self.s = scale

    @classmethod
    def make_unconstr(
        cls, scale: Union[Array, float], bij: Bijection = NonnegToLowerBd()
    ) -> "SimpleScaler":
        """Make a simple scaler with unconstrained parameters.

        Args:
            scale (Union[Array, float]): Scaling parameter(s).
            bij (Bijection, optional): Bijection. Defaults to NonnegToLowerBd().

        Returns:
            SimpleScaler: Simple scaler with unconstrained parameters.
        """
        return SimpleScaler(bij(scale))

    def __str__(self) -> str:
        """String representation of the scaler.

        Returns:
            str: String representation of the scaler.
        """
        return f"SimpleScaler({self.s})"

    def inv(self) -> Union[Array, float]:
        """Inverse of the scaling.

        Returns:
            Union[Array, float]: Inverse of the scaling.
        """
        return 1.0 / self.s

    def scale(self) -> Union[Array, float]:
        """Scaling.

        Returns:
            Union[Array, float]: Scaling.
        """
        return self.s

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Scale the input data.

        Args:
            inp (np.ndarray): Input data.

        Returns:
            np.ndarray: Scaled input data.
        """
        if inp is None:
            return None
        # either global scaling, meaning self.scale().size == 1,
        # or local scaling, in which case inp.shape[1] ==
        # check(self.scale().size == 1 or self.scale().size == inp.shape[1], "Scaling dimension mismatch")

        return self.s * inp


class ScaledPairwiseDistance:
    """A class for computing scaled pairwise distance for stationary/RBF kernels, depending only on

        dist = |X_i-Y_j|^p for all i, j

    For some power p.
    """

    def __init__(self, scaler: Scaler = NoScaler(), power: float = 2.0):
        """Constructor for ScaledPairwiseDistance.

        Args:
            scaler (Scaler, optional): Scaling module. Defaults to NoScaler().
            power (float, optional): Power p that the pairwise distance is taken to. Defaults to 2..
        """
        super().__init__()
        self.power = power
        if scaler.scale().size == 1:
            self.gs = scaler
            self.ds = NoScaler()
            self.is_global = True
        else:
            self.gs = NoScaler()
            self.ds = scaler
            self.is_global = False

    def __str__(self) -> str:
        """String representation of the `ScaledPairwiseDistance` instance."""
        return f"ScaledPairwiseDistance(scaler={self.gs if self.is_global else self.ds}, power={self.power})"

    @property
    def _scale(self) -> Union[Array, float]:
        """Scaling.

        Returns:
            Union[Array, float]: Scaling.
        """
        if self.is_global:
            return self.gs.inv()
        else:
            return self.ds.inv() ** (1.0 / self.power)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray = None,
        diag: bool = False,
    ) -> np.ndarray:
        """Compute the scaled pairwise distance.

        Args:
            X (np.ndarray): First input data array.
            Y (np.ndarray, optional): Second input data array. Defaults to None, in which case X = Y.
            diag (bool, optional): Whether to compute only the diagonal of the distance matrix. Defaults to False.

        Returns:
            np.ndarray: Scaled pairwise distance matrix or its diagonal.
        """
        if diag:
            if Y is None:
                rval = np.zeros(X.shape[0])
            else:
                # check(X.shape == Y.shape, "X and Y must have the same shape")
                rval = self.gs(np.sum(np.abs(self.ds(X) - self.ds(Y)) ** self.power, 1))
        else:
            rval = self.gs(dist(self.ds(X), self.ds(Y), power=1.0)) ** self.power
        return rval
