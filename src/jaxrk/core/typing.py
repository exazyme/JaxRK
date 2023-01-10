import jax.numpy as np
from typing import Any, Tuple, Callable, Union, TypeVar
from abc import ABC, abstractmethod

__all__ = [
    "Bijection",
    "PRNGKeyT",
    "Shape",
    "Dtype",
    "Array",
    "InitFn",
    "ConstOrInitFn",
    "FloatOrInitFn",
    "AnyOrInitFn",
]


PRNGKeyT = Any
Shape = Tuple[int]
Dtype = Any
Array = np.ndarray

InitFn = Callable[..., Any]

ConstOrInitFn = Union[float, InitFn, Any]
FloatOrInitFn = Union[float, InitFn]
AnyOrInitFn = Union[Any, InitFn]


ArrayOrFloatT = TypeVar("ArrayOrFloatT", np.ndarray, float)


class Bijection(ABC):
    """A bijection beween two spaces."""

    @abstractmethod
    def __call__(self, x: ArrayOrFloatT) -> ArrayOrFloatT:
        """Forward transformation.

        Args:
            x (ArrayOrFloatT): Input (in space A).

        Returns:
            ArrayOrFloatT: Output (in space B).
        """
        pass

    @abstractmethod
    def inv(self, y: ArrayOrFloatT) -> ArrayOrFloatT:
        """Inverse transformation.

        Args:
            y (ArrayOrFloatT): Input (in space B).

        Returns:
            ArrayOrFloatT: Output (in space A).
        """
        pass
