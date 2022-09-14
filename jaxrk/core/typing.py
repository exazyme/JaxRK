import jax.numpy as np
from typing import Any, Tuple, Callable, Union, TypeVar

__all__ = ["Bijection", "PRNGKeyT", "Shape", "Dtype", "Array", "InitFn", "ConstOrInitFn", "FloatOrInitFn", "AnyOrInitFn"]


PRNGKeyT = Any
Shape = Tuple[int]
Dtype = Any
Array = np.ndarray

InitFn = Callable[..., Any]

ConstOrInitFn = Union[float, InitFn, Any]
FloatOrInitFn = Union[float, InitFn]
AnyOrInitFn = Union[Any, InitFn]


ArrayOrFloatT = TypeVar("ArrayOrFloatT", np.ndarray, float)


class Bijection(object):
    def __call__(self, x:ArrayOrFloatT) -> ArrayOrFloatT:
        raise NotImplementedError
        
    def inv(self, y:ArrayOrFloatT) -> ArrayOrFloatT:
        raise NotImplementedError