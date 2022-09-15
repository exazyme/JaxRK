"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable, List
from abc import ABC, abstractmethod

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from ..utilities.views import tile_view
from ..core.typing import PRNGKeyT, Shape, Dtype, Array, ConstOrInitFn


class Reduce(Callable, ABC):
    """The basic reduction type."""

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        rval = self.reduce_first_ax(np.swapaxes(inp, axis, 0))
        return np.swapaxes(rval, axis, 0)

    @abstractmethod
    def reduce_first_ax(self, gram: np.array) -> np.array:
        pass

    @abstractmethod
    def new_len(self, original_len: int) -> int:
        pass

    @classmethod
    def apply(cls, inp, reduce: List["Reduce"] = None, axis=0):
        if reduce is None or len(reduce) == 0:
            return inp
        else:
            carry = np.swapaxes(inp, axis, 0)
            for gr in reduce:
                carry = gr.reduce_first_ax(carry)
            return np.swapaxes(carry, axis, 0)

    @classmethod
    def final_len(cls, original_len: int, reduce: List["Reduce"] = None):
        carry = original_len
        if reduce is not None:
            for gr in reduce:
                carry = gr.new_len(carry)
        return carry


class LinearizableReduce(Reduce):
    @abstractmethod
    def linearize(self, gram_shape: tuple) -> np.array:
        """Linearized version of reduce_first_ax.

        Args:
            gram_shape (tuple): The gram matrix
        """
        pass

# class SeqReduce(Reduce):
#     children:List[Reduce]

#     def __call__(self, inp:np.array, axis:int = 0) -> np.array:
#         carry = np.swapaxes(inp, axis, 0)
#         if self.children is not None:
#             for gr in self.children:
#                 carry = gr.reduce_first_ax(carry)
#         return np.swapaxes(carry, axis, 0)

#     def new_len(self, original_len:int):
#         carry = original_len
#         if self.children is not None:
#             for gr in self.children:
#                 carry = gr.new_len(carry)
#         return carry


class NoReduce(Reduce):
    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        return inp

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return inp

    def new_len(self, original_len: int) -> int:
        return original_len


class Prefactors(Reduce):
    def __init__(self, prefactors: Array):
        super().__init__()
        assert len(prefactors.shape) == 1
        self.prefactors = prefactors

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        assert self.prefactors.shape[0] == inp.shape[axis]
        return inp.astype(self.prefactors.dtype) * \
            np.expand_dims(self.prefactors, axis=(axis + 1) % 2)

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        assert original_len == len(self.prefactors)
        return original_len


class Scale(Reduce):
    def __init__(self, s: float):
        super().__init__()
        self.s = s

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        return inp * self.s

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        return original_len


class Repeat(Reduce):
    def __init__(self, times: int):
        super().__init__()
        assert times > 0
        self.times = times

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        return np.repeat(inp, axis)

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.call(inp, 0)

    def new_len(self, original_len: int) -> int:
        return original_len * self.times


class TileView(LinearizableReduce):
    def __init__(self, result_len: int):
        super().__init__()
        assert result_len > 0
        self.result_len = result_len

    def reduce_first_ax(self, inp: np.array) -> np.array:
        assert self.result_len % inp.shape[0] == 0, "Input can't be broadcasted to target length %d" % self.result_len
        return tile_view(inp, self.result_len // inp.shape[0])

    def linearize(self, gram_shape: tuple):
        return tile_view(
            np.eye(
                gram_shape[0]),
            self.result_len //
            gram_shape[0])

    def new_len(self, original_len: int) -> int:
        return self.result_len


class Sum(Reduce):
    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        return inp.sum(axis=axis, keepdims=True)

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        return 1

#    def linearize(self, gram_shape: tuple) -> np.array:
#        # sum
#        return np.ones((1, gram_shape[0]))


class Mean(Reduce):
    def __call__(self, inp: Array, axis: int = 0) -> np.array:
        return np.mean(inp, axis=axis, keepdims=True)

    def reduce_first_ax(self, inp: Array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        return 1

#    def linearize(self, gram_shape: tuple) -> np.array:
#        # mean
#        return np.ones((gram_shape[0], gram_shape[0])) / gram_shape[1]


class BalancedRed(Reduce):
    """Sum up even number of elements in input."""

    def __init__(self, points_per_split: int, average=False):
        super().__init__()
        assert points_per_split > 0
        self.points_per_split = points_per_split
        if average:
            self.red = np.mean
            self.factor = 1. / points_per_split
        else:
            self.red = np.sum
            self.factor = 1.

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        perm = list(range(len(inp.shape)))
        perm[0] = axis
        perm[axis] = 0

        inp = np.transpose(inp, perm)
        inp = self.red(np.reshape(
            inp, (-1, self.points_per_split, inp.shape[-1])), axis=1)
        inp = np.transpose(inp, perm)
        return inp

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.__call__(inp, 0)

    def linearize(self, inp_shape: tuple) -> np.array:
        new_len = self.new_len(inp_shape[0])
        rval = np.zeros((new_len, inp_shape[0]))
        for i in range(new_len):
            rval = rval.at[i, i *
                           self.points_per_split:(i +
                                                  1) *
                           self.points_per_split].set(self.factor)
        return rval

    def new_len(self, original_len: int) -> int:
        assert original_len % self.points_per_split == 0
        return original_len // self.points_per_split


class Center(Reduce):
    """Center input along axis."""

    def __call__(self, inp: np.array, axis: int = 0) -> np.array:
        return inp - np.mean(inp, axis, keepdims=True)

    def reduce_first_ax(self, inp: np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        return original_len
