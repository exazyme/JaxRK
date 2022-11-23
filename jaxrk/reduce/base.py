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
    """The abstract base class for reductions."""

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Call the reduction.

        Args:
            inp (Array): The array to reduce. Typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: The reduced array.
        """
        rval = self.reduce_first_ax(np.swapaxes(inp, axis, 0))
        return np.swapaxes(rval, axis, 0)

    @abstractmethod
    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input matrix.

        Args:
            inp (Array): The array to reduce. Typically a gram matrix.

        Returns:
            Array: The array reduced along the first axis.
        """
        pass

    @abstractmethod
    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        pass

    @classmethod
    def apply(cls, inp: Array, reduce: List["Reduce"] = None, axis=0) -> Array:
        """Apply a list of reductions to an array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            reduce (List[Reduce], optional): The reductions to apply. Defaults to None.
            axis (int, optional): Axis to apply reductions over. Defaults to 0.

        Returns:
            Array: Reduced array.
        """
        if reduce is None or len(reduce) == 0:
            return inp
        else:
            carry = np.swapaxes(inp, axis, 0)
            for gr in reduce:
                carry = gr.reduce_first_ax(carry)
            return np.swapaxes(carry, axis, 0)

    @classmethod
    def final_len(cls, original_len: int, reduce: List["Reduce"] = None) -> int:
        """Return the final length of an array after applying a list of reductions.

        Args:
            original_len (int): Original length of the array.
            reduce (List[Reduce], optional): List of reductions to apply. Defaults to None, in which case `original_len` is returned.

        Returns:
            int: Final length of the array after applying the reductions.
        """
        carry = original_len
        if reduce is not None:
            for gr in reduce:
                carry = gr.new_len(carry)
        return carry


class LinearReduce(Reduce):
    def __init__(self, linear_map: Array):
        super().__init__()
        self.linear_map = linear_map

    def reduce_first_ax(self, inp: Array):
        assert len(inp.shape) == 2
        assert self.linear_map.shape[-1] == inp.shape[0]
        return self.linear_map @ inp

    def new_len(self, original_len: int):
        assert (self.linear_map.shape[-1]) == original_len, (
            self.__class__.__name__
            + " expects a gram with %d columns" % self.linear_map.shape[1]
        )
        return self.linear_map.shape[-2]

    @classmethod
    def sum_from_unique(
        cls, input: Array, mean: bool = True
    ) -> tuple[np.array, np.array, "LinearReduce"]:
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        m = np.zeros((len(un_idx), input.shape[0]))
        for i, idx in enumerate(un_idx):
            b = np.ones(int(cts[i].squeeze())).squeeze()
            m = m.at[i, idx.squeeze()].set(b / cts[i].squeeze() if mean else b)
        return un, cts, LinearReduce(m)


class LinearizableReduce(Reduce):
    def linearize(self, gram_shape: tuple, axis: int = 0) -> LinearReduce:
        """Linearize the reduction.

        Args:
            gram_shape (tuple): Shape of the gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            LinearReduce: The linearized reduction.
        """
        return LinearReduce(self.linmap(gram_shape, axis))

    @abstractmethod
    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map equivalent to reduction.

        Args:
            gram_shape (tuple): The gram matrix shape.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.
        """
        pass


# class SeqReduce(Reduce):
#     children:List[Reduce]

#     def __call__(self, inp:Array, axis:int = 0) -> Array:
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
    """No reduction is actually applied."""

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Return the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: `inp`
        """
        return inp

    def reduce_first_ax(self, inp: Array) -> Array:
        """Return the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: `inp`
        """
        return inp

    def new_len(self, original_len: int) -> int:
        """Return the original length.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: `original_len`
        """
        return original_len


class Prefactors(Reduce):
    def __init__(self, prefactors: Array):
        """This reduction will multiply the input array with a set of prefactors.

        Args:
            prefactors (Array): The prefactors to multiply with, expected to have the same length as the axis of the input array along which the reduction is applied.
        """
        super().__init__()
        assert len(prefactors.shape) == 1
        self.prefactors = prefactors

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Multiply the input array with the prefactors.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Input array multiplied with the prefactors.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import Prefactors
            >>> p = Prefactors(jnp.array([1,2,3]))
            >>> m = jnp.array([[1,2,3],[4,5,6],[7,8,9]])
            >>> p(m)
            DeviceArray([[ 1,  2,  3],
                         [ 8, 10, 12],
                         [21, 24, 27]], dtype=int32)
            >>> p(m, axis=1)
            DeviceArray([[1,  4,  9],
                         [4, 10, 18],
                         [7, 16, 27]], dtype=int32)
        """
        assert self.prefactors.shape[0] == inp.shape[axis]
        return inp.astype(self.prefactors.dtype) * np.expand_dims(
            self.prefactors, axis=(axis + 1) % 2
        )

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array by multiplying with the prefactors.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        assert original_len == len(self.prefactors)
        return original_len


class Scale(Reduce):
    def __init__(self, s: float):
        """This reduction will multiply the input array with a constant.

        Args:
            s (float): Constant to multiply with.
        """
        super().__init__()
        self.s = s

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Scale the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Ignored.

        Returns:
            Array: Scaled input array.
        """
        return inp * self.s

    def reduce_first_ax(self, inp: Array) -> Array:
        """Scale the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Scaled input array.
        """
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction, i.e. `original_len`.
        """
        return original_len


class Repeat(Reduce):
    def __init__(self, times: int):
        """This reduction will repeat the input array `times` times.

        Args:
            times (int): Number of times to repeat the input array.
        """
        super().__init__()
        assert times > 0
        self.times = times

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Repeat the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Repeated input array.
        """
        return np.repeat(inp, axis)

    def reduce_first_ax(self, inp: Array) -> Array:
        """Repeat the input array along the first axis.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Repeated input array.
        """
        return self.call(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return original_len * self.times


class TileView(LinearizableReduce):
    def __init__(self, result_len: int):
        """Tile the input array to a new length.

        Args:
            result_len (int): Resulting length of the array.
        """
        super().__init__()
        assert result_len > 0
        self.result_len = result_len

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array by tiling it.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        assert self.result_len % inp.shape[0] == 0, (
            "Input can't be broadcasted to target length %d" % self.result_len
        )
        return tile_view(inp, self.result_len // inp.shape[0])

    def linmap(self, inp_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of reduce_first_ax for the tile view reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get a tiled result.
        """
        return tile_view(np.eye(inp_shape[axis]), self.result_len // inp_shape[axis])

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return self.result_len


class Sum(LinearizableReduce):
    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Sum the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Summed input array.

        Example:
            >>> from jax import numpy as jnp
            >>> from jaxrk.reduce import Sum
            >>> s = Sum()
            >>> m = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> s(m, axis=0)
            DeviceArray([[12, 15, 18]], dtype=int32)
            >>> s(m, axis=1)
            DeviceArray([[ 6],
                         [15],
                         [24]], dtype=int32)
        """
        return inp.sum(axis=axis, keepdims=True)

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array by summing over it.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of `Sum` reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to sum over `axis`.
        """
        return np.ones((1, gram_shape[axis]))


class Mean(LinearizableReduce):
    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Average the input array.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Reduced input array.
        """
        return np.mean(inp, axis=axis, keepdims=True)

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array by averaging over it.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return 1

    def linmap(self, gram_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of mean reduction.

        Args:
            gram_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix to average over `axis`.
        """
        return np.ones((1, gram_shape[axis])) / gram_shape[axis]


class BalancedRed(LinearizableReduce):
    def __init__(self, points_per_split: int, average=False):
        """Sum up a number of elements in the input.

        Args:
            points_per_split (int): Number of points per split, i.e. number of dimensions to sum up to a single result dimension.
            average (bool, optional): If True, average rather than sum up dimensions. Defaults to False.
        """
        super().__init__()
        assert points_per_split > 0
        self.points_per_split = points_per_split
        if average:
            self.red = np.mean
            self.factor = 1.0 / points_per_split
        else:
            self.red = np.sum
            self.factor = 1.0

    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Sum up a number of elements in the input.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Reduced input array.
        """
        perm = list(range(len(inp.shape)))
        perm[0] = axis
        perm[axis] = 0

        inp = np.transpose(inp, perm)
        inp = self.red(
            np.reshape(inp, (-1, self.points_per_split, inp.shape[-1])), axis=1
        )
        inp = np.transpose(inp, perm)
        return inp

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array by summing up even number of elements.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        return self.__call__(inp, 0)

    def linmap(self, inp_shape: tuple, axis: int = 0) -> Array:
        """Linear map version of `BalancedRed` reduction.

        Args:
            inp_shape (tuple): Shape of the input matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: A linear operator that can be applied to the input matrix and get the same result as the reduction.
        """
        new_len = self.new_len(inp_shape[axis])
        rval = np.zeros((new_len, inp_shape[axis]))
        for i in range(new_len):
            rval = rval.at[
                i, i * self.points_per_split : (i + 1) * self.points_per_split
            ].set(self.factor)
        return rval

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        assert original_len % self.points_per_split == 0
        return original_len // self.points_per_split


class Center(Reduce):
    def __call__(self, inp: Array, axis: int = 0) -> Array:
        """Center input along axis.

        Args:
            inp (Array): Input array, typically a gram matrix.
            axis (int, optional): Axis to apply reduction over. Defaults to 0.

        Returns:
            Array: Centered input array.
        """
        return inp - np.mean(inp, axis, keepdims=True)

    def reduce_first_ax(self, inp: Array) -> Array:
        """Reduce the first axis of the input array centering it.

        Args:
            inp (Array): Input array, typically a gram matrix.

        Returns:
            Array: Reduced array.
        """
        return self.__call__(inp, 0)

    def new_len(self, original_len: int) -> int:
        """Compute the new length of the array after reduction.

        Args:
            original_len (int): Original length of the array.

        Returns:
            int: Length of the array after reduction.
        """
        return original_len
