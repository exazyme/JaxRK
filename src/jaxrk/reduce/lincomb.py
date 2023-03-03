"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""

import collections.abc
import jax
from typing import Callable, List, TypeVar, Tuple

from functools import partial
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from ..core.typing import Array
from ..utilities.cv import vmatmul_fixed_inp
from .base import LinearizableReduce, Reduce


# ListOfArray_or_Array_T = TypeVar("CombT", List[Array], Array)


class SparseReduce(LinearizableReduce):
    """SparseReduce constructs a Gram matrix by summing/averaging over rows of its input"""

    def __init__(
        self, idcs: List[Array], average: bool = True, max_idx: int = None
    ) -> None:
        """Initialize SparseReduce.

        Args:
            idcs (List[np.array]): The indices of the rows to sum/average in the desired order. Each list element contains 2d arrays. The number of columns in the array is the number of summed/averaged elements. The number of rows is the number of rows in the output resulting from this list element.
            average (bool): If True average rows, else sum rows.
            max_idx (int, optional): The maximum index in the input. Defaults to None, in which case the maximum index is inferred from the idcs.
        """

        super().__init__()
        self.idcs = idcs
        if max_idx is not None:
            self.max_idx = max_idx
        else:
            max_list = []
            for i in idcs:
                if i.size > 0:
                    max_list.append(np.max(i))
            self.max_idx = np.array(max_list).max()
        self.average = average
        if average:
            self._reduce = np.mean
        else:
            self._reduce = np.sum

    def reduce_first_ax(self, inp: np.array) -> np.array:
        """Reduce the first axis of the input.

        Args:
            inp (np.array): Input to reduce. Typically a gram matrix.

        Returns:
            np.array: Reduced input.

        Examples:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce import SparseReduce
            >>> inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> r1 = SparseReduce([np.array([[0, 1]]), np.array([[0, 2]]), np.array([[0, 1, 2]])], True) # average rows 0 and 1, then 0 and 2, then all rows
            >>> r1.reduce_first_ax(inp)
            DeviceArray([[2.5, 3.5, 4.5],
                         [4. , 5. , 6. ],
                         [4. , 5. , 6. ]], dtype=float32)
            >>> r2 = SparseReduce([np.array([[0, 1], [0, 2]]), np.array([[0, 1, 2]])], True) # average rows 0 and 1, then 0 and 2, then all rows. Same as r1, just different input format, and probably more efficient.
            >>> r2.reduce_first_ax(inp)
            DeviceArray([[2.5, 3.5, 4.5],
                        [4. , 5. , 6. ],
                        [4. , 5. , 6. ]], dtype=float32)
            >>> r3 = SparseReduce([np.array([0, 0, 1, 1, 2])[:, np.newaxis]], False) # copy row 0 twice, then row 1 twice, then row 2
            >>> r3.reduce_first_ax(inp)
            DeviceArray([[ 1,  2,  3],
                            [ 1,  2,  3],
                            [ 4,  5,  6],
                            [ 4,  5,  6],
                            [ 7,  8,  9]], dtype=float32)
        """
        assert (self.max_idx + 1) <= len(inp), (
            self.__class__.__name__ + " expects a longer gram to operate on"
        )
        assert len(inp.shape) == 2
        rval = []

        for i in range(len(self.idcs)):
            if self.idcs[i].shape[1] == 0:
                rval.append(np.zeros((self.idcs[i].shape[0], inp.shape[1])))
            else:
                reduced = self._reduce(
                    inp[list(self.idcs[i].flatten()), :].reshape(
                        (-1, self.idcs[i].shape[1], inp.shape[1])
                    ),
                    1,
                )
                rval.append(reduced)
        return np.concatenate(rval, 0)

    def new_len(self, original_len: int) -> int:
        """Get the length of the reduced gram matrix.

        Args:
            original_len (int): The length of the original gram matrix.

        Returns:
            int: The length of the reduced gram matrix.
        """
        assert (self.max_idx + 1) <= original_len, (
            self.__class__.__name__ + " expects a longer gram to operate on"
        )
        return len(self.idcs)

    def linmap(self, inp_shape: tuple, axis: int = 0) -> np.array:
        """Get the linear map that reduces the first axis of the input.

        Args:
            inp_shape (tuple): The shape of the input.
            axis (int, optional): The axis to reduce. Defaults to 0.

        Returns:
            np.array: The linear map that reduces the first axis of the input.

        Example:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce.lincomb import SparseReduce
            >>> input = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
            >>> un, cts, sr = SparseReduce.sum_from_unique(input, mean=False)
            >>> print(sr.linmap((9, 1)))
            [[1. 0. 0. 1. 0. 0. 1. 0. 0.]
             [0. 1. 0. 0. 1. 0. 0. 1. 0.]
             [0. 0. 1. 0. 0. 1. 0. 0. 1.]]

        """
        n_in = self.max_idx + 1
        assert inp_shape[axis] == n_in, ValueError(
            "Input shape does not match reduction assumptions"
        )
        n_out = np.sum(np.array([len(i) for i in self.idcs]))
        offset = 0
        lin_map = np.zeros((n_out, n_in))
        for i in range(len(self.idcs)):
            if self.idcs[i].shape[1] != 0:
                idx1 = np.repeat(
                    np.arange(self.idcs[i].shape[0]) + offset, self.idcs[i].shape[1]
                )
                lin_map = lin_map.at[(idx1, self.idcs[i].flatten())].set(
                    1.0 / self.idcs[i].shape[1] if self.average else 1.0
                )
            offset += self.idcs[i].shape[0]
        return lin_map

    @classmethod
    def sum_from_block_example(cls, l: list[collections.abc.Sized], mean: bool = True):
        """Construct a SparseReduce object from an example list of arrays.
        The arrays in the list are assumed to be of the length of # of elements that should be reduced.

        Args:
            l (list[collections.abc.Sized]): Block example.
            mean (bool, optional): Whether to average the blocks. Defaults to True.
        """

        def collect_block_start_stop(l: list[np.ndarray]):
            rval = []
            total_len = 0
            for arr in l:
                arr_len = len(arr)
                rval.append((total_len, arr_len + total_len))
                total_len += arr_len
            return np.array(rval)

        def reduce_blocks(block_start_stop: np.ndarray):
            rval = []
            total_len = block_start_stop[-1, 1]
            for start, stop in block_start_stop:
                rval.append(np.arange(start, stop, dtype=np.uint32)[np.newaxis, :])
            return rval

        blocks = collect_block_start_stop(l)
        return cls(reduce_blocks(blocks), average=mean, max_idx=blocks[-1, 1] - 1)

    @classmethod
    def sum_from_unique(
        cls, input: np.array, mean: bool = True
    ) -> Tuple[np.array, np.array, "SparseReduce"]:
        """Construct a SparseReduce object from a 1d array values by summing/averaging over the indices of the unique values.

        Args:
            input (np.array): The input array.
            mean (bool, optional): Average the values if True, sum them if False. Defaults to True.

        Returns:
            Tuple[np.array, np.array, "SparseReduce"]: The unique values, the counts of the unique values, and the SparseReduce object.

        Example:
            >>> import jax.numpy as np
            >>> from jaxrk.reduce.lincomb import SparseReduce
            >>> input = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
            >>> un, cts, sr = SparseReduce.sum_from_unique(input)
            >>> print(un)
            [1 2 3]
            >>> print(cts)
            [3 3 3]
            >>> print(sr.idcs)
            [DeviceArray([[0, 3, 6],
                         [1, 4, 7],
                         [2, 5, 8]], dtype=int32)]
        """
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        l_arr = np.array([i.size for i in un_idx])
        argsort = np.argsort(l_arr)
        un_sorted = un[argsort]
        cts_sorted = cts[argsort]
        un_idx_sorted = [un_idx[i] for i in argsort]

        change = list(
            np.argwhere(l_arr[argsort][:-1] - l_arr[argsort][1:] != 0).flatten() + 1
        )
        change.insert(0, 0)
        change.append(len(l_arr))
        change = np.array(change)

        el = []
        for i in range(len(change) - 1):
            el.append(
                np.array([un_idx_sorted[j] for j in range(change[i], change[i + 1])])
            )

        # assert False
        return un_sorted, cts_sorted, SparseReduce(el, mean)
