"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""

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
            idcs (List[np.array]): The indices of the rows to sum/average in the desired order. Each list element contains 2d arrays. The number of columns in the array is the number of summed/averaged elements.
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
                lin_map = jax.ops.index_update(
                    lin_map,
                    (idx1, self.idcs[i].flatten()),
                    1.0 / self.idcs[i].shape[1] if self.average else 1.0,
                )
            offset += self.idcs[i].shape[0]
        return lin_map

    @classmethod
    def sum_from_unique(
        cls, input: np.array, mean: bool = True
    ) -> Tuple[np.array, np.array, "SparseReduce"]:
        """Construct a SparseReduce object from a 1d array values by summing/averaging over the indices of the unique values.

        Returns:
            Tuple[np.array, np.array, "SparseReduce"]: The unique values, the counts of the unique values, and the SparseReduce object.
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