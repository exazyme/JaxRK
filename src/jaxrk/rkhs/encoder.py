import abc
from typing import TypeVar, Generic
from jaxrk.rkhs import FiniteVec
from jaxrk.kern import Kernel
from jaxrk.core.typing import Array


InputT = TypeVar("InputT")


class RkhsVecEncoder(abc.ABC, Generic[InputT]):
    """Abstract class for RKHS vector encoders. These are used to encode input space data into RKHS vectors.

    Example:
        a = RkhsVecEncoder()
        rkhs_vec_train = a(inp_data_train)
        rkhs_vec_test = a(inp_data_test)
        gram_train = rkhs_vec_train.inner()
        gram_train_test = rkhs_vec_train.inner(rkhs_vec_test)
    """

    @abc.abstractmethod
    def __call__(self, inp: InputT) -> FiniteVec:
        """Encodes input data into an RKHS vector.

        Args:
            inp (InputT): Input data set.

        Returns:
            FiniteVec: RKHS vector.
        """
        pass


class StandardEncoder(RkhsVecEncoder[InputT]):
    """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.
    In other words, this is the standard mapping of classical kernel methods."""

    def __init__(self, k: Kernel) -> None:
        self.k = k

    def __call__(self, inp: InputT) -> FiniteVec:
        """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.

        Args:
            inp (Array): Input array.

        Returns:
            FiniteVec: RKHS vector.
        """
        return FiniteVec(self.k, inp)
