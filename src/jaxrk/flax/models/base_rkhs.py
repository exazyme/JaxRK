import flax.linen as nn
import abc
from ...rkhs import FiniteVec
from ...kern import Kernel
from ..factories import Factory
from ...core.typing import Array


class RkhsVecEncoder(nn.Module, abc.ABC):
    """Abstract class for RKHS vector encoders. These are used to encode input space data into RKHS vectors.

    Example:
        a = RkhsVecEncoder()
        rkhs_vec_train = a(inp_data_train)
        rkhs_vec_test = a(inp_data_test)
        gram_train = rkhs_vec_train.inner()
        gram_train_test = rkhs_vec_train.inner(rkhs_vec_test)
    """

    @abc.abstractmethod
    def __call__(self, inp: any) -> FiniteVec:
        """Encodes input data into an RKHS vector.

        Args:
            inp (any): Input data set.

        Returns:
            FiniteVec: RKHS vector.
        """
        pass


class OneToOneEncoder(RkhsVecEncoder):
    """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.
    In other words, this is the standard mapping of classical kernel methods."""

    kernel_fac: Factory[Kernel]

    def setup(
        self,
    ):
        """Flax setup method."""
        self.k = self.kernel_fac(self, "kernel")

    def __call__(self, inp: Array) -> FiniteVec:
        """Encodes input array into an RKHS vector, where the kernel is applied to each element of the input array.

        Args:
            inp (Array): Input array.

        Returns:
            FiniteVec: RKHS vector.
        """
        return FiniteVec(self.k, inp)
