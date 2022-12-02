import flax.linen as nn
import abc
from ...rkhs import FiniteVec
from ...kern import Kernel
from ..factories import Factory
from ...core.typing import Array


class RkhsVecEncoder(nn.Module, abc.ABC):
    """Abstract class for RKHS vector encoders. These are used to encode input space data into RKHS vectors."""

    @abc.abstractmethod
    def __call__(self, inp: any) -> FiniteVec:
        """Encodes input data into an RKHS vector.

        Args:
            inp (any): Input data.

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
