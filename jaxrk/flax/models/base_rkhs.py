import flax.linen as nn
import abc
from ...rkhs import FiniteVec
from typing import Iterator
from ...kern import Kernel
from ..factories import Factory
from ...core.typing import Array


class RkhsVecEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, inp) -> FiniteVec:
        pass


class OneToOneEncoder(RkhsVecEncoder):
    kernel_fac: Factory[Kernel]

    def setup(
        self,
    ):
        self.k = self.kernel_fac(self, "kernel")

    def __call__(self, inp: Array) -> FiniteVec:
        return FiniteVec(self.k, inp)
