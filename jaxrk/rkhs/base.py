from abc import abstractmethod, ABC
from jax.numpy import uint
from ..core.typing import Array
from typing import NewType, TypeVar, Generic, Sized, Union


class Vec(Sized, ABC):
    """Abstract base class for Vectors of RKHS elements."""

    @abstractmethod
    def reduce_gram(self, gram: Array, axis: uint = 0):
        """Reduce a gram matrix using the vector's reductions

        Args:
            gram (Array): The gram matrix to reduce
            axis (uint, optional): The axis to reduce over. Defaults to 0.
        """
        pass

    @abstractmethod
    def inner(self, Y: "Vec" = None):
        """Compute the inner product of the vector with itself or another vector.

        Args:
            Y (Vec, optional): The other vector. Defaults to None, in which case Y = self.
        """
        pass


InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

# The following is input to a map RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec)

CombT = TypeVar("CombT", "LinOp[RhInpVectT, InpVecT]", InpVecT, Array)


class LinOp(Vec, Generic[InpVecT, OutVecT], ABC):
    """Abstract base class for linear operators on RKHS vectors."""

    @abstractmethod
    def __matmul__(
        self, right_inp: CombT
    ) -> Union[OutVecT, "LinOp[RhInpVectT, OutVecT]"]:
        """Apply the linear operator to a vector or another linear operator.

        Args:
            right_inp (CombT): The vector or linear operator to apply the linear operator to.
        """
        pass


RkhsObject = Union[Vec, LinOp]


def inner(X: Vec, Y: Vec = None) -> Array:
    """Compute the inner product of two vectors in the RKHS.

    Args:
        X (Vec): The first vector
        Y (Vec, optional): The second vector. Defaults to None, in which case the inner product is computed with X with itself.

    Returns:
        Array: The inner product
    """
    return X.inner(Y)
