from copy import copy
from ..reduce import LinearReduce, Sum
from typing import Generic, TypeVar, Callable, Union

import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from .vector import FiniteVec
from ..core.typing import AnyOrInitFn, Array

from .base import LinOp, RkhsObject, Vec, InpVecT, OutVecT, RhInpVectT, CombT


class FiniteOp(LinOp[InpVecT, OutVecT]):
    """Finite rank LinOp in RKHS"""

    def __init__(
        self,
        inp_feat: InpVecT,
        outp_feat: OutVecT,
        matr: Array = None,
        normalize: bool = False,
    ):
        """Finite rank RKHS Operator.

        Args:
            inp_feat (InpVecT): Input feature vector.
            outp_feat (OutVecT): Output feature vector.
            matr (Array, optional): _description_. Defaults to None.
            normalize (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        if matr is not None:
            assert matr.shape == (len(outp_feat), len(inp_feat))
        self.matr = matr
        self.inp_feat, self.outp_feat = inp_feat, outp_feat
        self._normalize = normalize

    def __len__(self) -> int:
        """Length of operator when seen as a vector through the isometric isomorphism mapping between operator space and RKHS space."""
        return len(self.inp_feat) * len(self.outp_feat)

    def __matmul__(
        self, right_inp: CombT
    ) -> Union[OutVecT, "FiniteOp[RhInpVectT, OutVecT]"]:
        """Apply operator to input vector.

        Returns:
            Union[OutVecT, FiniteOp[RhInpVectT, OutVecT]]: Output vector or operator.
        """
        if isinstance(right_inp, FiniteOp):
            # right_inp is an operator
            # Op1 @ Op2
            matr = self.inp_feat.inner(right_inp.outp_feat) @ right_inp.matr
            if self.matr is not None:
                matr = self.matr @ matr
            return FiniteOp(right_inp.inp_feat, self.outp_feat, matr)
        else:
            if isinstance(right_inp, DeviceArray):
                right_inp = FiniteVec(self.inp_feat.k, np.atleast_2d(right_inp))
            # right_inp is a vector
            # Op @ vec
            right_gram = self.inp_feat.inner(right_inp)

            if self.matr is not None:
                right_gram = self.matr @ right_gram
            if self._normalize:
                right_gram = right_gram / right_gram.sum(1, keepdims=True)
            lr = LinearReduce(right_gram.T)
            rval = self.outp_feat.extend_reduce([lr])
            if len(right_inp) == 1:
                return rval
            else:
                return rval.sum()

    def reduce_gram(self, gram: np.ndarray, axis: np.uint = 0) -> np.ndarray:
        """Reduce gram matrix along axis using the operator's matrix..

        Args:
            gram (np.ndarray): Gram matrix.
            axis (uint, optional): Axis along which to reduce. Defaults to 0.

        Returns:
            np.uint: Reduced gram matrix.
        """
        return gram

    @property
    def T(self) -> "FiniteOp[OutVecT, InpVecT]":
        """Transpose operator."""
        return FiniteOp(self.outp_feat, self.inp_feat, self.matr.T, self._normalize)

    def __call__(self, inp: DeviceArray) -> RkhsObject:
        """Apply operator to input space points mapped to RKHS space.

        Args:
            inp (DeviceArray): Input space points.

        Returns:
            RkhsObject: Output vector or operator.
        """
        return self @ FiniteVec(self.inp_feat.k, np.atleast_2d(inp))

    def apply(self, inp: CombT) -> RkhsObject:
        """Apply operator to input vector.

        Args:
            inp (CombT): Input vector.

        Returns:
            RkhsObject: Output vector or operator.
        """
        return self @ inp

    def inner(self, Y: "FiniteOp[InpVecT, OutVecT]" = None) -> np.ndarray:
        """Inner product with another operator after mapping the operators to RKHS space using the isometric isomorphism relating the two spaces.

        Args:
            Y (FiniteOp[InpVecT, OutVecT], optional): Operator. Defaults to None, in which case `Y = self`.

        Returns:
            np.ndarray: Inner product.
        """
        assert NotImplementedError("This implementation has to be tested")
        if Y is None:
            Y = self
        G_i = self.inp_feat.inner(Y.inp_feat)
        G_o = self.outp_feat.inner(Y.outp_feat)

        # check the following expression again
        if Y.matr.size < self.matr.size:
            return np.sum((G_o.T @ self.matr @ G_i) * Y.matr)
        else:
            return np.sum((G_o @ Y.matr @ G_i.T) * self.matr)

        # is the kronecker product taken the right way around or do self and Y have to switch plaches?
        # return self.reduce_gram(Y.reduce_gram(G_i.T.reshape(1, -1) @
        # np.kron(self.matr, Y.matr) @ G_o.reshape(1, -1), 1), 0)
