from copy import copy
from ..reduce.centop_reductions import CenterInpFeat, DecenterOutFeat
from ..reduce.lincomb import LinearReduce
from ..reduce.base import Prefactors, Sum
from typing import Generic, TypeVar, Callable, Union

import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from ..rkhs.vector import FiniteVec, inner
from ..core.typing import AnyOrInitFn, Array

from .base import LinOp, RkhsObject, Vec, InpVecT, OutVecT, RhInpVectT, CombT


class FiniteOp(LinOp[InpVecT, OutVecT]):
    """Finite rank LinOp in RKHS"""
    
    

    def __init__(self, inp_feat:InpVecT, outp_feat:OutVecT, matr:Array = None, normalize:bool = False):
        super().__init__()
        if matr is not None:
            assert matr.shape == (len(outp_feat), len(inp_feat))
        self.matr = matr        
        self.inp_feat, self.outp_feat = inp_feat, outp_feat
        self._normalize = normalize

    def __len__(self):
        return len(self.inp_feat) * len(self.outp_feat)

    def __matmul__(self, right_inp:CombT) -> Union[OutVecT, "FiniteOp[RhInpVectT, OutVecT]"]:
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
            lin_LinOp = inner(self.inp_feat, right_inp)
            
            if self.matr is not None:
                lin_LinOp = self.matr @ lin_LinOp
            if self._normalize:
                lin_LinOp = lin_LinOp / lin_LinOp.sum(1, keepdims = True)
            lr = LinearReduce(lin_LinOp.T)
            if len(right_inp) != 1:
                return self.outp_feat.extend_reduce([lr])
            else:
                return self.outp_feat.extend_reduce([lr, Sum()])
    
    def inner(self, Y:"FiniteOp[InpVecT, OutVecT]"=None, full=True):
        assert NotImplementedError("This implementation as to be tested")
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
        #return self.reduce_gram(Y.reduce_gram(G_i.T.reshape(1, -1) @ np.kron(self.matr, Y.matr) @ G_o.reshape(1, -1), 1), 0)
    
    
    def reduce_gram(self, gram, axis = 0):
        return gram
    
    @property
    def T(self) -> "FiniteOp[OutVecT, InpVecT]":
        return FiniteOp(self.outp_feat, self.inp_feat, self.matr.T, self._normalize)

    def __call__(self, inp:DeviceArray) -> RkhsObject:
        return self @ FiniteVec(self.inp_feat.k, np.atleast_2d(inp))
    
    def apply(self, inp:CombT) -> RkhsObject:
        return self @ inp



