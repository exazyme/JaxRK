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
from ..utilities.cv import invert_loo, loo_I

from .base import LinOp, RkhsObject, Vec, InpVecT, OutVecT, RhInpVectT, CombT
from .cov import *
from .operator import FiniteOp

def Cmo(inp_feat:InpVecT, outp_feat:OutVecT, regul:float = None) -> FiniteOp[InpVecT, OutVecT]:
        if regul is not None:
            regul = np.array(regul, dtype=np.float32)
            assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
        return  CrossCovOp(Cov_solve(CovOp(inp_feat), inp_feat, regul=regul), outp_feat)

def RidgeCmo(inp_feat:InpVecT, outp_feat:OutVecT, regul:float = None) -> FiniteOp[InpVecT, OutVecT]:
        if regul is None:
            regul = Cov_regul(1, len(inp_feat))
        else:
            regul = np.array(regul, dtype=np.float32)
            assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
        matr = np.linalg.inv(inp_feat.inner() + regul * np.eye(len(inp_feat)))
        return  FiniteOp(inp_feat, outp_feat, matr)

def Cdo(inp_feat:InpVecT, outp_feat:OutVecT, ref_feat:OutVecT, regul = None) -> FiniteOp[InpVecT, OutVecT]:
        if regul is not None:
            regul = np.array(regul, dtype=np.float32)
            assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
        mo = Cmo(inp_feat, outp_feat, regul)
        rval =  Cov_solve(CovOp(ref_feat), mo, regul=regul)
        return rval