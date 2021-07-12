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
from .operator import FiniteOp

def CrossCovOp(inp_feat:InpVecT, outp_feat:OutVecT) -> FiniteOp[InpVecT, OutVecT]:
    assert len(inp_feat) == len(outp_feat)
    N = len(inp_feat)
    return FiniteOp(inp_feat, outp_feat, np.eye(N)/N)

def CovOp(inp_feat:InpVecT) -> FiniteOp[InpVecT, InpVecT]:
    N = len(inp_feat)
    return FiniteOp(inp_feat, inp_feat, np.eye(N)/N)
    
def CovOp_from_Samples(kern, inspace_points, prefactors = None) -> FiniteOp[InpVecT, InpVecT]:
    return CovOp(FiniteVec(kern, inspace_points, prefactors))
    
def Cov_regul(nsamps:int, nrefsamps:int = None, a:float = 0.49999999999999, b:float = 0.49999999999999, c:float = 0.1):
        """Compute the regularizer based on the formula from the Kernel Conditional Density operators paper (Schuster et al., 2020, Corollary 3.4).
        
        smaller c => larger bias, tighter stochastic error bounds
        bigger  c =>  small bias, looser stochastic error bounds

        Args:
            nsamps (int): Number of samples used for computing the RKHS embedding.
            nrefsamps (int, optional): Number of samples used for computing the reference distribution covariance operator. Defaults to nsamps.
            a (float, optional): Parameter a. Assume a > 0 and a < 0.5, defaults to 0.49999999999999.
            b (float, optional): Parameter b. Assume a > 0 and a < 0.5, defaults to 0.49999999999999.
            c (float, optional): Bias/variance tradeoff parameter. Assume c > 0 and c < 1, defaults to 0.1.

        Returns:
            [type]: [description]
        """
        if nrefsamps is None:
            nrefsamps = nsamps
            
        assert(a > 0 and a < 0.5)
        assert(b > 0 and b < 0.5)
        assert(c > 0 and c < 1)
        assert(nsamps > 0 and nrefsamps > 0)
        
        return max(nrefsamps**(-b*c), nsamps**(-2*a*c))

def Cov_inv(cov:FiniteOp[InpVecT, InpVecT], regul:float = None) -> "FiniteOp[InpVecT, InpVecT]":
    """Compute the inverse of this covariance operator with a certain regularization.

    Args:
        regul (float, optional): Regularization parameter to be used. Defaults to self.regul.

    Returns:
        FiniteOp[InpVecT, InpVecT]: The inverse operator
    """
    assert regul is not None
    gram = inner(cov.inp_feat)
    inv_gram = np.linalg.inv(gram + regul * np.eye(len(cov.inp_feat)))
    matr = (inv_gram @ inv_gram)
    if cov.matr is not None:
        #according to eq. 9 in appendix A.2 of "Kernel conditional density operators", the following line would rather be
        # matr = cov.matr @ cov.matr @ matr
        #this might be a mistake in the paper.
        matr = np.linalg.inv(cov.matr) @ matr
        
    rval = FiniteOp(cov.inp_feat, cov.inp_feat, matr)
        
    return rval
    
def Cov_solve(cov:FiniteOp[InpVecT, InpVecT], lhs:CombT, regul:float = None) -> RkhsObject:
    """If `inp` is an RKHS vector of length 1 (a mean embedding): Solve the inverse problem to find dP/dρ from equation
    μ_P = C_ρ dP/dρ
    where C_ρ is the covariance operator passed as `cov`, ρ is the reference distribution, and μ_P is given by `lhs`.
    If `lhs` is a `FiniteOp`: Solve the inverse problem to find operator B from equation
    A = C_ρ B
    where C_ρ is the covariance operator passed as `cov`, and A is given by `lhs`.
    
    Args:
        lhs (CombT): The embedding of the distribution of interest, or the LinOp of interest.
    """
    
    if isinstance(lhs, FiniteOp):
        reg_inp = lhs.outp_feat
    else:
        if isinstance(lhs, DeviceArray):
            lhs = FiniteVec(cov.inp_feat.k, np.atleast_2d(lhs))
        reg_inp = lhs
    if regul is None:
        regul = Cov_regul(1, len(cov.inp_feat))
    return (Cov_inv(cov, regul) @ lhs)