from copy import copy


import jax.numpy as np
from jax.interpreters.xla import DeviceArray
from scipy.optimize import minimize

from ..core.typing import AnyOrInitFn, Array

from ..rkhs import InpVecT, OutVecT, inner
from ..rkhs.cov import *
from ..rkhs.operator import FiniteOp


def Cmo(
    inp_feat: InpVecT, outp_feat: OutVecT, regul: float = None
) -> FiniteOp[InpVecT, OutVecT]:
    """Constructs a conditional mean operator (CMO) from input and output features. Assumes that input space points of input and output features were sampled from a joint distribution P(`inp`, `outp`), then mapped into the feature space, resulting in `inp_feat` and `outp_feat` respectively.
    Computes the result by solving a linear system of equations in the feature space.


    Args:
        inp_feat (InpVecT): Input features.
        outp_feat (OutVecT): Output features.
        regul (float, optional): Regularization parameter. Defaults to None, leaving it to Cov_solve to determine.

    Returns:
        FiniteOp[InpVecT, OutVecT]: Conditional mean operator.
    """
    if regul is not None:
        regul = np.array(regul, dtype=np.float32)
        assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
    return CrossCovOp(Cov_solve(CovOp(inp_feat), inp_feat, regul=regul), outp_feat)


def RidgeCmo(
    inp_feat: InpVecT, outp_feat: OutVecT, regul: float = None
) -> FiniteOp[InpVecT, OutVecT]:
    """Constructs a conditional mean operator (CMO) from input and output features. Assumes that input space points of input and output features were sampled from a joint distribution P(`inp`, `outp`), then mapped into the feature space, resulting in `inp_feat` and `outp_feat` respectively.
    Estimated using the classical closed form solution (see Muandet et al. Kernel Mean Embedding of Distributions 2016).

    Args:
        inp_feat (InpVecT): Input features.
        outp_feat (OutVecT): Output features.
        regul (float, optional): Regularization parameter. Defaults to None, leaving it to Cov_solve to determine.

    Returns:
        FiniteOp[InpVecT, OutVecT]: Conditional mean operator.
    """
    if regul is None:
        regul = Cov_regul(1, len(inp_feat))
    else:
        regul = np.array(regul, dtype=np.float32)
        assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
    matr = np.linalg.inv(inp_feat.inner() + regul * np.eye(len(inp_feat)))
    return FiniteOp(inp_feat, outp_feat, matr)


def Cdo(
    inp_feat: InpVecT, outp_feat: OutVecT, ref_feat: OutVecT, regul=None
) -> FiniteOp[InpVecT, OutVecT]:
    """Constructs a conditional density operator (CDO) from input and output features. Assumes that input space points of input and output features were sampled from a joint distribution P(`inp`, `outp`), then mapped into the feature space, resulting in `inp_feat` and `outp_feat` respectively.
    The resulting operator will map a new input directly to a density estimate over the output variable.
    Estimated using the closed form solution given in Schuster et al. 2019, "Kernel Conditional Density Operators".

    Args:
        inp_feat (InpVecT): Input features.
        outp_feat (OutVecT): Output features.
        regul (float, optional): Regularization parameter. Defaults to None, leaving it to Cov_solve to determine.

    Returns:
        FiniteOp[InpVecT, OutVecT]: Conditional mean operator.
    """
    if regul is not None:
        regul = np.array(regul, dtype=np.float32)
        assert regul.squeeze().size == 1 or regul.squeeze().shape[0] == len(inp_feat)
    mo = Cmo(inp_feat, outp_feat, regul)
    rval = Cov_solve(CovOp(ref_feat), mo, regul=regul)
    return rval
