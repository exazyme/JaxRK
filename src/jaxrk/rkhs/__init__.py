r"""
# RKHS module
This module contains the main classes for handling RKHS objects.
In particular, it contains implementations for vectors of RKHS elements and operators acting on such vectors.
"""
from .base import Vec, LinOp, inner
from .vector import FiniteVec, CombVec
from .operator import *
from .cov import *
from .encoder import *
