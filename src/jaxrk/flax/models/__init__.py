"""Flax based implementation of RKHS models. This allows to train the parameters by gradient descent, e.g. kernel parameters."""
from .gp import FlaxGP
from .base_rkhs import RkhsVecEncoder, OneToOneEncoder
