from typing import Any
from .base_rkhs import RkhsVecEncoder
import flax.linen as nn

from ...models import gp as jax_gp


class FlaxGP(nn.Module):

    encode_inp: RkhsVecEncoder
    train_inp: Any
    train_outp: Any
    regul: float

    def setup(self):
        self.train_inp_vec = self.encode_inp(self.train_inp)

    def get_gp(self) -> jax_gp.GP:
        # import pdb
        # pdb.set_trace()
        return jax_gp.GP(
            self.train_inp_vec, self.train_outp, self.regul, normalize_y=True
        )  # (self.train_inp_vec, self.train_outp_vec)

    def neg_llhood(self, cv_split) -> float:
        # Marginal likelihood:
        return -self.get_gp().marginal_loglhood()
