from typing import Any
from .base_rkhs import RkhsVecEncoder
import flax.linen as nn

from ...models import gp as jax_gp
from ...core.typing import Array


class FlaxGP(nn.Module):
    """GP model as a Flax module such that encoder (and thus kernel) can be trained by gradient descent."""

    encode_inp: RkhsVecEncoder
    train_inp: Any
    train_outp: Any
    regul: float

    def setup(self):
        """Flax setup method."""
        self.train_inp_vec = self.encode_inp(self.train_inp)

    def get_gp(self) -> jax_gp.GP:
        """Returns the vanilla GP object with the current parameters.

        Returns:
            jax_gp.GP: The GP object.
        """
        return jax_gp.GP(
            self.train_inp_vec, self.train_outp, self.regul, normalize_y=True
        )  # (self.train_inp_vec, self.train_outp_vec)

    def neg_llhood(self, cv_split: Array) -> float:
        """Marginal negative log-likelihood of the GP.

        Args:
            cv_split (Array): The cross-validation split indices.

        Returns:
            float: Negative log-likelihood.
        """  # Marginal likelihood:
        return -self.get_gp().marginal_loglhood()
