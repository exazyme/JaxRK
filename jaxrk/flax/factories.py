from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, Tuple, TypeVar, Union

import jax.numpy as np
import numpy as onp

from flax.linen import Module

from ..core import constraints
from ..core.typing import Array, Bijection, PRNGKeyT
from ..kern import DictKernel, GenGaussKernel

K = TypeVar("K")


class Factory(Generic[K], ABC):
    """Abstract base class for all JaxRK factories. These are used to create JaxRK objects while registering their parameters with Flax."""

    @abstractmethod
    def __call__(
        self,
        flax_mod: Module,
        param_name: str,
    ) -> K:
        """Create a JaxRK object, registering its parameters with Flax.

        Args:
            flax_mod (Module): The Flax module to register the parameters with.
            param_name (str): The name of the parameter to register.

        Returns:
            K: The JaxRK object.
        """
        pass


class ConstFactory(Generic[K]):
    """Factory for constant JaxRK objects, i.e. objects that are initialized with a constant value and do not have any parameters that should be registered as Flax parameters. This is useful if some parameters should not be trained."""

    def __init__(self, constant: K):
        """Initialize the factory.

        Args:
            constant (K): The constant value to return when the factory is called.
        """
        self.returned_instance = constant

    def __call__(
        self,
        flax_mod: Module,
        param_name: str,
    ) -> K:
        """Return the constant value. This does not register any parameters with Flax.

        Args:
            flax_mod (Module): The Flax module to register the parameters with.
            param_name (str): The name of the parameter to register.

        Returns:
            K: The constant value.
        """
        return self.returned_instance

    @staticmethod
    def wrap(factory_or_instance: Union[Factory[K], K]) -> Factory[K]:
        """Ensure that the given object is a factory. Constant values are wrapped in a ConstFactory, factories are returned unchanged.

        Args:
            factory_or_instance (Union[Factory[K], K]): The object to wrap.

        Returns:
            Factory[K]: The wrapped object.
        """
        if isinstance(factory_or_instance, Factory) or factory_or_instance is None:
            return factory_or_instance
        else:
            return ConstFactory(factory_or_instance)


class GenGaussKernelFactory(Factory[GenGaussKernel]):
    """Factory for GenGaussKernel objects."""

    def __init__(
        self,
        scale_init_fn: Callable[[PRNGKeyT], Array],
        shape_init_fn: Callable[[PRNGKeyT], Array],
        scale_bij: Bijection = constraints.NonnegToLowerBd(
            lower_bound=0.001, bij=constraints.SquarePlus()
        ),
        shape_bij: Bijection = constraints.SquashingToBounded(
            lower_bound=0.0, upper_bound=2.0
        ),
    ) -> None:
        """Initialize the factory.

        Args:
            scale_init_fn (Callable[[PRNGKeyT], Array]): Function to initialize the scale parameter.
            shape_init_fn (Callable[[PRNGKeyT], Array]): Function to initialize the shape parameter.
            scale_bij (Bijection, optional): Bijection to apply to the scale parameter. Defaults to constraints.NonnegToLowerBd(lower_bound=0.001, bij=constraints.SquarePlus()).
            shape_bij (Bijection, optional): Bijection to apply to the shape parameter. Defaults to constraints.SquashingToBounded(lower_bound=0.0, upper_bound=2.0).
        """
        super().__init__()
        self.scale_init_fn = scale_init_fn
        self.shape_init_fn = shape_init_fn
        self.scale_bij = scale_bij
        self.shape_bij = shape_bij

    def __call__(self, flax_mod: Module, param_name: str) -> K:
        """Create a GenGaussKernel object, registering its parameters with Flax.

        Args:
            flax_mod (Module): The Flax module to register the parameters with.
            param_name (str): The name of the parameter to register.

        Returns:
            K: The GenGaussKernel object.
        """
        k = GenGaussKernel.make(
            self.scale_bij(flax_mod.param(f"{param_name}_scale", self.scale_init_fn)),
            self.shape_bij(
                flax_mod.param(f"{param_name}_shape", self.shape_init_fn).flatten()[0]
            ),
        )
        # print(k)
        return k

    @staticmethod
    def from_constrained(
        scale_init: Array,
        shape_init: float,
        scale_init_noise: Callable[[PRNGKeyT], Array],
        shape_init_noise: Callable[[PRNGKeyT], Array],
        scale_lower_bound: float,
        shape_lower_bound: float,
        shape_upper_bound: float,
    ) -> "GenGaussKernelFactory":
        """Create a GenGaussKernelFactory from constrained parameters.

        Args:
            scale_init (Array): Bias for scale parameter initialization.
            shape_init (float): Bias for shape parameter initialization.
            scale_init_noise (Callable[[PRNGKeyT], Array]): Noise function for scale parameter initialization.
            shape_init_noise (Callable[[PRNGKeyT], Array]): Noise function for shape parameter initialization.
            scale_lower_bound (float): Lower bound for scale parameter.
            shape_lower_bound (float): Lower bound for shape parameter.
            shape_upper_bound (float): Upper bound for shape parameter.

        Returns:
            GenGaussKernelFactory: The factory.
        """
        sc_b: Bijection = constraints.NonnegToLowerBd(
            lower_bound=scale_lower_bound, bij=constraints.SquarePlus()
        )
        sh_b: Bijection = constraints.SquashingToBounded(
            lower_bound=shape_lower_bound, upper_bound=shape_upper_bound
        )

        return GenGaussKernelFactory(
            scale_init_fn=lambda rng: sc_b.inv(scale_init) + scale_init_noise(rng),
            shape_init_fn=lambda rng: sh_b.inv(shape_init) + shape_init_noise(rng),
            scale_bij=sc_b,
            shape_bij=sh_b,
        )


class DictKernFactory(Factory[DictKernel]):
    """Factory for DictKernel objects."""

    def __init__(
        self,
        inspace_vals: Sequence[str],
        similarity_init_fn: Callable[[PRNGKeyT], Array],
        chol_bij: Bijection = constraints.CholeskyBijection(
            diag_bij=constraints.NonnegToLowerBd(
                lower_bound=np.finfo(np.float32).tiny, bij=constraints.SquarePlus()
            )
        ),
    ) -> None:
        """Initialize the factory.

        Args:
            inspace_vals (Sequence[str]): The names of the input space values.
            similarity_init_fn (Callable[[PRNGKeyT], Array]): Function to initialize the similarity matrix.
            chol_bij (Bijection, optional): Bijection to apply to the similarity matrix. Defaults to constraints.CholeskyBijection(diag_bij=constraints.NonnegToLowerBd(lower_bound=np.finfo(np.float32).tiny, bij=constraints.SquarePlus())).
        """
        super().__init__()
        self.inspace_vals = inspace_vals
        self.init_fn = similarity_init_fn
        self.chol_bij = chol_bij

    def __call__(self, flax_mod: Module, param_name: str) -> K:
        """Create a DictKernel object, registering its parameters with Flax.

        Args:
            flax_mod (Module): The Flax module to register the parameters with.
            param_name (str): The name of the parameter to register.
        """
        return DictKernel(
            self.inspace_vals,
            cholesky_lower=self.chol_bij.param_to_chol(
                flax_mod.param(param_name, self.init_fn)
            ),
        )

    @staticmethod
    def from_similarity(
        alph: list[str],
        sm: Array,
        noise: Callable[[PRNGKeyT, tuple[int, int]], Array],
        diag_regul: float = 0.0,
        diag_lower_bound: float = np.finfo(np.float32).tiny,
    ) -> "DictKernFactory":
        """Create a DictKernFactory from a similarity matrix.

        Args:
            alph (list[str]): The names of the input space values.
            sm (Array): The similarity matrix.
            noise (Callable[[PRNGKeyT, tuple[int, int]], Array]): Noise function for the similarity matrix.
            diag_regul (float, optional): Regularization for the diagonal of the similarity matrix. Defaults to 0.0.
            diag_lower_bound (float, optional): Lower bound for the diagonal of the similarity matrix. Defaults to np.finfo(np.float32).tiny.
        """
        sm = sm + np.eye(sm.shape[0]) * diag_regul
        return DictKernFactory.from_psd_matrix(alph, sm, noise, diag_lower_bound)

    @staticmethod
    def from_diagonal(
        alph: list[str],
        diagonal_value: float,
        noise: Callable[[PRNGKeyT, tuple[int, int]], Array],
        diag_lower_bound: float = np.finfo(np.float32).tiny,
    ) -> "DictKernFactory":
        """Create a DictKernFactory from a diagonal similarity matrix.

        Args:
            alph (list[str]): The alphabet to be used.
            diagonal_value (float): The value of the diagonal of the similarity matrix.
            noise (Callable[[PRNGKeyT, tuple[int, int]], Array]): Noise function for the similarity matrix.
            diag_lower_bound (float, optional): Lower bound for the diagonal of the similarity matrix. Defaults to np.finfo(np.float32).tiny.
        """

        sm = np.eye(alph.size) * diagonal_value
        return DictKernFactory.from_psd_matrix(alph, sm, noise, diag_lower_bound)

    @staticmethod
    def from_psd_matrix(
        alphabet: Sequence[str],
        psd_matrix: np.ndarray,
        noise: Callable[[PRNGKeyT, tuple[int, int]], Array],
        diag_lower_bound: float = np.finfo(np.float32).tiny,
    ) -> "DictKernFactory":
        """Create a DictKernFactory from a positive semi-definite similarity matrix.

        Args:
            alphabet (Sequence[str]): The alphabet to use.
            psd_matrix (np.ndarray): The similarity matrix to be used as bias for initialization.
            noise (Callable[[PRNGKeyT, tuple[int, int]], Array]): Function for generating noise that will be used for the unconstrained initial parameters.
            diag_lower_bound (float, optional): Lower bound for the diagonal of the similarity matrix. Defaults to np.finfo(np.float32).tiny.

        Returns:
            DictKernFactory: The factory.
        """
        chol_bij = constraints.CholeskyBijection(
            diag_bij=constraints.NonnegToLowerBd(
                lower_bound=diag_lower_bound, bij=constraints.SquarePlus()
            )
        )

        def init_fn(rng):
            return np.tril(noise(rng, psd_matrix.shape)) + chol_bij.psd_to_param(
                psd_matrix
            )

        return DictKernFactory(alphabet, init_fn, chol_bij)
