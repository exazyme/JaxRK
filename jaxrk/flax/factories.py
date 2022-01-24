from mimetypes import init
import jax.numpy as np
from flax.linen import Module
from typing import Callable, Generic, Sequence, Tuple, TypeVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pedata.disk_cache import load_similarity

from ..kern import GenGaussKernel, DictKernel
from ..core.typing import Bijection, Array, PRNGKeyT
from ..core import constraints


K = TypeVar("K")

class Factory(Generic[K], ABC):
    @abstractmethod
    def __call__(self, flax_mod: Module, param_name:str, ) -> K:
        pass

class ConstFactory(Generic[K]):
    def __init__(self, constant:K):
        self.returned_instance = constant

    def __call__(self, flax_mod: Module, param_name:str, ) -> K:
        return self.returned_instance
    
    @staticmethod
    def wrap(factory_or_instance:Union[Factory[K], K]) -> Factory[K]:
        if isinstance(factory_or_instance, Factory):
            return factory_or_instance
        else:
            return ConstFactory(factory_or_instance)


class GenGaussFactory(Factory[GenGaussKernel]):

    def __init__(self,
                 scale_init_fn:Callable[[PRNGKeyT], Array],
                 shape_init_fn:Callable[[PRNGKeyT], Array],
                 scale_bij:Bijection = constraints.NonnegToLowerBd(lower_bound = 0.001, bij = constraints.SquarePlus()),
                 shape_bij:Bijection = constraints.SquashingToBounded(lower_bound = 0., upper_bound = 2.)) -> None:
        super().__init__()
        self.scale_init_fn = scale_init_fn
        self.shape_init_fn = shape_init_fn
        self.scale_bij = scale_bij
        self.shape_bij = shape_bij

    def __call__(self, flax_mod: Module, param_name: str) -> K:
        k = GenGaussKernel.make(self.scale_bij(flax_mod.param(f"{param_name}_scale", self.scale_init_fn)),
                                self.shape_bij(flax_mod.param(f"{param_name}_shape", self.shape_init_fn).flatten()[0]))
        #print(k)
        return k
    
    @staticmethod
    def from_constrained(scale_init:Array,
                         shape_init:float,
                         scale_init_noise:Callable[[PRNGKeyT], Array],
                         shape_init_noise:Callable[[PRNGKeyT], Array],
                         scale_lower_bound:float, shape_lower_bound:float, shape_upper_bound:float):
        sc_b:Bijection = constraints.NonnegToLowerBd(lower_bound = scale_lower_bound, bij = constraints.SquarePlus())
        sh_b:Bijection = constraints.SquashingToBounded(lower_bound = shape_lower_bound, upper_bound = shape_upper_bound)
                         
        return GenGaussFactory(scale_init_fn = lambda rng: sc_b.inv(scale_init) + scale_init_noise(rng),
                               shape_init_fn = lambda rng: sh_b.inv(shape_init) + shape_init_noise(rng),
                               scale_bij = sc_b,
                               shape_bij = sh_b)
    

class DictKernFactory(Factory[DictKernel]):
    def __init__(self,
                 inspace_vals:Sequence[str],
                 similarity_init_fn:Callable[[PRNGKeyT], Array],
                 chol_bij:Bijection = constraints.CholeskyBijection(diag_bij=constraints.NonnegToLowerBd(lower_bound = np.finfo(np.float32).tiny, bij = constraints.SquarePlus())),
                ) -> None:
        super().__init__()
        self.inspace_vals = inspace_vals
        self.init_fn = similarity_init_fn
        self.chol_bij = chol_bij
    
    def __call__(self, flax_mod: Module, param_name: str) -> K:
        return DictKernel(self.inspace_vals,
                          cholesky_lower = self.chol_bij.param_to_chol(flax_mod.param(param_name, self.init_fn)))
    
    @staticmethod
    def from_similarity(alphabet_type:str, similarity_name:str, noise:Callable[[PRNGKeyT, tuple[int, int]], Array], diag_regul:float = 0.):
        insp_vals, sm = load_similarity(alphabet_type, similarity_name)
        sm = sm + np.eye(sm.shape[0]) * diag_regul
        chol_bij = constraints.CholeskyBijection(diag_bij=constraints.NonnegToLowerBd(lower_bound = np.finfo(np.float32).tiny, bij = constraints.SquarePlus()))
        init_param_lower = chol_bij.psd_to_param(sm)
        init_fn = lambda rng: np.tril(noise(rng, sm.shape)) + init_param_lower
        return DictKernFactory(insp_vals, init_fn, chol_bij)
