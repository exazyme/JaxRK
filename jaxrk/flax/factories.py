from flax.linen import Module
from typing import Callable, Generic, Sequence, Tuple, TypeVar, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass


from ..kern import GenGaussKernel, DictKernel
from ..core.typing import Bijection, Array, PRNGKeyT
from ..core import constraints


K = TypeVar("K")

class Factory(Generic[K], ABC):
    @abstractmethod
    def __call__(self, flax_mod: Module, param_name:str, ) -> K:
        pass

# class ConstFactory(Generic[K], dataclass):
#     def __init__(self, constant:K):
#         self.returned_instance = constant

#     def __call__(self, flax_mod: Module, param_name:str, ) -> K:
#         return self.returned_instance

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
        return GenGaussKernel.make(self.scale_bij(flax_mod.param(f"{param_name}_scale", self.scale_init_fn)),
                                   self.shape_bij(flax_mod.param(f"{param_name}_shape", self.shape_init_fn).flatten()[0]))
    
    @staticmethod
    def from_constrained(scale_init:Array,
                         shape_init:float,
                         scale_init_noise:Callable[[PRNGKeyT], Array],
                         shape_init_noise:Callable[[PRNGKeyT], Array],
                         scale_lower_bound:float, shape_lower_bound:float, shape_upper_bound:float):
        sc_b:Bijection = constraints.NonnegToLowerBd(lower_bound = scale_lower_bound, bij = constraints.SquarePlus())
        sh_b:Bijection = constraints.SquashingToBounded(lower_bound = shape_lower_bound, upper_bound = shape_upper_bound)
                         
        return GenGaussFactory(scale_init_fn = lambda rng: sc_b.inv(scale_init) + scale_init_noise(rng),
                               shape_init_fn = lambda rng: sh_b.inv(scale_init) + shape_init_noise(rng),
                               scale_bij = sc_b,
                               shape_bij = sh_b)
    
    

