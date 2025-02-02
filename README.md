# JaxRK

JaxRK is a library for working with (vectors of) RKHS elements and RKHS operators using [JAX](https://github.com/google/jax) for automatic differentiation. This library includes implementations of [kernel transfer operators](https://arxiv.org/abs/1712.01572) and [conditional density operators](https://arxiv.org/abs/1905.11255).

## High level overview of software design

### Elementary Kernels
Elementary kernels can be applied directly to input space points $R^d$, such that $k: R^d \times R^d \mapsto R$. They adhere to the API defined in `jaxrk.kern.base.Kernel`.

### RKHS Elements
RKHS elements $φ ∈ H$ are linear combinations of elementary kernels with one fixed argument. For example, $Σ_i a_i k(x_i, .)$ is an RKHS element where $H$ is induced by the kernel $k$. If $a_i = 1/N$ for all $i$ and $\\{x_i\\}_{i=1}^N$ are samples from a distribution, this is the kernel mean embedding.

A more complex RKHS element would be $Σ_{i,j} a_{i,j} k_1(x_i, .) k_2(x´_j, .)$, involving two different elementary kernels $k_1$ and $k_2$ with two input spaces containing $x_i$ and $x´_j$. For example $k_1$ could be a kernel on nodes and $k_2$ a kernel on edges of a graph or in general elements of a data point with some complex inner structure.

### Vectors of RKHS Elements
Vectors of RKHS elements are simply represented as $[φ_1, …, φ_N] ∈ H^N$ and follow the API defined in `jaxrk.rkhs.base.Vec`. The most commonly used concrete implementation is `jaxrk.rhks.vec.FiniteVec`.

The simplest RKHS vector is one where each $φ_i$ corresponds to exactly one input space point, i.e., $φ_i = k(x_i, .)$. It can be constructed using `jaxrk.rhks.vec.FiniteVec(kernel_object, input_space_points)`. If the number of rows in `input_space_points` equals $N$, this means the constructed RKHS vector is in $H^N$.

#### Kernel models build upon vectors of RKHS elements

One commonly cited advantage of kernel models is that you only have to define a new kernel to apply the same upstream models like Gaussian Processes, Conditional Mean and conditional density operators, kernel based classifiers etc. to not only numbers but also more complicated data like graphs, sequences, etc.

Using RKHS vectors $\Phi = [φ_1, …, φ_N]$ as training data and RKHS vectors $\Phi´ = [φ´_1, …, φ´_M]$ as test data, this can become even easier. Concretely, each vector element could repesenting sequences, graphs for example as $Σ_{i,j} a_{i,j} k_1(x_i, .) k_2(x´_j, .)$ using only elementary kernels. This makes it unecessary to introduce new kernels. Instead, we need to define a way of ecoding the data into RKHS vectors.

The advantage is that all upstream models can automatically handle very complex data using only elementary kernels and a smart linear combination. For example, one can use all upstream models directly with distributions as input by simply representing each distribution by its mean embedding. Concretely, the training data would look like $\Phi = [\sum_i k(x^{(1)}_i, \cdot), …, \sum_i k(x^{(N)}_i, \cdot)]$, where $x^{(j)}_i$ is the $i$-th sample from the $j$-th distribution.

#### Reductions of RKHS Vectors
To construct an RKHS vector containing more complex RKHS elements, reductions can be used, which are implemented in `jaxrk.reduce.base.Reduce`. These take $[φ_1, …, φ_N]$ and often simply map them through a real matrix $A ∈ R^{M \times N}$. Concretely, $[φ_1, …, φ_N]$ would be mapped to $[Σ_i a_{1,i} \cdot φ_i, Σ_i a_{2,i} \cdot φ_i, …, Σ_i a_{M,i}\cdot φ_i]$.


#### Vectors of RKHS elements can themselves be input space points to a kernel

Given $\phi, \phi´ \in H$ one can define kernels such as $exp(-\|\phi-\phi´\|_H)$ (Laplace kernel with an RKHS input space) or $(-\langle\phi,\phi´\rangle_H + c)^d$ (polynomial kernel with an RKHS input space). Having introduced vectors of RKHS elements, implementing this becomes easier. The kernel implementation stays the same. Only the distances that are computed either as euclidean or RKHS distances, the inner products as  standard or RKHS inner products.

## Installation
First you have to make sure to have jax and jaxlib installed. Please follow the [JAX installation instructions](https://github.com/google/jax) depending on whether you want a CPU or GPU/TPU installation. After that you only need
```
$ pip install jaxrk
```

## Quick start examples

For some examples of what you can do with JaxRK, see [examples/Quick_start.ipynb](https://github.com/zalandoresearch/JaxRK/blob/master/examples/Quick_start.ipynb).


## Development

To help in developing JaxRK, clone the github repo and change to the cloned directory on the command line. Then 
```
$ pip install -e ".[ci]"
$ pytest test/
```
will install the package into your python path. Changes to files in the directory are reflected in the python package when loaded.
If you want to send pull requests, use the tool `black` for ensuring the correct style.
To browse the documentation, use

```
pdoc --docformat google jaxrk
```
