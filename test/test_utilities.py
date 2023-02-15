from jaxrk.utilities.eucldist import eucldist
from jaxrk.utilities.distances import dist
from jaxrk.utilities.linalg import inv_blockmatr
from scipy.spatial.distance import cdist
from numpy.random import randn
from jax.numpy import allclose
import jax.numpy as np
from jaxrk.kern import LinearKernel, GenGaussKernel
from jaxrk.rkhs import FiniteVec
import sys


def test_eucldist():
    a, b = randn(100, 3), randn(200, 3)
    variants = ("simple", "extension")
    truth_est = []
    for (name, pow) in [("sqeuclidean", 2), ("euclidean", 1)]:
        for dist_args in [(a, b), (a, a)]:
            ground_truth = cdist(*dist_args, name)
            for v in variants:
                d = eucldist(*dist_args, power=pow, variant=v)
                assert allclose(ground_truth, d, atol=1e-02)


def test_dist():
    a, b = randn(100, 3), randn(200, 3)
    print(
        "Testing distance computation in input space, i.e. standard arrays",
        file=sys.stderr,
    )
    for (name, pow) in [("sqeuclidean", 2), ("euclidean", 1)]:
        for dist_args in [(a, b), (a, a)]:
            ground_truth = cdist(*dist_args, name)
            d = dist(*dist_args, power=pow)
            print("*", name, pow, np.abs(ground_truth - d).max())
            assert allclose(ground_truth, d, atol=1e-03)

    print(
        "Testing distance computation between vectors of RKHS elements", file=sys.stderr
    )
    k = LinearKernel()
    va, vb = FiniteVec(k, a), FiniteVec(k, b)
    for pow in [2, 1]:
        for insp_args, dist_args in [[(a, b), (va, vb)], [(a, a), (va, va)]]:
            lin_rkhsdist_sq = (
                np.diag(insp_args[0] @ insp_args[0].T)[:, np.newaxis]
                - 2 * insp_args[0] @ insp_args[1].T
                + np.diag(insp_args[1] @ insp_args[1].T)[np.newaxis, :]
            )
            ground_truth = np.power(lin_rkhsdist_sq, pow / 2.0)
            d = dist(*dist_args, power=pow)
            print("*", pow, np.abs(ground_truth - d).max())
            assert allclose(ground_truth, d, atol=1e-05)


def test_inv_blockmatr():
    G = GenGaussKernel.make_gauss(1.0)(np.array([[1, 2, 3, 4]]).T + np.eye(4) * 3)
    inv_G = np.linalg.inv(G)
    for i in (0, 1, 2, 3, 4):
        P, Q, R, S = G[:i, :i], G[:i, i:], G[i:, :i], G[i:, i:]
        P_inv = np.linalg.inv(P)
        inv_Blk = inv_blockmatr(P, P_inv, Q, R, S)
        assert allclose(inv_G, inv_Blk, atol=1e-05)
        print(inv_G - inv_Blk)
