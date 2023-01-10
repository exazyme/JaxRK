import jax.numpy as np
import numpy.random as random


def nystrom_eigh(gram: np.ndarray, n_comp: int, regul: float = 0.0) -> tuple:
    """Eigenvalue decomposition of a gram matrix using the Nystrom method.

    Args:
        gram (np.ndarray): The gram matrix.
        n_comp (int): Number of components to keep in the nyström approximation.
        regul (float, optional): Regularization parameter. Defaults to 0.0.

    Returns:
        tuple: The eigenvectors of the used input points, the extrapolated eigenvectors, and the eigenvalues.
    """
    assert len(gram.shape) == 2
    assert gram.shape[0] == gram.shape[1]
    assert gram.shape[0] >= n_comp

    perm = np.arange(gram.shape[0])
    idx_in = perm[:n_comp]
    idx_out = perm[n_comp:]
    λ, vec_in = np.linalg.eigh(gram[idx_in, :][:, idx_in])
    vec_out = gram[idx_out, :][:, idx_in] @ vec_in @ np.diag(1.0 / (λ + regul))
    return (vec_in, vec_out, λ)


def nystrom_inv(gram: np.ndarray, n_comp: int, regul: float = 0.0) -> np.array:
    """Inverse of a gram matrix using the Nystrom method.

    Args:
        gram (np.ndarray): The gram matrix.
        n_comp (int): Number of components to keep in the nyström approximation.
        regul (float, optional): Regularization parameter. Defaults to 0.0.

    Returns:
        np.array: The inverse of the gram matrix as a dense matrix.
    """
    p = random.permutation(gram.shape[0])
    ip = np.argsort(p)
    (vec_in, vec_out, λ) = nystrom_eigh(gram[p, :][:, p], n_comp, regul)
    vec = np.vstack([vec_in, vec_out])
    rval = vec @ np.diag(1.0 / (λ + regul)) @ vec.T
    return rval[ip, :][:, ip]
