import jax.numpy as np


def tile_view(inp: np.ndarray, reps: int) -> np.ndarray:
    """Tile a view of an array

    Args:
        inp (np.ndarray): Array to tile
        reps (int): Repetitions of the array

    Returns:
        np.ndarray: Tile view of the array

    Example:
        >>> tile_view(np.array([[1, 2], [3, 4]]), 2)
        DeviceArray([[1, 2],
                     [3, 4],
                     [1, 2],
                     [3, 4]], dtype=int32)
    """
    return np.broadcast_to(inp.ravel(), (reps, inp.size)).reshape(
        (reps * inp.shape[0], inp.shape[1])
    )
