import numpy as np


def tensor(U, J):

    K_4 = np.zeros((4, 4, 4, 4), dtype=float)
    a, b, c, d = (
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([0, 2, 1, 3, 0, 2, 1, 3]),
        np.array([3, 3, 2, 2, 1, 1, 0, 0]),
        np.array([3, 1, 2, 0, 3, 1, 2, 0]),
    )
    K_4[a, b, c, d] += J

    a, b, c, d = (
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([1, 3, 0, 2, 1, 3, 0, 2]),
        np.array([3, 3, 2, 2, 1, 1, 0, 0]),
        np.array([2, 0, 3, 1, 2, 0, 3, 1]),
    )
    K_4[a, b, c, d] -= J

    a, b, c, d = (
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([1, 0, 3, 2]),
        np.array([1, 0, 3, 2]),
    )
    K_4[a, b, c, d] += 2 * J

    a, b, c, d = (
        np.array([0, 1, 2, 3]),
        np.array([1, 0, 3, 2]),
        np.array([1, 0, 3, 2]),
        np.array([0, 1, 2, 3]),
    )
    K_4[a, b, c, d] -= 2 * J

    a, b, c, d = (
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
        np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    )
    K_4[a, b, c, d] += U

    a, b, c, d = (
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
        np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
        np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]),
    )
    K_4[a, b, c, d] -= U

    return K_4
