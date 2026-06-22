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


# ----------------------------------------------------------------------
def split_quartic_tensor_in_charge_and_spin(U_4):
    """Assuming spin is the slow index and orbital is the fast index

    Using a rank 4 U_abcd tensor with composite (spin, orbital) indices
    as input, assuming c^+c^+ c c structure of the tensor

    Returns:

    U_c : Charge channel rank 4 interaction tensor
    U_s : Spin channel rank 4 interaction tensor"""

    shape_4 = np.array(U_4.shape)
    shape_8 = np.vstack(([2] * 4, shape_4 // 2)).T.flatten()

    U_8 = U_4.reshape(shape_8)

    U_8 = np.transpose(U_8, (0, 2, 4, 6, 1, 3, 5, 7))  # spin first

    # -- Check spin-conservation

    zeros = np.zeros_like(U_8[0, 0, 0, 0])

    np.testing.assert_array_almost_equal(U_8[0, 0, 0, 1], zeros)
    np.testing.assert_array_almost_equal(U_8[0, 0, 1, 0], zeros)
    np.testing.assert_array_almost_equal(U_8[0, 1, 0, 0], zeros)
    np.testing.assert_array_almost_equal(U_8[1, 0, 0, 0], zeros)

    np.testing.assert_array_almost_equal(U_8[1, 1, 1, 0], zeros)
    np.testing.assert_array_almost_equal(U_8[1, 1, 0, 1], zeros)
    np.testing.assert_array_almost_equal(U_8[1, 0, 1, 1], zeros)
    np.testing.assert_array_almost_equal(U_8[0, 1, 1, 1], zeros)

    np.testing.assert_array_almost_equal(U_8[1, 0, 1, 0], zeros)
    np.testing.assert_array_almost_equal(U_8[0, 1, 0, 1], zeros)

    # -- split in charge and spin
    
    U_c = U_8[0,0,1,1] + U_8[0,0,0,0]
    U_s = U_8[0,0,1,1] - U_8[0,0,0,0]

    return -U_c, -U_s
