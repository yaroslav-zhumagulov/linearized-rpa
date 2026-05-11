import numpy as np
from lrpa.chi_module import calculate_chi_ph as _calculate_chi_ph
from lrpa.chi_module import calculate_chi_q as _calculate_chi_q


def calculate_chi_ph(model):
    chi_ph = np.zeros((2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu

    na, nk, norb, nb = model.u.shape

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u.transpose(2, 0, 1, 3))  # (norb, na, nk, nb) — stride-1 for dot_product

    _calculate_chi_ph(chi_ph, e, u, model.beta, na, nk, norb, nb)
    model.chi_ph = np.ascontiguousarray(chi_ph) / (model.N * model.N)


def calculate_chi_q(model, q):
    """
    Compute static chi^{tau,tau'}(q; G, G') via the Fortran kernel.

    Parameters
    ----------
    model : RhombohedralMultilayer
        Must have calculate_bandstructure() called and mu, beta set.
    q : array-like, shape (2,)
        Momentum transfer in fractional mBZ coordinates.

    Returns
    -------
    chi : ndarray, shape (2, 2, NG, NG), complex
        Stored as model.chi_q.
    """
    na, nk, norb, nb = model.u.shape
    ng = model.Nqvec

    e = np.asfortranarray(model.e - model.mu)              # (na, nk, nb)
    u = np.asfortranarray(model.u.transpose(2, 0, 1, 3))  # (norb, na, nk, nb)

    # k+q indices: shift on the N×N grid, converted to 1-based Fortran indexing
    dkx = int(round(q[0] * model.N))
    dky = int(round(q[1] * model.N))
    kx_idx = np.arange(nk, dtype=np.int32) % model.N
    ky_idx = np.arange(nk, dtype=np.int32) // model.N
    kq_idx = ((ky_idx + dky) % model.N) * model.N + (kx_idx + dkx) % model.N + 1
    kq_idx = np.asfortranarray(kq_idx)

    # G-shift table: convert from Python 0-based (-1=invalid) to Fortran 1-based (0=invalid)
    G_shift = np.where(model._G_shift_idx >= 0, model._G_shift_idx + 1, 0).astype(np.int32)
    G_shift = np.asfortranarray(G_shift)  # (ng, norb) Fortran-order

    chi = np.zeros((na, na, ng, ng), dtype=np.complex128, order="F")
    _calculate_chi_q(chi, e, u, model.beta, kq_idx, G_shift, na, nk, norb, nb, ng)

    model.chi_q = np.ascontiguousarray(chi) / nk
    return model.chi_q
