import numpy as np
from lrpa.chi_module import calculate_chi_ph as _calculate_chi_ph
from lrpa.chi_module import calculate_chi_pp as _calculate_chi_pp
from lrpa.chi_module import calculate_chi_q as _calculate_chi_q


def calculate_chi_ph(model):
    chi_ph = np.zeros((2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu

    na, nk, norb, nb = model.u.shape

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u.transpose(2, 0, 1, 3))  # (norb, na, nk, nb) — stride-1 for dot_product

    _calculate_chi_ph(chi_ph, e, u, model.beta, na, nk, norb, nb)
    model.chi_ph = np.ascontiguousarray(chi_ph) / (model.N * model.N)

def calculate_chi_pp(model):
    chi_pp = np.zeros((2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu
    e_inv = model.e_inv - model.mu

    na, nk, norb, nb = model.u.shape
    

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u)
    
    e_inv = np.asfortranarray(e_inv)
    u_inv = np.asfortranarray(model.u_inv) 
    

    _calculate_chi_pp(chi_pp, e, u,e_inv, u_inv, model.beta, na, nk, norb, nb)
    model.chi_pp = np.ascontiguousarray(chi_pp) / (model.N * model.N)


def calculate_chi_q(model, q, Qcut_chi=0, bands=None):
    """
    Compute static chi^{tau,tau'}(q; G, G') via the Fortran kernel.

    Parameters
    ----------
    model : RhombohedralMultilayer
        Must have calculate_bandstructure() called and mu, beta set.
    q : array-like, shape (2,)
        Momentum transfer in fractional mBZ coordinates.
    Qcut_chi : int or None
        Truncation cutoff for the output G-vectors.  Only G-vectors with
        |n1| <= Qcut_chi and |n2| <= Qcut_chi are included.  Must be <= model.Qcut.
        Default Qcut_chi=0 returns only the G=G'=0 element, shape (2,2,1,1).
        Pass Qcut_chi=None to use the full model.Qcut.
        The G=(0,0) index in the result is always chi.shape[2]//2.
    bands : tuple (n_min, n_max) or None
        Python 0-based half-open slice [n_min:n_max] selecting which bands
        participate in the Lindhard sum.  None (default) uses all nb bands.
        For magic-angle TBG the two flat bands per valley are at indices
        Ndim//2-1 and Ndim//2, so pass bands=(Ndim//2-1, Ndim//2+1).

    Returns
    -------
    chi : ndarray, shape (2, 2, ng_chi, ng_chi), complex
        Stored as model.chi_q.  ng_chi = (2*Qcut_chi+1)**2.
    """
    na, nk, norb, nb_full = model.u.shape

    if Qcut_chi is None:
        Qcut_chi = model.Qcut

    if bands is None:
        n_min, n_max = 0, nb_full
    else:
        n_min, n_max = bands

    nb = n_max - n_min

    # Select rows of G_shift corresponding to |n1|,|n2| <= Qcut_chi
    mask = (np.abs(model.qvecs[0]) <= Qcut_chi) & (np.abs(model.qvecs[1]) <= Qcut_chi)
    ig_sel = np.where(mask)[0]
    ng = len(ig_sel)  # = (2*Qcut_chi+1)**2

    e = np.asfortranarray((model.e - model.mu)[:, :, n_min:n_max])   # (na, nk, nb)
    u = np.asfortranarray(model.u[:, :, :, n_min:n_max].transpose(2, 0, 1, 3))  # (norb, na, nk, nb)

    # k+q indices: shift on the N×N grid, converted to 1-based Fortran indexing
    dkx = int(round(q[0] * model.N))
    dky = int(round(q[1] * model.N))
    kx_idx = np.arange(nk, dtype=np.int32) % model.N
    ky_idx = np.arange(nk, dtype=np.int32) // model.N
    kq_idx = ((ky_idx + dky) % model.N) * model.N + (kx_idx + dkx) % model.N + 1
    kq_idx = np.asfortranarray(kq_idx)

    # G-shift table: select rows, convert Python 0-based to Fortran 1-based
    G_shift_full = np.where(model._G_shift_idx >= 0, model._G_shift_idx + 1, 0).astype(np.int32)
    G_shift = np.asfortranarray(G_shift_full[ig_sel])  # (ng, norb)

    chi = np.zeros((na, na, ng, ng), dtype=np.complex128, order="F")
    _calculate_chi_q(chi, e, u, model.beta, kq_idx, G_shift, na, nk, norb, nb, ng)

    model.chi_q = np.ascontiguousarray(chi) / nk
    return model.chi_q
