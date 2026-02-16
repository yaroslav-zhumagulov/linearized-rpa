import numpy as np
from lrpa.chi_module import calculate_chi_ph as _calculate_chi_ph
from lrpa.chi_module import calculate_chi_pp as _calculate_chi_pp


def calculate_chi_ph(model):
    chi_ph = np.zeros((2, 2, 2, 2, 2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu

    na, nk, ns, norb, nb = model.u.shape

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u)

    _calculate_chi_ph(chi_ph, e, u, model.beta, na, nk, norb, nb, ns)
    model.chi_ph = np.ascontiguousarray(chi_ph) / (model.N * model.N)
    model.chi_ph = model.chi_ph.reshape((4, 4, 4, 4))


def calculate_chi_pp(model):
    chi_pp = np.zeros((2, 2, 2, 2, 2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu
    e_inv = model.e_inv - model.mu

    if model.sigma is not None:
        e += model.sigma
        e_inv += model.sigma

    na, nk, ns, norb, nb = model.u.shape

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u)
    e_inv = np.asfortranarray(e_inv)
    u_inv = np.asfortranarray(model.u_inv)

    _calculate_chi_pp(chi_pp, e, u, e_inv, u_inv, model.beta, na, nk, norb, nb, ns)
    model.chi_pp = np.ascontiguousarray(chi_pp) / (model.N * model.N)
    model.chi_pp = model.chi_pp.reshape((4, 4, 4, 4))
