import numpy as np
from lrpa.chi_module import calculate_chi_ph as _calculate_chi_ph


def calculate_chi_ph(model):
    chi_ph = np.zeros((2, 2, 2, 2), dtype=np.complex128, order="F")

    e = model.e - model.mu

    na, nk, norb, nb = model.u.shape

    e = np.asfortranarray(e)
    u = np.asfortranarray(model.u)

    _calculate_chi_ph(chi_ph, e, u, model.beta, na, nk, norb, nb)
    model.chi_ph = np.ascontiguousarray(chi_ph) / (model.N * model.N)
