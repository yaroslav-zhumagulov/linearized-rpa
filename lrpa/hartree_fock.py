import numpy as np
from scipy.special import expit
from lrpa.constants import kB, sz
from lrpa.chemical_potential import calculate_mu


def hartree_fock_sym(model, U_2, T, occ, niter=20, alpha=0.7):

    model.sigma = 4 * 1e-3 * np.kron(sz, sz).diagonal()
    beta = kB / T

    for _ in range(niter):
        e = model.e + model.sigma[:, None, None]
        calculate_mu(model, occ, T, e=e)
        f = expit(-beta * (e - model.mu)) - 0.5
        rho = np.sum(f, axis=(1, 2)) / model.N / model.N
        sigma = np.dot(-U_2, rho)
        sigma -= sigma.mean()
        model.sigma = alpha * sigma + (1 - alpha) * model.sigma
        print(1e3 * model.sigma)

    model.rho = rho
