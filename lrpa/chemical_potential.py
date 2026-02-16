import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from lrpa.constants import kB


def calculate_mu(model, occ: float = 0, T: float = 0.4, e=None, tol: float = 1e-12):

    if e is None:
        e = model.e
    bounds = [e.min(), e.max()]

    beta = kB / T
    model.beta = beta
    model.occ = occ

    def func(mu):
        return model.factor * np.sum(expit(-beta * (e - mu)) - 0.5) - occ

    model.mu = brentq(func, *bounds, xtol=tol)
