import numpy as np
from lrpa.constants import s0, sx, sy, sz
from lrpa.soc_parameters import SOCParameters

# Lattice parameters
la = 2.46  # [A]
d = 3.35  # [A]

# Hopping parameters [eV]
t0 = 2.6
t1 = 0.339
t3 = 0.28
t4 = -0.140
delta = 0.0097

# Unit cell
cell = la * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
vol = np.linalg.det(cell)
icell = 2 * np.pi * np.linalg.inv(cell).T

# Derived
factor = np.sqrt(3) / 2 * la


class BernalBilayer(object):
    def __init__(self, V: float = 0, soc: SOCParameters = SOCParameters()) -> None:
        self.V = V
        self.mu = None
        self.sigma = None
        self.soc = soc

    # ----------------------------------------------------------------------------------------------------------------------
    def hamiltonian(self, k: np.ndarray, tau: int = 1) -> np.ndarray:

        kx, ky = k

        nk = len(kx)

        f1 = -factor * (tau * kx - 1j * ky)
        f2 = -factor * (tau * kx + 1j * ky)

        h0 = np.empty((nk, 4, 4), dtype=np.complex128)

        h0[:, 0, 0] = delta + 1e-3 * self.V
        h0[:, 1, 1] = +1e-3 * self.V
        h0[:, 2, 2] = -1e-3 * self.V
        h0[:, 3, 3] = delta - 1e-3 * self.V

        h0[:, 0, 1] = t0 * f1
        h0[:, 1, 0] = t0 * f2

        h0[:, 0, 2] = t4 * f2
        h0[:, 2, 0] = t4 * f1

        h0[:, 0, 3] = t1
        h0[:, 3, 0] = t1

        h0[:, 1, 2] = t3 * f1
        h0[:, 2, 1] = t3 * f2

        h0[:, 1, 3] = t4 * f2
        h0[:, 3, 1] = t4 * f1

        h0[:, 2, 3] = t0 * f1
        h0[:, 3, 2] = t0 * f2

        hx = -self.soc.rashba * np.kron(np.diag([0, 1]), sy)
        hy = +self.soc.rashba * tau * np.kron(np.diag([0, 1]), sx)
        hz = self.soc.valley_zeeman * tau * np.kron(np.diag([0, 1]), sz)

        return np.kron(s0, h0) + np.kron(sz, hz) + np.kron(sx, hx) + np.kron(sy, hy)

    # ----------------------------------------------------------------------------------------------------------------------
    def init_mesh(self, N: int = 12000, kmax: float = 0.04, cutoff: float = 0.025):

        self.N = N
        self.factor = 1e4 / vol / self.N / self.N
        mesh = np.fft.fftfreq(N)
        mesh = mesh[np.abs(mesh) <= cutoff]
        x, y = np.meshgrid(mesh, mesh)
        k = np.array([x, y]).T.dot(icell)
        i, j = np.where(np.linalg.norm(k, axis=2) <= kmax)
        n = np.vstack([i, j]).T
        k = k[i, j]
        n[n > N / 2] -= N
        nk = len(k)
        self.k = k.T
        self.nk = nk

    # ----------------------------------------------------------------------------------------------------------------------
    def calculate_bandstructure(self):

        h = np.empty((2, self.nk, 8, 8), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            h[t] = self.hamiltonian(self.k, tau=tau)
        e, u = np.linalg.eigh(h)

        self.h = h
        self.e = e[:, :, 2:6]
        self.u = u[:, :, :, 2:6]
        self.u = self.u.reshape((2, self.nk, 2, 4, 4))

    # ----------------------------------------------------------------------------------------------------------------------
    def calculate_inverse_bandstructure(self):

        h_inv = np.empty((2, self.nk, 8, 8), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            h_inv[t] = self.hamiltonian(-self.k, tau=tau)
        e_inv, u_inv = np.linalg.eigh(h_inv)

        self.h_inv = h_inv
        self.e_inv = e_inv[:, :, 2:6]
        self.u_inv = u_inv[:, :, :, 2:6]
        self.u_inv = self.u_inv.reshape((2, self.nk, 2, 4, 4))
