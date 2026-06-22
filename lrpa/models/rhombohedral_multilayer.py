import numpy as np
from scipy.special import expit
from lrpa.constants import s0, sx, sy

# lattice constant
la = 2.46  # Angstrom
a0 = la / np.sqrt(3)
dc = 3.35

gamma_3 = 0
gamma_4 = 0
v3 = gamma_3 * np.sqrt(3) / 2 * la
v4 = gamma_4 * np.sqrt(3) / 2 * la

# hopping
u_AA = 0.070  # eV
u_AB = 0.110  # eV
vppsigma = 0.40  # eV
vf = 5.817


class RhombohedralMultilayer(object):
    def __init__(
        self,
        V: float = 0,
        nlayer: int = 2,
        theta: float = 1.08,
        Qcut: int = 3,
        reverse: bool = True,
        twist: bool = True,
    ) -> None:
        self.V = V
        self.nlayer = int(nlayer)
        self.theta = theta
        self.Qcut = int(Qcut)
        self.reverse = bool(reverse)
        self.twist = bool(twist)

        if self.twist:
            if np.isclose(self.theta, 0.0):
                raise ValueError("twist=True requires theta != 0.")
            self.lm = la / (2 * np.sin(np.deg2rad(self.theta) / 2))
            self.cell = self.lm * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
            self.kD = self._moire_len()
        else:
            if not np.isclose(self.theta, 0.0):
                raise ValueError("twist=False describes the untwisted system, so use theta=0.")
            self.Qcut = 0
            self.lm = la
            self.cell = la * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
            self.kD = 0.0

        self.icell = 2 * np.pi * np.linalg.inv(self.cell).T
        self.Nqvec = (2 * self.Qcut + 1) ** 2
        self.Ndim = 2 * self.Nqvec * self.nlayer
        self.qvecs = np.mgrid[
            -self.Qcut:self.Qcut + 1,
            -self.Qcut:self.Qcut + 1,
        ].reshape(2, self.Nqvec)
        self._system_in()

    def _moire_len(self):
        th = np.deg2rad(self.theta)
        K = (4 * np.pi) / (3 * la) * np.array([1.0, 0.0])
        rot1 = np.array([
            [np.cos(th / 2), -np.sin(th / 2)],
            [np.sin(th / 2),  np.cos(th / 2)],
        ])
        rot2 = np.array([
            [ np.cos(th / 2), np.sin(th / 2)],
            [-np.sin(th / 2), np.cos(th / 2)],
        ])
        return np.linalg.norm(rot2 @ K - rot1 @ K)

    def _system_in(self):
        """
        reverse=True/False gives the (ABC-CBA) / (ABC-ABC) type twisted structure.
        twist=False removes the twisted interface and gives an untwisted ABC stack.
        """
        slg = self.nlayer // 2 if self.twist else self.nlayer
        ang = [0] * slg + [1] * (self.nlayer - slg)
        seq1 = ['A', 'B', 'C'] * (slg // 3) + ['A', 'B', 'C'][:slg % 3]
        seq2 = ['A', 'B', 'C'] * ((self.nlayer - slg) // 3) + ['A', 'B', 'C'][:(self.nlayer - slg) % 3]
        seq2 = seq2[::-1] if self.reverse else seq2
        self.ang = ang
        self.seq = seq1 + seq2

    @staticmethod
    def _stacking_chirality(bottom, top):
        pair = bottom + top
        if pair in ['AB', 'BC', 'CA']:
            return 1
        if pair in ['BA', 'CB', 'AC']:
            return -1
        if pair in ['AA', 'BB', 'CC']:
            return 0
        raise ValueError(
            f"ERROR in stacking_chirality: Unsupported stacking sequence '{pair}'. "
            "Supported sequences: AB, BC, CA, BA, CB, AC, AA, BB, CC."
        )

    def hamiltonian(self, km, valley):
        tau = int(valley)
        if tau not in (-1, +1):
            raise ValueError("valley must be +1 or -1.")

        km = np.asarray(km, dtype=float)
        # Centered displacement potential:
        # U_l = -(V/1000) * (l - (N_layer - 1)/2)  [eV]
        E_field = lambda ilayer: -(self.V / 1000.0) * (ilayer - 0.5 * (self.nlayer - 1))

        if self.twist:
            b1m = np.array([-0.5, -np.sqrt(3) / 2]) * np.sqrt(3) * self.kD
            b2m = np.array([+1.0, 0.0]) * np.sqrt(3) * self.kD
            K1 = np.array([np.sqrt(3) / 2, +0.5]) * self.kD
            K2 = np.array([np.sqrt(3) / 2, -0.5]) * self.kD
            Ks = (K2, K1)

            def get_k(ilayer, q):
                K = Ks[self.ang[ilayer]]
                return -tau * K + (km[0] + q[0]) * b1m + (km[1] + q[1]) * b2m
        else:
            def get_k(ilayer, q):
                return km

        H = np.zeros((self.Ndim, self.Ndim), dtype=complex)
        idx = lambda layer, iq: 2 * (layer * self.Nqvec + iq)

        phi = tau * 2 * np.pi / 3
        T1 = np.array([[u_AA, u_AB], [u_AB, u_AA]], dtype=complex)
        T2 = np.array([[u_AA, u_AB * np.exp(+1j * phi)],
                       [u_AB * np.exp(-1j * phi), u_AA]], dtype=complex)
        T3 = np.array([[u_AA, u_AB * np.exp(-1j * phi)],
                       [u_AB * np.exp(+1j * phi), u_AA]], dtype=complex)
        qmap = {(int(q[0]), int(q[1])): iq for iq, q in enumerate(self.qvecs.T)}

        # Intralayer Dirac blocks.
        for ilayer in range(self.nlayer):
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                k = get_k(ilayer, q)
                H[i:i + 2, i:i + 2] = (
                    vf * (tau * k[0] * sx - k[1] * sy) + E_field(ilayer) * s0
                )

        # Same-angle interlayer hopping inside each untwisted multilayer block.
        for ilayer in range(self.nlayer - 1):
            if self.ang[ilayer] != self.ang[ilayer + 1]:
                continue
            chirality = self._stacking_chirality(self.seq[ilayer], self.seq[ilayer + 1])
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                j = idx(ilayer + 1, iq)
                k = get_k(ilayer, q)
                kminus = -(tau * k[0] - 1j * k[1])
                kplus = -(tau * k[0] + 1j * k[1])

                if chirality == 1:
                    T = np.array([[v4 * kminus, vppsigma],
                                  [v3 * kplus,  v4 * kminus]], dtype=complex)
                elif chirality == -1:
                    T = np.array([[v4 * kplus,  v3 * kminus],
                                  [vppsigma,    v4 * kplus]], dtype=complex)
                else:
                    T = vppsigma * s0

                H[i:i + 2, j:j + 2] = T
                H[j:j + 2, i:i + 2] = T.conj().T

        # BM tunneling at the twisted interface.
        for ilayer in range(self.nlayer - 1):
            if self.ang[ilayer] == self.ang[ilayer + 1]:
                continue
            for iq, q in enumerate(self.qvecs.T):
                i = idx(ilayer, iq)
                for dq, T in (((0, 0), T1), ((-tau, 0), T2), ((-tau, -tau), T3)):
                    jq = qmap.get((int(q[0] + dq[0]), int(q[1] + dq[1])))
                    if jq is None:
                        continue
                    j = idx(ilayer + 1, jq)
                    H[i:i + 2, j:j + 2] = T
                    H[j:j + 2, i:i + 2] = T.conj().T

        return H

    def init_mesh(self, N=1000, kmax=0.04, cutoff=0.025):
        kn = np.fft.fftfreq(N)

        if self.twist:
            kx, ky = np.meshgrid(kn, kn)
            self.k = np.stack([kx, ky], axis=-1).reshape(-1, 2).T
        else:
            mesh = kn[np.abs(kn) <= cutoff]
            x, y = np.meshgrid(mesh, mesh)
            k = np.array([x, y]).T.dot(self.icell)
            i, j = np.where(np.linalg.norm(k, axis=2) <= kmax)
            self.k = k[i, j].T

        self.nk = self.k.shape[1]
        self.N = N
        self.factor = 1e4 / abs(np.linalg.det(self.cell)) / self.N / self.N

    def calculate_bandstructure(self):
        h = np.empty((2, self.nk, self.Ndim, self.Ndim), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            for ki, k in enumerate(self.k.T):
                h[t, ki] = self.hamiltonian(k, valley=tau)
        e, u = np.linalg.eigh(h)
        self.h = h
        self.e = e
        self.u = u

    def calculate_inverse_bandstructure(self):
        h_inv = np.empty((2, self.nk, self.Ndim, self.Ndim), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            for ki, k in enumerate(-self.k.T):
                h_inv[t, ki] = self.hamiltonian(k, valley=tau)
        e_inv, u_inv = np.linalg.eigh(h_inv)
        self.h_inv = h_inv
        self.e_inv = e_inv
        self.u_inv = u_inv

    def calculate_chi_ph_spinless(self, eps=1e-10):
        fermi = lambda x: expit(-self.beta * x)
        e = self.e - self.mu
        f = fermi(e)
        chi = np.zeros((2, 2, 2, 2), dtype=np.float64)

        for a in range(2):
            for b in range(2):
                # PH form factor: |<u_{a,n}(k)|u_{b,m}(k)>|^2
                W = np.abs(self.u[a].conj().transpose(0, 2, 1) @ self.u[b]) ** 2

                # Lindhard factor: (f_a - f_b) / (e_b - e_a)
                numer = f[a][:, :, None] - f[b][:, None, :]
                denom = e[b][:, None, :] - e[a][:, :, None]
                factor = np.empty_like(denom)
                mask = np.abs(denom) < eps
                np.divide(numer, denom, out=factor, where=~mask)
                if np.any(mask):
                    # Limit of (f(ea) - f(eb)) / (eb - ea) when eb -> ea:
                    # -df/de = beta * f(e) * (1 - f(e))
                    ki, ni, _ = np.where(mask)
                    factor[mask] = self.beta * f[a][ki, ni] * (1.0 - f[a][ki, ni])
                chi[a, b, b, a] = np.sum(factor * W)

        self.chi_ph = chi / self.N**2

    def calculate_chi_pp_spinless(self, eps=1e-10):
        fermi = lambda x: expit(-self.beta * x)

        if not hasattr(self, "e_inv") or not hasattr(self, "u_inv"):
            self.calculate_inverse_bandstructure()

        e = self.e - self.mu
        e_inv = self.e_inv - self.mu
        f = fermi(e)
        f_inv = fermi(e_inv)
        chi = np.zeros((2, 2, 2, 2), dtype=np.float64)

        for a in range(2):
            for b in range(2):
                # PP anomalous form factor:
                # A[k,n,m] = sum_i u_a[k,i,n] * u_inv_b[k,i,m]
                A = self.u[a].transpose(0, 2, 1) @ self.u_inv[b]
                W = np.abs(A) ** 2

                # Pairing factor: (1 - f(ea) - f(eb)) / (-(ea + eb))
                numer = 1.0 - f[a][:, :, None] - f_inv[b][:, None, :]
                denom = -e_inv[b][:, None, :] - e[a][:, :, None]
                factor = np.empty_like(denom)
                mask = np.abs(denom) < eps
                np.divide(numer, denom, out=factor, where=~mask)
                if np.any(mask):
                    # Limit of (1 - f(ea) - f(eb)) / (-(ea + eb))
                    # when ea + eb -> 0.
                    da = f[a][:, :, None] * (1.0 - f[a][:, :, None])
                    db = f_inv[b][:, None, :] * (1.0 - f_inv[b][:, None, :])
                    factor[mask] = -0.5 * self.beta * (da + db)[mask]
                chi[a, b, b, a] = np.sum(factor * W)

        self.chi_pp = chi / self.N**2

    def V00(self, eps=1.0):  # fit from 10.1103/PhysRevB.100.235424 Fig.3(a)
        if not self.twist:
            raise ValueError("V00 is a moire fit and is not valid for twist=False.")
        val = 18.0 * (self.theta - 1.0) + 1.0  # meV for eps=1
        return val / eps / 1000