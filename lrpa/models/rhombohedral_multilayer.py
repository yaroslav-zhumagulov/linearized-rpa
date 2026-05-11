import numpy as np
from scipy.special import expit
from lrpa.constants import s0, sx, sy, sz

# lattice constant
la = 2.46 # Å
a0 = la/np.sqrt(3) # Å
dc = 3.35 # Å
gamma_3 = 0
gamma_4 = 0
v3=gamma_3*np.sqrt(3)/2*la
v4=gamma_4*np.sqrt(3)/2*la

# hopping
u_AA = 0.070 # eV
u_AB = 0.110 # eV
vppsigma = 0.40 # eV
vf = 5.817

class RhombohedralMultilayer(object):
    def __init__(self, 
                 V: float = 0, 
                 nlayer: int = 2, 
                 theta:float = 1.08, 
                 Qcut:int = 3) -> None:
        
        self.V = V
        self.mu = None
        self.beta = None
        self.nlayer = nlayer
        self.theta = theta
        self.Qcut = Qcut

        self.lm = la / (2*np.sin(np.deg2rad(theta)/2))
        self.cell = self.lm * np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2]])
        # factor = np.sqrt(3) / 2 * self.lm

        def _moire_len(self):
            th = np.deg2rad(self.theta)
            a1 = np.array([ 0.5, np.sqrt(3) / 2]) * la
            a2 = np.array([-0.5, np.sqrt(3) / 2]) * la
            K_valley  = (4 * np.pi) / (3 * la) * np.array([ 1.0, 0.0])
            Kp_valley = (4 * np.pi) / (3 * la) * np.array([-1.0, 0.0])
            
            rot1 = np.array([
                [np.cos(th / 2), -np.sin(th / 2)],
                [np.sin(th / 2),  np.cos(th / 2)]
            ])
            rot2 = np.array([
                [ np.cos(th / 2), np.sin(th / 2)],
                [-np.sin(th / 2), np.cos(th / 2)]
            ])
        
            qb = np.dot(rot2, K_valley) - np.dot(rot1, K_valley)
            kD = np.linalg.norm(qb)
            return kD
            
        self.kD = _moire_len(self)
        self.Nqvec = (2 * self.Qcut + 1) ** 2
        self.Ndim = 2 * self.Nqvec * self.nlayer
        self.qvecs = np.mgrid[-self.Qcut:self.Qcut+1,
                               -self.Qcut:self.Qcut+1].reshape(2, self.Nqvec)
        self._qvec_to_idx = {
            (int(self.qvecs[0, iq]), int(self.qvecs[1, iq])): iq
            for iq in range(self.Nqvec)
        }
        self._G_shift_idx = self._build_G_shift_idx()

    
    def _build_G_shift_idx(self):
        """
        Precompute G-shift index table.
        Returns int32 array of shape (Nqvec, Ndim):
          idx[ig, alpha] = basis index of alpha shifted by G[ig], or -1 if out of basis.
        """
        Nqvec, Ndim = self.Nqvec, self.Ndim
        idx = np.full((Nqvec, Ndim), -1, dtype=np.int32)
        for ig in range(Nqvec):
            dq0 = int(self.qvecs[0, ig])
            dq1 = int(self.qvecs[1, ig])
            for alpha in range(Ndim):
                l  = alpha // (Nqvec * 2)
                iq = (alpha % (Nqvec * 2)) // 2
                s  = alpha % 2
                key = (int(self.qvecs[0, iq]) + dq0,
                       int(self.qvecs[1, iq]) + dq1)
                iq_new = self._qvec_to_idx.get(key)
                if iq_new is not None:
                    idx[ig, alpha] = l * Nqvec * 2 + iq_new * 2 + s
        return idx

    def hamiltonian(self, km, valley):
        
        def _system_in(self,reverse=1, twist=True):
            # assert self.nlayer%2==0, 'The number of the layers should be even'
            # reverse means from ABC-ABC to ABC-CBA
            slg = self.nlayer//2 if twist else self.nlayer
            ang = [0] * slg + [1] * (self.nlayer - slg)
            seq1 = ['A', 'B', 'C'] * (slg // 3) + ['A', 'B', 'C'][:slg % 3]
            seq2 = ['A', 'B', 'C'] * ((self.nlayer-slg) // 3) + ['A', 'B', 'C'][:(self.nlayer-slg) % 3]
            seq2 = seq2[::-1] if reverse else seq2
            seq = seq1 + seq2
            # ratio = [1] * (self.nlayer - 1)
            return ang, seq#, ratio

        def _stacking_chirality(bottom, top):
            stacking_sequence = bottom + top
        
            if stacking_sequence in ['AB', 'BC', 'CA']:
                return 1
            elif stacking_sequence in ['BA', 'CB', 'AC']:
                return -1
            elif stacking_sequence in ['AA', 'BB', 'CC']:
                return 0
            else:
                raise ValueError(
                    f"ERROR in stacking_chirality: Unsupported stacking sequence '{stacking_sequence}'. "
                    "Supported sequences: AB, BC, CA, BA, CB, AC, AA, BB, CC."
                )
            

        self.E_field = self.V / 1000 / dc
        ang, seq = _system_in(self)
        
        b1m = np.array([-0.5, -np.sqrt(3)/2]) * np.sqrt(3) * self.kD
        b2m = np.array([ 1.0,  0.0         ]) * np.sqrt(3) * self.kD
        
        K_valley1 = np.array([np.sqrt(3)/2, 0.5]) * self.kD
        K_valley2 = np.array([np.sqrt(3)/2,-0.5]) * self.kD
        
        # Generate Q-vectors
        Nqvec = (2 * self.Qcut + 1) ** 2
        qvecs = np.mgrid[-self.Qcut:self.Qcut+1, -self.Qcut:self.Qcut+1].reshape(2,Nqvec)
        
        Ndim = 2 * Nqvec * self.nlayer
        
        # mono-layer graphene Hamiltonian 
        # H(k)=-vf*(valley*sx-k*sy)
        Hmnk = np.zeros((self.Ndim,self.Ndim),dtype=complex)
    
        phi = valley*2*np.pi/3
        T1 = np.array([[u_AA,u_AB],[u_AB,u_AA]],dtype=complex)
        T2 = np.array([[u_AA,u_AB*np.exp( 1j*phi)],[u_AB*np.exp(-1j*phi),u_AA]])
        T3 = np.array([[u_AA,u_AB*np.exp(-1j*phi)],[u_AB*np.exp( 1j*phi),u_AA]])
    
        # diagonal term: intra-layer hopping
        for ilayer in range(self.nlayer):
            for iq in range(Nqvec):
                qvec = qvecs[:, iq]
                if ang[ilayer]:
                    kvec = -K_valley1 * valley + (km[0] + qvec[0]) * b1m + (km[1] + qvec[1]) * b2m
                else:
                    kvec = -K_valley2 * valley + (km[0] + qvec[0]) * b1m + (km[1] + qvec[1]) * b2m
                H_SLG = vf * (kvec[0] * valley * sx - kvec[1] * sy)
                H_SLG -= self.E_field * (ilayer - self.nlayer / 2.0) * dc * s0
                idx1 = ilayer * Nqvec * 2 + iq * 2
                Hmnk[idx1:idx1+2, idx1:idx1+2] = H_SLG
    
        # diagonal term: inter-layer hopping for non-TBG pair layers
        for ilayer in range(self.nlayer - 1):
            if ang[ilayer] == ang[ilayer+1]:
                for iq in range(Nqvec):
                    idx1 = ilayer * Nqvec * 2 + iq * 2
                    idx2 = (ilayer + 1) * Nqvec * 2 + iq * 2
                    qvec = qvecs[:, iq]
                    if ang[ilayer]:
                        kvec = -K_valley1 * valley + (km[0] + qvec[0]) * b1m + (km[1] + qvec[1]) * b2m
                    else:
                        kvec = -K_valley2 * valley + (km[0] + qvec[0]) * b1m + (km[1] + qvec[1]) * b2m
                    kminus = -(kvec[0] * valley - 1j * kvec[1])
                    kplus = -(kvec[0] * valley + 1j * kvec[1])
                    i = _stacking_chirality(seq[ilayer], seq[ilayer + 1])
                    if i == 1:
                        H_interlayer = np.array([[v4 * kminus, vppsigma], [v3 * kplus, v4 * kminus]], dtype=complex)
                    elif i == -1:
                        H_interlayer = np.array([[v4 * kplus, v3 * kminus], [vppsigma, v4 * kplus]], dtype=complex)
                    elif i == 0:
                        H_interlayer = np.array([[vppsigma, 0], [0, vppsigma]], dtype=complex)
                    else:
                        continue
                    Hmnk[idx1:idx1+2, idx2:idx2+2] = H_interlayer
                    Hmnk[idx2:idx2+2, idx1:idx1+2] = H_interlayer.conj().T
    
        # non-diagonal term: inter-layer hopping for TBG layers
        for ilayer in range(self.nlayer - 1):
            if ang[ilayer] != ang[ilayer+1]:
                for iq in range(Nqvec):
                    q1 = qvecs[:, iq]
                    for jq in range(Nqvec):
                        q2 = qvecs[:, jq]
                        idx1 = ilayer * Nqvec * 2 + iq * 2
                        idx2 = (ilayer + 1) * Nqvec * 2 + jq * 2
                        if iq == jq:
                            Hmnk[idx1:idx1+2, idx2:idx2+2] = T1
                            Hmnk[idx2:idx2+2, idx1:idx1+2] = T1.conj().T
                        if (q2[0] - q1[0] == -valley) and (q1[1] == q2[1]):
                            Hmnk[idx1:idx1+2, idx2:idx2+2] = T2
                            Hmnk[idx2:idx2+2, idx1:idx1+2] = T2.conj().T
                        if (q2[0] - q1[0] == -valley) and (q2[1] - q1[1] == -valley):
                            Hmnk[idx1:idx1+2, idx2:idx2+2] = T3
                            Hmnk[idx2:idx2+2, idx1:idx1+2] = T3.conj().T
        return Hmnk


    def init_mesh(self, N=1000):
        # b1 = np.array([-0.5, -np.sqrt(3)/2])
        # b2 = np.array([ 1.0,  0.0         ])
        # A_BZ = abs(b1[0]*b2[1] - b1[1]*b2[0])
        
        # kn = np.linspace(-0.5, 0.5, N, endpoint=False)
        kn = np.linspace(0,1,N,endpoint=False)
        # kn=np.fft.fftfreq(N)
        kx, ky = np.meshgrid(kn, kn)
        self.k = np.stack([kx, ky], axis=-1).reshape(-1, 2).T
        self.nk = self.k.shape[1]
        self.N = N
        self.factor = 1e4 / np.linalg.det(self.cell) / self.N / self.N


    def calculate_bandstructure(self):
        h = np.empty((2, self.nk, self.Ndim, self.Ndim), dtype=np.complex128)
        for t, tau in enumerate([-1, +1]):
            for ki, k in enumerate(self.k.T):
                h[t,ki] = self.hamiltonian(k, valley=tau)
        e, u = np.linalg.eigh(h)

        self.h = h
        self.e = e
        self.u = u
        # self.u = u[:, :, None, :, :]

    def calculate_chi_ph_spinless(self, eps=1e-10):
        fermi = lambda x: expit(-self.beta * x)
        
        # nvalley, nk, norb, nb = self.u.shape
        e = self.e - self.mu
        f = fermi(e)        
        chi = np.zeros((2, 2, 2, 2), dtype=np.float64)
        
        for _a in range(2):
            for _b in range(2):
                W = np.abs(self.u[_a].conj().transpose(0, 2, 1) @ self.u[_b]) ** 2
    
                numer = f[_a][:, :, None] - f[_b][:, None, :]
                denom = e[_b][:, None, :] - e[_a][:, :, None]
                factor = np.empty_like(denom)
                mask = np.abs(denom) < eps
                np.divide(numer, denom, out=factor, where=~mask)
                if np.any(mask):
                    ki, ni, _ = np.where(mask)
                    factor[mask] = self.beta * f[_a][ki, ni] * (1.0 - f[_a][ki, ni])
    
                chi[_a, _b, _b, _a] = np.sum(factor * W)
    
        self.chi_ph = chi/self.N**2

    def calculate_chi_q(self, q, Qcut_chi=0, bands=None):
        """
        Static chi^{tau,tau'}(q; G, G') in the particle-hole bubble approximation.
        Valley index is treated as pseudospin.

        Parameters
        ----------
        q : array-like, shape (2,)
            Momentum transfer in fractional mBZ coordinates (same units as self.k).
        Qcut_chi : int or None
            Cutoff for the output G-vectors (|n1|, |n2| <= Qcut_chi).
            Default Qcut_chi=0 gives only G=G'=0, shape (2, 2, 1, 1).
            Pass Qcut_chi=None to use the full model.Qcut.
            The G=(0,0) index in the result is always chi.shape[2]//2.
        bands : tuple (n_min, n_max) or None
            0-based half-open slice [n_min:n_max] of bands included in the
            Lindhard sum.  None uses all bands.  For magic-angle TBG the two
            flat bands per valley are at Ndim//2-1 and Ndim//2, so pass
            bands=(Ndim//2-1, Ndim//2+1).

        Returns
        -------
        chi : ndarray, shape (2, 2, ng_chi, ng_chi), complex
            Also stored as self.chi_q.  ng_chi = (2*Qcut_chi+1)**2.
        """
        from lrpa.susceptibility import calculate_chi_q as _ftn
        return _ftn(self, q, Qcut_chi=Qcut_chi, bands=bands)

    def V00(self, eps=1.0): # fit from 10.1103/PhysRevB.100.235424 Fig.3(a)
        val = 18.0 * (self.theta - 1.0) + 1.0  # meV for eps=1 
        return val / eps / 1000
