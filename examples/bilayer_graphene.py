"""
Linearized RPA calculation for Bernal bilayer graphene.

Computes the particle-hole and particle-particle susceptibilities
at a given displacement field, doping, and temperature, with
Hartree-Fock self-consistency.
"""

import numpy as np
from lrpa import BernalBilayer, tensor, calculate_mu
from lrpa.susceptibility import calculate_chi_ph, calculate_chi_pp

# --- Parameters ---
V = 30  # displacement field [meV]
occ = -0.6  # doping [10^12 cm^-2]
T = 0.4  # temperature [K]
N = 18000  # k-mesh density
kmax = 0.06  # momentum cutoff [1/A]

# Interaction: Hubbard U with Hund's coupling J = -U/10
U = 12  # [eV]
J = -U / 10
U_4 = tensor(U=U, J=J)

# --- Model setup ---
model = BernalBilayer(V=V)
model.init_mesh(N=N, kmax=kmax)
model.calculate_bandstructure()
model.calculate_inverse_bandstructure()

# --- Chemical potential ---
calculate_mu(model, occ=occ, T=T)

# --- Susceptibilities ---
calculate_chi_ph(model)
calculate_chi_pp(model)

print(f"\nChemical potential: {model.mu:.6f} eV")
print(f"chi_ph shape: {model.chi_ph.shape}")
print(f"chi_pp shape: {model.chi_pp.shape}")

# Largest eigenvalue of chi_ph @ U (RPA instability criterion)
U_ph = U_4.transpose((0, 1, 3, 2)).reshape((16, 16))
chi_ph_mat = model.chi_ph.transpose((0, 1, 3, 2)).reshape((16, 16))
lambda_rpa, phase_rpa = np.linalg.eig(chi_ph_mat @ U_ph)
print(f"Largest RPA eigenvalue: {lambda_rpa.max():.4f} (>1 = instability)")
phase_rpa = phase_rpa[:, lambda_rpa.argmax()].reshape((4, 4))
phase_rpa = np.round(phase_rpa, decimals=3)
print(f"RPA phase structure:\n{phase_rpa}")
