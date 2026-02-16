"""
Linearized RPA calculation for Bernal bilayer graphene.

Computes the particle-hole and particle-particle susceptibilities
at a given displacement field, doping, and temperature, with
Hartree-Fock self-consistency.
"""

import numpy as np
from linearized_rpa import BernalBilayer, tensor
from linearized_rpa.hartree_fock import hartree_fock_sym
from linearized_rpa.susceptibility import calculate_chi_ph, calculate_chi_pp

# --- Parameters ---
V = 30       # displacement field [meV]
occ = -0.6   # doping [10^12 cm^-2]
T = 0.4      # temperature [K]
N = 18000    # k-mesh density
kmax = 0.06  # momentum cutoff [1/A]

# Interaction: Hubbard U with Hund's coupling J = -U/10
U = 12  # [eV]
J = -U / 10
U_4 = tensor(U=U, J=J)

# Extract density-density part for Hartree-Fock
U_2 = np.zeros((4, 4))
for a in range(4):
    for b in range(4):
        U_2[a, b] = U_4[a, a, b, b]

# --- Model setup ---
model = BernalBilayer(V=V)
model.init_mesh(N=N, kmax=kmax)
model.calculate_bandstructure()
model.calculate_inverse_bandstructure()

# --- Self-consistent Hartree-Fock ---
hartree_fock_sym(model, U_2, T, occ)

# --- Susceptibilities ---
calculate_chi_ph(model)
calculate_chi_pp(model)

print(f"\nChemical potential: {model.mu:.6f} eV")
print(f"chi_ph shape: {model.chi_ph.shape}")
print(f"chi_pp shape: {model.chi_pp.shape}")

# Largest eigenvalue of chi_ph @ U (RPA instability criterion)
U_ph = U_4.transpose((0, 1, 3, 2)).reshape((16, 16))
chi_ph_mat = model.chi_ph.transpose((0, 1, 3, 2)).reshape((16, 16))
lambda_rpa = np.max(np.abs(np.linalg.eigvals(chi_ph_mat @ U_ph)))
print(f"Largest RPA eigenvalue: {lambda_rpa:.4f} (>1 = instability)")
