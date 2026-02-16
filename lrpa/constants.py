import numpy as np

# Unit conversions
Hartree = 27.211386024367243
Bohr = 0.5291772105638411
kB = 11604.5250061657

# Magnetic constants
M = 5.7883818060e-5
gs = 2.002319304386

# Pauli matrices
s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = [s0, sx, sy, sz]
