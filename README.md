# Linearized RPA

Linearized random phase approximation (RPA) for computing static susceptibilities in Bernal bilayer graphene. Includes particle-hole and particle-particle channels with self-consistent Hartree-Fock and Fortran-accelerated bubble summation.

## Features

- Fortran-accelerated computation of the Lindhard susceptibility in both particle-hole and particle-particle channels
- Self-consistent Hartree-Fock with spin/valley symmetry breaking
- Kanamori-type interaction tensor with Hubbard U and Hund's coupling J

## Installation

Requires a Fortran compiler (e.g. `gfortran`) and Python >= 3.10.

```bash
pip install .
```

For development:

```bash
pip install --no-build-isolation -e ".[dev]"
```

## Usage

```python
import numpy as np
from lrpa import BernalBilayer, tensor
from lrpa.hartree_fock import hartree_fock_sym
from lrpa.susceptibility import calculate_chi_ph, calculate_chi_pp

# Set up interaction
U_4 = tensor(U=12, J=-1.2)
U_2 = np.zeros((4, 4))
for a in range(4):
    for b in range(4):
        U_2[a, b] = U_4[a, a, b, b]

# Build model at V = 30 meV displacement field
model = BernalBilayer(V=30)
model.init_mesh(N=18000, kmax=0.06)
model.calculate_bandstructure()
model.calculate_inverse_bandstructure()

# Self-consistent Hartree-Fock at T = 0.4 K, doping = -0.6 * 10^12 cm^-2
hartree_fock_sym(model, U_2, T=0.4, occ=-0.6)

# Compute susceptibilities
calculate_chi_ph(model)   # particle-hole channel
calculate_chi_pp(model)   # particle-particle channel
```

See [examples/bilayer_graphene.py](examples/bilayer_graphene.py) for a complete example with RPA instability analysis.

## Project structure

```
linearized-rpa/
├── pyproject.toml              # Build configuration (meson-python)
├── meson.build                 # Top-level Meson build
├── lrpa/
│   ├── __init__.py
│   ├── constants.py            # Universal constants and Pauli matrices
│   ├── interaction.py          # Kanamori interaction tensor
│   ├── chemical_potential.py   # Chemical potential solver
│   ├── hartree_fock.py         # Self-consistent Hartree-Fock
│   ├── susceptibility.py       # Chi calculation (particle-hole & particle-particle)
│   ├── calculate_chi.f90       # Fortran susceptibility kernels
│   ├── meson.build             # f2py extension build
│   └── models/
│       ├── __init__.py
│       └── bernal_bilayer.py   # BernalBilayer class (Hamiltonian, bands, lattice constants)
└── examples/
    └── bilayer_graphene.py     # Example calculation
```

## Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| `V` | Displacement field | meV |
| `occ` | Carrier doping | 10^12 cm^-2 |
| `T` | Temperature | K |
| `N` | k-mesh grid size | - |
| `kmax` | Momentum cutoff around K/K' | 1/A |
| `rashba` | Rashba SOC strength | eV |
| `valley_zeeman` | Valley Zeeman coupling | eV |
