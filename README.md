# Linearized RPA

Linearized random phase approximation (RPA) for computing static susceptibilities in Bernal bilayer graphene. Includes a particle-hole channel with self-consistent Hartree-Fock and Fortran-accelerated bubble summation. This branch (`spinless`) implements the spinless formulation.

## Features

- Fortran-accelerated computation of the Lindhard susceptibility in the particle-hole channel (spinless)
- Self-consistent Hartree-Fock with spin/valley symmetry breaking
- Kanamori-type interaction tensor with Hubbard U and Hund's coupling J

## Installation

Requires a Fortran compiler (e.g. `gfortran`) and Python >= 3.10. On macOS, OpenMP support requires `libomp` (`brew install libomp`).

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/yaroslav-zhumagulov/linearized-rpa.git
```

Or install a specific branch / tag:

```bash
pip install git+https://github.com/yaroslav-zhumagulov/linearized-rpa.git@spinless
```

From a local clone:

```bash
git clone https://github.com/yaroslav-zhumagulov/linearized-rpa.git
cd linearized-rpa
pip install .
```

For development (editable install with dev extras):

```bash
pip install --no-build-isolation -e ".[dev]"
```

## Usage

```python
import numpy as np
from lrpa import BernalBilayer, tensor, calculate_mu
from lrpa.susceptibility import calculate_chi_ph

# Build model at V = 30 meV displacement field
model = BernalBilayer(V=30)
model.init_mesh(N=12000, kmax=0.04)
model.calculate_bandstructure()

# Chemical potential at T = 0.4 K, doping = -0.5 * 10^12 cm^-2
calculate_mu(model, occ=-0.5, T=0.4)

# Compute particle-hole susceptibility
calculate_chi_ph(model)
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
│   ├── susceptibility.py       # Chi calculation (particle-hole channel)
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
