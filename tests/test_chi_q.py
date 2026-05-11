import numpy as np
import pytest
from lrpa.chemical_potential import calculate_mu
from lrpa.models.rhombohedral_multilayer import RhombohedralMultilayer


@pytest.fixture
def model():
    m = RhombohedralMultilayer(nlayer=2, theta=1.08, Qcut=1)
    m.init_mesh(N=8)
    m.calculate_bandstructure()
    calculate_mu(m, occ=0, T=10)   # sets m.mu and m.beta (T=10 K)
    return m


def test_q0_G0_matches_chi_ph_spinless(model):
    """
    At q=0 the G=G'=(0,0) diagonal element must reproduce chi_ph_spinless.

    Both methods compute the same Lindhard sum; the G=0 form factor is the
    full orbital overlap |<psi_tau|psi_tau'>|^2 that chi_ph_spinless uses.
    """
    model.calculate_chi_ph_spinless()
    chi_ph = model.chi_ph.copy()

    chi_q = model.calculate_chi_q([0.0, 0.0])
    G0 = model._qvec_to_idx[(0, 0)]

    for tau in range(2):
        for taup in range(2):
            np.testing.assert_allclose(
                chi_q[tau, taup, G0, G0].real,
                chi_ph[tau, taup, taup, tau],
                rtol=1e-10,
                err_msg=f"q=0 G=G'=0 mismatch for tau={tau}, taup={taup}",
            )


def test_each_block_hermitian_q0(model):
    """
    For each (tau, taup) the NG x NG matrix chi_q[tau, taup] is Hermitian:
    chi[G, G'] == chi*[G', G].

    Proof: conjugating the sum Σ M_G W M*_G' swaps G <-> G'.
    """
    chi_q = model.calculate_chi_q([0.0, 0.0])
    for tau in range(2):
        for taup in range(2):
            block = chi_q[tau, taup]
            np.testing.assert_allclose(
                block, block.conj().T, atol=1e-12,
                err_msg=f"Hermiticity violated for tau={tau}, taup={taup}",
            )


def test_diagonal_elements_real_q0(model):
    """
    Diagonal G=G' elements are real (direct consequence of Hermiticity).
    """
    chi_q = model.calculate_chi_q([0.0, 0.0])
    for tau in range(2):
        for taup in range(2):
            diag_imag = np.diag(chi_q[tau, taup]).imag
            np.testing.assert_allclose(
                diag_imag, 0.0, atol=1e-12,
                err_msg=f"Diagonal not real for tau={tau}, taup={taup}",
            )


def test_each_block_hermitian_finite_q(model):
    """
    Hermiticity holds for a finite q as well.
    """
    q = [1.0 / model.N, 0.0]
    chi_q = model.calculate_chi_q(q)
    for tau in range(2):
        for taup in range(2):
            block = chi_q[tau, taup]
            np.testing.assert_allclose(
                block, block.conj().T, atol=1e-12,
                err_msg=f"Hermiticity violated at q={q} for tau={tau}, taup={taup}",
            )


def test_intravalley_positive_semidefinite(model):
    """
    The intra-valley charge response chi[tau, tau] is positive semidefinite.

    At omega=0 the Lindhard weight (f_n - f_m)/(E_m - E_n) >= 0 and the
    accumulation is a sum of rank-1 PSD updates, so all eigenvalues >= 0.
    """
    chi_q = model.calculate_chi_q([0.0, 0.0])
    for tau in range(2):
        block = chi_q[tau, tau]
        eigvals = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
        assert np.all(eigvals >= -1e-12), (
            f"Negative eigenvalue {eigvals.min():.3e} in chi[{tau},{tau}]"
        )
