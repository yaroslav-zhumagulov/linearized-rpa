import numpy as np
import pytest
from lrpa.chemical_potential import calculate_mu
from lrpa.models.rhombohedral_multilayer import RhombohedralMultilayer


@pytest.fixture
def model():
    m = RhombohedralMultilayer(nlayer=2, theta=1.08, Qcut=1)
    m.init_mesh(N=8)
    m.calculate_bandstructure()
    calculate_mu(m, occ=0, T=10)
    return m


def magic_bands(model):
    """Indices of the two flat bands per valley at the magic angle."""
    nb = model.Ndim
    return (nb // 2 - 1, nb // 2 + 1)


def test_all_bands_equals_default(model):
    """bands=(0, nb) must give the same result as bands=None."""
    nb = model.Ndim
    chi_default = model.calculate_chi_q([0.0, 0.0])
    chi_all = model.calculate_chi_q([0.0, 0.0], bands=(0, nb))
    np.testing.assert_allclose(chi_all, chi_default, rtol=1e-12,
                               err_msg="explicit full band range differs from default")


def test_magic_bands_shape(model):
    """Selecting a band subset does not change the output shape."""
    chi_full = model.calculate_chi_q([0.0, 0.0])
    chi_magic = model.calculate_chi_q([0.0, 0.0], bands=magic_bands(model))
    assert chi_full.shape == chi_magic.shape, (
        f"Shape mismatch: {chi_full.shape} vs {chi_magic.shape}"
    )


def test_magic_bands_smaller_than_full(model):
    """
    The magic-band chi is a strict subset of the full Lindhard sum.

    Each intravalley chi[tau,tau] is PSD, and restricting bands can only
    reduce the trace (fewer transitions contribute).
    """
    chi_full = model.calculate_chi_q([0.0, 0.0])
    chi_magic = model.calculate_chi_q([0.0, 0.0], bands=magic_bands(model))

    for tau in range(2):
        tr_full = np.trace(chi_full[tau, tau].real)
        tr_magic = np.trace(chi_magic[tau, tau].real)
        assert tr_magic <= tr_full + 1e-12, (
            f"tau={tau}: magic-band trace {tr_magic:.4e} > full trace {tr_full:.4e}"
        )
        assert tr_magic > 0, f"tau={tau}: magic-band trace is zero or negative"


def test_magic_bands_psd(model):
    """Intravalley magic-band chi is still positive semidefinite."""
    chi_magic = model.calculate_chi_q([0.0, 0.0], bands=magic_bands(model))
    for tau in range(2):
        block = chi_magic[tau, tau]
        eigvals = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
        assert np.all(eigvals >= -1e-12), (
            f"tau={tau}: negative eigenvalue {eigvals.min():.3e} in magic-band chi"
        )


def test_magic_bands_hermitian(model):
    """Each (tau, taup) block of magic-band chi is Hermitian."""
    chi_magic = model.calculate_chi_q([0.0, 0.0], bands=magic_bands(model))
    for tau in range(2):
        for taup in range(2):
            block = chi_magic[tau, taup]
            np.testing.assert_allclose(
                block, block.conj().T, atol=1e-12,
                err_msg=f"Hermiticity violated for tau={tau}, taup={taup}"
            )


def test_single_band_consistent(model):
    """
    Calling with a single band gives a non-negative intravalley G=G'=0 element.

    Also checks that two consecutive single-band calls produce different results
    (the two flat bands are physically distinct).
    """
    nb = model.Ndim

    chi_lo = model.calculate_chi_q([0.0, 0.0], bands=(nb // 2 - 1, nb // 2))
    chi_hi = model.calculate_chi_q([0.0, 0.0], bands=(nb // 2,     nb // 2 + 1))
    G0 = chi_lo.shape[2] // 2

    for tau in range(2):
        assert chi_lo[tau, tau, G0, G0].real >= -1e-12
        assert chi_hi[tau, tau, G0, G0].real >= -1e-12

    # The two flat bands contribute differently (not identical)
    assert not np.allclose(chi_lo, chi_hi), (
        "Lower and upper flat band give identical chi -- unexpected"
    )


def test_bands_combined_with_Qcut_chi(model):
    """bands and Qcut_chi can be used together; shape is (2,2,1,1) for Qcut_chi=0."""
    chi = model.calculate_chi_q([0.0, 0.0],
                                Qcut_chi=0,
                                bands=magic_bands(model))
    assert chi.shape == (2, 2, 1, 1), f"Expected (2,2,1,1), got {chi.shape}"
    # G=0 element must be real and non-negative for intravalley
    for tau in range(2):
        assert chi[tau, tau, 0, 0].real >= -1e-12
        assert abs(chi[tau, tau, 0, 0].imag) < 1e-12
