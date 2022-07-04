"""
Test functionals module.
"""
import pytest
import numpy as np
from pyscf import gto

from taco.methods.scf_pyscf import ScfPyScf
from taco.embedding.functionals import compute_kinetic_tf, compute_exchange_slater
from taco.embedding.functionals import compute_kinetic_weizsacker_potential
from taco.embedding.functionals import compute_kinetic_weizsacker_modified, _kinetic_ndsd_potential
from taco.embedding.functionals import compute_kinetic_ndsd, RDensFunc


def test_rdensfunc_base():
    """Test base SCFMethod class."""
    name = 'tf'
    grid_points = np.arange(0, 10, 0.1)
    grid_weigths = np.ones(grid_points.shape)
    myfunc = compute_kinetic_tf
    with pytest.raises(TypeError):
        RDensFunc(1, grid_points, grid_weigths)
    with pytest.raises(TypeError):
        RDensFunc(name, 1, grid_weigths)
    with pytest.raises(TypeError):
        RDensFunc(name, grid_points, 1)


def test_dft_co_sto3g():
    """Test functions of HFPySCF class."""
    atom = """C        -3.6180905689    1.3768035675   -0.0207958979
              O        -4.7356838533    1.5255563000    0.1150239130"""
    basis = 'sto-3g'
    mol = gto.M(atom=atom, basis=basis)
    method = 'dft'
    xc_code = 'LDA,VWN'
    dft = ScfPyScf(mol, method, xc_code)
    dft.solve_scf(conv_tol=1e-12)
    dm0 = dft.get_density()
    unperturbed_fock = dft.get_fock()
    assert 'scf' in dft.energy
    assert abs(dft.energy["scf"] - -110.86517923) < 1e-5
    vemb = np.zeros_like(dm0)
    dft.perturb_fock(vemb)
    dft.solve_scf()
    assert abs(dft.energy["scf"] - -110.86517923) < 1e-5
    perturbed_fock = dft.get_fock()
    np.testing.assert_allclose(unperturbed_fock, perturbed_fock, atol=1e-9)


if __name__ == "__main__":
    test_rdensfunc_base()
