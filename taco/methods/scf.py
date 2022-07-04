#!/usr/bin/env python3
"""
FDE SCF Methods Base Class.
"""
from pyscf import gto


class ScfMethod():
    """Base class for SCF method objects.

    Attributes
    ----------
    density : list[np.ndarray(dtype=float)]
        List with densities obtained with this method.
    energy : dict('name', np.ndarray)
        Energies obtained with this method.

    Properties
    ----------
    restricted : bool
        Wheter the wavefunction is restricted or not.

    Methods
    ---------
    __init__(self, mol)
        Initialize the method.
    get_density(self) :
        Return density/densities.
    solve_scf :
        Perform the SCF calculation.
    perturb_fock(self, pot) :
        Add a potential to the Fock matrix.

    """
    def __init__(self, mol):
        """ SCFMethod object.

        Parameters
        ----------
        mol : PySCF gto.M object
            Contains molecule information: geometry, basis, spin, charge.
        """
        if not isinstance(mol, gto.Mole):
            raise TypeError('Molecule object should be instance of gto.M')
        self.mol = mol
        self.new = True
        self.density = None
        self.energy = {}

    @property
    def restricted(self):
        """Whether it is Restricted case."""
        return self.mol.spin == 0

    # TODO: probably replace with a @cached_property
    def get_density(self):
        """Return the DM(s)."""
        if self.density is not None:
            return self.density
        else:
            self.solve_scf(conv_tol=1e-12)
            return self.density

    def get_energy(self):
        """Return the SCF energy."""
        if self.energy.get("scf", 0):
            return self.energy["scf"]
        else:
            self.solve_scf(conv_tol=1e-12)
            return self.energy["scf"]

    def get_fock(self):
        """Contruct Fock matrix."""
        raise NotImplementedError("get_fock not implemented.")

    def perturb_fock(self, pot):
        """Add an effective potential to the Fock matrix.

        Parameters
        ----------
        pot : np.ndarray(dtype=float)
            Effective potential in the form of a Fock matrix.

        """
        raise NotImplementedError("perturb_fock not implemented.")

    def solve_scf(self):
        """Perform SCF calculation."""
        raise NotImplementedError("solve_scf not implemented.")
