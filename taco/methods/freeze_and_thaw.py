"""All needed for Freeze-and-thaw Embedding calculations with PySCF.

Author: Cristina E. Gonzalez-Espinoza
Date: Dec. 2020

"""

import numpy as np
from copy import copy

from pyscf import gto, scf, dft
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_mat
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_coulomb_repulsion
from taco.embedding.pyscf_wrap_single import compute_nuclear_repulsion
from taco.embedding.pyscf_wrap_single import get_nad_energy


def nad_printings(info):
    """
    """
    print("          Non-Additive terms             ")
    print("E_xc_nad   : \t %.10f a.u." % self.exc_nad)
    print("Ts_nad     : \t %.10f a.u.", self.ts_nad)
    print("     Integrals of Non-Additive terms     ")
    print("Int. v_xc_nad   : \t %.10f a.u." % self.int_vxc_nad)
    print("Int. v_ts_nad   : \t %.10f a.u.", self.int_vts_nad)
    print(line)

def elst_printings(info):
    """
    """

def compute_embedding_potential(mola, dma, molb, dmb, points, xct_nad, plambda=50):
    """
    """
    # Evaluate electron densities
    rhoa_devs = get_density_from_dm(mola, dma, points, deriv=3, xctype='meta-GGA')
    rhob_devs = get_density_from_dm(molb, dmb, points, deriv=3, xctype='meta-GGA')
    # Coulomb repulsion potential
    v_coul = get_coulomb_repulsion(molb, dmb, points)
    # Nuclear-electron attraction potential
    molb_charges, molb_coords = get_charges_and_coords(molb)
    vb_nuca = np.zeros(rhoa_devs[0].shape)
    for j, point in enumerate(points):
        for i in range(len(molb_charges)):
            vb_nuca[j] += - molb_charges[i]/np.linalg.norm(point-molb_coords[i]) 
    # DFT nad potential
    rho_tot = rhoa_devs[0] + rhob_devs[0]
    # XC term
    exc_tot, vxc_tot = compute_corr_pyscf(rho_tot, xc_code)
    exc_a, vxc_a = compute_corr_pyscf(rhoa_devs[0], xc_code)
    exc_b, vxc_b = compute_corr_pyscf(rhob_devs[0], xc_code)
    vxc_nad = vxc_tot - vxc_a
    # Ts term
    ets_tot, vts_tot = compute_kinetic_tf(rho_tot)
    ets_a, vts_a = compute_kinetic_tf(rhoa_devs[0])
    vts_nad = vts_tot - vts_a
 #  vts_nad = compute_kinetic_ndsd_potential(rhoa_devs, rhob_devs, plambda)

    vemb_tot = v_coul + vb_nuca + vxc_nad + vts_nad
    return vemb_tot


class FDETFandT:
    def __init__(self, scf0, scf1, embpot, embfunc, conv=1e-7, grid_level=4):
        """
        Initialize the FDETFandT class.

        Parameters
        ----------
        Args :
            scf0, scf1 : instance of SCF from PySCF
                Fragments' mean-field information.
            emb_pot : callable
                Function to compute the embedding potential.
                Should return a Fock-like matrix, a AO representation of the potential.
            emb_func : callable
                Function to compute the energies of the functionals used in the
                embedding potential.
        Kwargs :
            conv : float
                Convergence criterium for the density difference.
            """
        self.mol0 = scf0.mol
        self.mol1 = scf1.mol
        self.ref_scf0 = scf0
        self.ref_scf1 = scf1
        self.conv = conv
        self.emb_pot = emb_pot
        self.emb_func = emb_func
        self.scf_emb0 = None
        self.scf_emb1 = None
        self.tot_ene = None
        self.converged = False
        # Construct grids for integration
        # This could be any grid, but we use the default from PySCF
        self.grid0 = gen_grid.Grids(self.mol0)
        self.grid0.level = grid_level
        self.grid0.build()
        self.grid1 = gen_grid.Grids(self.mol1)
        self.grid1.level = grid_level
        self.grid1.build()
        # Evaluate AOs on grid
        self.ao_mol0 = eval_ao(self.mol0, self.grid0.coords, deriv=0)
        self.ao_mol1 = eval_ao(self.mol1, self.grid1.coords, deriv=0)

    def compute_embedding_potential(self, dm0, dm1, ismol0=True):
        """Compute the embedding potential from density matrices.

        Parameters
        ----------
        dm0, dm1 :  np.ndarray(dtype=float)
            Reference density matrices of fragments 0 and 1.
        grid0, grid1 : Grids instance
            Molecular integration grids.
        """
        #############################################################
        # Make embedding potential 
        #############################################################
        if ismol0:
            mol0 = self.mol0
            mol1 = self.mol1
            ao_mol0 = self.ao_mol0
            grid0 = self.grid0
        else:
            mol0 = self.mol1
            mol1 = self.mol0
            ao_mol0 = self.ao_mol1
            grid0 = self.grid1
        vemb_mat = self.embpot(mol0, dm0, mol1, dm1, grid0.coords)
        return vemb_mat

    def kernel(self, maxcycle=15, conv=None):
        """Run the Freeze-and-Thaw cycles.
        """
        if self.converged == True:
            return self.energies
        if conv:
            self.conv = conv
        # Check if individual fragments are converged already
        # If not, evaluate the SCFs
        if self.scf0.converged is False:
            self.scf0.kernel()
        if self.scf1.converged is False:
            self.scf1.kernel()
        dm0 = self.scf0.make_rdm1()
        dm1 = self.scf1.make_rdm1()
        # Initial embedding potential
        rho0 = get_density_from_dm(self.mol0, dm0, self.grid0.coords)
        vemb_mat = self.compute_embedding_potential(dm0, dm1, ismol0=True)
        #############################################################
        # Run freeze and thaw
        #############################################################
        count = 0
        # Loop until embedded density and energy converges
        while True:
            print(" => Freeze and Thaw SCF cycle: ", count)
            if count >= maxcycle:
                print(" === Reached MAXCYCLE - NOT CONVERGED === ")
                break
            if count % 2 == 0:
                mol0 = self.mol0
                mol1 = self.mol1
                ao_mol1 = self.ao_mol1
                fock_ref = self.scf0.get_hcore()
                fock = fock_ref.copy()
                grid1 = self.grid1
                if count == 0:
                    dm0 = dm0.copy()
                    dm1 = dm1.copy()
                else:
                    dm0 = dm1.copy()
                    dm1 = dm_final.copy()
            else:
                mol0 = self.mol1
                mol1 = self.mol0
                ao_mol1 = self.ao_mol0
                fock_ref = self.scf1.get_hcore()
                fock = fock_ref.copy()
                dm0 = dm1.copy()
                dm1 = dm_final.copy()
                grid1 = self.grid0
            scfemb = dft.RKS(mol0)
            fock += vemb_mat
            scfemb.get_hcore = lambda *args: fock
            scfemb.conv_tol = 1e-11
            # Solve
            scfemb.kernel()
            dm_final = scfemb.make_rdm1()
            energy_final = scfemb.e_tot
            if count == 0:
                energy_old = energy_final
                self.scf_emb0 = copy(scfemb)
            elif count % 2 == 0:
                denergy = abs(energy_final - energy_old)
                self.scf_emb0 = copy(scfemb)
                print(" Energy Difference: ", denergy)
                if denergy <= self.conv:
                    print(" === Freeze and Thaw Converged === ", denergy)
                    self.dm0_final = dm_final
                    self.dm1_final = dm1
                    self.tot_ene = energy_final
                    self.converged = True
                    del scfemb
                    del fock_ref, fock
                    break
                else:
                    energy_old = energy_final
            else:
                self.scf_emb1 = copy(scfemb)
            del vemb_mat
            # Re-evaluate the embedding potential
            # But now for the other molecule
            isfrag0 = True if count % 2 == 0 else False
            vemb_mat = self.compute_embedding_potential(dm1, dm_final, ismol0=isfrag0)
            count += 1
            del scfemb
            del fock_ref, fock

    @property
    def energies(self):
        if self.energies is None:
            self._energies
        else:
            self.print_energies()

    def _energies(self):
        """Evaluate embedding energy terms.
        """
        #############################################################
        # Evaluate final energy 
        #############################################################
        rho0 = get_density_from_dm(self.mol0, self.dm0_final, self.grid0.coords)
        vemb_tot = self.compute_embedding_potential(self.dm0_final, self.dm1_final,
                                                    ismol0=True)
        vemb_mat = eval_mat(self.mol0, self.ao_mol0, self.grid0.weights, rho0, vemb_tot, xctype='LDA')
        int_vemb = np.einsum('ab,ba', vemb_mat, self.dm0_final)
        scf_energy = self.tot_ene - int_vemb
        a_charges, a_coords = get_charges_and_coords(self.mol0)
        b_charges, b_coords = get_charges_and_coords(self.mol1)
        enuc = compute_nuclear_repulsion(a_charges, a_coords, b_charges, b_coords)
        # Nuclear-electron attraction integrals
        vbnuca = 0
        for i, q in enumerate(a_charges):
            self.mol1.set_rinv_origin(a_coords[i])
            vbnuca += self.mol1.intor('int1e_rinv') * -q
        vanucb = 0
        for i, q in enumerate(b_charges):
            self.mol0.set_rinv_origin(b_coords[i])
            vanucb += self.mol0.intor('int1e_rinv') * -q
        evbnuca = np.einsum('ab,ba', vbnuca, self.dm1_final)
        evanucb = np.einsum('ab,ba', vanucb, self.dm0_final)
        # Coulomb repulsion
        mol1234 = self.mol1 + self.mol1 + self.mol0 + self.mol0
        shls_slice = (0, self.mol1.nbas,
                  self.mol1.nbas, self.mol1.nbas+self.mol1.nbas,
                  self.mol1.nbas+self.mol1.nbas, self.mol1.nbas+self.mol1.nbas+self.mol0.nbas,
                  self.mol1.nbas+self.mol1.nbas+self.mol0.nbas, mol1234.nbas)
        eris = mol1234.intor('int2e', shls_slice=shls_slice)
        v_coulomb = np.einsum('ab,abcd->cd', self.dm1_final, eris)
        ecoulomb = np.einsum('ab,ba', v_coulomb, self.dm0_final)
        # Non-additive terms
        # DFT nad potential
        rhoa_devs = get_density_from_dm(self.mol0, self.dm0_final, self.grid0.coords,
                                        deriv=3, xctype='meta-GGA')
        rhob_devs = get_density_from_dm(self.mol1, self.dm1_final, self.grid0.coords,
                                        deriv=3, xctype='meta-GGA')
        rho_tot = rhoa_devs[0] + rhob_devs[0]
        # XC terms
        enes = self.emb_func(self.dm0_final, self.dm1_final, self.grid0.coords)
        exc_tot, vxc_tot = compute_corr_pyscf(rho_tot, xc_code)
        exc_a, vxc_a = compute_corr_pyscf(rhoa_devs[0], xc_code)
        exc_b, vxc_b = compute_corr_pyscf(rhob_devs[0], xc_code)
        exc_nad = get_nad_energy(self.grid0.weights, [exc_tot, exc_a, exc_b],
                                 rho_tot, rhoa_devs[0], rhob_devs[0])
        #ts_nad, vts_nad = compute_kinetic_ndsd(rhoa_devs, rhob_devs, plambda, grids)
        
    def log(self, info):
        #############################################################
        # Printings 
        #############################################################
        if info == 'energies':
            print("""
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            Embedding Final energy
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++""")
            line = "-"*59
            print('SCF Energy:\t %.10f a.u.' % self.scf_energy)
            print(line)
            print("          Electrostatic terms            ")
            print("J[rho_A, rho_B] : \t %.10f a.u." % self.ecoulomb)
            print("V_A[rho_B] : \t %.10f a.u." % self.evanucb)
            print("V_B[rho_A] : \t %.10f a.u." % self.evbnuca)
            print("V_AB       : \t %.10f a.u." % self.enuc)
            print(line)

if __name__ == '__main__':
    #############################################################
    # Define Molecules of each fragment
    #############################################################
    geoa = """Li        0.0000000000    0.0000000000   -1.4877074476"""
    geob = """O         0.0000000000    0.0000000000    0.3352231084
              H         0.7935480381    0.0000000000    0.8968960962
              H        -0.7935480381    0.0000000000    0.8968960962"""
    # Define arguments
    basis = 'cc-pvdz'
    xc_code = 'LDA,VWN'
    plambda = 50
    mola = gto.M(atom=geoa, basis=basis, charge=1)
    molb = gto.M(atom=geob, basis=basis)

    #############################################################
    # Get reference densities
    #############################################################
    # TIP: For HF you only need: scfres1 = scf.RHF(mol)
    # Li+
    scfres = dft.RKS(mola)
    scfres.xc = xc_code
    scfres.conv_tol = 1e-12
    scfres.kernel()
    # H2O
    scfres1 = dft.RKS(molb)
    scfres1.xc = xc_code
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    fat = FDETFandT(mola, molb)
