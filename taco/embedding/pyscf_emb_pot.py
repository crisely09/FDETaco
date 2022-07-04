"""PySCF Utilities for Embedding calculations."""

import numpy as np
from pyscf import gto
from pyscf.dft import libxc, gen_grid
# from pyscf.dft.fdet import eval_mat_emb  #  For future development in PySCF
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat, _format_uks_dm

from taco.embedding.emb_pot import EmbPotBase
from taco.data.units import BOHR
from taco.embedding.pyscf_fdet import eval_mat_emb


def get_charges_and_coords(mol):
    """Return arrays with charges and coordinates."""
    coords = []
    charges = []
    for i in range(mol.natm):
        # Avoid ghost atoms
        if mol._atm[i][0] == 0:
            next
        else:
            if isinstance(mol.atom, str):
                atm_str = mol.atom.split()
                if mol.unit == 'Bohr':
                    tmp = [float(f) for f in atm_str[i*4+1:(i*4)+4]]
                else:
                    tmp = [float(f)/BOHR for f in atm_str[i*4+1:(i*4)+4]]
            else:
                if mol.unit == 'Bohr':
                    tmp = [mol.atom[i][1][j] for j in range(3)]
                else:
                    tmp = [mol.atom[i][1][j]/BOHR for j in range(3)]
            coords.append(tmp)
            charges.append(mol._atm[i][0])
    coords = np.array(coords)
    charges = np.array(charges, dtype=int)
    return charges, coords


def compute_nuclear_repulsion(mol0, mol1):
    """Compute nuclear repulsion between two fragments.

    Parameters
    ----------
    mol1, mol2 : gto.M
        Molecule objects

    """
    result = 0
    charges0, coord0 = get_charges_and_coords(mol0)
    charges1, coord1 = get_charges_and_coords(mol1)
    for i, q0 in enumerate(charges0):
        for j, q1 in enumerate(charges1):
            d = np.linalg.norm(coord0[i]-coord1[j])
            result += q0*q1/d
    return result


def compute_attraction_potential(mol0, mol1):
    """Compute the nuclei-electron attraction potentials.

    Returns
    -------
    v0nuc1 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.
    v1nuc0 : np.ndarray(NAO,NAO)
        Attraction potential between electron density of fragment0
        and nuclei of fragment1.

    """
    # Nuclear-electron attraction integrals
    mol0_charges, mol0_coords = get_charges_and_coords(mol0)
    mol1_charges, mol1_coords = get_charges_and_coords(mol1)
    v0_nuc1 = 0
    for i, q in enumerate(mol1_charges):
        mol0.set_rinv_origin(mol1_coords[i])
        v0_nuc1 += mol0.intor('int1e_rinv') * -q
    v1_nuc0 = 0
    for i, q in enumerate(mol0_charges):
        mol1.set_rinv_origin(mol0_coords[i])
        v1_nuc0 += mol1.intor('int1e_rinv') * -q
    return v0_nuc1, v1_nuc0


def compute_coulomb_potential(mol0, mol1, dm1):
    """Compute the electron-electron repulsion potential.

    Returns
    -------
    v_coulomb : np.ndarray(NAO,NAO)
        Coulomb repulsion potential.

    """
    mol1234 = mol1 + mol1 + mol0 + mol0
    shls_slice = (0, mol1.nbas,
                  mol1.nbas, mol1.nbas+mol1.nbas,
                  mol1.nbas+mol1.nbas, mol1.nbas+mol1.nbas+mol0.nbas,
                  mol1.nbas+mol1.nbas+mol0.nbas, mol1234.nbas)
    eris = mol1234.intor('int2e', shls_slice=shls_slice)
    v_coulomb = np.einsum('ab,abcd->cd', dm1, eris)
    return v_coulomb


def get_dft_grid_stuff(code, rho_both, rho0, rho1):
    """Evaluate energy densities and potentials on a grid.

    Parameters
    ----------
    code : str
        String with density functional code for PySCF.
    rho_both :  np.ndarray(npoints, dtype=float)
        Total density evaluated on n grid points.
    rho1, rho2 :  np.ndarray(npoints, dtype=float)
        Density of each fragment evaluated on n grid points.

    """
    exc, vxc = libxc.eval_xc(code, rho_both)[:2]
    exc2, vxc2 = libxc.eval_xc(code, rho0)[:2]
    exc3, vxc3 = libxc.eval_xc(code, rho1)[:2]
    return (exc, exc2, exc3), (vxc, vxc2, vxc3)


def get_nad_energy(grid, energies, rho_both, rho1, rho2):
    """Calculate non-additive energy.

    Parameters
    ----------
    grid : len_grids.grids
        Integration grid object.
    energies : list
        List of individual energies: total, [fragment1, fragment2]
    rho_both :  np.ndarray(npoints, dtype=float)
        Total density evaluated on n grid points.
    rho1, rho2 :  np.ndarray(npoints, dtype=float)
        Density of each fragment evaluated on n grid points.

    """
    e_nad = np.dot(rho_both*grid.weights, energies[0])
    e_nad -= np.dot(rho1*grid.weights, energies[1])
    e_nad -= np.dot(rho2*grid.weights, energies[2])
    return e_nad


def make_potential_matrix(mol0, mol1, system, dm0, dm1, dm_both, grids, xc_code):
    """Make Fock-like matrix potential.

    Parameters
    ----------
    mol0, mol1 : gto.Molecule
        PySCF molecule objects.
    system :  gto.Molecule
        PySCF object combining mol0 and mol1.
    dm0, dm1, dm_both : np.ndarray
        Density matrices of fragment0, fragment1 and both.
    grids : libgen Grids
        PySCF integration grid.
    xc_code : str
        Name of density functional for exchange and correlation or kinetic.

    Returns
    -------
    tuple(exc_nad, v_nad_xc)
        Energies and matrices with the potentials in a matrix form.
    """
    xctype = libxc.xc_type(xc_code)
    if xctype == "LDA":
        ao_mol0 = eval_ao(mol0, grids.coords, deriv=0)
        ao_mol1 = eval_ao(mol1, grids.coords, deriv=0)
        # Make Complex DM
        ao_both = eval_ao(system, grids.coords, deriv=0)
        # Compute DFT non-additive potential and energies
        rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
        rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='LDA')
        rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
        # Compute all densities on a grid
        excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
        vxc_emb = vxcs[0][0] - vxcs[1][0]
        # Energy functionals:
        exc_nad = get_nad_energy(grids, excs, rho_both, rho_mol0, rho_mol1)
        v_nad_xc = eval_mat(mol0, ao_mol0, grids.weights, rho_mol0, vxc_emb, xctype='LDA')
    else:  # xctype == "GGA"
        ao_mol0 = eval_ao(mol0, grids.coords, deriv=1)
        ao_mol1 = eval_ao(mol1, grids.coords, deriv=1)
        # Make Complex DM
        ao_both = eval_ao(system, grids.coords, deriv=1)
        # Compute DFT non-additive potential and energies
        rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='GGA')
        rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='GGA')
        rho_both = eval_rho(system, ao_both, dm_both, xctype='GGA')
        # Compute all densities on a grid
        excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
        # Energy functionals:
        exc_nad = get_nad_energy(grids, excs, rho_both[0], rho_mol0[0], rho_mol1[0])
        v_nad_xc = eval_mat_emb(mol0, ao_mol0, grids.weights, rho_both, rho_mol0, vxcs[0],
                                vxcs[1], xctype='GGA')
    return exc_nad, v_nad_xc


def make_both_potential_matrices(mol0, mol1, system, dm0, dm1, dm_both, grids, xc_code, t_code):
    """Make Fock-like matrix potential when both functionals are the same type.

    Parameters
    ----------
    mol0, mol1 : gto.Molecule
        PySCF molecule objects.
    system :  gto.Molecule
        PySCF object combining mol0 and mol1.
    dm0, dm1, dm_both : np.ndarray
        Density matrices of fragment0, fragment1 and both.
    grids : libgen Grids
        PySCF integration grid.
    xc_code : str
        Name of density functional for exchange and correlation.
    t_code : str
        Name of density functional for kinetic term.

    Returns
    -------
    tuple(exc_nad, et_nad, v_nad_xc, v_nad_t)
        Energies and matrices with the potentials in a matrix form.
    """
    xctype = libxc.xc_type(xc_code)
    if xctype == "LDA":
        ao_mol0 = eval_ao(mol0, grids.coords, deriv=0)
        ao_mol1 = eval_ao(mol1, grids.coords, deriv=0)
        # Make Complex DM
        ao_both = eval_ao(system, grids.coords, deriv=0)
        # Compute DFT non-additive potential and energies
        rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
        rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='LDA')
        rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
        # Compute all densities on a grid
        excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
        ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_mol0, rho_mol1)
        vxc_emb = vxcs[0][0] - vxcs[1][0]
        vt_emb = vts[0][0] - vts[1][0]
        # Energy functionals:
        exc_nad = get_nad_energy(grids, excs, rho_both, rho_mol0, rho_mol1)
        et_nad = get_nad_energy(grids, ets, rho_both, rho_mol0, rho_mol1)
        v_nad_xc = eval_mat(mol0, ao_mol0, grids.weights, rho_mol0, vxc_emb, xctype='LDA')
        v_nad_t = eval_mat(mol0, ao_mol0, grids.weights, rho_mol0, vt_emb, xctype='LDA')
    else:  # xctype == "GGA"
        ao_mol0 = eval_ao(mol0, grids.coords, deriv=1)
        ao_mol1 = eval_ao(mol1, grids.coords, deriv=1)
        # Make Complex DM
        ao_both = eval_ao(system, grids.coords, deriv=1)
        # Compute DFT non-additive potential and energies
        rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='GGA')
        rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='GGA')
        rho_both = eval_rho(system, ao_both, dm_both, xctype='GGA')
        # Compute all densities on a grid
        excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
        ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_mol0, rho_mol1)
        # Energy functionals:
        exc_nad = get_nad_energy(grids, excs, rho_both[0], rho_mol0[0], rho_mol1[0])
        et_nad = get_nad_energy(grids, ets, rho_both[0], rho_mol0[0], rho_mol1[0])
        v_nad_xc = eval_mat_emb(mol0, ao_mol0, grids.weights, rho_both, rho_mol0, vxcs[0],
                                vxcs[1], xctype='GGA')
        v_nad_t = eval_mat_emb(mol0, ao_mol0, grids.weights, rho_both, rho_mol0, vts[0],
                               vts[1], xctype='GGA')
    return (exc_nad, et_nad, v_nad_xc, v_nad_t)


def compute_uxc_potential(mol0, mol1, grids, xc_code, dm0, dm1, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        mols : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
            This should be the grid of the full system
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha,beta) spin.

    Examples:

    '''
    ni = dft.numint.NumInt()
    xctype = ni._xc_type(xc_code)
    system = gto.M(atom=mol0.atom + mol1.atom, basis=mol0.basis) 

    # First for fragment A
    shls_slice = (0, mol0.nbas)
    ao_loc = mol0.ao_loc_nr()

    dma0, dmb0 = _format_uks_dm(dm0)
    nao0 = dma0.shape[-1]
    make_rhoa0, nset0 = ni._gen_rho_evaluator(mol0, dma0, hermi)[:2]
    make_rhob0       = ni._gen_rho_evaluator(mol0, dmb0, hermi)[0]

    nelec0 = np.zeros((2,nset0))
    excsum0 = np.zeros(nset0)

    # Now for fragment B
    shls_slice = (0, mol1.nbas)
    ao_loc = mol1.ao_loc_nr()

    dma1, dmb1 = _format_uks_dm(dm1)
    nao1 = dma1.shape[-1]
    make_rhoa1, nset1 = ni._gen_rho_evaluator(mol1, dma1, hermi)[:2]
    make_rhob1       = ni._gen_rho_evaluator(mol1, dmb1, hermi)[0]

    nelec1 = np.zeros((2,nset1))
    excsum1 = np.zeros(nset1)

    # Final matrix
    vmat = np.zeros((2,nset0,nao0,nao0), dtype=np.result_type(dma0, dmb0))

    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        ao0 = eval_ao(mol0, grids.coords, ao_deriv)
        aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
        for idm in range(nset):
            rho_a = make_rhoa(idm, ao, mask, xctype)
            rho_b = make_rhob(idm, ao, mask, xctype)
            exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                  relativity=relativity, deriv=1,
                                  verbose=verbose)[:2]
            vrho = vxc[0]
            den = rho_a * weight
            nelec[0,idm] += den.sum()
            excsum[idm] += numpy.dot(den, exc)
            den = rho_b * weight
            nelec[1,idm] += den.sum()
            excsum[idm] += numpy.dot(den, exc)

            # *.5 due to +c.c. in the end
            #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,0], out=aow)
            aow = _scale_ao(ao, .5*weight*vrho[:,0], out=aow)
            vmat[0,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
            #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
            aow = _scale_ao(ao, .5*weight*vrho[:,1], out=aow)
            vmat[1,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
            rho_a = rho_b = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            aow = numpy.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += numpy.dot(den, exc)

                wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
                #:aow = numpy.einsum('npi,np->pi', ao, wva, out=aow)
                aow = _scale_ao(ao, wva, out=aow)
                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                #:aow = numpy.einsum('npi,np->pi', ao, wvb, out=aow)
                aow = _scale_ao(ao, wvb, out=aow)
                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = wva = wvb = None

    for i in range(nset):
        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
        vmat[1,i] = vmat[1,i] + vmat[1,i].conj().T
    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    return nelec, excsum, vmat


class PyScfEmbPot(EmbPotBase):
    """Base class for embedding potentials.

    Attributes
    ----------
    mol0, mol1 :
        Molecule objects.
    main_info : dict
        Main information to be used in other places:
        atoms, basis
    vemb_dict :  dict
        Container for matrices involved in embedding calculation.
    energy_dict :  dict
        Container for energies involved in embedding calculation.

    Methods
    -------
    __init__(self, mol0, mol1, frag1_args, emb_args)
    check_molecules(self, mol0, mol1)
    check_emb_arguments(emb_args)
    save_maininfo(self)
    assign_dm(self, nfrag, dm)
    compute_coulomb_potential(self)
    compute_attraction_potential(self)
    compute_nad_potential(self)
    compute_embedding_potential(self)
    export_matrices(self)

    """
    def __init__(self, mol0, mol1, emb_args):
        """Embedding potential Object.

        Parameters
        ----------
        mol0, mol1 : Depending on the program
            Molecule objects.
        dm0, dm1 : np.ndarray(NAO,NAO)
            One-electron density matrices.
        emb_args : dict
            Parameters for the embedding calculation:
            x_func, c_func, t_func.

        """
        self.check_molecules(mol0, mol1)
        self.save_maininfo(mol0)
        EmbPotBase.__init__(self, mol0, mol1, emb_args)

    @staticmethod
    def check_molecules(mol0, mol1):
        """Verify they are PySCF gto.M objects."""
        if not isinstance(mol0, gto.Mole):
            raise TypeError("mol0 must be gto.M PySCF object.")
        if not isinstance(mol1, gto.Mole):
            raise TypeError("mol1 must be gto.M PySCF object.")

    def save_maininfo(self, mol0):
        """Save in a dictionary basic information of mol0 in a simple format."""
        mol0_charges, mol0_coords = get_charges_and_coords(mol0)
        mol0_basis = mol0.basis
        mol0_nbas = mol0.nao_nr()
        self.maininfo = dict(inprog="pyscf", atoms=mol0_charges, basis=mol0_basis,
                             nbas=mol0_nbas)

    def compute_coulomb_potential(self):
        """Compute the electron-electron repulsion potential.

        Returns
        -------
        v_coulomb : np.ndarray(NAO,NAO)
            Coulomb repulsion potential.

        """
        return compute_coulomb_potential(self.mol0, self.mol1, self.dm1)

    def compute_attraction_potential(self):
        """Compute the nuclei-electron attraction potentials.

        Returns
        -------
        v0nuc1 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.
        v1nuc0 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.

        """
        return compute_attraction_potential(self.mol0, self.mol1)

    def compute_nad_potential(self):
        """Compute the non-additive potentials and energies.

        Returns
        -------
        vxc_nad : np.ndarray(NAO,NAO)
            Non-additive Exchange-Correlation + Kinetic potential.

        """
        # Create supersystem
        newatom = '\n'.join([self.mol0.atom, self.mol1.atom])
        system = gto.M(atom=newatom, basis=self.mol0.basis)
        # Construct grid for complex
        grids = gen_grid.Grids(system)
        grids.level = 4
        grids.build()
        nao_mol0 = self.mol0.nao_nr()
        nao_mol1 = self.mol1.nao_nr()
        nao_tot = nao_mol0 + nao_mol1
        dm_both = np.zeros((nao_tot, nao_tot))
        dm_both[:nao_mol0, :nao_mol0] = self.dm0
        dm_both[nao_mol0:, nao_mol0:] = self.dm1
        xc_code = self.emb_args["xc_code"]
        t_code = self.emb_args["t_code"]
        # Check type of functional
        xctype = libxc.xc_type(xc_code)
        tstype = libxc.xc_type(t_code)
        if xctype == tstype:
            return make_both_potential_matrices(self.mol0, self.mol1, system, self.dm0,
                                                self.dm1, dm_both, grids, xc_code, t_code)
        else:
            exc_nad, v_nad_xc = make_potential_matrix(self.mol0, self.mol1, system, self.dm0,
                                                      self.dm1, dm_both, grids, xc_code)
            et_nad, v_nad_t = make_potential_matrix(self.mol0, self.mol1, system, self.dm0,
                                                    self.dm1, dm_both, grids, t_code)
            return (exc_nad, et_nad, v_nad_xc, v_nad_t)

    def compute_embedding_potential(self, dm0=None, dm1=None):
        """Compute embedding potential.

        Parameters
        ----------
        dm0, dm1 : np.ndarray(NAO,NAO)
            One-electron density matrices.

        Returns
        -------
        vemb : np.ndarray
            Embedding potential as a Fock-like matrix.

        """
        if dm0 is None:
            if self.dm0 is None:
                raise AttributeError("Density matrix for fragment 0 is missing.")
            else:
                dm0 = self.dm0
        else:
            self.assign_dm(0, dm0)
        if dm1 is None:
            if self.dm1 is None:
                raise AttributeError("Density matrix for fragment 1 is missing.")
            else:
                dm1 = self.dm1
        else:
            self.assign_dm(1, dm1)
        # Get DFT non-additive terms
        ref_vnad = self.compute_nad_potential()
        exc_nad, et_nad, v_nad_xc, v_nad_t = ref_vnad
        self.vemb_dict["exc_nad"] = exc_nad
        self.vemb_dict["et_nad"] = et_nad
        # Electrostatic part
        v_coulomb = self.compute_coulomb_potential()
        # Nuclear-electron integrals
        v0_nuc1, v1_nuc0 = self.compute_attraction_potential()
        vemb = v_coulomb + v_nad_xc + v_nad_t + v0_nuc1
        self.vemb_dict["v_coulomb"] = v_coulomb
        self.vemb_dict["v_nad_t"] = v_nad_t
        self.vemb_dict["v_nad_xc"] = v_nad_xc
        self.vemb_dict["v0_nuc1"] = v0_nuc1
        self.vemb_dict["v1_nuc0"] = v1_nuc0
        self.vemb_dict["vemb"] = vemb
        return vemb
