"""
"""

import numpy as np

def compute_kinetic_tf(rho):
    """Thomas-Fermi kinetic energy functional."""
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    et = cf*(np.power(rho, 5./3.))
    vt = cf*5./3.*(np.power(rho, 2./3.))
    return (et, vt)


def compute_exchange_slater(rho):
    """Slater exchange energy functional."""
    #from scipy.special import cbrt
    cx = (3./4) * (3/np.pi)**(1./3)
    ex = - cx * (np.power(rho, 4./3))
    vx = -(4./3) * cx * pow(np.fabs(rho),1./3)
    return (ex, vx)


def compute_ldacorr_pyscf(rho, xc_code=',VWN'):
    """Correlation energy functional."""
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return (exc, vxc[0])


def compute_kinetic_weizsacker_potential(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    # A zero mask is added to exclude areas where rho=0
    zero_mask = np.where(abs(rho_devs[0] - 0.0) > 1e-10)[0]
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += 1.0/8.0*(np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - 1.0/4.0*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot


def compute_kinetic_weizsacker_modified(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    # Experimental functional derivative with the 1/8 factor:
    # A zero mask is added to exclude areas where rho=0
    zero_mask = np.where(abs(rho_devs[0] - 0.0) > 1e-10)[0]
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += 1.0/8.0*(np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - 1.0/8.0*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot


def ndsd_switch_factor(rho_devs, plambda):
    """Compute the NDSD switch factor.

    This formula follows eq. 21 from Garcia-Lastra 2008.

    Paramters
    ---------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    plambda :  float
        Smoothing parameter.
    """
    #CHANGES: last sfactor line added a * instead of +; zero mask for rho>rho_min; sb /= pow term;

    rhob = rho_devs[0]
    sb_min = 0.3
    sb_max = 0.9
    rhob_min = 0.7
    # A zero mask is added to exclude areas where rho=0
    zero_mask = np.where(abs(rho_devs[0] - 0.0) > rhob_min)[0]
    sfactor = np.zeros(rho_devs[0].shape)
    sb = np.linalg.norm(rho_devs[1:4].T[zero_mask])
    sb /= 2.0*((3*np.pi**2)**(1./3.))*pow(rhob[zero_mask], 4./3.)
    sfactor[zero_mask] = 1.0/(np.exp(plambda*(-sb+sb_min)) + 1.0)
    sfactor[zero_mask] *= 1.0-(1.0/(np.exp(plambda * (-sb + sb_max)) + 1.0))
    sfactor[zero_mask] *= 1.0/(np.exp(plambda * (-rhob[zero_mask] + rhob_min)) + 1.0)
    return sfactor


def _kinetic_ndsd_potential(rho0_devs, rho1_devs, plambda):
    """Compute the NDSD potential.

    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    rho_tot = rho0_devs[0] + rho1_devs[0]
    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot)
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])
    etf_1, vtf_1 = compute_kinetic_tf(rho1_devs[0])
    sfactor = ndsd_switch_factor(rho1_devs, plambda)
    # sfactor = np.zeros(rho0_devs[0].shape)
    wpot = compute_kinetic_weizsacker_potential(rho1_devs)
    v_ndsd = vtf_tot - vtf_0 + sfactor*wpot
    return v_ndsd


def _kinetic_ndsd2_potential(rho0_devs, rho1_devs, plambda):
    """Compute the NDSD2 potential.

    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    rho_tot = rho0_devs[0] + rho1_devs[0]
    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot)
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])
    etf_1, vtf_1 = compute_kinetic_tf(rho1_devs[0])
    sfactor = ndsd_switch_factor(rho1_devs, plambda)
    wpot = compute_kinetic_weizsacker_modified(rho1_devs)
    v_ndsd = vtf_tot - vtf_0 + sfactor*wpot
    return v_ndsd


def compute_kinetic_ndsd(rho0_devs, rho1_devs, plambda, grid, version=1):
    """Compute the NDSD(1,2) energy and potential.

    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    grid : Grids
        Molecular integration grid from PySCF
    """
    rho_tot = rho0_devs[0] + rho1_devs[0]
    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot)
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])
    etf_1, vtf_1 = compute_kinetic_tf(rho1_devs[0])
    # sfactor = np.zeros(rho0_devs[0].shape)
    if version == 1:
        wpot = compute_kinetic_weizsacker_potential(rho1_devs)
        sfactor = ndsd_switch_factor(rho1_devs, plambda)
    elif version == 2:
        wpot = compute_kinetic_weizsacker_modified(rho1_devs)
        sfactor = ndsd_switch_factor(rho1_devs, plambda)
    v_ndsd = vtf_tot - vtf_0 + sfactor*wpot
    e_nad = np.dot(grid.weights, etf_tot - etf_0 - etf_1)
    e_ndsd = e_nad + np.dot(grid.weights, rho0_devs[0]*wpot*sfactor)
    return e_ndsd, v_ndsd


class RDensFunc:
    """(Restricted) Density functionals.

    Description
    -----------
    An object to store the information of density functionals
    and its derivatives.

    Attributes
    ----------
    name :  str
        Standard or made up name for the functional.
    rho_ndevs : int
        Number of derivatives of the density it requires for its
        evaluation.
    """
    def __init__(self, name, grid_points, grid_weigths, function=None):
        """
        Initialize a density functional.

        Parameters
        ----------
        name : str
            Name of the density functional. If using PySCF it should be
            a valid `xc_code`.
        functions : list(callables)
            The function that evaluates the functional, should return a tuple
            with (density-energy, potential).
            If None is given, then the name of the functional
            is searched in the LibXC library and constructed
            with PySCF.
        """
        if not isinstance(name, str):
            raise TypeError('`name` must be a string.')
        if not isinstance(grid_points, np.ndarray):
            if isinstance(grid_points, list):
                grid_points = np.array(grid_points)
            else:
                raise TypeError('`grid_points` should be either a list or a numpy array.')
        if not isinstance(grid_weigths, np.ndarray):
            if isinstance(grid_weigths, list):
                grid_weigths = np.array(grid_weigths)
            else:
                raise TypeError('`grid_weigths` should be either a list or a numpy array.')
        self.name = name
        self.gpoints = grid_points
        self.gweigths = grid_weigths
        self.from_libxc = False
        if function is None:
            self.from_libxc = True
        else:
            self.eval_xc_func = function
        self.results = None

    def __call__(self, rho_devs=None, ndevs=1):
        """Returns the values of the Functional and derivatives on a grid.

        Parameters
        ----------
        rho_devs : list of arrays
            Density and density-derivatives evaluated at on a grid.
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            where N is number of grid points.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2

        Returns
        -------
        results : tuple((N), (N))
            The density-energy, potential and kernel (if ndevs=2).
        """
        if rho_devs is None:
            if self.results is None:
                raise ValueError('Please provide the density and density derivatives.')
            else:
                return self.results
        else:
            return self._eval_xc(rho_devs, ndevs)

    def _eval_xc(self, rho_devs, ndevs):
        if self.from_libxc:
            results = libxc.eval_xc(self.name, rho_devs, deriv=ndevs, spin=0)
            if ndevs > 1:
                self.results = (results[0], results[1], results[2])
            else:
                self.results = (results[0], results[1])
        else:
            self.results = self.eval_xc_func(rho_devs)
        return self.results

    @property
    def energy(self):
        """Evaluate the energy functional from the energy-density."""
        return self._energy()

    def _energy(self):
        """Actually evaluate the energy functional.
        """
        ene_dens = self.results[0]
        return np.dot(self.gweigths, ene_dens)

    @property
    def expectation_value(self, rho=None):
        """Return the integration of the potential and the density rho.
        """
        return self._exp_val(rho)

    def _exp_val(self, rho):
        r"""Return the integration of the potential and the density rho.

        Parameters
        ----------
        rho : np.ndarray(Npoints)
            Density evaluated on integration grid, used to evaluate
            the integral of $\int rho(\mathbf{r}) v_nad(\mathbf{r}) d\mathbf{r}$
        """
        potential = self.results[1]
        return np.dot(self.gweigths, rho*potential)
