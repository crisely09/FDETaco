"""Compute the embedding potential on a grid and export it."""
import numpy as np

from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.dft.numint import eval_ao, eval_rho
from taco.embedding.pyscf_wrap import PyScfWrap
from taco.embedding.pyscf_wrap_single import get_density_from_dm, get_dft_grid_stuff
from taco.embedding.pyscf_wrap_single import get_charges_and_coords
#from taco.embedding.cc_gridfns import coulomb_potential_grid, nuclear_attraction_energy

# Define Molecules with QCElemental
atom_co = """C        -3.6180905689    1.3768035675   -0.0207958979
             O        -4.7356838533    1.5255563000    0.1150239130"""
atom_h2o = """O  -7.9563726699    1.4854060709    0.1167920007
              H  -6.9923165534    1.4211335985    0.1774706091
              H  -8.1058463545    2.4422204631    0.1115993752"""
# Define arguments
basis = 'cc-pvdz'
co = gto.M(atom=atom_co, basis=basis)
h2o = gto.M(atom=atom_h2o, basis=basis)
method = 'dft'
xc_code = 'LDA,VWN'
args0 = {"mol": co, "method": method, "xc_code": xc_code}
args1 = {"mol": h2o, "method": method, "xc_code": xc_code}
embs = {"mol": co, "method": 'dft',
        "xc_code": xc_code, "t_code": 'LDA_K_TF'}
# Make a wrap
wrap = PyScfWrap(args0, args1, embs)

# Make molecule in pyscf
co_mol = co
h2o_mol = h2o
dm0 = wrap.method0.get_density()
dm1 = wrap.method1.get_density()
# Create supersystem
newatom = '\n'.join([co_mol.atom, h2o_mol.atom])
system = gto.M(atom=newatom, basis=basis)
# Construct grid for integration
grids = gen_grid.Grids(system)
grids.level = 4
grids.build()
# Grid for plot
points = np.array([[0., 0., z] for z in np.arange(-10., 10., 0.1)])
rho0 = get_density_from_dm(wrap.mol0, dm0, points)
rho1 = get_density_from_dm(wrap.mol1, dm1, points)
rho1_grid = get_density_from_dm(wrap.mol1, dm1, grids.coords)
# TODO: The electrostatic potential needs to be fixed. Switch from cc_grids to compiled executable
# Coulomb repulsion potential
#v_coul = coulomb_potential_grid(points, grids.coords, grids.weights, rho1_grid)
## Nuclear-electron attraction potential
#mol1_charges, mol1_coords = get_charges_and_coords(h2o_mol)
#v1_nuc0 = np.zeros(rho0.shape)
#for i in range(len(mol1_charges)):
#    v1_nuc0 += - mol1_charges[i]*rho0/np.linalg.norm(points-mol1_coords[i]) 
# DFT nad potential
xc_code = embs["xc_code"]
t_code = embs["t_code"]
rho_both = rho0 + rho1
excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho0, rho1)
ets, vts = get_dft_grid_stuff(t_code, rho_both, rho0, rho1)
vxc_emb = vxcs[0][0] - vxcs[1][0]
vt_emb = vts[0][0] - vts[1][0]

#vemb_tot = v_coul + v1_nuc0 + vxc_emb + vt_emb
vemb_nad = vxc_emb + vt_emb


# Save it to use it later
np.save('vnad_co_h2o.npy', vemb_nad)
