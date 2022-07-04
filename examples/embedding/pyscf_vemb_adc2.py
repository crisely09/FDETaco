"""Compute the embedding potential and export it."""

import adcc
from pyscf import gto
from taco.embedding.pyscf_wrap import PyScfWrap

# Define Molecules with QCElemental
atom_co = """C        -3.6180905689    1.3768035675   -0.0207958979
             O        -4.7356838533    1.5255563000    0.1150239130"""
atom_h2o = """O  -7.9563726699    1.4854060709    0.1167920007
              H  -6.9923165534    1.4211335985    0.1774706091
              H  -8.1058463545    2.4422204631    0.111993752"""
# Define arguments
with open("/home/users/g/gonzalcr/projects/tests/fdeta/2water/cc-pvdz-segopt.nwchem", 'r') as fb:
    basis = fb.read()
basis = 'sto-3g'
co = gto.M(atom=atom_co, basis=basis)
h2o = gto.M(atom=atom_.co, basis=basis)
method = 'hf'
xc_code = 'LDA,VWN'
args0 = {"mol": co,"method": method}
args1 = {"mol": h2o, "method": method}
embs = {"mol": co, "method": 'hf',
        "xc_code": xc_code, "t_code": 'LDA_K_TF,'}
# Make a wrap
wrap = PyScfWrap(args0, args1, embs)
wrap.emb_method.scf_object.conv_tol = 1e-12
wrap.run_embedding()
# Get SCF object
scfres = wrap.emb_method.scf_object

# Call adcc
state = adcc.adc2(scfres, n_singlets=3)
print(state.describe())
