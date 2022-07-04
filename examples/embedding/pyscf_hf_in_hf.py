"""Run embedded HF-in-HF case."""

from pyscf import gto
from taco.embedding.pyscf_wrap import PyScfWrap

# Define Molecules
atom_co = """C        -3.6180905689    1.3768035675   -0.0207958979
             O        -4.7356838533    1.5255563000    0.1150239130"""
atom_h2o = """O  -7.9563726699    1.4854060709    0.1167920007
              H  -6.9923165534    1.4211335985    0.1774706091
              H  -8.1058463545    2.4422204631    0.1115993752"""
# Define arguments
basis = 'cc-pvdz'
co = gto.M(atom=atom_co, basis=basis)
h2o = gto.M(atom=atom_h2o, basis=basis)
method = 'hf'
args0 = {"mol": co, "method": method}
args1 = {"mol": h2o, "method": method}
# Use LDA functionals for embedding potential, and solve final energy with HF
embs = {"mol": co, "method": 'hf',
        "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF'}
# Make a wrap
wrap = PyScfWrap(args0, args1, embs)
# Run the embedding calculation
wrap.run_embedding()
# Save information to files
wrap.print_embedding_information(to_csv=True)
wrap.export_matrices()
