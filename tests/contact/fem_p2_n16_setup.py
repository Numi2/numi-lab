import json
from fem_p2_dynamics import Model
m=Model(16,18)
print(json.dumps({'dofs':int(m.b.N),'tetrahedra':int(m.mesh.nelements),
 'mass_nonzeros':int(m.M.nnz),'factor_nonzeros':int(m.factor.nnz),
 'total_mass_kg':float(m.massload.sum())}),flush=True)
