import json
import math
import numpy as np
from scipy.optimize import linprog
P=np.array([
[-1.1165707282527606,0.8907405935581916,0.38954359053015475],
[-1.1961322028105807,0.7504209914329159,0.25329830570039436],
[-0.9336157717673905,-0.22187878718069287,1.9706943205416514],
],dtype=float)
C=np.array([
[-1.4360573094831204,-1.053497591019691,0.3752449571409362],
[1.3597123033165000,1.6868391130867888,-0.40333112014312666],
[1.0754457208618984,-1.5259818399516214,-0.4553717768811505],
],dtype=float)
radius=4.0
cost=np.array([[(np.linalg.norm(P[i]-C[j])**2)/(radius*radius) for j in range(3)] for i in range(3)])
parent=np.full(3,1.0e-3)
child_lo=np.full(3,0.9e-3)
child_hi=np.full(3,1.1e-3)
rank=np.argsort(cost,axis=1,kind="stable")
seed=np.zeros((3,3),dtype=bool)
for i in range(3): seed[i,rank[i,:2]]=True
def solve(allowed):
    c=cost.ravel()
    Aeq=np.zeros((3,9))
    for i in range(3): Aeq[i,i*3:(i+1)*3]=1
    bub=[]; b=[]; 
    for j in range(3):
        row=np.zeros(9); row[j::3]=1; bub.append(row); b.append(child_hi[j])
        bub.append(-row); b.append(-child_lo[j])
    bounds=[(0,None) if allowed.flat[k] else (0,0) for k in range(9)]
    r=linprog(c,A_ub=np.array(bub),b_ub=np.array(b),A_eq=Aeq,b_eq=parent,bounds=bounds,method="highs")
    return r
full=solve(np.ones((3,3),dtype=bool))
sparse=solve(seed)
if not full.success or not sparse.success or not (full.fun < sparse.fun-1e-6):
    raise SystemExit("oracle fixture did not establish a feasible but suboptimal nearest-two graph")
print(json.dumps({
 "solver":"scipy.optimize.linprog(method=highs)",
 "parent_mass_kg":parent.tolist(),
 "child_reference_volume_m3":[1e-6]*3,
 "density_bounds_kg_m3":[900,1100],
 "max_transport_distance_m":radius,
 "cost_matrix_normalized_squared_distance":cost.tolist(),
 "nearest_two_child_rows":rank[:,:2].tolist(),
 "full_optimum_objective":float(full.fun),
 "full_optimum_transport_kg":full.x.reshape(3,3).tolist(),
 "nearest_two_optimum_objective":float(sparse.fun),
 "nearest_two_transport_kg":sparse.x.reshape(3,3).tolist(),
 "status":"PASS independent optimum and feasible-suboptimal seed graph"
},indent=2,sort_keys=True))

