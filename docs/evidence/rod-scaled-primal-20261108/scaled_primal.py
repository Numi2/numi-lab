#!/usr/bin/env python3
"""FP32 scaled tridiagonal primal axial solve versus FP64 dense equilibrium."""
import json
import math
import numpy as np

REST_TOTAL = 0.11758755785232791
WORLD_LENGTH = .122
EA = 2.3818718058345718
DENSITY_AREA = 1100.0 * math.pi * (0.000020 ** 2)
H = .001
T = np.float32

def setup(n, case, dtype):
    typ = dtype
    rest = typ(REST_TOTAL / (n - 1))
    p = np.linspace(0, WORLD_LENGTH, n, dtype=typ)
    if case in ('moving', 'asymmetric'):
        p[n//2] = typ(p[n//2] - typ(.001))
    if case == 'moving':
        p[n//2] = typ(p[n//2] + typ(typ(H) * typ(.01)))
    target = np.array([0, WORLD_LENGTH + (.00005 if case == 'moving_endpoint' else 0)], dtype=typ)
    edge_mass = typ(DENSITY_AREA * (REST_TOTAL / (n - 1)))
    mass = np.full(n, edge_mass, dtype=typ)
    mass[[0, -1]] *= typ(.5)
    return rest, p, target, mass

def candidate(n, case):
    rest, p, target, mass = setup(n,case,T)
    k = T(T(EA) / rest)
    stiffness_scale = T(T(H)*T(H)*k)
    strain = np.asarray(p[1:] - p[:-1] - rest, dtype=T)
    # -B.T strain on the free nodes, with a Dirichlet displacement at each end.
    rhs = np.asarray(strain[1:] - strain[:-1], dtype=T)
    delta = np.zeros(n, dtype=T)
    delta[0] = T(target[0]-p[0]); delta[-1] = T(target[1]-p[-1])
    rhs[0] = T(rhs[0] + delta[0]); rhs[-1] = T(rhs[-1] + delta[-1])
    diag = np.asarray(T(2) + mass[1:-1]/stiffness_scale, dtype=T)
    modified_rhs = rhs.copy(); modified_diag = diag.copy()
    # Thomas elimination, tridiagonal off-diagonals both -1.
    for i in range(1,n-2):
        factor = T(-T(1)/modified_diag[i-1])
        modified_diag[i] = T(modified_diag[i] + factor)
        modified_rhs[i] = T(modified_rhs[i] - T(factor*modified_rhs[i-1]))
    delta[-2] = T(modified_rhs[-1]/modified_diag[-1])
    for i in range(n-4,-1,-1):
        delta[i+1] = T(T(modified_rhs[i]+delta[i+2])/modified_diag[i])
    x = np.asarray(p + delta, dtype=T)
    # Constrained positions are directly owned by the attachment targets.
    x[[0,-1]] = target
    tension_left = T(k*T(T(x[1]-x[0])-rest))
    tension_right = T(k*T(T(x[-1]-x[-2])-rest))
    forces = np.array([T(tension_left - T(mass[0]*T(x[0]-p[0])/T(H*H))),
                       T(-tension_right - T(mass[-1]*T(x[-1]-p[-1])/T(H*H)))],dtype=T)
    # Linear residual for the assembled scaled displacement system.
    d = delta[1:-1]
    r = np.asarray(diag*d - rhs, dtype=T)
    if len(d)>1:
        r[1:] = np.asarray(r[1:] - d[:-1], dtype=T)
        r[:-1] = np.asarray(r[:-1] - d[1:], dtype=T)
    return x, forces, float(np.max(np.abs(r)))

def reference(n, case):
    rest, p, target, mass = setup(n,case,np.float64)
    b = np.zeros((n-1,n),dtype=np.float64)
    idx=np.arange(n-1); b[idx,idx]=-1; b[idx,idx+1]=1
    k=EA/rest
    matrix=k*(b.T@b)+np.diag(mass/H**2)
    rhs=mass/H**2*p+k*(b.T@np.full(n-1,rest))
    x=np.empty(n); x[[0,-1]]=target
    x[1:-1]=np.linalg.solve(matrix[1:-1,1:-1],rhs[1:-1]-matrix[1:-1][:,[0,-1]]@target)
    gradient=(mass/H**2)*(x-p)+k*(b.T@(b@x-rest))
    return x,-gradient[[0,-1]]

spec=[(n,case) for n in (3,5,9,17,33,65,127) for case in ('centered','moving')]
spec.extend([(9,'asymmetric'),(9,'moving_endpoint')])
rows=[]
for n,case in spec:
    x,forces,residual=candidate(n,case)
    expected,expected_forces=reference(n,case)
    pos_error=float(np.max(np.abs(x.astype(np.float64)-expected)))
    force_error=np.abs(forces.astype(np.float64)-expected_forces)
    limits=np.maximum(1e-6,.01*np.abs(expected_forces))
    rows.append(dict(nodes=n,case=case,position_error_m=pos_error,
        reactions_N=list(map(float,forces)),reference_reactions_N=list(map(float,expected_forces)),
        reaction_error_N=list(map(float,force_error)),reaction_limit_N=list(map(float,limits)),
        scaled_system_residual_m=residual,
        passed=bool(pos_error <= 1e-6 and np.all(force_error <= limits))))
print(json.dumps(dict(cases=rows,all_pass=all(r['passed'] for r in rows)),indent=2,sort_keys=True))
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
