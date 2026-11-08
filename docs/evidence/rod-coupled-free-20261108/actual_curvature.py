import ast,json,subprocess,sys
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
native=sys.argv[1]
src=ast.parse((p/'free_3d_globalized.py').read_text())
ns=dict(np=np,subprocess=subprocess,native=native,L=.11758755785232791,R=2e-5,RHO=1100.)
for name in ('invoke','line','gradient','native_rows','mass'):
    f=next(x for x in src.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'reference', 'exec'),ns)
native_rows=ns['native_rows'];gradient=ns['gradient'];mass=ns['mass']
cases=json.loads((p/'free-3d-global-1-inputs.json').read_text())
results=json.loads((p/'free-3d-global-1-results.json').read_text())
out=[]
for c,r in zip(cases,results):
    n=c['n'];M=3*n;N=4*n-1;h=c['h'];a=r['actual']
    q=np.array(a['q']);t=np.array(a['theta'])
    m,I=mass(c);scale=1/np.sqrt(np.r_[np.repeat(m/h**2,3),I/h**2])
    mats=[]
    for fraction in (1.,.5):
        states=[]
        for j in range(N):
            eps=fraction*(1e-6 if j<M else 1e-4)
            for sign in (1.,-1.):
                qq=q.copy();tt=t.copy()
                if j<M:qq[j//3,j%3]+=sign*eps
                else:tt[j-M]+=sign*eps
                states.append((qq,tt))
        rows=native_rows(c,states); J=np.empty((N,N))
        for j in range(N):
            eps=fraction*(1e-6 if j<M else 1e-4)
            J[:,j]=(gradient(rows[2*j],n)-gradient(rows[2*j+1],n))/(2*eps)
        J=(J+J.T)*.5;J+=np.diag(1/(scale*scale))
        mats.append(scale[:,None]*J*scale[None,:])
    eig=[float(np.min(np.linalg.eigvalsh(x))) for x in mats]
    out.append(dict(seed=c['seed'],eigenvalues=eig,max_matrix_difference=float(np.max(np.abs(mats[0]-mats[1]))),passed=bool(min(eig)>0)))
print(json.dumps(out,indent=2,allow_nan=False))
sys.exit(0 if len(out)==9 and all(x['passed'] for x in out) else 1)
