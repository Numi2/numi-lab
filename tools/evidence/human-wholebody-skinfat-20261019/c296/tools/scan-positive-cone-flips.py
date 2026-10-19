import csv,subprocess,numpy as np
from scipy.optimize import linprog
base='/Users/n/numi-human-separation-20261018/C/wholebody-skin-layer-fitting-296/mechanical-edge-collapse-009/mechanical-capitate-fairing-capped-020'
t=open(base+'/candidate.txt').read().split();nv,nf=map(int,t[:2]);sv=np.asarray([int(x) for x in t[2:2+nv]]);k=2+nv;F=np.asarray([int(x) for x in t[k:k+3*nf]],dtype=np.int32).reshape(nf,3);k+=3*nf;par=np.asarray([int(x) for x in t[k:k+nf]])
rows={}
with open(base+'/registered-skin-mechanical.csv') as f:
 r=csv.reader(f);next(r)
 for a in r:rows[int(float(a[0]))]=np.asarray(list(map(float,a)))
P=np.stack([rows[int(x)][2:5] for x in sv]);D=np.asarray([int(rows[int(x)][1]) for x in sv])
auditor='/Users/n/numi-human-separation-20261018/C/wholebody-skin-layer-fitting-296/persistent-source-audit'
def audit(Fc):
 proc=subprocess.run([auditor],input=f'{nv} {len(Fc)}\n'+'\n'.join(' '.join(f'{x:.17g}' for x in p) for p in P)+'\n'+'\n'.join(' '.join(map(str,f)) for f in Fc)+'\n',text=True,capture_output=True)
 return proc.stdout.strip(),proc.returncode
def cone(Fc,v):
 raw=np.cross(P[Fc[:,1]]-P[Fc[:,0]],P[Fc[:,2]]-P[Fc[:,0]])
 lengths=np.linalg.norm(raw,axis=1);N=raw/lengths[:,None]
 inc=N[np.any(Fc==v,axis=1)]
 r=linprog(np.zeros(3),A_ub=-inc,b_ub=-np.ones(len(inc)),bounds=[(None,None)]*3,method='highs')
 return bool(r.success),len(inc),r.x if r.success else None
v=int(np.where(sv==27019)[0][0]);neighbors=sorted(set(int(x) for f in F[np.any(F==v,axis=1)] for x in f if x!=v))
print('target',v,'source',sv[v],'neighbors',[(u,int(sv[u]),float(np.linalg.norm(P[u]-P[v]))) for u in neighbors])
for u in neighbors:
 edgefaces=np.where(np.sum(np.isin(F,[v,u]),axis=1)==2)[0]
 if len(edgefaces)!=2:continue
 f1=F[edgefaces[0]].tolist();f2=F[edgefaces[1]].tolist()
 if D[v]!=D[u]:continue
 # orient faces so first has v->u and second has u->v
 def rotate_edge(f,a,b):
  for i in range(3):
   if f[i]==a and f[(i+1)%3]==b:return f[i:]+f[:i]
  return None
 a1=rotate_edge(f1,v,u);a2=rotate_edge(f2,u,v)
 if a1 is None and a2 is not None:a1=rotate_edge(f2,v,u);a2=rotate_edge(f1,u,v);edgefaces=edgefaces[::-1]
 if a1 is None or a2 is None:continue
 a=a1[2];b=a2[2]
 if a==b or any(len(set(map(int,f)))<3 for f in ((a,v,b),(a,b,u))):continue
 # reject duplicate edge a-b elsewhere
 if any(np.sum(np.isin(f,[a,b]))==2 for i,f in enumerate(F) if i not in edgefaces):continue
 Fc=F.copy();parc=par.copy();Fc[edgefaces[0]]=[a,v,b];Fc[edgefaces[1]]=[a,b,u]
 # local star orientation/check geometry
 auditResult,rc=audit(Fc);okv,nvfaces,x=cone(Fc,v);oku,nufaces,x2=cone(Fc,u)
 print('flip edge',u,int(sv[u]),'faces',edgefaces.tolist(),'new',Fc[edgefaces].tolist(),'parents',par[edgefaces].tolist(),'audit',auditResult,'cone target',okv,nvfaces,'other',oku,nufaces)
