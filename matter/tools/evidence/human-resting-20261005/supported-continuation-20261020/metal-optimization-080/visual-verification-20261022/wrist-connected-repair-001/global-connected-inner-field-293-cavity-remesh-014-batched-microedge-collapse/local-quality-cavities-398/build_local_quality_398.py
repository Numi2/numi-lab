#!/usr/bin/env python3
"""Bounded local quality retriangulation of the frozen positive395 background."""
from pathlib import Path
import numpy as np, json, hashlib, ast, collections, subprocess, time, sys
D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse')
OUT=D/'local-quality-cavities-398'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
IN=D/'p1-interior-face-support-395/refined-background-checkpoint.npz'
assert sha(IN)=='8d48aeae163c30826f16a78f0cf231244bb932103b0e4b74315515ec57ff4536'
# Reuse the retained local cavity and overlap owners verbatim, not their driver.
src=D/'build_014-replay.py'
tree=ast.parse(src.read_text());wanted={'dets','quality','volume','face_rows','edge_hist','read_nodes','read_ele','barycentric','tet_intersection_volume'}
ns=dict(np=np,collections=collections)
from scipy.spatial import ConvexHull,QhullError,cKDTree
ns.update(ConvexHull=ConvexHull,QhullError=QhullError)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[]),str(src),'exec'),ns)
globals().update({k:ns[k] for k in wanted})
z=np.load(IN,allow_pickle=False);X=z['points_world_m'].astype(np.float32);T=z['tetrahedra'].astype(np.int64)
print('keys',z.files,flush=True)
A=z['child_region_attributes'].astype(np.int64)
q=quality(X,T);bad=np.flatnonzero(q<1e-5)
assert len(bad)==25 and np.all(dets(X,T)>0)
seednodes=np.unique(T[bad]);sel=set(map(int,np.flatnonzero(np.isin(T,seednodes).any(axis=1))))
rings=[]
for iteration in range(4):
 bf,_=face_rows(X,T,sel);problems=[e for e,n in edge_hist(bf).items() if n!=2]
 if not problems:break
 for a,b in problems:sel.update(map(int,np.flatnonzero((T==a).any(1)&(T==b).any(1))))
 rings.append({'closure_iteration':iteration+1,'problem_edges':len(problems),'selected':len(sel)})
else:raise RuntimeError('cavity boundary closure did not converge')
bf,_=face_rows(X,T,sel)
if any(n!=2 for n in edge_hist(bf).values()):raise RuntimeError('cavity boundary notclosed')
if len(sel)>2500:raise RuntimeError('bounded cavity budget exceeded')
rows=np.array(sorted(sel),np.int64);LT=T[rows];nodes=np.unique(LT)
boundarynodes=np.unique(np.asarray(bf).ravel())
# Only boundary points constrain the retriangulation. Removed interior volume
# coordinates retain explicit old-row ancestry; the immutable input stays intact.
localnodes=boundarynodes
assert len(set(X[localnodes].tobytes()[i:i+12] for i in range(0,len(localnodes)*12,12)))==len(localnodes)
lid={int(g):i+1 for i,g in enumerate(localnodes)}
# Connected components of the selected volume by shared faces, for region seeds.
adj=collections.defaultdict(list)
for ti in rows:
 for k in range(4):adj[tuple(sorted(np.delete(T[ti],k)))].append(int(ti))
neighbors=collections.defaultdict(set)
for ids in adj.values():
 if len(ids)==2:neighbors[ids[0]].add(ids[1]);neighbors[ids[1]].add(ids[0])
unseen=set(sel);components=[]
while unseen:
 start=min(unseen);stack=[start];unseen.remove(start);comp=[]
 while stack:
  a=stack.pop();comp.append(a)
  for b in neighbors[a]:
   if b in unseen:unseen.remove(b);stack.append(b)
 components.append(sorted(comp))
TETGEN=D.parent/'arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen'
# Fixed facets, conservative local volume controls, no boundary Steiner points.
maxvol=max(volume(X,LT)/max(500,len(LT)*2),1e-12)
poly=OUT/'cavities.poly'
with poly.open('w') as f:
 f.write(f'{len(localnodes)} 3 0 1\n')
 for i,g in enumerate(localnodes,1):f.write(f'{i} '+' '.join(format(float(v),'.17g') for v in X[g])+' 1\n')
 f.write(f'{len(bf)} 1\n')
 for tri in bf:f.write('1 0\n3 '+' '.join(str(lid[int(v)]) for v in tri)+'\n')
 f.write(f'0\n{len(components)}\n')
 for k,c in enumerate(components,1):
  seed=X[T[c[0]]].astype(float).mean(0)
  assert len(set(map(int,A[c])))==1
  f.write(f'{k} '+' '.join(format(float(v),'.17g') for v in seed)+f' {int(A[c[0]])} {maxvol:.17g}\n')
report={'input':str(IN),'input_sha256':sha(IN),'owner014_sha256':sha(src),'tetgen_sha256':sha(TETGEN),'bad_rows':bad.tolist(),'selected_rows':rows.tolist(),'selected_tets':len(rows),'boundary_faces':len(bf),'components':[len(c) for c in components],'boundary_node_count':len(boundarynodes),'removed_interior_coordinate_ids':sorted(set(map(int,nodes))-set(map(int,boundarynodes))),'closure':rings,'volume_before_m3':volume(X,LT),'native_admission':False}
(OUT/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['selected_rows','removed_interior_coordinate_ids']}),flush=True)
if '--run' not in sys.argv:raise SystemExit(0)
cmd=[str(TETGEN),f'-pYq1.2/10a{maxvol:.17g}AS5000',str(poly)]
t=time.monotonic()
try:p=subprocess.run(cmd,cwd=OUT,capture_output=True,text=True,timeout=45)
except subprocess.TimeoutExpired as e:
 (OUT/'failure.json').write_text(json.dumps({'failure':'tetgen45stimeout','native_admission':False}));raise
(OUT/'tetgen.stdout.txt').write_text(p.stdout);(OUT/'tetgen.stderr.txt').write_text(p.stderr)
report['command']=cmd;report['elapsed_s']=time.monotonic()-t;report['tetgen_exit']=p.returncode
if p.returncode!=0:
 report['failure']='tetgenfailed';(OUT/'failure.json').write_text(json.dumps(report,indent=2));raise SystemExit(1)
local=read_nodes(OUT/'cavities.1.node');tet,attr=read_ele(OUT/'cavities.1.ele')
order=sorted(local);idrow={v:i for i,v in enumerate(order)}
LX=np.asarray([local[i] for i in order],np.float32);TT=np.asarray([[idrow[int(v)] for v in f] for f in tet],np.int64)
neg=np.flatnonzero(dets(LX,TT)<0);TT[neg,0],TT[neg,1]=TT[neg,1].copy(),TT[neg,0].copy()
lookup={X[int(g)].tobytes():int(g) for g in localnodes}
mapping=[];extra=[]
for pnt in LX:
 key=pnt.tobytes()
 if key in lookup:mapping.append(lookup[key])
 else:mapping.append(len(X)+len(extra));extra.append(pnt)
mapping=np.asarray(mapping,np.int64);NX=np.vstack((X,np.asarray(extra,np.float32).reshape(-1,3)))
NTlocal=mapping[TT];outbf,_=face_rows(LX,TT,set(range(len(TT))))
expected={tuple(sorted(map(int,f))) for f in bf}
actual={tuple(sorted(map(int,mapping[list(f)]))) for f in outbf}
report['local_boundary_exact']=expected==actual
report['local_volume_delta_m3']=volume(LX,TT)-volume(X,LT)
keep=np.ones(len(T),bool);keep[rows]=False
NT=np.vstack((T[keep],NTlocal));NA=np.r_[A[keep],attr]
nq=quality(NX,NT);nd=dets(NX,NT)
report['result']={'nodes':len(NX),'tets':len(NT),'nonpositive':int((nd<=0).sum()),'min_quality':float(nq.min()),'below1e_5':int((nq<1e-5).sum()),'oldnodes_exact':np.array_equal(X,NX[:len(X)]),'global_volume_delta_m3':volume(NX,NT)-volume(X,T)}
passed=report['local_boundary_exact'] and np.all(nd>0) and np.min(nq)>=1e-5 and abs(report['local_volume_delta_m3'])<=max(volume(X,LT)*2e-6,1e-14)
report['geometry_stage_pass']=bool(passed);report['field_ancestry_mass_pending']=True
np.savez_compressed(OUT/'candidate-geometry-only.npz',points_world_m=NX,tetrahedra=NT,region_attributes=NA,replaced_395_tet_rows=rows,new_local_tets=NTlocal,unchanged_395_tet_rows=np.flatnonzero(keep),removed_local_interior_coordinate_ids=np.asarray(report['removed_interior_coordinate_ids'],np.int64),physical_outer_faces=z['physical_outer_faces'],physical_shell_faces=z['physical_shell_faces'],construction_closure_faces=z['construction_closure_faces'])
report['candidate_sha256']=sha(OUT/'candidate-geometry-only.npz')
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['result']),flush=True)
if not passed:raise SystemExit(2)
