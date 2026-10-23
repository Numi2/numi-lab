from pathlib import Path
import importlib.util,json,subprocess,re,hashlib,time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,eye,csr_matrix
r=Path('/Users/n/numi-human-aortic-contact-opt-20261019');e=r/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
p=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
out=e/'middle-pip-singular-strain-repair-080';out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('m',r/'tools/rebind_human_skin_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=m.decode((p/'restored-pre020-weights-current-reference-001.nhskin').read_bytes())
a0,b0=m.reference_transforms(base,json.loads((p/'neutral.json').read_text()),[0,0,.35]);ap,bp=m.reference_transforms(base,json.loads((p/'right-middle-pip-53.json').read_text()),[0,0,.35])
lanes={int(b['body']):i for i,b in enumerate(base['binds'])};prox,mid=lanes[64],lanes[65]
seeds=[2011,2016,2017,2019,2023,2024,2025,2026,2027,2030,2031,2032,2033,2034,2035,2036,2040,2413]
adj=[set() for _ in range(base['nv'])]
for a,b,c in base['inds']:
 for i,j in [(int(a),int(b)),(int(b),int(c)),(int(c),int(a))]:adj[i].add(j);adj[j].add(i)
distance={i:0 for i in seeds};front=set(seeds)
for level in [1,2,3]:
 front={j for i in front for j in adj[i] if j not in distance}
 distance.update((j,level) for j in front)
rows=np.array(sorted(distance));n=len(rows);lookup={int(v):i for i,v in enumerate(rows)}
faces=base['inds'];ids=np.flatnonzero(np.isin(faces,rows).any(axis=1));tris=faces[ids]
edges=np.unique(np.sort(np.concatenate([tris[:,[0,1]],tris[:,[1,2]],tris[:,[2,0]]]),axis=1),axis=0)
x0=m.positions(base,a0,b0);ybase=m.positions(base,ap,bp)
e0=np.stack([x0[tris[:,1]]-x0[tris[:,0]],x0[tris[:,2]]-x0[tris[:,0]]],axis=2);invref=np.linalg.inv(np.linalg.qr(e0,mode='reduced')[1])
probe=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-002/build/probe');export=p/'export-pose-lbs-csv';fixture=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-001/candidate-fixture-001.txt');template=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/native-native-root-pose-003/registered-skin.csv')
wit=json.loads((p/'posed-surface-audit-001/right-middle-pip-witness-detail.json').read_text())
# Face rows in detailed witness correspond to unchanged NHSKIN face order.
pairs=[]
for rec in wit['pairs']:
 keys=[]
 for side in ['a','b']:
  vertices=tuple(sorted(rec[side]['payload_vertices']))
  keys.append(vertices)
 pairs.append(tuple(keys))
face_lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(faces)}
pairs=[(face_lookup[a],face_lookup[b]) for a,b in pairs]
current=m.decode((e/"middle-pip-constrained-weight-repair-079/iteration-1/candidate.nhskin").read_bytes());history=[]
original_pairs=pairs.copy()
for iteration in range(4):
 d=out/('iteration-'+str(iteration));d.mkdir()
 w=current['weights'].astype(float);y=m.positions(current,ap,bp);x=current['verts'][:,:3].astype(float)
 aweighted=np.einsum('nk,kij->nij',w[rows],a0);pweighted=np.einsum('nk,kij->nij',w[rows],ap)
 dx=-np.linalg.solve(aweighted,(((a0[mid]-a0[prox])@x[rows].T).T+(b0[mid]-b0[prox]))[...,None])[...,0]
 derivative=((ap[mid]-ap[prox])@x[rows].T).T+(bp[mid]-bp[prox])+np.einsum('nij,nj->ni',pweighted,dx)
 # Constraints are in millimetres to avoid tiny metre-scale LP rows.
 rr=[];cc=[];vv=[];rhs=[];kind=[]
 def add(coeff,bound,tag):
  k=len(rhs);rhs.append(bound*1000);kind.append(tag)
  for j,z in coeff.items():
   if abs(z)>1e-14:rr.append(k);cc.append(j);vv.append(z*1000)
 def vec_coeff(v,axis):
  j=lookup.get(int(v));return {} if j is None else {j:float(derivative[j]@axis)}
 def combine(a,b):
  z=dict(a)
  for k,v in b.items():z[k]=z.get(k,0)+v
  return z
 for fa,fb in pairs:
  ta,tb=faces[fa],faces[fb];pa,pb=y[ta],y[tb]
  ea=[pa[1]-pa[0],pa[2]-pa[1],pa[0]-pa[2]];eb=[pb[1]-pb[0],pb[2]-pb[1],pb[0]-pb[2]]
  na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
  axes=[na,nb]+[np.cross(a,b) for a in ea for b in eb]
  axes += [np.cross(na,a) for a in ea]+[np.cross(nb,b) for b in eb]
  choices=[]
  for axis in axes:
   ln=np.linalg.norm(axis)
   if ln<1e-14:continue
   axis=axis/ln;aa=pa@axis;bb=pb@axis
   for ax,depth in [(axis,aa.max()-bb.min()),(-axis,bb.max()-aa.min())]:
    # Already separated axes remain eligible to preserve clearance.
    sens=max([abs(float(derivative[lookup[int(v)]]@ax)) for v in np.r_[ta,tb] if int(v) in lookup]+[1e-12])
    choices.append(((depth+2e-5)/sens,ax))
  if not choices:continue
  axis=min(choices,key=lambda z:z[0])[1]
  for va in ta:
   for vb in tb:
    coeff=combine(vec_coeff(va,axis),vec_coeff(vb,-axis))
    add(coeff,-2e-5-float((y[va]-y[vb])@axis),'collision')
 # Hard surface metric barriers relative to the original restored posed patch.
 for ia,ib in edges:
  edge=ybase[ib]-ybase[ia];length=np.linalg.norm(edge)
  if length<1e-10:continue
  axis=edge/length;coeff=combine(vec_coeff(ib,axis),vec_coeff(ia,-axis));now=float((y[ib]-y[ia])@axis)
  add(coeff,1.20*length-now,'strain');add({j:-z for j,z in coeff.items()},now-.80*length,'strain')
 for ta in tris:
  pa=y[ta];original=ybase[ta]
  normal=np.cross(original[1]-original[0],original[2]-original[0]);norm=np.linalg.norm(normal)
  if norm<1e-14:continue
  normal/=norm;area0=norm
  e1,e2=pa[1]-pa[0],pa[2]-pa[0];area=float(np.cross(e1,e2)@normal)
  coeff={}
  for corner,v in enumerate(ta):
   if int(v) not in lookup:continue
   j=lookup[int(v)];dv=derivative[j]
   de1=dv if corner==1 else -dv if corner==0 else np.zeros(3)
   de2=dv if corner==2 else -dv if corner==0 else np.zeros(3)
   coeff[j]=float((np.cross(de1,e2)+np.cross(e1,de2))@normal)
  # area has m² units; normalize into a 1 mm equivalent length.
  coeff={j:z/area0*.001 for j,z in coeff.items()};now=area/area0*.001
  add(coeff,.00135-now,'strain');add({j:-z for j,z in coeff.items()},now-.00065,'strain')
 # Explicit singular-stretch cuts, since edge and area bounds alone allow shear collapse.
 for ti,ta in enumerate(tris):
  pa=y[ta]; ep=np.stack([pa[1]-pa[0],pa[2]-pa[0]],axis=1)
  U,S,Vh=np.linalg.svd(ep@invref[ti],full_matrices=False)
  for k,target,lower in [(1,.38,True),(0,2.35,False)]:
   if lower and S[k]>.55:continue
   coeff={}
   for corner,v in enumerate(ta):
    if int(v) not in lookup:continue
    j=lookup[int(v)];dv=derivative[j]
    dep=np.zeros((3,2))
    if corner==0:dep[:,0]=-dv;dep[:,1]=-dv
    elif corner==1:dep[:,0]=dv
    else:dep[:,1]=dv
    coeff[j]=float(U[:,k]@(dep@invref[ti])@Vh[k])*.001
   if lower:add({j:-z for j,z in coeff.items()},(S[k]-target)*.001,'strain')
   else:add(coeff,(target-S[k])*.001,'strain')
 a=coo_matrix((vv,(rr,cc)),shape=(len(rhs),n)).tocsr()
 nc=sum(k=='collision' for k in kind)
 # Slack only in collision cuts; finite-strain restrictions cannot be traded away.
 slack=coo_matrix(([-1.]*nc,(list(range(nc)),list(range(nc)))),shape=(len(rhs),nc)).tocsr()
 z=csr_matrix((len(rhs),n))
 A=hstack([a,z,slack],format='csr')
 I=eye(n,format='csr');tail=csr_matrix((n,nc))
 A=vstack([A,hstack([I,-I,tail]),hstack([-I,-I,tail])],format='csr')
 cost=np.r_[np.zeros(n),np.array([1+distance[int(v)] for v in rows]),np.full(nc,1e5)]
 bounds=[]
 for v in rows:
  cap=[.20,.15,.08,.03][distance[int(v)]]
  bounds.append((-min(cap,w[v,mid]),min(cap,w[v,prox])))
 bounds +=[(0,max(abs(lo),abs(hi))) for lo,hi in bounds]+[(0,None)]*nc
 sol=linprog(cost,A_ub=A,b_ub=np.r_[rhs,np.zeros(2*n)],bounds=bounds,method='highs',options={'time_limit':45})
 rec={'iteration':iteration,'variables':n,'collision_constraints':nc,'total_constraints':len(rhs),'solver_success':bool(sol.success),'solver_message':sol.message}
 if not sol.success:
  (d/'result.json').write_text(json.dumps(rec,indent=2)+'\n');history.append(rec);print(rec,flush=True);break
 weights=current['weights'].copy()
 weights[rows,mid]=(w[rows,mid]+sol.x[:n]).astype('<f4');weights[rows,prox]=(w[rows,prox]-sol.x[:n]).astype('<f4')
 blob,inv=m.rebind(current,weights,a0,b0);candidate=m.decode(blob);asset=d/'candidate.nhskin';asset.write_bytes(blob)
 Y=m.positions(candidate,ap,bp);ee=np.stack([Y[tris[:,1]]-Y[tris[:,0]],Y[tris[:,2]]-Y[tris[:,0]]],axis=2)
 stretch=np.linalg.svd(ee@invref,compute_uv=False);area=np.linalg.norm(np.cross(ee[:,:,0],ee[:,:,1]),axis=1)/np.linalg.norm(np.cross(e0[:,:,0],e0[:,:,1]),axis=1)
 csvp=d/'posed.csv';z=subprocess.run([str(export),str(asset),str(p/'right-middle-pip-53.json'),str(template),str(csvp)],capture_output=True,text=True);assert z.returncode==0,z.stderr
 z=subprocess.run([str(probe),str(asset),str(fixture),str(csvp)],capture_output=True,text=True);match=re.search(r'source_crossing_count=(\d+)',z.stdout)
 rawpairs=re.findall(r'crossing,(\d+),(\d+)',z.stderr);pairs=[(int(a),int(b)) for a,b in rawpairs]
 rec.update(candidate_sha256=hashlib.sha256(blob).hexdigest(),inverse=inv,minimum_stretch=float(stretch[:,1].min()),maximum_stretch=float(stretch[:,0].max()),minimum_area_ratio=float(area.min()),maximum_collision_slack_mm=float(sol.x[2*n:].max()) if nc else 0,probe_returncode=z.returncode,crossings=int(match.group(1)) if match else None,probe_stdout=z.stdout,probe_stderr=z.stderr)
 rec['surface_gate']=bool(rec['minimum_stretch']>=.35 and rec['maximum_stretch']<=2.4 and rec['minimum_area_ratio']>=.25)
 (d/'result.json').write_text(json.dumps(rec,indent=2)+'\n');history.append(rec);csvp.unlink()
 print({k:rec[k] for k in ['iteration','crossings','minimum_stretch','maximum_stretch','minimum_area_ratio','maximum_collision_slack_mm','surface_gate']},flush=True)
 if rec['crossings']==0 and rec['surface_gate']:break
 if not pairs:pairs=original_pairs.copy()
 current=candidate
(out/'report.json').write_text(json.dumps({'scope':'Bounded minimal source-weight correction, neutral-invariant. Not physical admission. Strain barriers and exact intersections are both mandatory.', 'results':history},indent=2)+'\n')
