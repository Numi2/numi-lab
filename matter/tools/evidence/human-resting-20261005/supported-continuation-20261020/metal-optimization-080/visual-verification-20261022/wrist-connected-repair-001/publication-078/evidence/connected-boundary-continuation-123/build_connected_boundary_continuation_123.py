"""Global barrier continuation for the connected O/I/D boundary target.
Static reference construction only; no runtime forces, locks or physics solver.
"""
from pathlib import Path
import sys,hashlib,json,importlib.util,time
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 P=D/'connected-boundary-continuation-123/run-001';assert not P.exists();P.mkdir(parents=True)
 C=D/'connected-boundary-refinement-114/run-002-source-normal-all-146/surface-proposal.npz';assert sha(C)=='fdcb88d99dd488afb59b5f49ad3b6f5975f37fc46479f74dd8d31e24a0e5d5b8'
 z=np.load(C);B=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz';b=np.load(B)
 F=z['surface_faces_zero_based'];sid=z['surface_face_surface_ids'];selected=z['selected_outer_edges'];mid=z['selected_edge_midpoint_rows'];eps=z['selected_edge_layer_endpoint_rows'];N=len(z['reference_points_m'])
 # Components use source-face edge adjacency, not independently moved vertices.
 SF=b['source_O_faces'];em={}
 for i,f in enumerate(SF):
  for a,c in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):em.setdefault(tuple(sorted((int(a),int(c)))),[]).append(i)
 touched=set()
 for e in selected:
  inc=em[tuple(sorted(map(int,e)))];assert len(inc)==2;touched.update(inc)
 parent={i:i for i in touched}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 for inc in em.values():
  active=[i for i in inc if i in touched]
  if len(active)==2:parent[find(active[1])]=find(active[0])
 roots=sorted(set(find(i)for i in touched));labels={r:i for i,r in enumerate(roots)}
 edge_comp=np.asarray([labels[find(em[tuple(sorted(map(int,e)))][0])]for e in selected],np.int64)
 node_comp=np.full(N,-1,np.int64)
 for e,rows in enumerate(mid):node_comp[rows]=edge_comp[e]
 face_comps=[]
 for f in F:
  comps=set(node_comp[f].tolist())-{-1};assert len(comps)<=1;face_comps.append(next(iter(comps))if comps else-1)
 face_comps=np.asarray(face_comps);alpha=np.ones(len(roots))
 linear={};target={}
 for frame,key in [('reference','reference_points_m'),('current037','current037_points_m_f32')]:
  target[frame]=z[key].astype(float);L=target[frame].copy()
  for i in range(len(selected)):
   for j in range(3):L[mid[i,j]]=.5*(L[eps[i,j,0]]+L[eps[i,j,1]])
  linear[frame]=L
 sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'));pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
 spec=importlib.util.spec_from_file_location('numilab_human._continuation123',pred);s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
 logs=[];t0=time.monotonic();passed=False
 for it in range(10):
  bad=set();frames={};Xs={}
  for frame in ('reference','current037'):
   L=linear[frame];X=L.copy();rr=np.flatnonzero(node_comp>=0);X[rr]+=alpha[node_comp[rr],None]*(target[frame][rr]-L[rr]);Xs[frame]=X
   V=X.astype(np.float32)[F].astype(float);U=L.astype(np.float32)[F].astype(float)
   cn=np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]);on=np.cross(U[:,1]-U[:,0],U[:,2]-U[:,0]);den=np.linalg.norm(on,axis=1);cur=np.linalg.norm(cn,axis=1)
   assert np.all(den>0)
   area=cur/den;cos=np.einsum('ij,ij->i',cn,on)/np.maximum(cur*den,np.finfo(float).tiny)
   fail=np.flatnonzero((area<.2)|(cos<.2)|~np.isfinite(area)|~np.isfinite(cos))
   for f in fail:
    assert face_comps[f]>=0;bad.add(int(face_comps[f]))
   rec,_=s.float32_surface_records(X.astype(np.float32),F,source_seam_groups=());a=s._audit_self_pairs_touching_faces(rec,set(range(len(F))));pairs=np.asarray(a['triangle_pairs'],np.int64).reshape(-1,2)
   for pair in pairs:
    cs=set(face_comps[pair].tolist())-{-1};assert cs,('baseline intersection has no refinement component',pair);bad.update(cs)
   np.save(P/f'{it:02d}-{frame}-self-pairs.npy',pairs)
   frames[frame]={'self_pairs':len(pairs),'quality_faces':len(fail),'minimum_area_ratio':float(area.min()),'minimum_cosine':float(cos.min()),'complete_surface_audit':a['complete_surface_audit'],'invalid_face_rows':fail.tolist()}
  logs.append({'iteration':it,'alpha':alpha.tolist(),'bad_components':sorted(bad),'frames':frames})
  (P/'continuation-log.json').write_text(json.dumps(logs,indent=2)+'\n')
  if not bad:passed=True;break
  alpha[list(bad)]*=.5
 out={k:z[k]for k in z.files};out['reference_points_m']=Xs['reference'];out['current037_points_m_f32']=Xs['current037'].astype(np.float32)
 out.update(edge_connected_component=edge_comp,component_alpha=alpha,node_connected_component=node_comp)
 np.savez_compressed(P/'surface-proposal.npz',**out)
 report={'input_sha256':sha(C),'builder_sha256':sha(Path(__file__)),'candidate_sha256':sha(P/'surface-proposal.npz'),'connected_components':len(roots),'component_alpha':alpha.tolist(),'linear_surface_geometry_preserved_at_alpha_zero':True,'iteration_count':len(logs),'surface_and_local_orientation_gate_pass':passed,'final':logs[-1],'wall_seconds':time.monotonic()-t0,'accepted_physical_steps':0,'geometry_admitted':False,'limitations':'Whole connected barrier continuation toward curved detailed-source midpoint targets. Preserves old source vertices and shared O/I/D subdivision; not full exact-source patch reconstruction, bone/solid certification, mass/ancestry transfer or native qualification. Explicit local area/cosine construction thresholds 0.2; original physical quality threshold unchanged.'}
 (P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
