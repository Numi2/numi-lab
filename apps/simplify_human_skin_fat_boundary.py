#!/usr/bin/env python3
"""Initialization-only joint O/I/D skin/fat boundary and volume simplification.
Uses retained Float32 geometry and production exact intersection predicates.
Outputs existing typed surface and volume proposal NPZs, never physical admission.
"""
from pathlib import Path
import argparse,ctypes,hashlib,itertools,json,time,collections,shutil
import numpy as np
from scipy.spatial import cKDTree
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
V=H/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz"
C=H/"regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz"
S={"O":H/"all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz","I":H/"all-quality-1009-point-update-1018-root/actual-001/candidate-I-point-updated.npz","D":H/"all-quality-958-D014-root/proposal-003/candidate-D-typed-current.npz"}
W=Path("/Users/numi/Work/human-integrated-gaps-20261104")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def measure(p,t):
 x=p[t].astype(float);e=x[:,1:]-x[:,:1]
 det=np.einsum("ij,ij->i",np.cross(e[:,0],e[:,1]),e[:,2])
 mx=np.stack([np.sum((x[:,i]-x[:,j])**2,axis=1) for i,j in itertools.combinations(range(4),2)]).max(0)**.5
 return det,det/mx**3
def link(cells,vs):
 vs=set(vs);out=set()
 for cell in cells:
  if not vs.issubset(cell):continue
  rest=sorted(set(map(int,cell))-vs)
  for k in range(len(rest)+1):out.update(itertools.combinations(rest,k))
 return out
def incidence(rows,n):
 order=np.argsort(rows.ravel(),kind="stable");off=np.r_[0,np.cumsum(np.bincount(rows.ravel(),minlength=n))]
 return lambda v:np.unique(order[off[v]:off[v+1]]//rows.shape[1])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=180);ap.add_argument("--max-operations",type=int,default=2048);ap.add_argument("--maximum-edge-m",type=float,default=.0001);ap.add_argument("--replay-report",type=Path);ap.add_argument("--exclude-operation-rows",type=Path);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.output/"executed-runner.py");start=time.monotonic()
 with np.load(V) as z:
  p=z["points_reference_world_m_f64"].astype(np.float32);ot=z["volume_tetrahedra"];om=z["material_index"]
  rawlabs=np.column_stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]])
  maps={r:z["current_typed_role_source_row_to_joint_node_"+r] for r in S}
 with np.load(C) as z:
  assert str(z["source_volume_sha256"].item())==sha(V)
  kept=z["retained_input_cell_rows"];new=z["replacement_tetrahedra"];po=z["replacement_parent_offsets"];pr=z["replacement_parent_cell_rows"]
  t=np.concatenate([ot[kept],new]);labs=np.concatenate([rawlabs[kept],rawlabs[pr[po[:-1]]]])
  original_parents=[np.array([i],np.int64) for i in kept]+[pr[po[i]:po[i+1]] for i in range(len(new))]
  assert all(np.all(rawlabs[pr[po[i]:po[i+1]]]==labs[len(kept)+i]) for i in range(len(new)))
 det,q=measure(p,t);assert np.all((det/6>1e-18)&(q>=np.float32(1e-5)))
 getcells=incidence(t,len(p));alive=np.ones(len(t),bool);used=set();reject=collections.Counter();ops=[]
 raw={r:dict(np.load(path,allow_pickle=False)) for r,path in S.items()}
 faces={r:maps[r][raw[r]["faces"]] for r in S};assert all(np.all(f>=0) for f in faces.values())
 getfaces={r:incidence(f,len(p)) for r,f in faces.items()};fa={r:np.ones(len(f),bool) for r,f in faces.items()}
 sourcep=p.copy()
 for r in S:
  active=maps[r]>=0;sourcep[maps[r][active]]=raw[r]["points_world_m_f32"][active]
 positions=[np.ascontiguousarray(x,dtype=np.float64) for x in [p,sourcep]]
 spatial=[]
 for pp in positions:
  xyz=np.concatenate([pp[faces[r]] for r in S]);ctr=xyz.mean(1);rad=np.linalg.norm(xyz-ctr[:,None,:],axis=2).max(1)
  spatial.append((cKDTree(ctr),float(rad.max())))
 allfaces=np.concatenate(list(faces.values()));offsets=np.r_[0,np.cumsum([len(faces[r]) for r in S])]
 lib=ctypes.CDLL(str(W/"connected-fold-repair-62/exact-pair.dylib"))
 U=ctypes.POINTER(ctypes.c_uint32);D=ctypes.POINTER(ctypes.c_double)
 lib.beyond.argtypes=[D,U,U];lib.beyond.restype=ctypes.c_uint
 protected=set()
 for r,f in faces.items():
  edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);un,ct=np.unique(edges,axis=0,return_counts=True)
  protected.update(map(int,un[ct!=2].ravel()))
 back=json.loads((H/"fixed-boundary-quality-topology-196/collapse-batch-004-from-003-001/back412-scope-audit-001/report.json").read_text())
 for box in back["back412_world_aabbs"].values():
  lo=np.array(box["min_world_m"]);hi=np.array(box["max_world_m"])
  for pp in positions:
   for f in faces.values():
    x=pp[f];touch=np.all(x.max(1)>=lo,axis=1)&np.all(x.min(1)<=hi,axis=1);protected.update(map(int,f[touch].ravel()))
 routes=json.loads((H/"verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json").read_text())
 for route in routes["surface_routes"]:
  r=route["role"];sp=raw[r]["points_world_m_f32"];rows=np.flatnonzero(np.all(sp==np.array(route["target_world_m_f32"],np.float32),axis=1));protected.update(map(int,maps[r][rows]))
 edges=np.unique(np.sort(np.concatenate([f[:,e] for f in faces.values() for e in [[0,1],[1,2],[2,0]]]),axis=1),axis=0)
 length=np.linalg.norm(p[edges[:,0]].astype(float)-p[edges[:,1]].astype(float),axis=1);edges=edges[length<=a.maximum_edge_m];length=length[length<=a.maximum_edge_m]
 hits=np.bincount(t[q<.001].ravel(),minlength=len(p))
 known_fold=np.all(edges==[22677,22679],axis=1)
 rank=np.lexsort((edges[:,1],edges[:,0],length,-(hits[edges[:,0]]+hits[edges[:,1]]),~known_fold));edges=edges[rank]
 replay_directions={}
 if a.replay_report:
  replay=json.loads(a.replay_report.read_text())["operations"]
  excluded=set(json.loads(a.exclude_operation_rows.read_text())["failed_operation_rows"]) if a.exclude_operation_rows else set()
  selected=[op for i,op in enumerate(replay) if i not in excluded]
  edges=np.asarray([[op["drop"],op["keep"]] for op in selected],np.int64)
  replay_directions={tuple(map(int,e)):True for e in edges}
 total_volume=det.sum()/6;changed_volume=0.
 def crossing(changes,deleted):
  excluded=set(deleted);excluded.update(changes)
  for pp,(tree,maxr) in zip(positions,spatial):
   for row,f in changes.items():
    xyz=pp[f];ctr=xyz.mean(0);rr=np.linalg.norm(xyz-ctr,axis=1).max()
    cand=tree.query_ball_point(ctr,rr+maxr+2*a.maximum_edge_m)
    cand=np.asarray(cand,np.int64)
    cf=allfaces[cand].copy()
    for k,j in enumerate(cand):
     if j in changes:cf[k]=changes[j]
    xx=pp[cf]
    mask=np.all(xx.max(1)>=xyz.min(0),axis=1)&np.all(xx.min(1)<=xyz.max(0),axis=1)
    mask &= ~np.any(np.diff(np.sort(cf,axis=1),axis=1)==0,axis=1)
    for j,g in zip(cand[mask],cf[mask]):
     if j==row or j in deleted:continue
     aa=np.asarray(f,np.uint32);bb=np.asarray(g,np.uint32)
     if lib.beyond(pp.ctypes.data_as(D),aa.ctypes.data_as(U),bb.ctypes.data_as(U)):return False
  return True
 for u,v in edges:
  if time.monotonic()-start>a.seconds or len(ops)>=a.max_operations:break
  u=int(u);v=int(v)
  rows=np.union1d(getcells(u),getcells(v));nodes=set(map(int,t[rows].ravel()))
  if nodes&used or u in protected or v in protected:reject["protected_or_dependent"]+=1;continue
  if len(rows)>256:reject["unbounded_star"]+=1;continue
  if not np.all(labs[rows,1:]==labs[rows[0],1:]):reject["donor_or_region_interface"]+=1;continue
  if link(t[rows],[u])&link(t[rows],[v])!=link(t[rows],[u,v]):reject["volume_link"]+=1;continue
  for drop,keep in ([(u,v)] if replay_directions else [(u,v),(v,u)]):
   n=t[rows].copy();n[n==drop]=keep;live=~np.any(np.diff(np.sort(n,axis=1),axis=1)==0,axis=1)
   nd,nq=measure(p,n[live])
   if not len(nd) or np.any((nd/6<=1e-18)|(nq<np.float32(1e-5))):reject["native_cell_quality"]+=1;continue
   oldq=q[rows];cost=lambda z:float(np.maximum(np.log(.001/z),0).sum())
   good_coarsening=oldq.min()>=.001 and nq.min()>=.001 and int((~live).sum())>0
   conditioning_gain=nq.min()>=oldq.min() and cost(nq)<cost(oldq)-1e-6
   if not (good_coarsening or conditioning_gain):reject["no_conditioning_gain"]+=1;continue
   delta=float((nd.sum()-det[rows].sum())/6)
   if abs(delta)>.01*det[rows].sum()/6 or changed_volume+abs(delta)>1e-4*total_volume:reject["geometric_volume_budget"]+=1;continue
   changes={};deleted=set();affected={}
   valid=True
   for ri,r in enumerate(S):
    fr=getfaces[r](drop)
    if not len(fr):continue
    star=np.union1d(getfaces[r](drop),getfaces[r](keep))
    if link(faces[r][star],[u])&link(faces[r][star],[v])!=link(faces[r][star],[u,v]):valid=False;break
    nf=faces[r][fr].copy();nf[nf==drop]=keep;fl=~np.any(np.diff(np.sort(nf,axis=1),axis=1)==0,axis=1)
    for pp in positions:
     x=pp[faces[r][fr[fl]]];y=pp[nf[fl]];na=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);nb=np.cross(y[:,1]-y[:,0],y[:,2]-y[:,0])
     den=np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1)
     if np.any(den<=0) or np.any(np.einsum("ij,ij->i",na,nb)<.9*den):valid=False;break
    if not valid:break
    changes.update({int(offsets[ri]+i):ff for i,ff in zip(fr[fl],nf[fl])});deleted.update(map(int,offsets[ri]+fr[~fl]));affected[r]=(fr,nf,fl)
   if not valid:reject["surface_link_or_orientation"]+=1;continue
   if not crossing(changes,deleted):reject["exact_surface_crossing"]+=1;continue
   # Retained source cell ancestry is a union, not a nearest-cell guess.
   parents=np.unique(np.concatenate([original_parents[i] for i in rows]))
   for i in rows[live]:original_parents[int(i)]=parents[om[parents]==labs[i,0]]
   t[rows]=n;alive[rows[~live]]=False;q[rows[live]]=nq;det[rows[live]]=nd
   for r,(fr,nf,fl) in affected.items():faces[r][fr]=nf;fa[r][fr[~fl]]=False
   for i,f in changes.items():allfaces[i]=f
   for i in deleted:allfaces[i]=[0,0,0]
   used.update(nodes);changed_volume+=abs(delta)
   ops.append(dict(drop=drop,keep=keep,source_cell_rows=rows.tolist(),removed_cells=int((~live).sum()),roles=list(affected),source_face_rows={r:fr.tolist() for r,(fr,_,_) in affected.items()},signed_volume_change_m3=delta,minimum_quality_before=float(oldq.min()),minimum_quality_after=float(nq.min())))
   with (a.output/"operations.jsonl").open("a") as ledger:ledger.write(json.dumps(ops[-1])+"\n")
   if len(ops)%10==0:print(json.dumps(dict(operations=len(ops),elapsed_s=time.monotonic()-start)),flush=True)
   break
 # Export joint volume with immediate parent unions in the established CSR representation.
 rows=np.flatnonzero(alive);parents=[original_parents[i] for i in rows];off=np.r_[0,np.cumsum([len(x) for x in parents])]
 np.savez_compressed(a.output/"candidate-volume.npz",points_reference_world_m_f64=p.astype(float),volume_tetrahedra=t[rows],material_index=labs[rows,0],geometric_donor_body=labs[rows,1],anatomical_region_id=labs[rows,2],replacement_parent_offsets=off,replacement_parent_cell_rows=np.concatenate(parents),source_volume_sha256=np.array(sha(V)))
 # Typed surface emission and face-union composition happen only after accepted operations.
 for r in S:
  old=raw[r];keepf=np.flatnonzero(fa[r]);reverse=np.full(len(p),-1,np.int64);active=maps[r]>=0;reverse[maps[r][active]]=np.flatnonzero(active)
  nf=reverse[faces[r][keepf]];assert np.all(nf>=0)
  payload={k:old[k][keepf] for k in ["face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1"] if k in old}
  payload.update(points_world_m=old["points_world_m"],points_world_m_f32=old["points_world_m_f32"],faces=nf,source_input_face_rows=keepf,typed_parent_candidate_path=np.array(str(S[r])),typed_parent_candidate_sha256=np.array(sha(S[r])))
  parent_for_face={}
  for op in ops:
   if r in op["source_face_rows"]:
    fr=op["source_face_rows"][r]
    for f in fr:parent_for_face[f]=fr
  for offset,lanes in [("quality_star_patch_union_offsets",["quality_star_patch_union_face_source_"+k for k in ["layer","body","bone","face_id","original_id","identity0","identity1"]]),("face_ancestry_offsets",["face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"])]:
   oo=old[offset];chunks=[]
   for f in keepf:
    ff=parent_for_face.get(int(f),[int(f)]);ix=np.concatenate([np.arange(oo[i],oo[i+1]) for i in ff]);chunks.append(np.unique(np.column_stack([old[k][ix] for k in lanes]),axis=0))
   merged=np.concatenate(chunks);payload[offset]=np.r_[0,np.cumsum([len(x) for x in chunks])]
   for j,k in enumerate(lanes):payload[k]=merged[:,j]
  for k in old:
   if "schema" in k and k not in payload:payload[k]=old[k]
  quotient=np.arange(len(maps[r]))
  for op in ops:
   if r in op["roles"]:quotient[reverse[op["drop"]]]=reverse[op["keep"]]
  payload["source_input_vertex_quotient"]=quotient
  np.savez_compressed(a.output/f"candidate-{r}-typed-current.npz",**payload)
 report=dict(status="JOINT_BOUNDARY_VOLUME_PROPOSAL_NOT_PHYSICAL_ADMISSION",physical_admission=False,native_steps=0,mass_mapping="PENDING",input_pins={str(x):sha(x) for x in [V,C,*S.values(),a.output/"executed-runner.py",*([a.replay_report] if a.replay_report else []),*([a.exclude_operation_rows] if a.exclude_operation_rows else [])]},operations=ops,rejections=dict(reject),input_cells=len(t),output_cells=len(rows),input_nodes=len(p),active_output_nodes=len(np.unique(t[rows])),low_quality_cells_before=int((measure(p,np.concatenate([ot[kept],new]))[1]<.001).sum()),low_quality_cells_after=int((q[rows]<.001).sum()),native_quality_failures=int(((det[rows]/6<=1e-18)|(q[rows]<np.float32(1e-5))).sum()),all_point_coordinates_unchanged=True,absolute_geometric_volume_change_m3=changed_volume,elapsed_s=time.monotonic()-start,remaining=["complete anatomical context including bones and containment","independent topology, F32, typed ancestry and volume checks","canonical parent and conservative source mass compiler","complete native scene comparison"])
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="operations"}),flush=True)
if __name__=="__main__":main()
