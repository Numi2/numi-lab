#!/usr/bin/env python3
"""Initialization-only joint O/I/D skin/fat boundary and volume simplification.
Uses retained Float32 geometry and production exact intersection predicates.
Outputs existing typed surface and volume proposal NPZs, never physical admission.
"""
from pathlib import Path
import argparse,ctypes,hashlib,itertools,json,time,collections,shutil,csv
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
def compose_local_cell_mass(old_mass,old_material,new_material,new_volume,density_bounds):
 """Conserve each cavity material mass without touching other cells.
 This is construction eligibility, not source-flow or native admission.
 Roundoff matches densityBoundRoundoff in the existing C++ mass owner.
 """
 old_mass=np.asarray(old_mass,dtype=np.float64)
 old_material=np.asarray(old_material)
 new_material=np.asarray(new_material)
 new_volume=np.asarray(new_volume,dtype=np.float64)
 if (old_mass.ndim!=1 or old_material.shape!=old_mass.shape or
     new_volume.ndim!=1 or new_material.shape!=new_volume.shape or
     not len(old_mass) or not len(new_volume) or
     not np.all(np.isfinite(old_mass)&(old_mass>0)) or
     not np.all(np.isfinite(new_volume)&(new_volume>0))):
  raise ValueError("invalid local mass or volume domain")
 if set(map(int,old_material))!=set(map(int,new_material)):
  raise ValueError("cavity loses a material")
 result=np.empty(len(new_volume),dtype=np.float64)
 for material in np.unique(old_material):
  rows=np.flatnonzero(new_material==material)
  lower,upper=map(float,density_bounds[int(material)])
  if not (np.isfinite(lower) and np.isfinite(upper) and 0<lower<=upper):
   raise ValueError("invalid material density bounds")
  total=float(np.sum(old_mass[old_material==material],dtype=np.float64))
  volumes=new_volume[rows];volume=float(np.sum(volumes,dtype=np.float64))
  density=total/volume
  tolerance=64*np.finfo(np.float64).eps*max(1.,abs(lower),abs(upper),abs(density))
  if not np.isfinite(density) or density<lower-tolerance or density>upper+tolerance:
   raise ValueError("local material mass cannot fit unchanged density bounds")
  portions=(volumes/volume)*total
  # Put the final rounding residual on the largest cell. Assigning it to an
  # arbitrarily tiny last cell can move its density outside the original bound.
  pivot=int(np.argmax(volumes))
  portions[pivot]=total-float(np.sum(np.delete(portions,pivot),dtype=np.float64))
  actual=portions/volumes
  tol=64*np.finfo(np.float64).eps*np.maximum.reduce([np.ones(len(rows)),np.full(len(rows),abs(lower)),np.full(len(rows),abs(upper)),np.abs(actual)])
  if not np.all(np.isfinite(portions)&(portions>0)&(actual>=lower-tol)&(actual<=upper+tol)):
   raise ValueError("rounded local cell mass violates unchanged density bounds")
  result[rows]=portions
 return result

def load_reference_cell_mass(path,expected_sha,count):
 if not expected_sha or sha(path)!=expected_sha:
  raise ValueError("reference flow SHA mismatch")
 # This is the existing flow CSV: child_row,source_parent_row,mass_kg,...
 with Path(path).open() as stream:
  if stream.readline().strip().split(",")[:3]!=["child_row","source_parent_row","mass_kg"]:
   raise ValueError("reference flow column identities differ")
 flow=np.loadtxt(path,delimiter=",",skiprows=1,usecols=(0,2),ndmin=2)
 rows=flow[:,0].astype(np.int64)
 if (not np.all(np.isfinite(flow)) or not np.array_equal(rows.astype(float),flow[:,0]) or
     np.any(rows<0) or np.any(rows>=count) or np.any(flow[:,1]<=0)):
  raise ValueError("invalid reference flow")
 mass=np.bincount(rows,weights=flow[:,1],minlength=count)
 if len(mass)!=count or not np.all(np.isfinite(mass)&(mass>0)):
  raise ValueError("reference flow does not cover every original cell")
 return mass

def build_triangle_bins(points,faces):
 """Conservative centroid/radius index with size-local query radii.
 Bins partition triangle rows; exact predicates remain the acceptance authority.
 """
 valid=~np.any(np.diff(np.sort(faces,axis=1),axis=1)==0,axis=1)
 rows=np.flatnonzero(valid)
 if not len(rows):return []
 xyz=points[faces[rows]];centers=xyz.mean(axis=1)
 radii=np.linalg.norm(xyz-centers[:,None,:],axis=2).max(axis=1)
 exponent=np.frexp(radii)[1]
 return [(cKDTree(centers[exponent==e]),float(radii[exponent==e].max()),
          rows[exponent==e]) for e in np.unique(exponent)]

def query_triangle_bins(bins,triangle,padding=0.):
 if not np.isfinite(padding) or padding<0:raise ValueError("invalid motion padding")
 center=triangle.mean(axis=0);radius=float(np.linalg.norm(triangle-center,axis=1).max())
 chunks=[]
 for tree,maximum,rows in bins:
  bound=radius+maximum+padding
  bound=np.nextafter(bound+32*np.finfo(float).eps*max(1.,bound),np.inf)
  ids=tree.query_ball_point(center,bound)
  if ids:chunks.append(rows[np.asarray(ids,np.intp)])
 return np.concatenate(chunks) if chunks else np.empty(0,np.int64)

def quotient_closure(quotient):
 """Resolve repeated contractions to final live vertices, rejecting cycles."""
 q=np.asarray(quotient,dtype=np.int64).copy()
 if q.ndim!=1 or np.any(q<0) or np.any(q>=len(q)):raise ValueError("invalid quotient domain")
 for _ in range(max(1,len(q).bit_length()+1)):
  nxt=q[q]
  if np.array_equal(nxt,q):
   if not np.all(np.asarray(quotient)[q]==q):raise ValueError("cyclic vertex quotient")
   return q
  q=nxt
 raise ValueError("cyclic vertex quotient")

def compose_face_ancestry(operations,role):
 parents={}
 for op in operations:
  if role not in op["source_face_rows"]:continue
  rows=op["source_face_rows"][role]
  union=sorted(set().union(*(parents.get(int(f),{int(f)}) for f in rows)))
  for f in rows:parents[int(f)]=union
 return parents

def load_bone_barriers():
 """Read the existing370 pinned bone banks without Python triangle records."""
 root=H.parent
 native=root/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
 manifest=root/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
 native_sha="e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5"
 manifest_sha="0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c"
 if sha(native)!=native_sha or sha(manifest)!=manifest_sha:raise ValueError("bone barrier pins changed")
 with np.load(native,allow_pickle=False) as z:
  if len(z["bank_body"])!=185:raise ValueError("incomplete native bone inventory")
  points=[z["points_world_f32"].astype(np.float64)];faces=[z["faces"].astype(np.int64)]
 offset=len(points[0]);operands=json.loads(manifest.read_text())["operands"]
 if len(operands)!=185 or len({(int(b["body"]),int(b["bone"])) for b in operands})!=185:
  raise ValueError("incomplete pinned Float64 bone inventory")
 for bank in operands:
  path=Path(bank["operand_path"])
  if sha(path)!=bank["operand_sha256"]:raise ValueError("Float64 bone pin changed")
  with np.load(path,allow_pickle=False) as z:
   pp=z["points_world_m_f64"];ff=z["faces_f64"].astype(np.int64)
   if not np.all(np.isfinite(pp)) or ff.min()<0 or ff.max()>=len(pp):raise ValueError("invalid bone geometry")
   points.append(pp);faces.append(ff+offset);offset+=len(pp)
 points=np.ascontiguousarray(np.concatenate(points),np.float64);faces=np.concatenate(faces)
 if len(faces)!=938272:raise ValueError("bone triangle census differs")
 return points,faces,{"native_path":str(native),"native_sha256":native_sha,
  "manifest_path":str(manifest),"manifest_sha256":manifest_sha,
  "coordinate_domains":370,"triangles":len(faces),"exact_intersection_predicate_unchanged":True}

def attachment_nodes(path,expected_sha,mapping):
 if sha(path)!=expected_sha:raise ValueError("attachment request SHA mismatch")
 result=set();identities=set()
 with Path(path).open() as stream:
  reader=csv.DictReader(stream)
  if set(reader.fieldnames or [])!={"stable_attachment_id","donor_body","stable_source_vertex","source_domain"}:
   raise ValueError("attachment request columns differ")
  for row in reader:
   identity=int(row["stable_attachment_id"]);vertex=int(row["stable_source_vertex"])
   if identity in identities or row["source_domain"]!="D37_coarse_parent" or not 0<=vertex<len(mapping) or mapping[vertex]<0:
    raise ValueError("invalid attachment identity or source domain")
   identities.add(identity);result.add(int(mapping[vertex]))
 if not identities:raise ValueError("empty attachment requests")
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=180);ap.add_argument("--max-operations",type=int,default=2048);ap.add_argument("--maximum-edge-m",type=float,default=.0001);ap.add_argument("--replay-report",type=Path);ap.add_argument("--exclude-operation-rows",type=Path);ap.add_argument("--reference-flow-csv",type=Path);ap.add_argument("--reference-flow-sha256");ap.add_argument("--passes",type=int,default=1);ap.add_argument("--bone-barriers",action="store_true");ap.add_argument("--attachment-requests-csv",type=Path);ap.add_argument("--attachment-requests-sha256");a=ap.parse_args()
 if bool(a.attachment_requests_csv)!=bool(a.attachment_requests_sha256):raise ValueError("attachment path/SHA must be paired")
 if a.passes<1 or a.seconds<=0 or a.max_operations<0 or not np.isfinite(a.maximum_edge_m) or a.maximum_edge_m<=0:raise ValueError("invalid construction budget")
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
  cell_mass=None
  density_bounds={15:(900.,1100.),16:(850.,950.)}
  if bool(a.reference_flow_csv)!=bool(a.reference_flow_sha256):
   raise ValueError("reference flow path and SHA must be provided together")
  if a.reference_flow_csv:
   original_mass=load_reference_cell_mass(a.reference_flow_csv,a.reference_flow_sha256,len(ot))
   new_mass=np.empty(len(new),dtype=np.float64)
   patch_offsets=z["replacement_tet_offsets"]
   new_volume=measure(p,new)[0]/6
   # Match the existing fixed-cavity owner: only repartition a patch's own mass.
   for first,last in zip(patch_offsets[:-1],patch_offsets[1:]):
    parents=pr[po[first]:po[first+1]]
    # Reconstruct the already retained fixed-cavity baseline's own masses.
    # Its exact geometry/density/source-flow certification remains the native
    # fixed-cavity owner's authority. New quotient operations are screened below.
    assert np.all(om[parents]==om[parents[0]])
    total=float(original_mass[parents].sum())
    portions=new_volume[first:last]/new_volume[first:last].sum()*total
    portions[-1]=total-float(portions[:-1].sum())
    assert np.all(np.isfinite(portions)&(portions>0))
    new_mass[first:last]=portions
   cell_mass=np.r_[original_mass[kept],new_mass]
   input_cell_mass=cell_mass.copy()
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
 bone_bins=[];bone_faces=None;bone_receipt=None
 if a.bone_barriers:
  bp,bf,bone_receipt=load_bone_barriers()
  bone_bins=build_triangle_bins(bp,bf)
  bone_faces=bf+len(p)
  positions=[np.ascontiguousarray(np.concatenate([pp,bp]),np.float64) for pp in positions]
 spatial=[]
 motion_padding=[]
 allfaces=np.concatenate(list(faces.values()));offsets=np.r_[0,np.cumsum([len(faces[r]) for r in S])]
 lib=ctypes.CDLL(str(W/"connected-fold-repair-62/exact-pair.dylib"))
 U=ctypes.POINTER(ctypes.c_uint32);D=ctypes.POINTER(ctypes.c_double)
 lib.beyond.argtypes=[D,U,U];lib.beyond.restype=ctypes.c_uint
 protected=set()
 if a.attachment_requests_csv:protected.update(attachment_nodes(a.attachment_requests_csv,a.attachment_requests_sha256,maps["D"]))
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
 total_volume=det.sum()/6;changed_volume=0.
 def crossing(changes,deleted):
  excluded=set(deleted);excluded.update(changes)
  for pp,bins,padding in zip(positions,spatial,motion_padding):
   for row,f in changes.items():
    xyz=pp[f]
    cand=query_triangle_bins(bins,xyz,padding)
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
    if bone_faces is not None:
     candidates=query_triangle_bins(bone_bins,xyz)
     cf=bone_faces[candidates];xx=pp[cf]
     near=np.all(xx.max(1)>=xyz.min(0),axis=1)&np.all(xx.min(1)<=xyz.max(0),axis=1)
     for g in cf[near]:
      aa=np.asarray(f,np.uint32);bb=np.asarray(g,np.uint32)
      if lib.beyond(pp.ctypes.data_as(D),aa.ctypes.data_as(U),bb.ctypes.data_as(U)):return False
  return True
 replay_passes={}
 if a.replay_report:
  excluded=set(json.loads(a.exclude_operation_rows.read_text())["failed_operation_rows"]) if a.exclude_operation_rows else set()
  for i,op in enumerate(json.loads(a.replay_report.read_text())["operations"]):
   if i not in excluded:replay_passes.setdefault(int(op.get("pass_index",0)),[]).append(op)
 passes=[]
 for pass_index in range(a.passes):
  if time.monotonic()-start>a.seconds or len(ops)>=a.max_operations:break
  pass_start=time.monotonic();before_ops=len(ops)
  used=set()
  active_cells=np.flatnonzero(alive)
  localcells=incidence(t[active_cells],len(p))
  getcells=lambda v:active_cells[localcells(v)]
  getfaces={}
  for r in S:
   active_faces=np.flatnonzero(fa[r]);localfaces=incidence(faces[r][active_faces],len(p))
   getfaces[r]=(lambda v,rows=active_faces,lookup=localfaces:rows[lookup(v)])
  spatial=[build_triangle_bins(pp,allfaces) for pp in positions]
  edges=np.unique(np.sort(np.concatenate([f[:,e] for f in [faces[r][fa[r]] for r in S] for e in [[0,1],[1,2],[2,0]]]),axis=1),axis=0)
  length=np.linalg.norm(p[edges[:,0]].astype(float)-p[edges[:,1]].astype(float),axis=1);edges=edges[length<=a.maximum_edge_m];length=length[length<=a.maximum_edge_m]
  hits=np.bincount(t[alive&(q<.001)].ravel(),minlength=len(p))
  known_fold=np.all(edges==[22677,22679],axis=1)
  rank=np.lexsort((edges[:,1],edges[:,0],length,-(hits[edges[:,0]]+hits[edges[:,1]]),~known_fold));edges=edges[rank]
  replay_directions={}
  if pass_index in replay_passes:
   selected=replay_passes[pass_index]
   edges=np.asarray([[op["drop"],op["keep"]] for op in selected],np.int64).reshape(-1,2)
   replay_directions={tuple(map(int,e)):True for e in edges}
  motion_padding=[float(np.linalg.norm(pp[edges[:,0]]-pp[edges[:,1]],axis=1).max()) if len(edges) else 0. for pp in positions]
  for u,v in edges:
   if time.monotonic()-start>a.seconds or len(ops)>=a.max_operations:break
   u=int(u);v=int(v)
   rows=np.union1d(getcells(u),getcells(v));nodes=set(map(int,t[rows].ravel()))
   if nodes&used or u in protected or v in protected:reject["protected_or_dependent"]+=1;continue
   if not len(rows):reject["inactive_replay_edge"]+=1;continue
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
    proposed_mass=None
    if cell_mass is not None:
     try:
      proposed_mass=compose_local_cell_mass(cell_mass[rows],labs[rows,0],
       labs[rows[live],0],nd/6,density_bounds)
     except ValueError:
      reject["local_mass_density_infeasible"]+=1;continue
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
    if not crossing(changes,deleted):reject["exact_surface_or_bone_crossing" if a.bone_barriers else "exact_surface_crossing"]+=1;continue
    # Retained source cell ancestry is a union, not a nearest-cell guess.
    parents=np.unique(np.concatenate([original_parents[i] for i in rows]))
    for i in rows[live]:original_parents[int(i)]=parents[om[parents]==labs[i,0]]
    t[rows]=n;alive[rows[~live]]=False;q[rows[live]]=nq;det[rows[live]]=nd
    if proposed_mass is not None:cell_mass[rows[live]]=proposed_mass
    for r,(fr,nf,fl) in affected.items():faces[r][fr]=nf;fa[r][fr[~fl]]=False
    for i,f in changes.items():allfaces[i]=f
    for i in deleted:allfaces[i]=[0,0,0]
    used.update(nodes);changed_volume+=abs(delta)
    ops.append(dict(pass_index=pass_index,drop=drop,keep=keep,source_cell_rows=rows.tolist(),removed_cells=int((~live).sum()),roles=list(affected),source_face_rows={r:fr.tolist() for r,(fr,_,_) in affected.items()},signed_volume_change_m3=delta,minimum_quality_before=float(oldq.min()),minimum_quality_after=float(nq.min())))
    with (a.output/"operations.jsonl").open("a") as ledger:ledger.write(json.dumps(ops[-1])+"\n")
    if len(ops)%10==0:print(json.dumps(dict(operations=len(ops),elapsed_s=time.monotonic()-start)),flush=True)
    break
  passes.append(dict(pass_index=pass_index,operations=len(ops)-before_ops,
   active_cells=int(alive.sum()),elapsed_s=time.monotonic()-pass_start))
  print(json.dumps({"completed_pass":passes[-1]}),flush=True)
  if len(ops)==before_ops:break
 # Export joint volume with immediate parent unions in the established CSR representation.
 rows=np.flatnonzero(alive);parents=[original_parents[i] for i in rows];off=np.r_[0,np.cumsum([len(x) for x in parents])]
 np.savez_compressed(a.output/"candidate-volume.npz",points_reference_world_m_f64=p.astype(float),volume_tetrahedra=t[rows],material_index=labs[rows,0],geometric_donor_body=labs[rows,1],anatomical_region_id=labs[rows,2],replacement_parent_offsets=off,replacement_parent_cell_rows=np.concatenate(parents),source_volume_sha256=np.array(sha(V)))
 if cell_mass is not None:
  # Same established child-mass CSV columns; never a replacement for source-flow admission.
  np.savetxt(a.output/"local-child-mass-proposal.csv",
   np.column_stack((np.arange(len(rows)),cell_mass[rows])),delimiter=",",
   header="child_row,mass_kg",comments="",fmt=["%d","%.17g"])
 # Typed surface emission and face-union composition happen only after accepted operations.
 for r in S:
  old=raw[r];keepf=np.flatnonzero(fa[r]);reverse=np.full(len(p),-1,np.int64);active=maps[r]>=0;reverse[maps[r][active]]=np.flatnonzero(active)
  nf=reverse[faces[r][keepf]];assert np.all(nf>=0)
  payload={k:old[k][keepf] for k in ["face_source_layer","face_source_body","face_source_bone","face_source_face_id","face_source_original_id","face_source_identity0","face_source_identity1"] if k in old}
  payload.update(points_world_m=old["points_world_m"],points_world_m_f32=old["points_world_m_f32"],faces=nf,source_input_face_rows=keepf,typed_parent_candidate_path=np.array(str(S[r])),typed_parent_candidate_sha256=np.array(sha(S[r])))
  parent_for_face=compose_face_ancestry(ops,r)
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
  payload["source_input_vertex_quotient"]=quotient_closure(quotient)
  np.savez_compressed(a.output/f"candidate-{r}-typed-current.npz",**payload)
 report=dict(status="JOINT_BOUNDARY_VOLUME_PROPOSAL_NOT_PHYSICAL_ADMISSION",physical_admission=False,native_steps=0,mass_mapping="PENDING",input_pins={str(x):sha(x) for x in [V,C,*S.values(),a.output/"executed-runner.py",*([a.replay_report] if a.replay_report else []),*([a.exclude_operation_rows] if a.exclude_operation_rows else [])]},operations=ops,passes=passes,bone_barrier=bone_receipt,protected_attachment_requests=(dict(path=str(a.attachment_requests_csv),sha256=a.attachment_requests_sha256) if a.attachment_requests_csv else None),rejections=dict(reject),input_cells=len(t),output_cells=len(rows),input_nodes=len(p),active_output_nodes=len(np.unique(t[rows])),low_quality_cells_before=int((measure(p,np.concatenate([ot[kept],new]))[1]<.001).sum()),low_quality_cells_after=int((q[rows]<.001).sum()),native_quality_failures=int(((det[rows]/6<=1e-18)|(q[rows]<np.float32(1e-5))).sum()),all_point_coordinates_unchanged=True,absolute_geometric_volume_change_m3=changed_volume,elapsed_s=time.monotonic()-start,remaining=["complete anatomical context including bones and containment","independent topology, F32, typed ancestry and volume checks","canonical parent and conservative source mass compiler","complete native scene comparison"])
 if cell_mass is not None:
  touched=np.unique(np.concatenate([np.asarray(o["source_cell_rows"],np.int64) for o in ops])) if ops else np.empty(0,np.int64)
  unchanged=np.ones(len(t),bool);unchanged[touched]=False
  assert np.array_equal(cell_mass[unchanged],input_cell_mass[unchanged])
  report["local_mass_construction"]={"reference_flow_path":str(a.reference_flow_csv),
   "reference_flow_sha256":a.reference_flow_sha256,"density_bounds_kg_m3":density_bounds,
   "input_mass_kg":float(input_cell_mass.sum()),"output_mass_kg":float(cell_mass[rows].sum()),
   "unchanged_cells":int(unchanged.sum()),"unchanged_cell_masses_exact":True,
   "source_flow_and_donor_composition":"PENDING_EXISTING_CONSERVATIVE_COMPILER",
   "proposal_path":str(a.output/"local-child-mass-proposal.csv")}
 (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="operations"}),flush=True)
if __name__=="__main__":main()
