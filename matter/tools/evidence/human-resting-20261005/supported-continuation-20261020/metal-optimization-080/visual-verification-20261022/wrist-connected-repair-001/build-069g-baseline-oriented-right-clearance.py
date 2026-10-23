from pathlib import Path
from functools import lru_cache
from collections import defaultdict
import sys,json,hashlib,time,math
import numpy as np
class NumpyJSONEncoder(json.JSONEncoder):
 def default(self,obj):
  if isinstance(obj,np.generic):return obj.item()
  return super().default(obj)
from scipy.sparse import coo_matrix,lil_matrix
from scipy.optimize import least_squares,minimize,linprog
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,"/tmp")
import wrist_sections_20261022 as ws
E=ws.E
BASE=E/"finger-profile-068-whole-hand-minimax"; SRCREF=E/"replacement-hand-reference-038"; PRIOR=E/"finger-profile-065-sourcefit-conservative"
OUT=E/"finger-profile-069g-baseline-oriented-right-clearance"
OUT.mkdir(exist_ok=True)
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
D0=decode_payload((BASE/"replacement-bilateral-hands038.nhskin").read_bytes())
Fglobal=np.asarray(D0["indices"],int).reshape(-1,3)
pose=json.loads(next((E/"native-pose-checks/repair-005/neutral").glob("*.skin-poses.json")).read_text())
H={};exec((E/"wrist-loft-rebuild-019/audit_wrist_bones035.py").read_text().split("report={'schema'")[0],H)
MANIFEST=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
BODY={int(x["core_body_index"]):x["myosim_body"] for x in json.loads(MANIFEST.read_text())["source"]["anchors"]}
def nearest(p,tris):
 a,b,c=tris[:,0],tris[:,1],tris[:,2];ab=b-a;ac=c-a;n=np.cross(ab,ac);n2=np.einsum("ij,ij->i",n,n);ap=p-a;signed=np.einsum("ij,ij->i",ap,n)/np.maximum(n2,1e-30);proj=p-signed[:,None]*n;v=proj-a;d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac);d20=np.einsum("ij,ij->i",v,ab);d21=np.einsum("ij,ij->i",v,ac);den=d00*d11-d01*d01;u=(d11*d20-d01*d21)/np.maximum(den,1e-30);vv=(d00*d21-d01*d20)/np.maximum(den,1e-30);inside=(u>=0)&(vv>=0)&(u+vv<=1)&(n2>1e-24);d2=np.where(inside,np.einsum("ij,ij->i",p-proj,p-proj),np.inf)
 for x,y in ((a,b),(b,c),(c,a)):
  e=y-x;t=np.clip(np.einsum("ij,ij->i",p-x,e)/np.maximum(np.einsum("ij,ij->i",e,e),1e-30),0,1);z=x+t[:,None]*e;ds=np.einsum("ij,ij->i",p-z,p-z);d2=np.minimum(d2,ds)
 return float(np.sqrt(max(float(d2.min()),0)))
def closest_point(p,tris):
 a,b,c=tris[:,0],tris[:,1],tris[:,2];ab=b-a;ac=c-a;n=np.cross(ab,ac);n2=np.einsum("ij,ij->i",n,n);ap=p-a;signed=np.einsum("ij,ij->i",ap,n)/np.maximum(n2,1e-30);proj=p-signed[:,None]*n;v=proj-a;d00=np.einsum("ij,ij->i",ab,ab);d01=np.einsum("ij,ij->i",ab,ac);d11=np.einsum("ij,ij->i",ac,ac);d20=np.einsum("ij,ij->i",v,ab);d21=np.einsum("ij,ij->i",v,ac);den=d00*d11-d01*d01;u=(d11*d20-d01*d21)/np.maximum(den,1e-30);vv=(d00*d21-d01*d20)/np.maximum(den,1e-30);inside=(u>=0)&(vv>=0)&(u+vv<=1)&(n2>1e-24);best=np.where(inside,np.einsum("ij,ij->i",p-proj,p-proj),np.inf);q=proj.copy()
 for x,y in ((a,b),(b,c),(c,a)):
  e=y-x;t=np.clip(np.einsum("ij,ij->i",p-x,e)/np.maximum(np.einsum("ij,ij->i",e,e),1e-30),0,1);z=x+t[:,None]*e;d=np.einsum("ij,ij->i",p-z,p-z);m=d<best;best=np.where(m,d,best);q=np.where(m[:,None],z,q)
 i=int(np.argmin(best));return q[i],float(np.sqrt(max(float(best[i]),0.))),i
def edge_info(V,F):
 N=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);A=.5*np.linalg.norm(N,axis=1);N/=np.maximum(np.linalg.norm(N,axis=1)[:,None],1e-30);edges=defaultdict(list);adj=[set() for _ in range(len(V))]
 for fi,f in enumerate(F):
  for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
   edges[tuple(sorted((a,b)))].append(fi);adj[a].add(b);adj[b].add(a)
 ang={e:float(np.degrees(np.arccos(np.clip(N[fs[0]]@N[fs[1]],-1,1)))) for e,fs in edges.items() if len(fs)==2}
 boundary={v for e,fs in edges.items() if len(fs)==1 for v in e}
 return ang,A,N,edges,adj,boundary
def combine_full(parts):
 c={};f={}
 for (bc,bf),scale in parts:
  for k,v in bc:c[k]=c.get(k,0.)+v*scale
  for k,v in bf:f[k]=f.get(k,0.)+v*scale
 return (tuple(sorted((k,v) for k,v in c.items() if abs(v)>1e-14)),
         tuple(sorted((k,v) for k,v in f.items() if abs(v)>1e-14)))
def make_rowbasis(side,p49,p56,p58):
 a56=json.loads((E/"replacement-hands-056"/(side+"-source-ancestry.json")).read_text())
 a58=json.loads((E/"replacement-wrists-058"/(side+"-source-ancestry.json")).read_text())
 anc=np.load(E/"replacement-mechanical-055"/(side+"-template-control-ancestry049.npz"))
 ptr=anc["template_ancestry_indptr"];ix=anc["template_ancestry_control_index"];vv=anc["template_ancestry_weight"]
 bptr=anc["boundary_ancestry_indptr"];bix=anc["boundary_ancestry_index"];bvv=anc["boundary_ancestry_weight"]
 tids=anc["template_control_ids"];boundary_slots=anc["boundary_source_slots_049"]
 slot49={int(s):i for i,s in enumerate(p49["source_slots"])};maps={}
 for level,path in ((56,E/"replacement-hands-056"),(58,E/"replacement-wrists-058")):
  g=np.load(path/"replacement-geometry.npz")
  if "internal_new_vertex_ids" in g.files:maps[level]=dict(zip(g["internal_new_vertex_ids"].tolist(),g["emitted_source_slots"].tolist()))
  else:
   both={}
   for ss in ("left","right"):both.update(json.loads((path/(ss+"-source-ancestry.json")).read_text()))
   maps[level]={61950+i:s for i,s in enumerate(sorted(map(int,both)))}
 def b49(row):
  lo,hi=ptr[row:row+2];blo,bhi=bptr[row:row+2]
  c=tuple((int(tids[ix[j]]),float(vv[j])) for j in range(lo,hi))
  f=tuple((int(boundary_slots[bix[j]]),float(bvv[j])) for j in range(blo,bhi))
  total=sum(w for _,w in c)+sum(w for _,w in f)
  if abs(total-1.)>1e-8:raise RuntimeError((side,"049 partition failure",int(row),total))
  return c,f
 def b49slot(slot):
  row=slot49.get(int(slot))
  if row is None:raise RuntimeError((side,"049 source slot missing",int(slot)))
  return b49(row)
 def baseline_endpoint(slot):
  row=slot49.get(int(slot))
  if row is not None:return b49(row)
  if int(slot) in set(map(int,boundary_slots)):return ((),((int(slot),1.),))
  if int(slot)>=0:return ((),((int(slot),1.),))
  raise RuntimeError((side,"invalid fixed baseline slot",int(slot)))
 def map_slot(level,internal):
  slot=int(maps[level].get(int(internal),int(internal)));src={56:a56,58:a58}[level]
  if str(slot) not in src:raise RuntimeError((side,"internal ancestry missing",level,int(internal),slot))
  return slot
 @lru_cache(None)
 def b56(slot):
  rec=a56.get(str(int(slot)))
  if rec is None:raise RuntimeError((side,"unknown 056 slot",int(slot)))
  q=rec.get("reference_candidate049_vertex")
  if q is not None:return b49slot(q)
  rc=rec.get("reference_cut")
  if rc:
   x,y=rc["edge"];w=float(rc["weight"])
   return combine_full(((b49(x),1-w),(b49(y),w)))
  bc=rec.get("baseline_cut")
  if bc:
   x,y=bc["edge"];w=float(bc["weight"])
   return combine_full(((baseline_endpoint(x),1-w),(baseline_endpoint(y),w)))
  loft=rec.get("loft_sections")
  if loft:
   x,y=loft;w=float(rec["blend"])
   return combine_full(((b56(map_slot(56,x)),1-w),(b56(map_slot(56,y)),w)))
  f=rec.get("refined_face")
  if f:return combine_full((b56(map_slot(56,x)),1/len(f)) for x in f)
  raise RuntimeError((side,"unclassified 056 ancestry",int(slot),rec))
 @lru_cache(None)
 def b58(slot):
  rec=a58.get(str(int(slot)))
  if rec is None:raise RuntimeError((side,"unknown 058 slot",int(slot)))
  q=rec.get("reference_candidate056_vertex")
  if q is not None:return b56(int(q))
  rc=rec.get("reference_cut")
  if rc:
   x,y=rc["edge"];w=float(rc["weight"])
   return combine_full(((b56(int(p56["source_slots"][x])),1-w),(b56(int(p56["source_slots"][y])),w)))
  bc=rec.get("baseline_cut")
  if bc:
   x,y=bc["edge"];w=float(bc["weight"])
   return combine_full(((baseline_endpoint(x),1-w),(baseline_endpoint(y),w)))
  loft=rec.get("loft_sections")
  if loft:
   x,y=loft;w=float(rec["blend"])
   return combine_full(((b58(map_slot(58,x),),1-w),(b58(map_slot(58,y),),w)))
  f=rec.get("refined_face")
  if f:return combine_full((b58(map_slot(58,x)),1/len(f)) for x in f)
  raise RuntimeError((side,"unclassified 058 ancestry",int(slot),rec))
 full=[b58(int(s)) for s in p58["source_slots"]]
 errs=[];fixed_rows=0;cmass=[]
 for r,(c,f) in enumerate(full):
  total=sum(w for _,w in c)+sum(w for _,w in f)
  errs.append(abs(total-1.));fixed_rows+=bool(f);cmass.append(sum(w for _,w in c))
  if abs(total-1.)>1e-8:raise RuntimeError((side,"full ancestry partition failure",r,total))
 summary={"rows":len(full),"max_partition_error":float(max(errs)),"rows_with_fixed_boundary_basis":int(fixed_rows),"control_mass_minmax":[float(min(cmass)),float(max(cmass))],"unknown_records":0}
 return [x[0] for x in full],full,summary
TARGET={"left":[(1577,1578),(883,884),(882,884)],"right":[(1616,1617),(887,888)]}
MECH_IDS={"left":[9396],"right":[2831,3045,3652]}
CLEARANCE_SAMPLE_DEFS={"left":[{"name":"source-row-6652","face":106972,"slots":[6171,6170,53929],"bary":[1.,0.,0.]},{"name":"source-row-6781","face":106982,"slots":[4436,6170,53932],"bary":[0.,0.,1.]}],"right":[]}
BILATERAL_MAX_SEVERE_BASE_DEG=129.0069364642264
Qneutral0=ws.ns["posed"](D0,pose).astype(np.float64)
results={}
for side in ("right",):
 t0=time.time()
 p0=np.load(BASE/(side+"-patch.npz"));p65=np.load(PRIOR/(side+"-patch.npz"));p49=np.load(E/"replacement-hands-049"/(side+"-patch.npz"));p56=np.load(E/"replacement-hands-056"/(side+"-patch.npz"));p58=np.load(E/"replacement-wrists-058"/(side+"-patch.npz"))
 for key in ("source_slots","faces","weights"):
  if not np.array_equal(p0[key],p65[key]):raise AssertionError((side,key))
 V0=p0["world"].astype(float);F=p0["faces"].astype(int);slots=np.asarray(p0["source_slots"],int);rowbasis,fullbasis,ancestry_summary=make_rowbasis(side,p49,p56,p58)
 ang0,area0,N0,edges0,adj0,boundary=edge_info(V0,F);direct=defaultdict(list)
 for r,b in enumerate(rowbasis):
  if len(b)==1 and abs(b[0][1]-1)<1e-9:direct[int(b[0][0])].append(r)
 pinned=set()
 for r in boundary:
  for t,w in rowbasis[r]:
   if abs(w)>1e-10:pinned.add(int(t))
 slot_to_row={int(s):i for i,s in enumerate(slots)}
 sample_defs=[]
 for sd in CLEARANCE_SAMPLE_DEFS[side]:
  rr=np.asarray([slot_to_row[int(x)] for x in sd["slots"]],int);bb=np.asarray(sd["bary"],float)
  sample_defs.append({"name":sd["name"],"face":int(sd["face"]),"rows":rr,"bary":bb})
 targets=[tuple(map(int,e)) for e in TARGET[side]];witnesses=[]
 hinfo=json.loads((BASE/(side+"-remaining-hinge-anatomy-classified-v3.json")).read_text())
 source_angles={tuple(sorted(h["edge_patch_rows"])):float(h["registered_source049_hinge_deg"]) for h in hinfo["hinges"] if h.get("registered_source049_hinge_deg") is not None}
 targetgoal={tuple(sorted(e)):source_angles[tuple(sorted(e))] for e in targets if tuple(sorted(e)) in source_angles}
 if len(targetgoal)!=len(targets):raise RuntimeError((side,"registered source angle missing",targets,source_angles))
 seed=set()
 for a,b in targets:
  key=tuple(sorted((a,b)))
  if key not in ang0:raise RuntimeError((side,"missing target edge",key))
  seed.update((a,b))
  seed.update(adj0[a]);seed.update(adj0[b])
 ring=set(seed)
 for _ in range(3):
  nxt=set()
  for r in ring:nxt.update(adj0[r])
  ring.update(nxt)
 region_controls={int(t) for r in ring for t,w in rowbasis[int(r)] if abs(w)>1e-12 and int(t) in direct}
 for sd in sample_defs:
  for r,w0 in zip(sd["rows"],sd["bary"]):
   if w0>0:region_controls.update(int(t) for t,w in rowbasis[int(r)] if abs(w)>1e-12 and int(t) in direct)
 mech_controls=set(map(int,MECH_IDS[side]))
 missing_mech=mech_controls-set(direct)
 if missing_mech:raise RuntimeError((side,"mechanical controls absent from ancestry",sorted(missing_mech)))
 pinned.difference_update(mech_controls)
 active=sorted(t for t in direct if (int(t) in region_controls or int(t) in mech_controls) and int(t) not in pinned)
 if not active:raise RuntimeError((side,"no active local direct controls"))
 ctrl_index={t:i for i,t in enumerate(active)}
 # Include every edge incident to any descendant of active controls, not only selected target hinges.
 moved_rows={r for r,b in enumerate(rowbasis) if any(int(t) in ctrl_index for t,w in b if abs(w)>1e-12)}
 local_edges=[e for e,fs in edges0.items() if len(fs)==2 and (e[0] in moved_rows or e[1] in moved_rows)]
 unavailable_targets=[list(e) for e in targets if tuple(sorted(e)) not in set(local_edges)]
 targets=[e for e in targets if tuple(sorted(e)) in set(local_edges)]
 targetgoal={tuple(sorted(e)):source_angles[tuple(sorted(e))] for e in targets}
 target_keys={tuple(sorted(e)) for e in targets}
 if not target_keys.issubset(set(local_edges)):raise RuntimeError((side,"target missing from local edge graph"))
 edge_index={e:i for i,e in enumerate(local_edges)}
 facepairs=np.asarray([edges0[e] for e in local_edges],int);base_ang=np.asarray([ang0[e] for e in local_edges],float);target_idx=np.asarray([edge_index[tuple(sorted(e))] for e in targets],dtype=int);goals=np.asarray([targetgoal[tuple(sorted(e))] for e in targets],float)
 if len(target_idx)!=len(targets):raise RuntimeError((side,"target edge map incomplete"))
 # Sparse ancestry operator: every edited fine row receives its exact full 049 basis-weighted control motion.
 br=[];bc=[];bv=[]
 for r,b in enumerate(rowbasis):
  for t,w in b:
   if int(t) in ctrl_index:
    br.append(r);bc.append(ctrl_index[int(t)]);bv.append(float(w))
 B=coo_matrix((bv,(br,bc)),shape=(len(V0),len(active))).tocsr()
 sample_constraints=[]
 for sd in sample_defs:
  coeff=np.zeros(len(active),float)
  for rr,ww in zip(sd["rows"],sd["bary"]):coeff+=float(ww)*B.getrow(int(rr)).toarray().ravel()
  pbase=np.sum(V0[sd["rows"]]*sd["bary"][:,None],axis=0)
  sample_constraints.append({"name":sd["name"],"face":sd["face"],"rows":sd["rows"],"bary":sd["bary"],"coeff":coeff,"point0":pbase,"required":.00102})
 ancestry=np.load(E/("replacement-mechanical-055/"+side+"-template-control-ancestry049.npz"))
 srcmap={int(t):np.asarray(x,float) for t,x in zip(ancestry["template_control_ids"],ancestry["template_control_world_049"])}
 prior=np.zeros((len(active),3),float);active_rows=[]
 for t,j in ctrl_index.items():
  r=direct[t][0];active_rows.append(r)
  if t in srcmap:prior[j]=srcmap[t]-V0[r]
 bv,bf,bowners=H["bone_mesh"](pose,side);bo=np.asarray(bowners);handmask=np.asarray([any(k in nm for k in ("carpal","metacarp","mc_","thumb","proxph","midph","distph")) for nm in bo]);boneT=bv[bf[handmask]].astype(float);boneT_all=bv[bf].astype(float)
 active_rows=np.asarray(active_rows,int);before_clear=np.asarray([nearest(V0[r],boneT_all) for r in active_rows]);required=np.where(before_clear>=.001,.001,np.maximum(before_clear-.00001,0.))
 low_clearance=np.flatnonzero((before_clear<.001)&np.asarray([int(t) not in mech_controls for t in active]))
 mech_info={}
 for t in sorted(mech_controls):
  j=ctrl_index[t];r=direct[t][0];cp,dist,tri=closest_point(V0[r],boneT_all);direction=(V0[r]-cp)/max(dist,1e-30);need=max(0.,.00103-dist)
  mech_info[t]={"row":int(r),"control_index":int(j),"clearance_before_m":float(dist),"closest_point":cp.tolist(),"outward_unit":direction.tolist(),"nearest_triangle_index":int(tri),"required_m":0.00102}
  prior[j]=direction*need
  required[j]=.00102;before_clear[j]=dist
 prior_mm=prior*1000.
 # Neighbor control pairs for smooth local motion.
 row_to_ctrl={direct[t][0]:t for t in active if t in direct}
 smooth=set()
 for e in edges0:
  u,v=e
  if u in row_to_ctrl and v in row_to_ctrl and row_to_ctrl[u]!=row_to_ctrl[v]:smooth.add(tuple(sorted((ctrl_index[row_to_ctrl[u]],ctrl_index[row_to_ctrl[v]]))))
 smooth=sorted(smooth)
 # Build sparse Jacobian pattern: each hinge sees the ancestry controls of its four incident-face vertices.
 e_ctrl=[]
 for e,fs in zip(local_edges,facepairs):
  verts=set(F[int(fs[0])].tolist()+F[int(fs[1])].tolist());cs=set()
  for r in verts:
   for t,w in rowbasis[int(r)]:
    if int(t) in ctrl_index:cs.add(ctrl_index[int(t)])
  e_ctrl.append(sorted(cs))
 n=len(active);ne=len(local_edges);nt=len(target_idx);nsmooth=len(smooth);nvar=3*n
 nr=3*ne+nt+6*n+3*nsmooth+1
 J=lil_matrix((nr,nvar),dtype=np.int8);off=0
 for i,cs in enumerate(e_ctrl):
  for j in cs:
   J[off+i,3*j:3*j+3]=1;J[off+ne+i,3*j:3*j+3]=1;J[off+2*ne+i,3*j:3*j+3]=1
 off=3*ne
 for k,ei in enumerate(target_idx):
  for j in e_ctrl[int(ei)]:J[off+k,3*j:3*j+3]=1
 off+=nt
 for j in range(n):
  J[off+3*j:off+3*j+3,3*j:3*j+3]=1
 off+=3*n
 for k,(i,j) in enumerate(smooth):
  for r in range(3):J[off+3*k+r,3*i+r]=1;J[off+3*k+r,3*j+r]=1
 off+=3*nsmooth
 # Max/softmax row depends on every active control touching the included edge neighborhood.
 for j in set(c for cs in e_ctrl for c in cs):J[off,3*j:3*j+3]=1
 J=J.tocsr()
 def make_points(xmm):
  return V0 + B.dot(np.asarray(xmm,float).reshape(n,3)*0.001)
 mapping_3176_fd={}
 if side=="right" and 3176 in ctrl_index:
  j3176=ctrl_index[3176];r3176=direct[3176][0];br3176=B.getrow(r3176)
  xp=np.zeros(nvar);xp[3*j3176]=0.01
  fd=(make_points(xp)[r3176]-V0[r3176])/(0.01e-3)
  expected=np.zeros(n);expected[j3176]=1.
  mapping_3176_fd={"row":int(r3176),"row_basis":[[int(t),float(w)] for t,w in rowbasis[r3176]],"B_indices":br3176.indices.tolist(),"B_values":br3176.data.tolist(),"variable_index":int(j3176),"Brow_identity_error":float(np.max(np.abs(br3176.toarray().ravel()-expected))),"finite_difference_delta_per_10um":fd.tolist(),"finite_difference_identity_error":float(np.max(np.abs(fd-np.eye(3)[0])))}
 def calc_angles(V):
  f0=F[facepairs[:,0]];f1=F[facepairs[:,1]]
  a=V[f0[:,0]];b=V[f0[:,1]];c=V[f0[:,2]];d=V[f1[:,0]];e=V[f1[:,1]];f=V[f1[:,2]]
  na=np.cross(b-a,c-a);nb=np.cross(e-d,f-d);na/=np.maximum(np.linalg.norm(na,axis=1)[:,None],1e-30);nb/=np.maximum(np.linalg.norm(nb,axis=1)[:,None],1e-30)
  return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",na,nb),-1,1)))
 def residual(xmm):
  V=make_points(xmm);aa=calc_angles(V);diff=aa-base_ang
  # Jointly reduce all existing severe hinges and prevent any formerly mild edge from becoming severe.
  severe0=base_ang>90.;mild0=~severe0;targetmask=np.zeros(ne,bool);targetmask[target_idx]=True
  coh=.015*diff/10.
  hinge=.55*np.where(severe0 & ~targetmask,np.maximum(0.,aa-base_ang)/10.,0.)
  new=18.0*np.where(mild0,np.maximum(0.,aa-90.)/1.,0.)
  target_res=4.0*(aa[target_idx]-goals)/15.
  x=np.asarray(xmm).reshape(n,3)
  src=.85*(x-prior_mm)/.4
  reg=.10*x/.5
  sm=.10*np.asarray([(x[i]-x[j])/.5 for i,j in smooth]).reshape(-1) if smooth else np.zeros(0)
  global_excess=max(0.,float(np.max(aa))-BILATERAL_MAX_SEVERE_BASE_DEG)
  return np.concatenate((coh,hinge,new,target_res,src.ravel(),reg.ravel(),sm,np.array([10.*global_excess/5.])))
 x0=np.clip(prior_mm,-.9,.9).reshape(-1);lb=np.full(nvar,-1.0);ub=np.full(nvar,1.0)
 for j in low_clearance:lb[3*j:3*j+3]=-1e-10;ub[3*j:3*j+3]=1e-10
 bounds=(lb,ub)
 opt=least_squares(residual,x0,bounds=bounds,jac_sparsity=J,diff_step=1e-3,method="trf",tr_solver="lsmr",max_nfev=16,ftol=3e-3,xtol=3e-3,gtol=3e-3,verbose=0)
 ctrlrows=np.asarray(active_rows,int)
 # Constrained sequential quadratic refinement: every direct-control nearest-bone floor is a moving tangent half-space.
 # Refresh all planes after each solve so changed nearest triangles become new witnesses.
 # Exact tangent-plane barriers for active direct controls and source-face interpolants.
 slsqp_rounds=[];xseed=opt.x.copy();lastsol=None;plane_rows=[];plane_rhs=[];plane_names=[]
 witness_specs=[]
 for j,r in enumerate(active_rows):
  cc=np.zeros(n,float);cc[j]=1.
  witness_specs.append({"name":"control-"+str(active[j]),"coeff":cc,"point0":V0[int(r)].copy(),"required":float(required[j]),"kind":"direct"})
 for sd in sample_constraints:witness_specs.append({"name":sd["name"],"coeff":sd["coeff"],"point0":sd["point0"],"required":float(sd["required"]),"kind":"source_interpolant"})
 def point_for(spec,x):return spec["point0"]+(spec["coeff"]@np.asarray(x).reshape(n,3))*.001
 def add_plane(spec,x,round_id):
  # Keep every tangent plane on the physically valid exterior branch established at the 068 baseline.
  pt=point_for(spec,x);cp,dist,tri=closest_point(pt,boneT_all)
  p0=np.asarray(spec["point0"],float);cp0,d0,tri0=closest_point(p0,boneT_all);n0=(p0-cp0)/max(d0,1e-30)
  ncur=(pt-cp)/max(dist,1e-30);branch_flip=bool(float(np.dot(ncur,n0))<=0.)
  if branch_flip: cp,dist,tri,normal=cp0,d0,tri0,n0
  else: normal=ncur
  row=np.zeros(nvar,float)
  for j,c0 in enumerate(spec["coeff"]):row[3*j:3*j+3]=float(c0)*normal
  rhs0=(spec["required"]-float(np.dot(normal,p0-cp)))*1000.
  plane_rows.append(row);plane_rhs.append(rhs0);plane_names.append({"name":spec["name"],"round":int(round_id),"triangle":int(tri),"distance_mm":float(dist*1000),"baseline_distance_mm":float(d0*1000),"baseline_triangle":int(tri0),"exterior_branch_flip_corrected":branch_flip})
 xzero=np.zeros(nvar,float)
 for spec in witness_specs:add_plane(spec,xzero,-1)
 Cseed=np.asarray(plane_rows,float);rhsseed=np.asarray(plane_rhs,float)
 lp=linprog(np.zeros(nvar),A_ub=-Cseed,b_ub=-rhsseed,bounds=list(zip(lb,ub)),method="highs")
 linear_feasibility={"success":bool(lp.success),"status":int(lp.status),"message":str(lp.message),"plane_count":int(len(plane_rhs)),"initialization":"all tangent planes anchored at valid 068 baseline (x=0); refreshed planes preserve baseline exterior branch","baseline_min_plane_residual_mm":float(np.min(Cseed@xzero-rhsseed)),"baseline_violations":[{"plane":plane_names[i],"residual_mm":float((Cseed@xzero-rhsseed)[i])} for i in np.flatnonzero(Cseed@xzero-rhsseed < -2e-6)]}
 if lp.success:
  xseed=np.asarray(lp.x,float)
  linear_feasibility["min_plane_residual_mm"]=float(np.min(Cseed@xseed-rhsseed))
  linear_feasibility["row3176_plane_residual_mm"]=float((Cseed@xseed-rhsseed)[next(i for i,nm in enumerate(plane_names) if nm["name"]=="control-3176")]) if side=="right" and any(nm["name"]=="control-3176" for nm in plane_names) else None
  np.savez_compressed(OUT/"right-lp-feasible-point.npz",x=lp.x,C=Cseed,rhs=rhsseed)
 else:
  Cp=np.column_stack((Cseed,np.ones(len(rhsseed))))
  bp=linprog(np.r_[np.zeros(nvar),1.],A_ub=-Cp,b_ub=-rhsseed,bounds=list(zip(lb,ub))+[(0,None)],method="highs")
  linear_feasibility.update({"phase1_success":bool(bp.success),"phase1_status":int(bp.status),"phase1_message":str(bp.message)})
  if bp.success:
   viol=np.maximum(rhsseed-Cseed@bp.x[:nvar],0.);order=np.argsort(viol)[::-1]
   linear_feasibility["phase1_minimax_violation_mm"]=float(bp.x[-1])
   linear_feasibility["worst_plane_violations"]=[{"plane":plane_names[int(i)],"rhs_mm":float(rhsseed[i]),"lhs_mm":float(Cseed[i]@bp.x[:nvar]),"violation_mm":float(viol[i])} for i in order[:12]]
   np.savez_compressed(OUT/"right-lp-phase1-point.npz",x=bp.x[:nvar],slack=bp.x[-1],C=Cseed,rhs=rhsseed)
 (OUT/"right-linear-feasibility.json").write_text(json.dumps(linear_feasibility,indent=2,cls=NumpyJSONEncoder)+"\n")
 if not lp.success:
  print("RIGHT_BASELINE_BRANCH_LP_FAIL",json.dumps(linear_feasibility,cls=NumpyJSONEncoder),flush=True)
  raise SystemExit(0)
 for round_i in range(4):
  C=np.asarray(plane_rows,float);rhs=np.asarray(plane_rhs,float)
  def fobj(x):
   rr=residual(x);return .5*float(np.dot(rr,rr))
  cache={"x":None,"r":None,"j":None}
  def fgrad(x):
   if cache["x"] is None or not np.array_equal(x,cache["x"]):
    rr=residual(x);jr=approx_derivative(residual,x,method="2-point",sparsity=J)
    cache["x"]=np.array(x,copy=True);cache["r"]=rr;cache["j"]=jr
   return np.asarray(cache["j"].T@cache["r"]).reshape(-1)
  cons={"type":"ineq","fun":lambda x,C=C,rhs=rhs:C@x-rhs,"jac":lambda x,C=C:C}
  sol=minimize(fobj,xseed,method="SLSQP",jac=fgrad,bounds=list(zip(lb,ub)),constraints=[cons],options={"maxiter":20,"ftol":2e-4,"disp":False})
  lastsol=sol;xseed=sol.x.copy()
  dtest=np.asarray([closest_point(point_for(z,xseed),boneT_all)[1] for z in witness_specs])
  bad=np.flatnonzero(dtest<np.asarray([z["required"] for z in witness_specs])-2e-6);residual_mm=C@xseed-rhs
  slsqp_rounds.append({"round":round_i,"success":bool(sol.success),"message":str(sol.message),"nfev":int(sol.nfev),"objective":float(sol.fun),"planes_retained":len(plane_rows),"min_plane_residual_mm":float(np.min(residual_mm)) if len(residual_mm) else None,"violating_witnesses":[{"name":witness_specs[i]["name"],"clearance_mm":float(dtest[i]*1000),"required_mm":float(witness_specs[i]["required"]*1000)} for i in bad]})
  if len(bad)==0:break
  for i in bad:add_plane(witness_specs[int(i)],xseed,round_i)
 mapping_3176={}
 if side=="right" and 3176 in ctrl_index:
  j=ctrl_index[3176];r=direct[3176][0];br=B.getrow(r)
  mapping_3176={"row":int(r),"basis":[[int(t),float(w)] for t,w in rowbasis[r]],"B_indices":br.indices.tolist(),"B_values":br.data.tolist(),"variable_index":int(j),"plane_residual_mm_min":float(np.min(np.asarray(plane_rows)@xseed-np.asarray(plane_rhs))) if plane_rows else None}
 opt.x=xseed;opt.success=bool(lastsol.success);opt.message=str(lastsol.message);opt.nfev=int(lastsol.nfev);opt.cost=float(lastsol.fun)
 vopt=make_points(opt.x);dopt=vopt-V0
 # Exact bone clearance barrier on every edited direct control.
 ctrlrows=np.asarray(active_rows,int)
 # Hard post-solve barriers with a descending line search; do not accept a step that sharpens any incident hinge.
 accepted=None;attempts=[]
 for scale in (1.,.85,.7,.55,.4,.3,.2,.12,.08):
  V1=V0+scale*dopt
  aa=calc_angles(V1);deltaang=aa-base_ang
  moved=np.flatnonzero(np.linalg.norm(V1-V0,axis=1)>1e-12);changed_faces=np.flatnonzero(np.any(np.isin(F,moved),axis=1))
  N1raw=np.cross(V1[F[:,1]]-V1[F[:,0]],V1[F[:,2]]-V1[F[:,0]]);A1=.5*np.linalg.norm(N1raw,axis=1);N1=N1raw/np.maximum(np.linalg.norm(N1raw,axis=1)[:,None],1e-30)
  ratios=A1[changed_faces]/np.maximum(area0[changed_faces],1e-30);dots=np.einsum("ij,ij->i",N0[changed_faces],N1[changed_faces])
  ctrlpts=V1[ctrlrows];clr=np.asarray([nearest(q,boneT_all) for q in ctrlpts]);sample_clr={z["name"]:float(closest_point(np.sum(V1[z["rows"]]*z["bary"][:,None],axis=0),boneT_all)[1]) for z in sample_defs}
  boneok=bool(np.all(clr>=required))
  special_clear={}
  for t,z in mech_info.items():
   cp1,d1,tr1=closest_point(V1[z["row"]],boneT_all);special_clear[int(t)]=float(d1);clr[ctrl_index[t]]=d1
  boneok=boneok and all(v>=.00102-2e-6 for v in special_clear.values()) and all(v>=.00102-2e-6 for v in sample_clr.values()) and bool(np.all(clr>=required-2e-6))
  boundary_move=max((np.linalg.norm(V1[r]-V0[r]) for r in boundary),default=0.)
  new_severe=int(np.sum((base_ang<=90.)&(aa>90.05)));severe_before=int(np.sum(base_ang>90.));severe_after=int(np.sum(aa>90.));max_severe_before=float(base_ang[base_ang>90.].max()) if severe_before else 0.;max_severe_after=float(aa[aa>90.].max()) if severe_after else 0.
  creaseok=(new_severe==0 and severe_after<=severe_before and max_severe_after<=BILATERAL_MAX_SEVERE_BASE_DEG+.02)
  target_after=aa[target_idx];target_before=base_ang[target_idx];target_err_before=np.abs(target_before-goals);target_err_after=np.abs(target_after-goals)
  palm_idx=[i for i,e in enumerate(targets) if (side=="left" and tuple(sorted(e))==(1577,1578)) or (side=="right" and tuple(sorted(e))==(1616,1617))]
  targetok=bool(np.all(target_err_after<target_err_before-.01) and all(target_after[i]<90. for i in palm_idx))
  faceok=bool((len(dots)==0 or np.min(dots)>0) and (len(ratios)==0 or np.min(ratios)>=.25))
  moveok=float(np.linalg.norm(V1-V0,axis=1).max())<=.002
  criteria={"scale":scale,"max_incident_edge_increase_deg":float(deltaang.max()),"new_severe_edges_over90":new_severe,"severe_edges_before_after":[severe_before,severe_after],"max_severe_angle_before_after_deg":[max_severe_before,max_severe_after],"severe_crease_improvement_pass":creaseok,"target_source_fit_pass":targetok,"target_angles_before_after_deg":{str(e):[float(target_before[i]),float(target_after[i]),float(goals[i])] for i,e in enumerate(targets)},"target_source_error_before_after_deg":[float(np.max(target_err_before)) if len(target_err_before) else 0.,float(np.max(target_err_after)) if len(target_err_after) else 0.],"p95_abs_edge_delta_deg":float(np.percentile(abs(deltaang),95)),"target_angle_reductions_deg":{str(e):float(ang0[tuple(sorted(e))]-aa[edge_index[tuple(sorted(e))]]) for e in targets},"min_changed_face_area_ratio":float(ratios.min()) if len(ratios) else 1.,"min_changed_face_normal_dot":float(dots.min()) if len(dots) else 1.,"boundary_move_mm":boundary_move*1000,"preexisting_controls_below_1mm":int(len(low_clearance)),"min_control_clearance_before_mm":float(before_clear.min()*1000),"min_control_clearance_after_mm":float(clr.min()*1000),"mechanical_control_clearance_after_mm":{str(k):float(v*1000) for k,v in special_clear.items()},"mechanical_controls":mech_info,"source_interpolant_clearance_after_mm":{k:v*1000 for k,v in sample_clr.items()},"source_interpolant_clearance_pass":all(v>=.00102-2e-6 for v in sample_clr.values()),"bone_barrier_pass":boneok,"face_orientation_area_pass":faceok,"max_vertex_move_mm":float(np.linalg.norm(V1-V0,axis=1).max()*1000),"move_cap_pass":moveok,"clearance_violating_controls":[{"template_control_id":int(active[i]),"row":int(ctrlrows[i]),"before_mm":float(before_clear[i]*1000),"after_mm":float(clr[i]*1000),"required_mm":float(required[i]*1000)} for i in np.flatnonzero(clr<required-2e-6)],"local_edges_over90_before_after":[int(np.sum(base_ang>90)),int(np.sum(aa>90))]}
  passed=creaseok and targetok and faceok and boneok and moveok and boundary_move<1e-10
  attempts.append({"criteria":criteria,"passed":passed})
  if passed:accepted=(scale,V1,aa,moved,changed_faces,ratios,dots,clr,criteria);break
 if accepted is None:
  result={"status":"no_safe_step","active_controls":n,"local_edges":ne,"solver":{"success":bool(opt.success),"message":opt.message,"nfev":opt.nfev,"cost":float(opt.cost),"linear_feasibility":linear_feasibility,"slsqp_clearance_rounds":slsqp_rounds},"attempts":attempts,"target_before_deg":{str(e):ang0[tuple(sorted(e))] for e in targets}}
  results[side]=result;print(side,json.dumps(result,cls=NumpyJSONEncoder),flush=True);continue
 scale,V1,aa,moved,changed_faces,ratios,dots,clr,criteria=accepted
 patch_delta=V1-V0
 fields={k:p0[k].copy() for k in p0.files};fields["world"]=V1;np.savez_compressed(OUT/(side+"-patch.npz"),**fields)
 cids=np.asarray(active,int);cpos=np.asarray([V0[direct[t][0]] for t in active])+scale*opt.x.reshape(n,3)*.001;np.savez_compressed(OUT/(side+"-control-positions.npz"),template_vertex_ids=cids,candidate_positions_world_m=cpos)
 lineage=[]
 for r in np.flatnonzero(np.linalg.norm(patch_delta,axis=1)>1e-12):
  lineage.append({"patch_row_060":int(r),"source_slot_060":int(slots[r]),"template_control_basis":[[int(t),float(w)] for t,w in rowbasis[int(r)]],"propagated_delta_m":patch_delta[r].tolist()})
 (OUT/(side+"-ancestry-lineage.json")).write_text(json.dumps(lineage,indent=2,cls=NumpyJSONEncoder)+"\n")
 # Registered-reference contour distance on exactly the moved patch rows.
 ref=np.load(SRCREF/(side+"-patch.npz"));refT=ref["world"].astype(float)[ref["faces"].astype(int)]
 touched=np.flatnonzero(np.linalg.norm(patch_delta,axis=1)>1e-12);refbefore=[nearest(V0[r],refT) for r in touched];refafter=[nearest(V1[r],refT) for r in touched]
 # Emit positions only for changed global rows from exact 060 source; retain all other rest position floats bitwise.
 Q=Qneutral0.copy();Q[slots]+=patch_delta
 glob_changed=slots[touched]
 all_changed=set(map(int,glob_changed))
 results[side]={"status":"candidate_generated_not_admitted","ancestry":ancestry_summary,"control_3176_mapping":mapping_3176,"linear_feasibility":linear_feasibility,"unavailable_source_targets":unavailable_targets,"source_interpolant_witnesses":[{"name":z["name"],"face":z["face"],"rows":z["rows"].tolist(),"bary":z["bary"].tolist(),"coefficient_nonzero":int(np.count_nonzero(z["coeff"]))} for z in sample_constraints],"active_controls":n,"support_rows":len(moved_rows),"incident_edge_count":ne,"target_edges":len(targetgoal),"witness_edges":[list(e) for e in witnesses],"solver":{"success":bool(opt.success),"message":str(opt.message),"nfev":int(opt.nfev),"cost":float(opt.cost),"slsqp_clearance_rounds":slsqp_rounds},"accepted":criteria,"attempts":attempts,"changed_patch_rows":int(len(touched)),"max_control_move_mm":float(np.linalg.norm(scale*opt.x.reshape(n,3)*.001,axis=1).max()*1000),"max_patch_move_mm":float(np.linalg.norm(patch_delta,axis=1).max()*1000),"reference_distance_median_before_after_mm":[float(np.median(refbefore)*1000),float(np.median(refafter)*1000)],"reference_distance_max_before_after_mm":[float(np.max(refbefore)*1000),float(np.max(refafter)*1000)],"selected_edges_before_after_deg":{str(e):[ang0[tuple(sorted(e))],aa[edge_index[tuple(sorted(e))]]] for e in targets},"patch_sha256":hashlib.sha256((OUT/(side+"-patch.npz")).read_bytes()).hexdigest(),"lineage_rows":len(lineage)}
 print(side,json.dumps(results[side],cls=NumpyJSONEncoder),flush=True)
# If both sides have a feasible step, emit full native payload with exact 060 positions retained on all unchanged rows.
if results.get("left",{}).get("status")=="candidate_generated_not_admitted":
 raw0=(BASE/"replacement-bilateral-hands038.nhskin").read_bytes();D=decode_payload(raw0);F=np.asarray(D["indices"],int).reshape(-1,3);Q=Qneutral0.copy();changed_global=set();side_data={}
 for side in ("right",):
  if side in results and results[side].get("status")=="candidate_generated_not_admitted":
   p=np.load(OUT/(side+"-patch.npz"));b=np.load(BASE/(side+"-patch.npz"))
  else:
   p=np.load(BASE/(side+"-patch.npz"));b=p
  slots=np.asarray(p["source_slots"],int);delta=p["world"].astype(float)-b["world"].astype(float);Q[slots]+=delta;changed_global.update(map(int,slots[np.linalg.norm(delta,axis=1)>1e-12]));side_data[side]=(slots,p["world"])
 changed=np.asarray(sorted(changed_global),int);W=np.asarray(D["full_weights"],float);BF=np.asarray(D["bindings_f"],float);bindings=np.asarray(D["bindings_u"][:,0],int);state={int(x["body_index"]):x for x in pose["bodies"]};A=np.zeros((len(Q),3,3));N=np.zeros_like(A);tr=np.zeros_like(Q)
 for j,b in enumerate(bindings):
  st=state[int(b)];rc=ws.ns["mat"](st["current"]["orientation_world_xyzw"]);restR=ws.ns["mat"](st["rest"]["orientation_world_xyzw"]);A+=W[:,j,None,None]*(rc*BF[j,8]@ws.ns["mat"](BF[j,4:8]))[None];tr+=W[:,j,None]*(BF[j,1:4]@rc.T+st["current"]["position_world_m"]);N+=W[:,j,None,None]*(rc@restR.T)[None]
 raw=bytearray(raw0);arr=np.ndarray((len(Q),14),dtype="<f4",buffer=raw,offset=60+36*D["binding_count"]);arr[changed,:3]=np.linalg.solve(A[changed],(Q[changed]-tr[changed])[:,:,None])[:,:,0];affected=np.unique(F[np.any(np.isin(F,changed),axis=1)]);fn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]]);wn=np.zeros_like(Q);np.add.at(wn,F.ravel(),np.repeat(fn,3,axis=0));wn/=np.maximum(np.linalg.norm(wn,axis=1)[:,None],1e-30);ln=np.linalg.solve(N[affected],wn[affected,:,None])[:,:,0];ln/=np.maximum(np.linalg.norm(ln,axis=1)[:,None],1e-30);arr[affected,3:6]=ln;payload=bytes(raw);(OUT/"replacement-bilateral-hands038.nhskin").write_bytes(payload);DC=decode_payload(payload);Qcheck=ws.ns["posed"](DC,pose).astype(float);proj={s:float(np.linalg.norm(Qcheck[sd[0]]-sd[1],axis=1).max()) for s,sd in side_data.items()};unchanged=np.setdiff1d(np.arange(len(Q)),changed);pos_preserved=bool(np.array_equal(DC["vertices_f"][unchanged,:3],D["vertices_f"][unchanged,:3]));meta=all(np.array_equal(DC[k],D[k]) for k in ("indices","bindings_f","bindings_u","full_weights"));assert max(proj.values())<2e-6 and pos_preserved and meta
 g0=np.load(BASE/"replacement-geometry.npz");gg={k:g0[k] for k in g0.files};gg["world"]=Qcheck;np.savez(OUT/"replacement-geometry.npz",**gg)
 report={"schema":"numi.human.finger-local-crease-repair.v1","candidate":"069g-baseline-oriented-right-clearance","status":("bilateral_candidate_not_admitted_pending_lead_audit" if all(results.get(x,{}).get("status")=="candidate_generated_not_admitted" for x in ("left","right")) else "left_only_candidate_not_admitted_pending_lead_audit"),"parent_payload_sha256":hashlib.sha256(raw0).hexdigest(),"candidate_payload_sha256":hashlib.sha256(payload).hexdigest(),"global_index_sha256":hashlib.sha256(np.asarray(DC["indices"],dtype="<u4").tobytes()).hexdigest(),"changed_global_rows":int(len(changed)),"outside_changed_rows_rest_position_bit_exact":pos_preserved,"indices_bindings_weights_bit_exact":meta,"patch_reprojection_error_m":proj,"sides":results,"method":"069 warm-starts from 068 and locally targets the two palm folds and three distal-thumb hinge angles at their registered 049 source values. Direct controls come from exact 049 ancestry of palm/thumb neighborhoods plus four mechanical capacity controls. The source-position prior uses exact template-control identities and registered 049 positions; non-target severe hinges are restrained near attained 068 angles. Mechanical direct controls and the two left source-face interpolants receive exact neutral all-bone point-to-triangle tangent-plane barriers with retained witness planes and 1.02 mm nearest-bone clearance floors; all other active direct controls retain their local >=1.0 mm or baseline-minus-0.01 mm floor. The recursive 056/058 ancestry explicitly resolves reference vertices, cuts, refined faces, loft sections, and baseline cuts into the exact 049 control basis plus fixed-boundary basis; partition of unity is checked for every patch row, and unknown records raise. Each accepted displacement propagates through all descendants. The line search creates no new >90-degree hinge, does not increase the local >90-degree count, keeps the severe maximum below the bilateral 068 maximum, improves the three left source-angle deviations and lowers the left palm fold below 90 degrees; the right side remains byte-preserved at 068. The left patch preserves positive face normals, area ratio >=0.25 and boundary, and moves at most 2.0 mm. This is a left-only diagnostic payload; full pose and collision audit not run."}
 (OUT/"build-report.json").write_text(json.dumps(report,indent=2,cls=NumpyJSONEncoder)+"\n");print("EMITTED",json.dumps({k:report[k] for k in ("status","candidate_payload_sha256","changed_global_rows","outside_changed_rows_rest_position_bit_exact","indices_bindings_weights_bit_exact","patch_reprojection_error_m")},cls=NumpyJSONEncoder),flush=True)
else:
 (OUT/"build-report.json").write_text(json.dumps({"schema":"numi.human.finger-local-crease-repair.v1","candidate":"069g-baseline-oriented-right-clearance","status":"no_safe_bilateral_candidate","sides":results},indent=2,cls=NumpyJSONEncoder)+"\n")
