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
from scipy.optimize import least_squares
sys.path.insert(0,"/tmp")
import wrist_sections_20261022 as ws
E=ws.E
BASE=E/"finger-profile-060"; SRCREF=E/"replacement-hand-reference-038"; PRIOR=E/"finger-profile-065-sourcefit-conservative"
OUT=E/"finger-profile-068-whole-hand-minimax"
OUT.mkdir(exist_ok=False)
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
def edge_info(V,F):
 N=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]);A=.5*np.linalg.norm(N,axis=1);N/=np.maximum(np.linalg.norm(N,axis=1)[:,None],1e-30);edges=defaultdict(list);adj=[set() for _ in range(len(V))]
 for fi,f in enumerate(F):
  for a,b in ((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))):
   edges[tuple(sorted((a,b)))].append(fi);adj[a].add(b);adj[b].add(a)
 ang={e:float(np.degrees(np.arccos(np.clip(N[fs[0]]@N[fs[1]],-1,1)))) for e,fs in edges.items() if len(fs)==2}
 boundary={v for e,fs in edges.items() if len(fs)==1 for v in e}
 return ang,A,N,edges,adj,boundary
def combine(parts):
 d={}
 for b,s in parts:
  for k,v in b:d[k]=d.get(k,0.)+v*s
 return tuple(sorted((k,v) for k,v in d.items() if abs(v)>1e-14))
def make_rowbasis(side,p49,p56,p58):
 a56=json.loads((E/"replacement-hands-056"/(side+"-source-ancestry.json")).read_text());a58=json.loads((E/"replacement-wrists-058"/(side+"-source-ancestry.json")).read_text());anc=np.load(E/"replacement-mechanical-055"/(side+"-template-control-ancestry049.npz"));ptr=anc["template_ancestry_indptr"];ix=anc["template_ancestry_control_index"];vv=anc["template_ancestry_weight"];tids=anc["template_control_ids"];slot49={int(s):i for i,s in enumerate(p49["source_slots"])}
 maps={}
 for level,path in ((56,E/"replacement-hands-056"),(58,E/"replacement-wrists-058")):
  g=np.load(path/"replacement-geometry.npz")
  if "internal_new_vertex_ids" in g.files: maps[level]=dict(zip(g["internal_new_vertex_ids"].tolist(),g["emitted_source_slots"].tolist()))
  else:
   both={}
   for ss in ("left","right"):both.update(json.loads((path/(ss+"-source-ancestry.json")).read_text()))
   maps[level]={61950+i:s for i,s in enumerate(sorted(map(int,both)))}
 def basis49(row):return tuple((int(tids[ix[j]]),float(vv[j])) for j in range(ptr[row],ptr[row+1]))
 @lru_cache(None)
 def b56(slot):
  rec=a56[str(int(slot))]
  if rec.get("reference_candidate049_vertex") is not None:return basis49(slot49[int(rec["reference_candidate049_vertex"])])
  if rec.get("reference_cut"):
   rc=rec["reference_cut"];a,b=rc["edge"];w=rc["weight"];return combine(((basis49(a),1-w),(basis49(b),w)))
  if rec.get("refined_face"):
   f=rec["refined_face"];return combine((b56(int(maps[56].get(i,i)),),1/len(f)) for i in f)
  return ()
 @lru_cache(None)
 def b58(slot):
  rec=a58[str(int(slot))]
  if rec.get("reference_candidate056_vertex") is not None:return b56(int(rec["reference_candidate056_vertex"]))
  if rec.get("reference_cut"):
   rc=rec["reference_cut"];a,b=rc["edge"];w=rc["weight"];return combine(((b56(int(p56["source_slots"][a])),1-w),(b56(int(p56["source_slots"][b])),w)))
  if rec.get("refined_face"):
   f=rec["refined_face"];return combine((b58(int(maps[58].get(i,i)),),1/len(f)) for i in f)
  return ()
 return [b58(int(s)) for s in p58["source_slots"]]
TARGET={"left":[(303,311),(687,695),(687,1109),(303,1031),(311,1184),(883,884),(887,971)],"right":[(691,699),(691,1113),(307,315),(307,1035),(887,888),(891,975)]}
SOURCE_TURNS={"left":{(303,311):90.85,(687,695):14.36,(687,1109):0.,(303,1031):78.09,(311,1184):76.34,(883,884):49.21,(887,971):9.85},"right":{(691,699):10.17,(691,1113):35.63,(307,315):89.93,(307,1035):78.45,(887,888):48.39,(891,975):9.4}}
Qneutral0=ws.ns["posed"](D0,pose).astype(np.float64)
results={}
for side in ("left","right"):
 t0=time.time()
 p0=np.load(BASE/(side+"-patch.npz"));p65=np.load(PRIOR/(side+"-patch.npz"));p49=np.load(E/"replacement-hands-049"/(side+"-patch.npz"));p56=np.load(E/"replacement-hands-056"/(side+"-patch.npz"));p58=np.load(E/"replacement-wrists-058"/(side+"-patch.npz"))
 for key in ("source_slots","faces","weights"):
  if not np.array_equal(p0[key],p65[key]):raise AssertionError((side,key))
 V0=p0["world"].astype(float);F=p0["faces"].astype(int);slots=np.asarray(p0["source_slots"],int);rowbasis=make_rowbasis(side,p49,p56,p58)
 ang0,area0,N0,edges0,adj0,boundary=edge_info(V0,F);direct=defaultdict(list)
 for r,b in enumerate(rowbasis):
  if len(b)==1 and abs(b[0][1]-1)<1e-9:direct[int(b[0][0])].append(r)
 pinned=set()
 for r in boundary:
  for t,w in rowbasis[r]:
   if abs(w)>1e-10:pinned.add(int(t))
 slot_to_row={int(s):i for i,s in enumerate(slots)}
 targets=list(TARGET[side]);wit_global=[(695,1262)] if side=="left" else [(47989,48862)]
 witnesses=[]
 for ga,gb in wit_global:
  if ga not in slot_to_row or gb not in slot_to_row:raise RuntimeError((side,"witness source slot missing",(ga,gb)))
  witnesses.append((slot_to_row[ga],slot_to_row[gb]))
 targets += witnesses
 targets=[tuple(map(int,e)) for e in targets]
 targetgoal={}
 for e in TARGET[side]:
  targetgoal[tuple(sorted(e))]=max(30.,min(70.,SOURCE_TURNS[side][e]+10.))
 for e in witnesses:
  key=tuple(sorted(e));targetgoal[key]=max(30.,ang0[key]-10.)
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
 active=sorted(t for t in direct if int(t) not in pinned)
 if not active:raise RuntimeError((side,"no active direct controls"))
 ctrl_index={t:i for i,t in enumerate(active)}
 # Include every edge incident to any descendant of active controls, not only selected target hinges.
 moved_rows={r for r,b in enumerate(rowbasis) if any(int(t) in ctrl_index for t,w in b if abs(w)>1e-12)}
 local_edges=[e for e,fs in edges0.items() if len(fs)==2 and (e[0] in moved_rows or e[1] in moved_rows)]
 target_keys={tuple(sorted(e)) for e in targets}
 if not target_keys.issubset(set(local_edges)):raise RuntimeError((side,"target missing from local edge graph"))
 edge_index={e:i for i,e in enumerate(local_edges)}
 facepairs=np.asarray([edges0[e] for e in local_edges],int);base_ang=np.asarray([ang0[e] for e in local_edges],float);target_idx=np.asarray([],dtype=int);goals=np.asarray([],float)
 # Sparse ancestry operator: every edited fine row receives its exact full 049 basis-weighted control motion.
 br=[];bc=[];bv=[]
 for r,b in enumerate(rowbasis):
  for t,w in b:
   if int(t) in ctrl_index:
    br.append(r);bc.append(ctrl_index[int(t)]);bv.append(float(w))
 B=coo_matrix((bv,(br,bc)),shape=(len(V0),len(active))).tocsr()
 # Directional source-contour prior from registered source-fit candidate 065, read at exact direct controls.
 prior=np.zeros((len(active),3),float)
 active_rows=[]
 for t,j in ctrl_index.items():
  r=direct[t][0];active_rows.append(r);prior[j]=p65["world"][r].astype(float)-V0[r]
  n=np.linalg.norm(prior[j])
  if n>.0005:prior[j]*=.0005/n
 prior_mm=prior*1000.
 # Mechanical reference floor: 1.0 mm. Already-subthreshold controls are pinned in this diagnostic.
 bv,bf,bowners=H["bone_mesh"](pose,side);bo=np.asarray(bowners);handmask=np.asarray([any(k in nm for k in ("carpal","metacarp","mc_","thumb","proxph","midph","distph")) for nm in bo]);boneT=bv[bf[handmask]].astype(float)
 active_rows=np.asarray(active_rows,int);before_clear=np.asarray([nearest(V0[r],boneT) for r in active_rows]);required=np.where(before_clear>=.001,.001,np.maximum(before_clear-.00001,0.))
 low_clearance=np.flatnonzero(before_clear<.001)
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
 def calc_angles(V):
  f0=F[facepairs[:,0]];f1=F[facepairs[:,1]]
  a=V[f0[:,0]];b=V[f0[:,1]];c=V[f0[:,2]];d=V[f1[:,0]];e=V[f1[:,1]];f=V[f1[:,2]]
  na=np.cross(b-a,c-a);nb=np.cross(e-d,f-d);na/=np.maximum(np.linalg.norm(na,axis=1)[:,None],1e-30);nb/=np.maximum(np.linalg.norm(nb,axis=1)[:,None],1e-30)
  return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i",na,nb),-1,1)))
 def residual(xmm):
  V=make_points(xmm);aa=calc_angles(V);diff=aa-base_ang
  # Jointly reduce all existing severe hinges and prevent any formerly mild edge from becoming severe.
  severe0=base_ang>90.;mild0=~severe0
  coh=.025*diff/10.
  hinge=5.0*np.where(severe0,np.maximum(0.,aa-90.)/10.,0.)
  new=15.0*np.where(mild0,np.maximum(0.,aa-90.)/1.,0.)
  x=np.asarray(xmm).reshape(n,3)
  src=.35*(x-prior_mm)/.5
  reg=.12*x/.5
  sm=.10*np.asarray([(x[i]-x[j])/.5 for i,j in smooth]).reshape(-1) if smooth else np.zeros(0)
  z=np.maximum(aa-90.,0.)/8.;zmax=float(np.max(z));soft=8.*(zmax+np.log(np.mean(np.exp(z-zmax))))
  return np.concatenate((coh,hinge,new,src.ravel(),reg.ravel(),sm,np.array([1.5*soft])))
 x0=(prior_mm*.65).reshape(-1);lb=np.full(nvar,-.55);ub=np.full(nvar,.55)
 for j in low_clearance:lb[3*j:3*j+3]=-1e-10;ub[3*j:3*j+3]=1e-10
 bounds=(lb,ub)
 opt=least_squares(residual,x0,bounds=bounds,jac_sparsity=J,diff_step=1e-3,method="trf",tr_solver="lsmr",max_nfev=16,ftol=3e-3,xtol=3e-3,gtol=3e-3,verbose=0)
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
  ctrlpts=V1[ctrlrows];clr=np.asarray([nearest(q,boneT) for q in ctrlpts]);boneok=bool(np.all(clr>=required))
  boundary_move=max((np.linalg.norm(V1[r]-V0[r]) for r in boundary),default=0.)
  new_severe=int(np.sum((base_ang<=90.)&(aa>90.05)));severe_before=int(np.sum(base_ang>90.));severe_after=int(np.sum(aa>90.));max_severe_before=float(base_ang[base_ang>90.].max()) if severe_before else 0.;max_severe_after=float(aa[aa>90.].max()) if severe_after else 0.;creaseok=(new_severe==0 and severe_after<=severe_before and (not severe_before or max_severe_after<max_severe_before-.02))
  faceok=bool((len(dots)==0 or np.min(dots)>0) and (len(ratios)==0 or np.min(ratios)>=.25))
  moveok=float(np.linalg.norm(V1-V0,axis=1).max())<=.00065
  criteria={"scale":scale,"max_incident_edge_increase_deg":float(deltaang.max()),"new_severe_edges_over90":new_severe,"severe_edges_before_after":[severe_before,severe_after],"max_severe_angle_before_after_deg":[max_severe_before,max_severe_after],"severe_crease_improvement_pass":creaseok,"p95_abs_edge_delta_deg":float(np.percentile(abs(deltaang),95)),"target_angle_reductions_deg":{str(e):float(ang0[tuple(sorted(e))]-aa[edge_index[tuple(sorted(e))]]) for e in targets},"min_changed_face_area_ratio":float(ratios.min()) if len(ratios) else 1.,"min_changed_face_normal_dot":float(dots.min()) if len(dots) else 1.,"boundary_move_mm":boundary_move*1000,"preexisting_controls_below_1mm":int(len(low_clearance)),"min_control_clearance_before_mm":float(before_clear.min()*1000),"min_control_clearance_after_mm":float(clr.min()*1000),"bone_barrier_pass":boneok,"face_orientation_area_pass":faceok,"max_vertex_move_mm":float(np.linalg.norm(V1-V0,axis=1).max()*1000),"move_cap_pass":moveok,"local_edges_over90_before_after":[int(np.sum(base_ang>90)),int(np.sum(aa>90))]}
  passed=creaseok and faceok and boneok and moveok and boundary_move<1e-10
  attempts.append({"criteria":criteria,"passed":passed})
  if passed:accepted=(scale,V1,aa,moved,changed_faces,ratios,dots,clr,criteria);break
 if accepted is None:
  result={"status":"no_safe_step","active_controls":n,"local_edges":ne,"solver":{"success":bool(opt.success),"message":opt.message,"nfev":opt.nfev,"cost":float(opt.cost)},"attempts":attempts,"target_before_deg":{str(e):ang0[tuple(sorted(e))] for e in targets}}
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
 results[side]={"status":"candidate_generated_not_admitted","active_controls":n,"support_rows":len(moved_rows),"incident_edge_count":ne,"target_edges":len(targetgoal),"witness_edges":[list(e) for e in witnesses],"solver":{"success":bool(opt.success),"message":str(opt.message),"nfev":int(opt.nfev),"cost":float(opt.cost)},"accepted":criteria,"attempts":attempts,"changed_patch_rows":int(len(touched)),"max_control_move_mm":float(np.linalg.norm(scale*opt.x.reshape(n,3)*.001,axis=1).max()*1000),"max_patch_move_mm":float(np.linalg.norm(patch_delta,axis=1).max()*1000),"reference_distance_median_before_after_mm":[float(np.median(refbefore)*1000),float(np.median(refafter)*1000)],"reference_distance_max_before_after_mm":[float(np.max(refbefore)*1000),float(np.max(refafter)*1000)],"selected_edges_before_after_deg":{str(e):[ang0[tuple(sorted(e))],aa[edge_index[tuple(sorted(e))]]] for e in targets},"patch_sha256":hashlib.sha256((OUT/(side+"-patch.npz")).read_bytes()).hexdigest(),"lineage_rows":len(lineage)}
 print(side,json.dumps(results[side],cls=NumpyJSONEncoder),flush=True)
# If both sides have a feasible step, emit full native payload with exact 060 positions retained on all unchanged rows.
if all(results.get(s,{}).get("status")=="candidate_generated_not_admitted" for s in ("left","right")):
 raw0=(BASE/"replacement-bilateral-hands038.nhskin").read_bytes();D=decode_payload(raw0);F=np.asarray(D["indices"],int).reshape(-1,3);Q=Qneutral0.copy();changed_global=set();side_data={}
 for side in ("left","right"):
  p=np.load(OUT/(side+"-patch.npz"));b=np.load(BASE/(side+"-patch.npz"));slots=np.asarray(p["source_slots"],int);delta=p["world"].astype(float)-b["world"].astype(float);Q[slots]+=delta;changed_global.update(map(int,slots[np.linalg.norm(delta,axis=1)>1e-12]));side_data[side]=(slots,p["world"])
 changed=np.asarray(sorted(changed_global),int);W=np.asarray(D["full_weights"],float);BF=np.asarray(D["bindings_f"],float);bindings=np.asarray(D["bindings_u"][:,0],int);state={int(x["body_index"]):x for x in pose["bodies"]};A=np.zeros((len(Q),3,3));N=np.zeros_like(A);tr=np.zeros_like(Q)
 for j,b in enumerate(bindings):
  st=state[int(b)];rc=ws.ns["mat"](st["current"]["orientation_world_xyzw"]);restR=ws.ns["mat"](st["rest"]["orientation_world_xyzw"]);A+=W[:,j,None,None]*(rc*BF[j,8]@ws.ns["mat"](BF[j,4:8]))[None];tr+=W[:,j,None]*(BF[j,1:4]@rc.T+st["current"]["position_world_m"]);N+=W[:,j,None,None]*(rc@restR.T)[None]
 raw=bytearray(raw0);arr=np.ndarray((len(Q),14),dtype="<f4",buffer=raw,offset=60+36*D["binding_count"]);arr[changed,:3]=np.linalg.solve(A[changed],(Q[changed]-tr[changed])[:,:,None])[:,:,0];affected=np.unique(F[np.any(np.isin(F,changed),axis=1)]);fn=np.cross(Q[F[:,1]]-Q[F[:,0]],Q[F[:,2]]-Q[F[:,0]]);wn=np.zeros_like(Q);np.add.at(wn,F.ravel(),np.repeat(fn,3,axis=0));wn/=np.maximum(np.linalg.norm(wn,axis=1)[:,None],1e-30);ln=np.linalg.solve(N[affected],wn[affected,:,None])[:,:,0];ln/=np.maximum(np.linalg.norm(ln,axis=1)[:,None],1e-30);arr[affected,3:6]=ln;payload=bytes(raw);(OUT/"replacement-bilateral-hands038.nhskin").write_bytes(payload);DC=decode_payload(payload);Qcheck=ws.ns["posed"](DC,pose).astype(float);proj={s:float(np.linalg.norm(Qcheck[sd[0]]-sd[1],axis=1).max()) for s,sd in side_data.items()};unchanged=np.setdiff1d(np.arange(len(Q)),changed);pos_preserved=bool(np.array_equal(DC["vertices_f"][unchanged,:3],D["vertices_f"][unchanged,:3]));meta=all(np.array_equal(DC[k],D[k]) for k in ("indices","bindings_f","bindings_u","full_weights"));assert max(proj.values())<2e-6 and pos_preserved and meta
 g0=np.load(BASE/"replacement-geometry.npz");gg={k:g0[k] for k in g0.files};gg["world"]=Qcheck;np.savez(OUT/"replacement-geometry.npz",**gg)
 report={"schema":"numi.human.finger-local-crease-repair.v1","candidate":"068-whole-hand-minimax","status":"candidate_not_admitted_pending_lead_audit","parent_payload_sha256":hashlib.sha256(raw0).hexdigest(),"candidate_payload_sha256":hashlib.sha256(payload).hexdigest(),"global_index_sha256":hashlib.sha256(np.asarray(DC["indices"],dtype="<u4").tobytes()).hexdigest(),"changed_global_rows":int(len(changed)),"outside_changed_rows_rest_position_bit_exact":pos_preserved,"indices_bindings_weights_bit_exact":meta,"patch_reprojection_error_m":proj,"sides":results,"method":"All neighboring hinges incident to the exact 049 ancestry descendants of a whole-hand direct-control set were included in a sparse least-squares objective. Source-contour prior comes from the registered-surface 065 sourcefit direction; target crease goals include all original shaft edges and both 066 neighbor regressions. Every incident hinge has a weak normalized angle-change residual; severe hinges are penalized above 90 degrees and new severe hinges are penalized. The accepted line-search scale is checked for no new >90-degree hinges, no increase in severe-hinge count, a lower maximum severe angle, positive normals, changed-face area ratio >=0.25, unchanged boundary, clearance floor retention, and displacement cap. There is no universal +0.10-degree cap on every incident edge. Control ancestry is propagated through supported reference and refinement records; loft_sections and baseline_cut records returned an empty basis, so complete recursive propagation was not established. Full pose audit not run by design."}
 (OUT/"build-report.json").write_text(json.dumps(report,indent=2,cls=NumpyJSONEncoder)+"\n");print("EMITTED",json.dumps({k:report[k] for k in ("status","candidate_payload_sha256","changed_global_rows","outside_changed_rows_rest_position_bit_exact","indices_bindings_weights_bit_exact","patch_reprojection_error_m")},cls=NumpyJSONEncoder),flush=True)
else:
 (OUT/"build-report.json").write_text(json.dumps({"schema":"numi.human.finger-local-crease-repair.v1","candidate":"068-whole-hand-minimax","status":"no_safe_bilateral_candidate","sides":results},indent=2,cls=NumpyJSONEncoder)+"\n")
