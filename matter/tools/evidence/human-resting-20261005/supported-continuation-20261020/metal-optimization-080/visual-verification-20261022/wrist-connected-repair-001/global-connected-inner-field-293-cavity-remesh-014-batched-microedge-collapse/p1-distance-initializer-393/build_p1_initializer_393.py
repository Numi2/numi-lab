#!/usr/bin/env python3
"""Bounded unsolved P1 initializer on c014 using the existing 277 distance/winding semantics."""
import ast, hashlib, json, math, signal, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
C14=ROOT/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/candidate.npz"
TRANSFER=ROOT/"global-connected-inner-field-293-cavity-remesh-014-independent-volume-001/field-transfer-preflight.npz"
BANK=ROOT/"registered-surrounding-tissue-native013-191/registered-tissue-banks.npz"
INV=ROOT/"registered-surrounding-tissue-native013-191/inventory.json"
POSE=ROOT/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json"
POSE_EXPECT="0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd"
BANK_EXPECT="39deb100a99601dc48550c3e5bcddfaac7aea266525e389abbf349d1d8c47792"
INV_EXPECT="b0823cf8f33df2586ba1cd45e5b260cd75d1e35b8d2682b67c15f64ea22bbcc7"
HELPER126=ROOT/"ankle-parent-tet-refinement-026/build_common_volume_126.py"
HELPER224=ROOT/"ankle-parent-tet-refinement-026/generated-224-commonfield-tetgen-volume-actual007/build_commonfield_224.py"
GUARD390=ROOT/"p1-material-field-attainability-owner-390/tools/human_skin_p1_material_volume.py"
OUT=ROOT/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/p1-distance-initializer-393"
MARGIN_BONE=0.00025
MARGIN_SOFT=0.00010
TARGETS={"skin":0.003452570642,"skin_plus_fat":0.015104913881}
BUDGET_S=115.0
STATE={}
T0=time.monotonic()
class BudgetStop(Exception): pass
def _alarm(_sig,_frame): raise BudgetStop("bounded 115s initializer budget reached")
signal.signal(signal.SIGALRM,_alarm)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def emit(stage,**kw): print(json.dumps({"stage":stage,**kw},sort_keys=True),flush=True)
def load_functions(path,names,extra):
 tree=ast.parse(Path(path).read_text())
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 if {n.name for n in funcs}!=set(names):raise RuntimeError(f"missing functions {set(names)-{n.name for n in funcs}} from {path}")
 ns=dict(extra)
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),"exec"),ns)
 return {n:ns[n] for n in names}
def api_preflight():
 c14=np.load(C14,allow_pickle=False); tr=np.load(TRANSFER,allow_pickle=False)
 assert c14["points_world_m"].shape==(35962,3)
 assert c14["tetrahedra"].shape==(164176,4)
 assert tr["phi"].shape==(len(c14["points_world_m"],),)
 assert np.array_equal(c14["physical_outer_faces"],tr["physical_outer_faces"])
 q=load_functions(HELPER126,{"closest_bary","closest_surface"},{"np":np,"cKDTree":cKDTree})
 x=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32);f=np.asarray([[0,1,2]],np.int64)
 d,fi,b,p=q["closest_surface"](np.asarray([[.25,.25,1.]],float),x,f,k=1)
 assert abs(float(d[0])-1)<1e-12 and int(fi[0])==0 and np.max(np.abs(p[0]-[.25,.25,0]))<1e-12
 v=load_functions(HELPER224,{"consistent_closed","solid_angle_winding"},{"np":np,"Counter":Counter,"math":math})
 closed,_=v["consistent_closed"](np.asarray([[0,1,2],[0,3,1],[1,3,2],[2,3,0]],np.int64))
 assert isinstance(closed,(bool,np.bool_))
 import importlib.util
 spec=importlib.util.spec_from_file_location("human_p1_guard390",GUARD390);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert callable(mod.check_material_volume_targets)
 # Positive-unit tetra smoke; the guard is necessary-only and must return.
 xt=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);tt=np.asarray([[0,1,2,3]],np.int64)
 gg=mod.check_material_volume_targets(xt,tt,np.asarray([0.,.2,.6,1.]),{"skin":.01,"skin_plus_fat":.02})
 assert "necessary_conditions_pass" in gg
 emit("preflight_pass",closest_surface="126 candidate-centroid exact-on-candidate triangles",inside="224 exact-coordinate-welded closed group + solid-angle winding",volume_guard="390 necessary-only")
def closest_surface_global(P,V,F,k,chunk=192):
 tris_cent=[]
 for lo in range(0,len(F),100000):
  tri=V[F[lo:lo+100000]].astype(np.float64)
  tris_cent.append(tri.mean(axis=1))
 centers=np.concatenate(tris_cent,axis=0)
 tree=cKDTree(centers)
 n=len(P);od=np.empty(n);of=np.empty(n,np.int64);ob=np.empty((n,3));oq=np.empty((n,3))
 for lo in range(0,n,chunk):
  pp=np.asarray(P[lo:lo+chunk],np.float64)
  _,ix=tree.query(pp,k=min(k,len(F)),workers=-1)
  if ix.ndim==1:ix=ix[:,None]
  TT=V[F[ix]].astype(np.float64)
  bary,ds=CB(pp,TT);kk=np.argmin(ds,axis=1);rr=np.arange(len(pp));ids=ix[rr,kk]
  od[lo:lo+len(pp)]=np.sqrt(ds[rr,kk]);of[lo:lo+len(pp)]=ids;ob[lo:lo+len(pp)]=bary[rr,kk]
  oq[lo:lo+len(pp)]=np.einsum("ni,nij->nj",ob[lo:lo+len(pp)],V[F[ids]].astype(np.float64))
 return od,of,ob,oq,centers

def write_npz(path,**kw):
 tmp=Path(str(path)+".tmp.npz");np.savez_compressed(tmp,**kw);tmp.replace(path)
def snapshot(status,extra=None):
 if not OUT.exists():OUT.mkdir(parents=True,exist_ok=True)
 payload={k:v for k,v in STATE.items() if isinstance(v,np.ndarray)}
 if payload:write_npz(OUT/"initializer-partial.npz",**payload)
 rep={"schema":"numi.human.p1-distance-initializer.393.v1","status":status,"native_admission":False,
      "elapsed_s":time.monotonic()-T0,"inputs_sha256":STATE.get("input_hashes",{}),"notes":STATE.get("notes",[])}
 if extra:rep.update(extra)
 (OUT/"run-report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
def main(smoke=False):
 global CB,CLOSE,WIND,GUARD
 apis=load_functions(HELPER126,{"closest_bary","closest_surface"},{"np":np,"cKDTree":cKDTree});CB=apis["closest_bary"]
 vw=load_functions(HELPER224,{"consistent_closed","solid_angle_winding"},{"np":np,"Counter":Counter,"math":math});CLOSE,WIND=vw["consistent_closed"],vw["solid_angle_winding"]
 import importlib.util
 spec=importlib.util.spec_from_file_location("human_p1_guard390",GUARD390);gm=importlib.util.module_from_spec(spec);spec.loader.exec_module(gm);GUARD=gm.check_material_volume_targets
 if smoke:return api_preflight()
 if OUT.exists():raise RuntimeError(f"refusing overwrite {OUT}")
 OUT.mkdir(parents=True)
 hashes={"candidate014":sha(C14),"field_transfer_preflight014":sha(TRANSFER),"registered_banks689":sha(BANK),"inventory689":sha(INV),"pose157":sha(POSE),"closest_helper126":sha(HELPER126),"winding_helper224":sha(HELPER224),"volume_guard390":sha(GUARD390)}
 if hashes["registered_banks689"]!=BANK_EXPECT or hashes["inventory689"]!=INV_EXPECT or hashes["pose157"]!=POSE_EXPECT:raise RuntimeError("registered-bank input pin mismatch")
 STATE["input_hashes"]=hashes;STATE["notes"]=["Initializer intentionally unsolved; no LP, level solve, material split, native handoff, or admission.","Distance values use k nearest triangle centroids followed by exact point-triangle distance on those candidates. They are candidate upper bounds, not globally certified nearest distances.","Closed-group interior labels use existing 224 exact-coordinate-welded topology test and solid-angle winding; open groups remain unsigned-distance-only.","Bank cache is a registered static source-pose obstacle set, not accepted deformed tissue state."]
 inv=json.loads(INV.read_text());
 if len(inv.get("banks",[]))!=689:raise RuntimeError("expected complete 689-bank inventory")
 if sha(POSE)!=inv.get("pose_sha256"):raise RuntimeError("pose/hash mismatch in bank inventory")
 c=np.load(C14,allow_pickle=False);X=np.asarray(c["points_world_m"],np.float32);T=np.asarray(c["tetrahedra"],np.int64)
 OF=np.asarray(c["physical_outer_faces"],np.int64);CF=np.asarray(c["construction_closure_faces"],np.int64)
 if X.shape!=(35962,3) or T.shape!=(164176,4):raise RuntimeError("candidate014 topology changed")
 outer=np.unique(OF);caps=np.unique(CF);cap_only=np.setdiff1d(caps,outer,assume_unique=False)
 STATE["physical_outer_node_ids"]=outer;STATE["construction_cap_node_ids"]=cap_only
 emit("inputs_pinned",nodes=len(X),tets=len(T),physical_outer_faces=len(OF),physical_outer_nodes=len(outer),cap_only_nodes=len(cap_only),banks=len(inv["banks"]))
 # Load all 689 banks once. Concatenated face rows retain bank kind and component IDs.
 z=np.load(BANK,allow_pickle=False);vparts=[];fparts=[];face_bank=[];groups=defaultdict(list);vbase=0;fbase=0;kind_codes=[];facecounts=[]
 for i,m in enumerate(inv["banks"]):
  vi=np.asarray(z[f"bank_{i:04d}_vertices"],np.float32);fi=np.asarray(z[f"bank_{i:04d}_faces"],np.int64)
  vparts.append(vi);fparts.append(fi+vbase);vbase+=len(vi);fbase+=len(fi);facecounts.append(len(fi))
  face_bank.append(np.full(len(fi),i,np.int32));kind_codes.append(np.full(len(fi),0 if m["kind"]=="anatomy" else 1,np.uint8))
  groups[tuple(sorted(map(int,m["body_indices"])))].append((i,vi,fi))
  if i%128==0:emit("banks_loaded",bank=i,faces=fbase,vertices=vbase)
 BV=np.concatenate(vparts,axis=0);BF=np.concatenate(fparts,axis=0);FB=np.concatenate(face_bank);FK=np.concatenate(kind_codes)
 del vparts,fparts,face_bank,kind_codes,z
 if len(BF)!=sum(facecounts):raise RuntimeError("bank face ownership table mismatch")
 emit("distance_db_begin",faces=len(BF),vertices=len(BV),groups=len(groups))
 db,bfi,bb,bq,bankcenters=closest_surface_global(X.astype(np.float64),BV,BF,k=64)
 np.savez_compressed(OUT/"distance-bank-checkpoint.npz",distance_bank_candidate_upper_m=db,nearest_bank_face_row=bfi,nearest_bank_barycentric=bb,nearest_bank_point_m=bq)
 emit("distance_db_done",elapsed_s=round(time.monotonic()-T0,3),q_m=np.quantile(db,[0,.01,.5,.95,.99,1]).tolist())
 dO,ofi,ob,oq,_=closest_surface_global(X.astype(np.float64),X,OF,k=48)
 np.savez_compressed(OUT/"distance-outer-checkpoint.npz",distance_outer_candidate_upper_m=dO,nearest_outer_face_row=ofi,nearest_outer_barycentric=ob,nearest_outer_point_m=oq)
 STATE.update(distance_bank_candidate_upper_m=db,nearest_bank_face_row=bfi,nearest_bank_barycentric=bb,nearest_bank_point_m=bq,distance_outer_candidate_upper_m=dO,nearest_outer_face_row=ofi,nearest_outer_barycentric=ob,nearest_outer_point_m=oq)
 emit("distance_outer_done",elapsed_s=round(time.monotonic()-T0,3),q_m=np.quantile(dO,[0,.01,.5,.95,.99,1]).tolist())
 # Existing 277 margins are applied by the nearest bank kind; winding uses closed per-body exact-coordinate unions.
 margin=np.where(FK[bfi]==0,MARGIN_BONE,MARGIN_SOFT);near=db<=margin
 inside=np.zeros(len(X),bool);closed_groups=0;open_groups=0;tested=0;inside_total=0;inside_o=0;wind_reports=[]
 STATE.update(inside_closed_union=inside,near_bank_candidate=near)
 for gi,(body_ids,items) in enumerate(groups.items()):
  vp=[];fp=[];off=0
  for _,vi,fi in items:vp.append(vi);fp.append(fi+off);off+=len(vi)
  Vcat=np.concatenate(vp,axis=0);Fcat=np.concatenate(fp,axis=0)
  Vg,invx=np.unique(Vcat,axis=0,return_inverse=True);Fg=invx[Fcat]
  closed,top=CLOSE(Fg)
  if not closed:
   open_groups+=1;wind_reports.append({"body_indices":list(body_ids),"closed":False,"banks":len(items),"topology":top});continue
  closed_groups+=1
  mn=Vg.min(0).astype(float)-1e-6;mx=Vg.max(0).astype(float)+1e-6
  ids=np.flatnonzero(np.all(X.astype(np.float64)>=mn,axis=1)&np.all(X.astype(np.float64)<=mx,axis=1))
  if len(ids):
   w=WIND(X[ids].astype(np.float64),Vg,Fg);hit=np.abs(w)>0.5
   inside[ids[hit]]=True;tested+=len(ids);inside_total+=int(hit.sum());inside_o+=int(np.sum(hit & np.isin(ids,outer)))
  else:w=np.zeros(0);hit=np.zeros(0,bool)
  wind_reports.append({"body_indices":list(body_ids),"closed":True,"banks":len(items),"vertices":int(len(Vg)),"faces":int(len(Fg)),"tested_points":int(len(ids)),"inside_points":int(hit.sum()),"max_abs_winding":float(np.max(np.abs(w))) if len(w) else 0.0})
  if gi%12==0:
   STATE["inside_closed_union"]=inside
   write_npz(OUT/"winding-progress.npz",inside_closed_union=inside,near_bank_candidate=near)
   emit("winding_progress",group=gi,groups=len(groups),closed=closed_groups,open=open_groups,tested_nodes=tested,inside_nodes=inside_total,elapsed_s=round(time.monotonic()-T0,3))
 phi=np.where(inside|near,1.0,dO/np.maximum(dO+np.maximum(db,1e-9),1e-9));phi=np.clip(phi,0.,1.)
 phi[outer]=0.0
 STATE.update(phi_initializer_unsolved=phi,inside_closed_union=inside,near_bank_candidate=near,physical_outer_node_mask=np.isin(np.arange(len(X)),outer),construction_cap_only_node_mask=np.isin(np.arange(len(X)),cap_only))
 volume_gate=GUARD(X,T,phi,TARGETS)
 STATE["notes"].append("390 guard is a necessary P1 volume-attainability screen only; passing does not solve levels or certify geometry.")
 nearest_kind=np.where(FK[bfi]==0,0,1)
 write_npz(OUT/"p1-distance-initializer-unsolved.npz",points_world_m=X,tetrahedra=T,phi_initializer_unsolved=phi,
  distance_outer_candidate_upper_m=dO,distance_bank_candidate_upper_m=db,nearest_outer_face_row=ofi,nearest_outer_barycentric=ob,
  nearest_bank_face_row=bfi,nearest_bank_barycentric=bb,nearest_bank_kind_code=nearest_kind,
  inside_closed_union=inside,near_bank_candidate=near,physical_outer_node_ids=outer,construction_cap_only_node_ids=cap_only)
 report={"schema":"numi.human.p1-distance-initializer.393.v1","status":"unsolved_candidate_initializer","native_admission":False,
  "inputs_sha256":hashes,"paths":{"candidate014":str(C14),"registered_banks":str(BANK),"inventory":str(INV),"pose":str(POSE),"distance_helper":str(HELPER126),"winding_helper":str(HELPER224),"volume_guard":str(GUARD390)},
  "nodes":len(X),"tets":len(T),"physical_outer_faces":len(OF),"physical_outer_nodes":len(outer),"construction_cap_only_nodes":len(cap_only),
  "distance_method":"277 semantics: k=48 physical O and k=64 concatenated registered-bank face-centroid candidate queries followed by exact point-to-triangle projection on those candidates; distances remain candidate upper bounds, not global nearest certificates.",
  "inside_classifier":"277 semantics: per-body-index groups exact-coordinate-welded, closed/oriented groups classified by 224 solid-angle winding abs(w)>0.5; open groups unsigned-distance only; inside is union across closed groups.",
  "margins_m":{"anatomy":MARGIN_BONE,"soft_tissue":MARGIN_SOFT},"closed_group_count":closed_groups,"open_group_count":open_groups,"inside_point_count_union":int(inside.sum()),"inside_physical_outer_nodes":int(np.sum(inside[outer])),"winding_group_reports":wind_reports,
  "distance_outer_quantiles_m":np.quantile(dO,[0,.01,.05,.5,.95,.99,1]).tolist(),"distance_bank_quantiles_m":np.quantile(db,[0,.01,.05,.5,.95,.99,1]).tolist(),
  "near_candidate_count":int(near.sum()),"nearest_kind_counts":{"anatomy":int(np.sum(nearest_kind==0)),"soft_tissue":int(np.sum(nearest_kind==1))},
  "phi_quantiles":np.quantile(phi,[0,.01,.05,.5,.95,.99,1]).tolist(),"field_definition":"phi=1 for closed-solid inside or estimated near-margin nodes; otherwise dO/(dO+dBank), clipped [0,1]; physical O nodes reset to 0. This is only an unsolved initializer.",
  "volume_target_necessary_gate_390":volume_gate,"elapsed_s":time.monotonic()-T0,
  "limitations":["Static source-pose tissue bank cache; not accepted deformed state.","Centroid-candidate distances can overestimate true global nearest distances; near-margin labels and ratio are provisional.","No obstacle LP, field solve, level selection, interface extraction, material split, conservative ancestry, native compile, or admission was attempted."]}
 (OUT/"run-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 emit("initializer_complete_unsolved",elapsed_s=round(report["elapsed_s"],3),inside=report["inside_point_count_union"],inside_outer=report["inside_physical_outer_nodes"],necessary_conditions=volume_gate["necessary_conditions_pass"],phi_q=report["phi_quantiles"])

if __name__=="__main__":
 if "--smoke" in sys.argv:
  main(smoke=True);sys.exit(0)
 signal.alarm(int(BUDGET_S))
 try:main(False)
 except BudgetStop as exc:
  snapshot("bounded_partial_unsolved_initialization",{"reason":str(exc)})
  emit("bounded_partial",elapsed_s=round(time.monotonic()-T0,3),reason=str(exc),checkpoint=str(OUT))
  sys.exit(124)
