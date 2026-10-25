#!/usr/bin/env python3
"""Per-face source-outward first-empty-union ray targets for frozen 005 alpha 0.125."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
RUN=D/"barycentric-connected-clearance-170/joint-fit-005-rebased-soft/attempt-001/scp-02-alpha-0p125"
CAND=RUN/"candidate.npz"
SURFACE=RUN/"audit-159/report.json"
DETAIL=D/"publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
ORIENT=D/"digit-source-diagnosis-110/source-path-orientation-audit-112.json"
BANKHELPER=D/"publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py"
EVIDENCE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANK=EVIDENCE/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BANK_FIX=EVIDENCE/"route-audit-counter-fix-030/canonical/matter"
EXACT=EVIDENCE/"incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py"
CONSTRAINT=D/"barycentric-virtual-point-170/human_skin_fat_virtual_point_constraints.py"
OLD172=D/"automatic-connected-continuation-169/run-002/round-001/barycentric-virtual-point-172-corrected-raw-vector-transform/run002-round001-certified-inside-face-rows.npz"
OLD172_SUM=OLD172.with_name("run002-round001-certified-inside-face-summary.json")
OUT=RUN/"coherent-source-face-union-exits-175"
CLEARANCE_M=5e-5

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def loadmod(p,n):
 spec=importlib.util.spec_from_file_location(n,p)
 m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m

def json_default(o):
 if isinstance(o,np.ndarray):return o.tolist()
 if isinstance(o,np.generic):return o.item()
 if isinstance(o,Fraction):return {"numerator":str(o.numerator),"denominator":str(o.denominator),"value":float(o)}
 raise TypeError(type(o).__name__)

def csvrows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))

def ray_box(q,d,lo,hi):
 t0=-math.inf;t1=math.inf
 for k in range(3):
  if d[k]==0:
   if q[k]<lo[k] or q[k]>hi[k]:return None
   continue
  a=(lo[k]-q[k])/d[k];b=(hi[k]-q[k])/d[k]
  if a>b:a,b=b,a
  t0=max(t0,a);t1=min(t1,b)
  if t0>t1:return None
 if t1<0:return None
 return max(0.0,t0),t1

def first_empty_union_interval(initial_inside,events,clearance_m):
 grouped={}
 for ev in events:grouped.setdefault(float(ev["distance_m"]),[]).append(ev)
 distances=sorted(grouped);active=set(initial_inside)
 for i,dexit in enumerate(distances):
  toggles={}
  for ev in grouped[dexit]:toggles[ev["bank"]]=toggles.get(ev["bank"],0)+1
  for key,count in toggles.items():
   if count%2:
    if key in active:active.remove(key)
    else:active.add(key)
  if not active:
   nxt=distances[i+1] if i+1<len(distances) else None
   gap=math.inf if nxt is None else nxt-dexit
   if gap<=0:continue
   margin=clearance_m if math.isinf(gap) else min(clearance_m,.5*gap)
   if margin>0:return {"exit_distance_m":dexit,"target_distance_m":dexit+margin,"next_event_distance_m":nxt,"margin_m":margin,"free_gap_m":gap}
 return None

def check_first_union_fixture():
 r=first_empty_union_interval({"A"},[
  {"distance_m":.002,"bank":"A"},{"distance_m":.5,"bank":"B"},{"distance_m":.501,"bank":"B"}],CLEARANCE_M)
 assert r and abs(r["target_distance_m"]-.00205)<1e-15 and r["next_event_distance_m"]==.5

def bank_orientation(tris):
 vids={};tri_vid=np.empty((len(tris),3),np.int64)
 for ti in range(len(tris)):
  for j in range(3):
   key=tuple(np.asarray(tris[ti,j],np.float32).view(np.uint32).tolist())
   tri_vid[ti,j]=vids.setdefault(key,len(vids))
 edge_count=defaultdict(int);edge_balance=defaultdict(int)
 for a,b,c in tri_vid:
  for u,v in ((int(a),int(b)),(int(b),int(c)),(int(c),int(a))):
   key=(min(u,v),max(u,v));edge_count[key]+=1;edge_balance[key]+=1 if u<v else -1
 vol6=float(np.einsum("ij,ij->i",tris[:,0],np.cross(tris[:,1],tris[:,2])).sum())
 closed=bool(edge_count) and all(v==2 for v in edge_count.values())
 oriented=closed and all(v==0 for v in edge_balance.values())
 sign=1 if vol6>0 else -1 if vol6<0 else 0
 return (sign if oriented else 0),{"vertex_count":len(vids),"edge_count":len(edge_count),"closed":closed,"consistently_oriented":oriented,"signed_six_volume_m3":vol6}

def fracpoint(x):
 return tuple(Fraction.from_float(float(v)) for v in x)

def exact_point_from_bary(lam,tri):
 return tuple(sum((lam[i]*tri[i][k] for i in range(3)),Fraction(0)) for k in range(3))

def build_source_outward_normals(candidate,detail,orientation):
 """kind2: absolute registered detailed outward sign; kind1: inward coarse winding is negated; kind0 unresolved."""
 P=np.asarray(candidate["points_world_m"],np.float32).astype(np.float64)
 F=np.asarray(candidate["source_O_faces"],np.int64)
 kinds=np.asarray(candidate["face_kind"],np.int8)
 drow=np.asarray(candidate["parent_fine_detail_face_row"],np.int64)
 DP=np.asarray(detail["detail_baseline_world_m"],np.float64)
 DF=np.asarray(detail["source_surface_faces_detail_rows"],np.int64)
 out=np.full((len(F),3),np.nan,np.float64); records=[]
 chosen=int(orientation["component_outward_sign_from_layer_order"]["0"]["chosen_outward_sign"])
 parity=int(orientation["component_fine_to_coarse_winding_anchors"]["0"]["chosen_outward_parity"])
 if chosen!=1 or parity!=-1: raise RuntimeError(f"orientation pins changed: absolute={chosen}, fine/coarse={parity}")
 for f,kind in enumerate(kinds):
  if kind==2:
   df=int(drow[f])
   if not 0<=df<len(DF): records.append({"face_row":f,"kind":2,"status":"MISSING_FINE_FACE"});continue
   tri=DP[DF[df]]
   n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);ln=float(np.linalg.norm(n))
   if not np.isfinite(ln) or ln<=0:records.append({"face_row":f,"kind":2,"status":"DEGENERATE_FINE_FACE","detail_face_row":df});continue
   n=n/ln*chosen
   out[f]=n
   current=np.cross(P[F[f,1]]-P[F[f,0]],P[F[f,2]]-P[F[f,0]])
   cn=float(np.linalg.norm(current));align=float(np.dot(n,current/cn)) if cn else None
   records.append({"face_row":f,"kind":2,"status":"SOURCE_DETAIL_OUTWARD","detail_face_row":df,"absolute_outward_sign":chosen,"fine_to_coarse_parity_diagnostic_only":parity,"current_face_normal_alignment":align})
  elif kind==1:
   tri=P[F[f]]
   n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);ln=float(np.linalg.norm(n))
   if not np.isfinite(ln) or ln<=0:records.append({"face_row":f,"kind":1,"status":"DEGENERATE_COARSE_FACE"});continue
   n=-n/ln
   out[f]=n
   records.append({"face_row":f,"kind":1,"status":"COARSE_INWARD_WINDING_NEGATED","current_face_normal_alignment":-1.0})
  else:records.append({"face_row":f,"kind":int(kind),"status":"UNRESOLVED_SOURCE_WINDING"})
 return out,records

def strict_interior_sample(face_local,bone_local,bankloc,bank_sign,exact,bary_fn):
 a=tuple(tuple(v for v in row) for row in (fracpoint(p) for p in face_local))
 b=tuple(tuple(v for v in row) for row in (fracpoint(p) for p in bone_local))
 pts=sorted(set(exact.triangle_intersection_points(a,b)))
 if len(pts)<2:return None,"NO_INTERSECTION_SEGMENT",None
 p0,p1=max(((x,y) for i,x in enumerate(pts) for y in pts[i+1:]),key=lambda xy:sum(float(xy[0][k]-xy[1][k])**2 for k in range(3)))
 l0=bary_fn(a,p0);l1=bary_fn(a,p1);base=tuple((l0[i]+l1[i])/2 for i in range(3))
 sn=np.cross(face_local[1]-face_local[0],face_local[2]-face_local[0]);sl=float(np.linalg.norm(sn))
 bn=np.cross(bone_local[1]-bone_local[0],bone_local[2]-bone_local[0]);bl=float(np.linalg.norm(bn))
 if sl<=0 or bl<=0:return None,"DEGENERATE_PAIR_TRIANGLE",None
 bn=bank_sign*bn/bl;sn/=sl
 inward=-bn+float(np.dot(bn,sn))*sn;il=float(np.linalg.norm(inward))
 if il<1e-14:return None,"NEAR_TANGENT_PLANES",None
 inward/=il
 e1=face_local[1]-face_local[0];e2=face_local[2]-face_local[0]
 uv=np.linalg.lstsq(np.column_stack((e1,e2)),inward,rcond=None)[0]
 dl=np.asarray([-uv[0]-uv[1],uv[0],uv[1]],np.float64)
 bf=np.asarray([float(x) for x in base])
 tmax=min((-bf[i]/dl[i] for i in range(3) if dl[i]<0),default=math.inf)
 if not math.isfinite(tmax) or tmax<=0:return None,"NO_INWARD_SIMPLEX_ROOM",None
 for frac in (.5,.25,.125,.0625,.03125,.015625,.0078125):
  lf=bf+(tmax*frac)*dl
  if np.min(lf)<=0:continue
  l0=Fraction.from_float(float(lf[0]));l1=Fraction.from_float(float(lf[1]));lam=(l0,l1,Fraction(1)-l0-l1)
  if min(lam)<=0:continue
  loc=exact.point_location_prepared_indexed(exact_point_from_bary(lam,a),bankloc)
  if loc.get("location")=="inside":return lam,"INSIDE",{"point0":p0,"point1":p1,"locator":loc}
 return None,"NO_CERTIFIED_INSIDE_FACE_SAMPLE",None

def main():
 if OUT.exists():raise FileExistsError(f"output exists: {OUT}")
 check_first_union_fixture()
 tstart=time.monotonic()
 z=np.load(CAND,allow_pickle=False);P=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["source_O_faces"],np.int64)
 surface=json.loads(SURFACE.read_text())
 if not surface.get("complete") or surface.get("candidate_sha256")!=sha(CAND) or surface.get("surface_point_count")!=len(P) or surface.get("surface_face_count")!=len(F):raise RuntimeError("matched 159 candidate/report pin mismatch")
 detail=np.load(DETAIL,allow_pickle=False);orient=json.loads(ORIENT.read_text())
 normals,normal_records=build_source_outward_normals(z,detail,orient)
 reader=loadmod(BANKHELPER,"bone_bank_reader_175")
 triangles,ids,gids,bodies,M,off,pos,bankkeys,bankpins,exact_unused=reader.read_bank_inputs(BANK,BANK_FIX)
 if len(bankkeys)!=185 or len(triangles)!=len(ids):raise RuntimeError("incomplete pinned 185-bank input")
 sys.path.insert(0,str(EVIDENCE/"incremental-exact-audit-owner-253/src"))
 exact=loadmod(EXACT,"numilab_human.cardiac_cavity_intersections")
 from sys import path
 path.insert(0,str(D/"barycentric-virtual-point-170"))
 cmod=loadmod(CONSTRAINT,"virtual_point_constraint_172")
 bary_fn=cmod.exact_triangle_barycentric
 banks={}
 for b0,k0 in bankkeys:
  b,k=int(b0),int(k0);mask=(ids[:,0]==b)&(ids[:,1]==k)
  T=np.asarray(triangles[mask],np.float64);TI=np.asarray(ids[mask,2],np.int64)
  if not len(T):raise RuntimeError(f"empty bank {b}/{k}")
  sign,top=bank_orientation(T)
  banks[(b,k)]={"T":T,"id":TI,"lo":T.min((0,1)),"hi":T.max((0,1)),"tl":T.min(1),"th":T.max(1),"M":M[b],"pos":pos[b],"off":off[b],"sign":sign,"topology":top,"locator":None,"tri_by_id":{int(i):T[j] for j,i in enumerate(TI)}}
  if not top["closed"] or not top["consistently_oriented"] or sign==0:raise RuntimeError(f"bank shell topology/orientation unverified {b}/{k}: {top}")
 # Keep the full 185-key ordered inventory; locators are prepared lazily.
 qreport=json.loads((RUN/"audit-130/query/query-receipt.json").read_text())
 if qreport.get("query_rows")!=len(P) or qreport.get("total_banks") not in (None,185):raise RuntimeError("matched 130 query receipt incomplete")
 # Build crossing-pair records from the complete matched 159 report.
 pairs=[]
 for bank in surface["banks"]:
  result=bank.get("result") or {}
  if not result.get("triangle_pairs"):continue
  body,bone=int(bank["body"]),int(bank["stable_bone_id"])
  for f,tid in result["triangle_pairs"]:pairs.append((int(f),body,bone,int(tid)))
 if len(pairs)!=3658:raise RuntimeError(f"unexpected matched crossing-pair count {len(pairs)}")
 # Cache complete bank point locators only when sample or target enters a bank AABB.
 def ensure_locator(key):
  rec=banks[key]
  if rec["locator"] is None:
   T=rec["T"];vertices=tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in T.reshape(-1,3))
   faces=tuple((3*i,3*i+1,3*i+2) for i in range(len(T)))
   rec["locator"]=exact.prepare_point_location(exact._records(vertices,faces))
  return rec["locator"]
 def classify_world(point):
  status={};unresolved=[]
  for key in bankkeys:
   key=(int(key[0]),int(key[1]));rec=banks[key]
   q=(np.asarray(point,np.float64)-rec["pos"])@rec["M"].T-rec["off"]
   if np.any(q<rec["lo"]) or np.any(q>rec["hi"]):status[key]="outside_aabb";continue
   rr=exact.point_location_prepared_indexed(fracpoint(q),ensure_locator(key))
   loc=rr.get("location")
   if loc not in ("inside","outside"):unresolved.append({"bank":list(key),"result":rr})
   status[key]=loc
  return status,unresolved
 # Prepare candidate face geometry and body-local transforms.
 body_local={}
 for b in {int(k[0]) for k in bankkeys}:
  rr=banks[next(k for k in banks if k[0]==b)]
  body_local[b]=(P.astype(np.float64)-rr["pos"])@rr["M"].T-rr["off"]
 output=[];fail=Counter();initial_classes=Counter();union_targets=0
 for pair_i,(face,body,bone,tid) in enumerate(pairs):
  key=(body,bone);rec=banks[key]
  if not np.all(np.isfinite(normals[face])):
   fail["UNRESOLVED_SOURCE_WINDING"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"UNRESOLVED_SOURCE_WINDING","face_kind":int(z["face_kind"][face])});continue
  tri=rec["tri_by_id"].get(tid)
  if tri is None:
   fail["MISSING_BANK_TRIANGLE_STABLE_ID"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"MISSING_BANK_TRIANGLE_STABLE_ID"});continue
  flocal=body_local[body][F[face]]
  lam,sample_status,sample_detail=strict_interior_sample(flocal,tri,ensure_locator(key),rec["sign"],exact,bary_fn)
  if sample_status!="INSIDE":
   fail[sample_status]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":sample_status,"face_kind":int(z["face_kind"][face])});continue
  lamf=np.asarray([float(v) for v in lam],np.float64)
  # Query position is reconstructed from emitted F32 candidate vertices, matching fit-row coordinates.
  start=np.asarray(lamf@P[F[face]].astype(np.float64),np.float32).astype(np.float64)
  direction=np.asarray(normals[face],np.float64)
  dl=float(np.linalg.norm(direction))
  if not np.isfinite(dl) or dl<=0:fail["INVALID_SOURCE_NORMAL"]+=1;continue
  direction/=dl
  classes,unresolved=classify_world(start)
  if unresolved:
   fail["INITIAL_UNION_CLASSIFICATION_UNRESOLVED"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"INITIAL_UNION_CLASSIFICATION_UNRESOLVED","witnesses":unresolved[:8]});continue
  if classes.get(key)!="inside":
   fail["RECONSTRUCTED_SAMPLE_NOT_INSIDE_IMPLICATED_BANK"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"RECONSTRUCTED_SAMPLE_NOT_INSIDE_IMPLICATED_BANK","implicated_class":classes.get(key)});continue
  initial={k for k,v in classes.items() if v=="inside"};initial_classes[len(initial)]+=1
  events=[];unresolved_ray=[];end_states={}
  for key2 in bankkeys:
   key2=(int(key2[0]),int(key2[1]));rr=banks[key2];st=classes[key2]
   q=(start-rr["pos"])@rr["M"].T-rr["off"];d=rr["M"]@direction
   iv=ray_box(q,d,rr["lo"],rr["hi"])
   if iv is None:
    if st=="inside":unresolved_ray.append({"bank":list(key2),"reason":"INSIDE_START_OUTSIDE_AABB"})
    end_states[key2]=st=="inside";continue
   t0,t1=iv;te=t1+CLEARANCE_M
   p0=q+t0*d;p1=q+te*d;seglo=np.minimum(p0,p1);seghi=np.maximum(p0,p1)
   candidate=np.flatnonzero(np.all(rr["th"]>=seglo,axis=1)&np.all(rr["tl"]<=seghi,axis=1))
   hits=[];hit_unresolved=[]
   for ti in candidate:
    h=exact_unused["classify_segment_triangle"](p0,p1,rr["T"][int(ti)]);cl=h["classification"]
    if cl=="proper_interior_hit":
     tv=t0+float(h["t"])*(te-t0)
     ev={"distance_m":tv,"bank":key2,"triangle_stable_id":int(rr["id"][int(ti)])};events.append(ev);hits.append(ev)
    elif cl not in ("disjoint","parallel_disjoint"):
     hit_unresolved.append({"body":key2[0],"bone":key2[1],"triangle_stable_id":int(rr["id"][int(ti)]),"classification":cl,"exact_result":h})
   if hit_unresolved:unresolved_ray.extend(hit_unresolved[:8])
   end_states[key2]=(st=="inside")^(len(hits)%2==1)
  if unresolved_ray:
   fail["UNRESOLVED_RAY_CONTACT"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"UNRESOLVED_RAY_CONTACT","witnesses":unresolved_ray[:16]});continue
  remain=[list(k) for k,v in end_states.items() if v]
  if remain:
   fail["RAY_PARITY_FAILURE"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"RAY_PARITY_FAILURE","remaining_inside_banks":remain[:32]});continue
  exitrec=first_empty_union_interval(initial,events,CLEARANCE_M)
  if exitrec is None:
   fail["NO_FIRST_EMPTY_UNION_INTERVAL"]+=1;output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"NO_FIRST_EMPTY_UNION_INTERVAL","initial_inside_banks":[list(k) for k in sorted(initial)],"event_count":len(events)});continue
  target=np.asarray(start+exitrec["target_distance_m"]*direction,np.float32)
  emitted=float(np.dot(target.astype(np.float64)-start,direction))
  if not emitted>exitrec["exit_distance_m"] or (exitrec["next_event_distance_m"] is not None and not emitted<exitrec["next_event_distance_m"]):
   fail["F32_TARGET_LEFT_FIRST_FREE_INTERVAL"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"F32_TARGET_LEFT_FIRST_FREE_INTERVAL","exit":exitrec,"emitted_distance_m":emitted});continue
  # Exact full-union endpoint membership; do not rely only on event parity.
  endclass,unresolved_end=classify_world(target.astype(np.float64))
  if unresolved_end or any(v=="inside" for v in endclass.values()):
   fail["TARGET_NOT_EXACTLY_OUTSIDE_UNION"]+=1
   output.append({"pair_index":pair_i,"face_row":face,"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"status":"TARGET_NOT_EXACTLY_OUTSIDE_UNION","target_inside_banks":[list(k) for k,v in endclass.items() if v=="inside"],"unresolved":unresolved_end[:8],"target":target.tolist()});continue
  union_targets+=1
  output.append({"pair_index":pair_i,"face_row":face,"face_kind":int(z["face_kind"][face]),"skin_node_rows":F[face].tolist(),"barycentric_weights":lamf.tolist(),"inside_sample_world_f32_m":start.tolist(),"bank_body":body,"bank_stable_bone_id":bone,"bank_triangle_stable_id":tid,"source_outward_world_unit":direction.tolist(),"source_normal_provenance":None,"initial_inside_union_banks":[list(k) for k in sorted(initial)],"initial_inside_union_count":len(initial),"union_exit_distance_m":exitrec["exit_distance_m"],"next_union_event_distance_m":exitrec["next_event_distance_m"],"free_interval_margin_m":emitted-exitrec["exit_distance_m"],"target_distance_m":emitted,"target_world_f32_m":target.tolist(),"target_exactly_outside_all185":True,"ray_events":len(events),"touched_banks":len({e["bank"] for e in events}),"status":"PASS_SOURCE_OUTWARD_FIRST_UNION_TARGET"})
 # Write heavy numeric records before metadata; keep all unmatched/failures in JSONL.
 outrows=OUT/"target-records.jsonl";npz=OUT/"targets.npz";reportp=OUT/"report.json"
 OUT.mkdir(parents=True,exist_ok=False)
 with outrows.open("w") as f:
  for r in output:f.write(json.dumps(r,separators=(",",":"),sort_keys=True,default=json_default)+"\n")
 passed=[r for r in output if r["status"]=="PASS_SOURCE_OUTWARD_FIRST_UNION_TARGET"]
 np.savez_compressed(npz,
  pair_index=np.asarray([r["pair_index"] for r in passed],np.int64),
  face_rows=np.asarray([r["face_row"] for r in passed],np.int64),
  face_node_rows=np.asarray([r["skin_node_rows"] for r in passed],np.int64).reshape(-1,3),
  barycentric_weights=np.asarray([r["barycentric_weights"] for r in passed],np.float64).reshape(-1,3),
  inside_sample_world_f32_m=np.asarray([r["inside_sample_world_f32_m"] for r in passed],np.float32).reshape(-1,3),
  bank_body=np.asarray([r["bank_body"] for r in passed],np.int32),
  bank_stable_bone_id=np.asarray([r["bank_stable_bone_id"] for r in passed],np.int32),
  bank_triangle_stable_id=np.asarray([r["bank_triangle_stable_id"] for r in passed],np.int32),
  source_outward_world_unit=np.asarray([r["source_outward_world_unit"] for r in passed],np.float64).reshape(-1,3),
  union_exit_distance_m=np.asarray([r["union_exit_distance_m"] for r in passed],np.float64),
  target_world_f32_m=np.asarray([r["target_world_f32_m"] for r in passed],np.float32).reshape(-1,3),
  target_distance_m=np.asarray([r["target_distance_m"] for r in passed],np.float64),
  target_exactly_outside_all185=np.ones(len(passed),np.bool_))
 summary={"schema":"numi.human.coherent-source-face-first-union-targets.175.v1","classification":"construction_target_proposals_only_no_fit_or_clearance_admission",
  "inputs":{"candidate_sha256":sha(CAND),"surface_report_sha256":sha(SURFACE),"detail_source_sha256":sha(DETAIL),"orientation_report_sha256":sha(ORIENT),"exact_bank_triangle_export_sha256":sha(BANK/"exact-emitted.triangles.csv"),"exact_bank_body_export_sha256":sha(BANK/"exact-emitted.bodies.csv"),"bank_reader_sha256":sha(BANKHELPER),"exact_predicate_sha256":sha(EXACT),"virtual_sample_helper_sha256":sha(CONSTRAINT),"previous_172_samples_sha256":sha(OLD172),"previous_172_summary_sha256":sha(OLD172_SUM)},
  "candidate_points":len(P),"candidate_faces":len(F),"matched_surface_bank_triangle_pairs":len(pairs),"source_face_kind_counts_in_pairs":dict(Counter(int(z["face_kind"][f]) for f,_,_,_ in pairs)),
  "source_winding_rule":{"kind2":"raw detailed face normal times absolute chosen_outward_sign=+1","kind1":"negative current coarse-only face area normal; candidate coarse winding is inward","kind0":"unresolved; no direction guessed","fine_to_coarse_winding_parity":int(orient["component_fine_to_coarse_winding_anchors"]["0"]["chosen_outward_parity"]),"parity_is_not_absolute_outward_sign":True},
  "banks":len(bankkeys),"matched_159_complete":bool(surface.get("complete")),"crossing_pairs":len(pairs),"inside_samples_certified_for_implicated_bank":sum(1 for r in output if r.get("status")!="UNRESOLVED_SOURCE_WINDING" and r.get("inside_sample_world_f32_m") is not None),
  "source_outward_first_union_targets":len(passed),"failure_counts":dict(fail),"initial_inside_union_count_histogram":dict(initial_classes),
  "full_union_endpoint_classification":"Exact point-location for every target against all185 banks; target accepted only if outside every complete bank; event-parity endpoint also required.",
  "limitations":["Source normals establish registered source orientation only; they do not establish anatomical outwardness at every deformed face.","First-union ray targets are construction proposals, not clearance or physical admission.","Kind-0 source faces remain unresolved; no inferred normal is substituted.","No fit or native physical step was run."],
  "elapsed_seconds":time.monotonic()-tstart,"result_rows":len(output),"passing_targets":len(passed),"output_sha256_pending":True}
 reportp.write_text(json.dumps(summary,indent=2,default=json_default)+"\n")
 print(json.dumps({"report":str(reportp),"report_sha256":sha(reportp),"npz":str(npz),"npz_sha256":sha(npz),"rows":len(output),"pairs":len(pairs),"targets":len(passed),"failures":dict(fail),"elapsed_seconds":summary["elapsed_seconds"]},indent=2))
if __name__=="__main__":main()

