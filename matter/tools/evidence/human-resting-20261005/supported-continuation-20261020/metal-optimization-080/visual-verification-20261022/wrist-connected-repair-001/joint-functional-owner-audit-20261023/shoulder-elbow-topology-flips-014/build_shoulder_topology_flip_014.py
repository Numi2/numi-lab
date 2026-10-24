#!/usr/bin/env python3
import sys, json, hashlib, time, runpy, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
IN=E/"joint-functional-owner-audit-20261023"
BASE=IN/"shoulder-elbow-topology-flips-013/candidate.nhskin"
PATCH=IN/"shoulder-elbow-witness-patch-input-001.npz"
OLD=E/"candidate-skin-registration-owner-081/080-all-joint-pose-audit.json"
IDENT=E/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json"
OUT=IN/"shoulder-elbow-topology-flips-014"
AUD=runpy.run_path("/tmp/audit_shoulder_010.py")
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
RB=runpy.run_path(str(ROOT/"tools/rebind_human_skin_reference.py"))
METRIC_PATH=E/"right-commonfield-layer-reconstruction-202/tools/human_skin_patch_geometry.py"
import importlib.util
spec=importlib.util.spec_from_file_location("human_skin_patch_geometry",METRIC_PATH)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
PATCH_SHA=""
PREVANC=IN/"shoulder-elbow-topology-flips-013/topology-flip-ancestry.npz"
BASE_SHA="01a4b88fc74013c5f9c2f668cabd7d13ef58f25f52c4fac9571620d7e2fb3eba"
PREVANC=IN/"shoulder-elbow-topology-flips-013/topology-flip-ancestry.npz"
BASE_SHA="01a4b88fc74013c5f9c2f668cabd7d13ef58f25f52c4fac9571620d7e2fb3eba"
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def oriented_rotation(tri,u,v):
 t=list(map(int,tri))
 for i in range(3):
  if t[i]==u and t[(i+1)%3]==v:return [u,v,t[(i+2)%3]]
 raise ValueError("directed edge not in triangle")
def quality(X,T):
 p=np.asarray(X[T],float)
 cr=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]); area2=np.linalg.norm(cr,axis=1)
 l2=(np.sum((p[:,1]-p[:,0])**2,axis=1)+np.sum((p[:,2]-p[:,1])**2,axis=1)+np.sum((p[:,0]-p[:,2])**2,axis=1))
 return np.divide(np.sqrt(3.0)*area2,l2,out=np.zeros_like(area2),where=l2>0)
def directed_edge_rotation(tri,edge):
 wanted=set(map(int,edge));t=list(map(int,tri))
 for i in range(3):
  a,b,c=t[i],t[(i+1)%3],t[(i+2)%3]
  if {a,b}==wanted:return a,b,c
 raise ValueError("edge not in triangle")
def new_triangles(F,fa,fb,edge):
 u,v,c=directed_edge_rotation(F[fa],edge)
 x,y,d=directed_edge_rotation(F[fb],edge)
 if (x,y)!=(v,u):raise ValueError("adjacent faces do not traverse shared edge oppositely")
 # New diagonal c-d, oriented consistently with the original shared-edge orientation.
 t0=np.array([c,u,d],dtype=np.int64)
 t1=np.array([c,d,v],dtype=np.int64)
 return t0,t1

def pair_is_crossing(a,b):
 hits=AUD["triangle_intersection_points"](a[0],b[0])
 if not hits:return False
 shared=set(a[4]) & set(b[4])
 common={a[0][a[4].index(k)] for k in shared}
 if len(shared) in (1,2) and all(AUD["_allowed_shared_point"](x,common) for x in hits):return False
 return True

def changed_row_crossings(ctx,Fcand,changed_faces):
 qmap=ctx["qmap"];qF=ctx["qF"].copy();changed_faces=np.asarray(changed_faces,dtype=np.int64)
 qF[changed_faces]=qmap[Fcand[changed_faces]]
 local,deg=AUD["records_nonzero"](ctx["pts"],qF,changed_faces)
 if deg:raise ValueError("candidate introduces degenerate face")
 changedset=set(map(int,changed_faces))
 unchanged=[r for r in ctx["all_records"] if int(r[3]) not in changedset]
 pairs=set()
 for a,b in AUD["_aabb_candidate_pairs"](local,unchanged,same_surface=False):
  fa,fb=int(a[3]),int(b[3]);key=(min(fa,fb),max(fa,fb))
  if pair_is_crossing(a,b):pairs.add(key)
 for a,b in itertools.combinations(local,2):
  fa,fb=int(a[3]),int(b[3]);key=(min(fa,fb),max(fa,fb))
  if pair_is_crossing(a,b):pairs.add(key)
 return pairs

def main():
 global PATCH_SHA
 if sha(BASE)!=BASE_SHA:raise SystemExit("topology-013 baseline hash mismatch")
 if not PREVANC.exists():raise SystemExit("missing prior flip ancestry")
 PATCH_SHA=sha(PATCH)
 base=decode_payload(BASE.read_bytes());p=RB["decode"](BASE.read_bytes())
 F=np.asarray(base["indices"],dtype=np.int64).reshape(-1,3)
 if p["nv"]!=61950 or F.shape!=(108987,3):raise SystemExit("unexpected source128 topology")
 prevz=np.load(PREVANC)
 prior_modified=set(map(int,prevz["modified_face_rows"]))
 z=np.load(PATCH)
 patch_faces=np.asarray(z["source_face_rows"],dtype=np.int64)
 patch_vertices=np.asarray(z["source_vertex_ids"],dtype=np.int64)
 fixed_edges=np.asarray(z["fixed_outer_boundary_edges"],dtype=np.int64)
 fixed_edge_set={tuple(sorted(map(int,e))) for e in fixed_edges}
 if len(patch_faces)!=357 or len(set(map(int,patch_faces)))!=357:raise SystemExit("unexpected patch/collar face map")
 if patch_faces.min()<0 or patch_faces.max()>=len(F):raise SystemExit("patch face row out of range")
 pose_meta=json.loads(OLD.read_text());identity=json.loads(IDENT.read_text())
 restpose={"source_poses":[{"core_body":int(r["body_index"]),"position":r["rest"]["position_world_m"],"quaternion_xyzw":r["rest"]["orientation_world_xyzw"]} for r in identity["all_bodies"]]}
 pose_by_name={}
 for item in pose_meta["results"]:
  if item["name"] in ("shoulder_elevation_bilateral_q36q74_0p5","shoulder_plane_bilateral_q35q73_m0p3_at_elevation0p5","elbow_bilateral_q39q77_0p8"):
   pp=Path(item["pose"]["path"])
   pose_by_name[item["name"]]=json.loads(pp.read_text())
 if len(pose_by_name)!=3:raise SystemExit("required pose fixtures absent")
 pose_names=sorted(pose_by_name)
 def getpose(pose):
  A,B=RB["reference_transforms"](p,pose,np.zeros(3))
  return RB["positions"](p,A,B).astype(np.float32)
 Xrest=getpose(restpose)
 Xposes={name:getpose(pose_by_name[name]) for name in pose_names}
 Xall={"native_rest":Xrest,**Xposes}
 allsets={};contexts={};baseline_quality={}
 patchset=set(map(int,patch_faces))
 for name,X in [("native_rest",Xrest),*Xposes.items()]:
  qv,qf,qmap=coordinate_quotient(X.astype(np.float32).tolist(),F.tolist())
  qF=np.asarray(qf,dtype=np.int64);pts=AUD["exact_points"](np.asarray(qv,dtype=np.float64))
  rec,deg=AUD["records_nonzero"](pts,qF,np.arange(len(F),dtype=np.int64))
  if deg:raise SystemExit(f"baseline degenerate triangles {name}: {deg}")
  patchrec,_=AUD["records_nonzero"](pts,qF,patch_faces)
  baseline=AUD["crossing_pairs"](X,F,patch_faces)
  pairs={(int(a["face_a"]),int(a["face_b"])) for a in baseline["pairs"]}
  allsets[name]=pairs
  contexts[name]={"pts":pts,"qF":qF,"qmap":np.asarray(qmap,dtype=np.int64),"all_records":rec}
  baseline_quality[name]=quality(X,F[patch_faces])
  print(json.dumps({"event":"baseline","pose":name,"patch_crossings":len(pairs),"patch_faces":len(patch_faces),"min_quality":float(baseline_quality[name].min())}),flush=True)
 # Only two-sided patch interior edges, not fixed boundary cuts.
 edge_faces=defaultdict(list)
 for fr in patch_faces:
  tri=F[int(fr)]
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   edge_faces[tuple(sorted((int(a),int(b))))].append(int(fr))
 all_edge_faces=defaultdict(list)
 for fr,tri in enumerate(F):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   all_edge_faces[tuple(sorted((int(a),int(b))))].append(fr)
 crossing_faces=set()
 for pairs in allsets.values():
  for a,b in pairs:
   if a in patchset:crossing_faces.add(a)
   if b in patchset:crossing_faces.add(b)
 low_quality_faces=set()
 for name,q in baseline_quality.items():
  ids=np.argsort(q)[:20]
  low_quality_faces.update(int(patch_faces[i]) for i in ids)
 focus_faces=crossing_faces|low_quality_faces
 existing_edges=set(all_edge_faces)
 candidates=[]
 rejected=defaultdict(int)
 for edge,fs in edge_faces.items():
  if len(fs)!=2:rejected["patch_edge_not_two_faces"]+=1;continue
  fa,fb=fs
  if all_edge_faces[edge]!=fs and set(all_edge_faces[edge])!=set(fs):rejected["nonlocal_edge_incidence"]+=1;continue
  if edge in fixed_edge_set:rejected["fixed_boundary_edge"]+=1;continue
  if fa not in patchset or fb not in patchset:rejected["face_outside_patch_collar"]+=1;continue
  if {fa,fb}&prior_modified:rejected["touches_prior_topology_edit"]+=1;continue
  if not ({fa,fb}&focus_faces):rejected["not_touching_crossing_or_worst_quality_face"]+=1;continue
  verts=set(map(int,F[fa]))|set(map(int,F[fb]))
  if len(verts)!=4:rejected["not_quad"]+=1;continue
  t0,t1=new_triangles(F,fa,fb,edge)
  newedge=tuple(sorted((int(next(v for v in t0 if v not in edge)),int(next(v for v in t1 if v not in edge)))))
  # The new diagonal joins the two opposite vertices; this expression is also independently checked below.
  opp=[v for v in verts if v not in edge]
  newedge=tuple(sorted(map(int,opp)))
  if newedge in existing_edges:rejected["diagonal_already_exists"]+=1;continue
  if any(len(set(map(int,t)))<3 for t in (t0,t1)):rejected["degenerate_indices"]+=1;continue
  if any(tuple(sorted(map(int,t))) in {tuple(sorted(map(int,F[i]))) for i in range(len(F)) if i not in (fa,fb)} for t in (t0,t1)):
   rejected["duplicate_existing_face"]+=1;continue
  faceids=np.array([fa,fb],dtype=np.int64)
  replacement=np.stack([t0,t1])
  quality_delta={};min_quality_gain=float("inf");quality_ok=True;min_area_ratio=float("inf");min_cosine=float("inf")
  for name,X in [("native_rest",Xrest),*Xposes.items()]:
   oldq=quality(X,F[faceids]);newq=quality(X,replacement)
   gain=float(newq.min()-oldq.min());quality_delta[name]={"old_min":float(oldq.min()),"new_min":float(newq.min()),"gain":gain}
   min_quality_gain=min(min_quality_gain,gain)
   tri0=X[replacement].astype(np.float64)
   tri_ref=Xrest[replacement].astype(np.float64)
   ar,co,ori=M.triangle_deformation_metrics(tri_ref,tri0)
   min_area_ratio=min(min_area_ratio,float(ar.min()));min_cosine=min(min_cosine,float(co.min()))
   if np.any(ar<=0) or np.any(co<=0):quality_ok=False
   if gain < -1e-8:quality_ok=False
  if not quality_ok:rejected["pose_area_orientation_or_quality"]+=1;continue
  candidates.append({"edge":edge,"faces":[fa,fb],"new_faces":replacement.tolist(),"new_diagonal":newedge,"quality_delta":quality_delta,"worst_quality_gain":min_quality_gain,"minimum_triangle_area_ratio":min_area_ratio,"minimum_normal_cosine":min_cosine})
 # prioritize balanced quality gains but test all admissible candidates (typically tens, bounded by patch).
 candidates.sort(key=lambda x:(x["worst_quality_gain"],x["minimum_triangle_area_ratio"],x["minimum_normal_cosine"]),reverse=True)
 print(json.dumps({"event":"flip_candidates","count":len(candidates),"rejected":dict(rejected),"crossing_faces":len(crossing_faces)}),flush=True)
 # Run exact local-to-global Float32 crossing comparisons for each candidate.
 evaluated=[]
 for ci,c in enumerate(candidates):
  Ftest=F.copy();ids=np.asarray(c["faces"],dtype=np.int64);Ftest[ids]=np.asarray(c["new_faces"],dtype=np.int64)
  poseeval={};zero_new=True;resolved_total=0
  for name,X in Xall.items():
   new_local=changed_row_crossings(contexts[name],Ftest,ids)
   old_local={pair for pair in allsets[name] if pair[0] in set(ids.tolist()) or pair[1] in set(ids.tolist())}
   resolved=old_local-new_local;new=new_local-old_local
   zero_new &= not bool(new);resolved_total+=len(resolved)
   poseeval[name]={"old_local":len(old_local),"candidate_local":len(new_local),"resolved":sorted(map(list,resolved)),"new":sorted(map(list,new))}
  c2={**c,"pose_crossings":poseeval,"resolved_total":resolved_total,"no_new_crossings":bool(zero_new)}
  evaluated.append(c2)
  if ci%20==0 or ci==len(candidates)-1:
   print(json.dumps({"event":"flip_eval_progress","done":ci+1,"total":len(candidates),"resolving":sum(x["resolved_total"]>0 and x["no_new_crossings"] for x in evaluated)}),flush=True)
 # Greedy non-adjacent batch with incremental full local-to-global exact checks.
 # Each proposal checks interactions among modified faces against every unmodified face.
 good=[c for c in evaluated if c["resolved_total"]>0]
 good.sort(key=lambda x:(bool(set(x["faces"])&low_quality_faces),x["worst_quality_gain"],x["resolved_total"],x["minimum_triangle_area_ratio"]),reverse=True)
 chosen=[];usedfaces=set();usedverts=set();Fout=F.copy();modified=[]
 for c in good:
  ids=np.asarray(c["faces"],dtype=np.int64)
  vs=set(map(int,F[ids].ravel()))
  if vs&usedverts or set(map(int,ids))&usedfaces:continue
  trial=Fout.copy();trial[ids]=np.asarray(c["new_faces"],dtype=np.int64)
  allchanged=np.asarray(modified+ids.tolist(),dtype=np.int64)
  no_new=True;resolved_total=0
  for name,X in Xall.items():
   newlocal=changed_row_crossings(contexts[name],trial,allchanged)
   changedset=set(map(int,allchanged))
   oldlocal={pair for pair in allsets[name] if pair[0] in changedset or pair[1] in changedset}
   resolved_total+=len(oldlocal-newlocal)
   if newlocal-oldlocal:no_new=False;break
  if not no_new or resolved_total==0:continue
  chosen.append(c);Fout=trial;usedverts|=vs;usedfaces|=set(map(int,ids));modified.extend(map(int,ids))
  if len(chosen)>=12:break
 if not chosen:raise SystemExit("no non-adjacent exact-safe flip batch with crossing reduction found")
 parents=[]
 for c in chosen:
  for _ in c["faces"]:parents.append([int(c["faces"][0]),int(c["faces"][1])])
 # Exact batch gate: full patch+collar against entire surface for all recorded poses.
 batch_pose={};batch_pass=True
 for name,X in Xall.items():
  r0=AUD["crossing_pairs"](X,F,patch_faces);r1=AUD["crossing_pairs"](X,Fout,patch_faces)
  p0={(int(a["face_a"]),int(a["face_b"])) for a in r0["pairs"]};p1={(int(a["face_a"]),int(a["face_b"])) for a in r1["pairs"]}
  new=sorted(p1-p0);resolved=sorted(p0-p1);retained=sorted(p0&p1)
  batch_pass &= len(new)==0 and (name=="native_rest" or len(p1)<len(p0))
  batch_pose[name]={"baseline_crossings":len(p0),"candidate_crossings":len(p1),"new_crossings":len(new),"resolved_crossings":len(resolved),"retained_crossings":len(retained),"new_pair_ids":[list(x) for x in new],"resolved_pair_ids":[list(x) for x in resolved],"retained_pair_ids":[list(x) for x in retained],"candidate_exact_report":r1}
 # Emitted native-rest triangle orientation + explicit source-face ancestry.
 rest_orientation={}
 for name,X in [("native_rest",Xrest),*Xposes.items()]:
  t=Fout[np.asarray(modified,dtype=np.int64)]
  ref=Xrest[t].astype(np.float64);cur=X[t].astype(np.float64)
  ar,co,ori=M.triangle_deformation_metrics(ref,cur)
  rest_orientation[name]={"modified_faces":len(modified),"min_area_ratio_vs_native_rest":float(ar.min()),"min_normal_cosine_vs_native_rest":float(co.min()),"min_oriented_ratio_vs_native_rest":float(ori.min()),"inverted_or_degenerate":int(np.count_nonzero((ori<=0)|(ar<=0)))}
  if name!="native_rest" and rest_orientation[name]["inverted_or_degenerate"]:batch_pass=False
 # Save candidate payload with same vertex records/weights/bindings, changed indices only.
 raw=bytearray(p["raw"])
 raw[p["indexoff"]:p["indexoff"]+p["ni"]*4]=Fout.astype("<u4").tobytes()
 candbytes=bytes(raw)
 outpayload=OUT/"candidate.nhskin";OUT.mkdir(parents=True,exist_ok=False);outpayload.write_bytes(candbytes)
 ancestry=OUT/"topology-flip-ancestry.npz"
 new_parent_map={int(fr):list(map(int,par)) for fr,par in zip(modified,parents)}
 old_rows=np.asarray(prevz["modified_face_rows"],dtype=np.int64)
 old_parents=np.asarray(prevz["parent_source_face_rows"],dtype=np.int64)
 all_rows=np.concatenate([old_rows,np.asarray(modified,dtype=np.int64)])
 all_parents=np.vstack([old_parents,np.asarray(parents,dtype=np.int64)])
 all_faces=Fout[all_rows]
 np.savez_compressed(ancestry,modified_face_rows=all_rows,parent_source_face_rows=all_parents,final_faces_for_modified_rows=all_faces,previous_ancestry_sha256=np.frombuffer(bytes.fromhex(sha(PREVANC)),dtype=np.uint8),flipped_edges=np.concatenate([np.asarray(prevz["flipped_edges"],dtype=np.int64),np.asarray([c["edge"] for c in chosen],dtype=np.int64)]),flipped_face_rows=np.concatenate([np.asarray(prevz["flipped_face_rows"],dtype=np.int64),np.asarray([c["faces"] for c in chosen],dtype=np.int64)]),replacement_faces=all_faces.copy(),source_vertex_identity=np.arange(p["nv"],dtype=np.int64))
 report={"schema":"numi.human.shoulder-elbow-local-surface-edge-flip-candidate.v1","scope":"Topology-only same-vertex candidate on source128. Source XYZ, vertex records, donor weights, bindings, and vertex IDs are byte-identical; only selected local face rows change. No physical/native qualification.","inputs":{"base_path":str(BASE),"base_sha256":sha(BASE),"original_source128_sha256":"4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5","previous_flip_ancestry_path":str(PREVANC),"previous_flip_ancestry_sha256":sha(PREVANC),"patch_path":str(PATCH),"patch_sha256":PATCH_SHA,"pose_manifest":str(OLD),"pose_manifest_sha256":sha(OLD),"native_rest_snapshot":str(IDENT),"native_rest_snapshot_sha256":sha(IDENT),"face_count":len(F),"vertex_count":p["nv"],"patch_and_collar_face_count":len(patch_faces),"fixed_outer_boundary_edge_count":len(fixed_edges),"shared_triangle_metric_helper":str(METRIC_PATH),"shared_triangle_metric_helper_sha256":sha(METRIC_PATH)},"search":{"eligible_flip_candidates":len(candidates),"exact_candidates_evaluated":len(evaluated),"selected_flip_count":len(chosen),"cumulative_modified_face_count":len(all_rows),"selection":"second-round vertex-disjoint same-vertex flips on unmodified faces, prioritized at worst-quality faces then crossing witnesses; all candidate pairs rechecked against the full surface in native rest and three poses","candidate_rejections":dict(rejected),"selected":chosen},"candidate":{"path":str(outpayload),"sha256":sha(outpayload),"ancestry_path":str(ancestry),"ancestry_sha256":sha(ancestry),"modified_face_rows":all_rows.astype(int).tolist(),"newly_modified_face_rows":modified,"unchanged_face_rows_exact":True,"source_vertex_records_exact":p["nv"],"bindings_weights_and_positions_byte_identical":True,"candidate_vs_base_face_rows_changed":len(all_rows),"orientation_metrics":rest_orientation},"exact_gate":{"all_patch_collar_faces_checked_against_full_surface":True,"baseline_is_candidate_013":True,"poses":batch_pose,"all_poses_no_new_and_three_poses_strictly_fewer_crossings":bool(batch_pass)},"overall_pass":bool(batch_pass and len(chosen)>0),"interpretation":"A pass only establishes improved exact Float32 self-intersection counts for this local patch in the three pinned static LBS poses, with positive orientation and no changed vertex records. It does not establish bone clearance, accepted FEM state, general ROM, or physical qualification."}
 rp=OUT/"candidate-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"event":"complete","pass":report["overall_pass"],"out":str(OUT),"payload_sha256":sha(outpayload),"report_sha256":sha(rp),"selected_flip_count":len(chosen),"pose_counts":{n:{k:v[k] for k in ("baseline_crossings","candidate_crossings","new_crossings","resolved_crossings","retained_crossings")} for n,v in batch_pose.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
