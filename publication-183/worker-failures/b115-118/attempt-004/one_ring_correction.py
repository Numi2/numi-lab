#!/usr/bin/env python3
"""One connected one-ring correction on the current 653-parent-pair candidate."""
from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json, sys, time
import numpy as np
from scipy import sparse

S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
W=S/"per-bank-workers-20261027/b115-118"
O=W/"attempt-004/run-001"
CURRENT=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
CURRENT_SHA="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
BASE=S/"edge-refinement-661-threebank-001/generation-002-all-banks-001/candidate.npz"
BASE_SHA="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8"
A159=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json"
A130=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-130/summary.json"
FAILED=S/"per-bank-workers-20261027/b115-118/attempt-003/run-001/exact-incremental.json"
TARGETS=S/"partial-midpoint-continuation-root-001/targets-001/nearest-union-exits.json"
QPATH=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
SOURCE=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz")
SOURCE_SHA="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m
def load(p):
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}
assert not O.exists()
O.mkdir(parents=True)
started=time.monotonic()
assert sha(CURRENT)==CURRENT_SHA and sha(BASE)==BASE_SHA and sha(SOURCE)==SOURCE_SHA
z=load(CURRENT); base=load(BASE)
x=z["points_world_m"]; faces=z["source_O_faces"]
N=len(x); assert x.dtype==np.float32 and faces.shape==(42379,3)
# Candidate compatibility: exact same face rows and all face/source ancestry arrays as the refined baseline.
for key in ("source_O_faces","refinement_face_parent_face_row","refinement_face_parent_barycentric",
            "parent_coarse_face_row","parent_fine_detail_face_row","parent_fine_subface_row",
            "face_source_ancestry_indptr","face_source_ancestry_face_rows",
            "face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",
            "face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows"):
    assert z[key].dtype==base[key].dtype and z[key].shape==base[key].shape and z[key].tobytes()==base[key].tobytes(),key
base_report=json.loads(A159.read_text())
self_report=json.loads(A130.read_text())
assert base_report["candidate_sha256"]==CURRENT_SHA and base_report["complete"] and base_report["bank_count"]==185
assert self_report["candidate_sha256"]==CURRENT_SHA and self_report["complete_surface_audit"]
assert int(self_report["exact_PLC_surface_pairs"])==0
failed=json.loads(FAILED.read_text())
forbid_bank={(int(a),int(b),int(c),int(d)) for a,b,c,d in failed["all_bank"]["new_pairs_skin_face_bone_face"]}
forbid_self={tuple(sorted(map(int,p))) for p in failed["self_surface"]["pairs"]}
assert len(forbid_bank)==2 and len(forbid_self)==18

# One-ring support around the two original-topology faces affected by the failed flip.
focus=np.asarray([31364,31367],dtype=np.int64)
core=np.unique(faces[focus].reshape(-1))
edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0),axis=1),axis=0)
adj=sparse.csr_matrix((np.ones(2*len(edges)),
                       (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                      shape=(N,N))
ring=set(map(int,core))
for row in core:ring.update(map(int,adj[int(row)].indices))
support=np.asarray(sorted(ring),dtype=np.int64)
local={int(row):i for i,row in enumerate(support)}
assert set(map(int,core)).issubset(local)
anchor_rows=z["typed_coarse_anchor_output_rows_018"]
# Target direction comes from row 21140 only after exact bitwise start-point parity with the current candidate.
target_json=json.loads(TARGETS.read_text())
target_record=next(t for t in target_json["targets"] if int(t["query_row"])==21140)
assert target_json["inputs"]["candidate_sha256"]==BASE_SHA
assert np.array_equal(np.asarray(target_record["start_world_f32_m"],dtype=np.float32).view(np.uint32),
                      x[21140].view(np.uint32))
direction=np.asarray(target_record["proposal_direction_world_unit"],dtype=np.float64)
assert abs(float(np.linalg.norm(direction))-1.0)<1e-10
target_core_mm=0.20; target_ring_mm=0.10
soft_target=np.zeros(3*len(support),dtype=np.float64)
core_set=set(map(int,core))
for row in support:
    weight=1.0 if int(row) in core_set else 0.5
    soft_target[3*local[int(row)]:3*local[int(row)]+3]=weight*target_core_mm*direction
# Ensure a zero-motion F32 emission is bitwise identical to the current source.
assert np.array_equal((x.astype(np.float64)+np.zeros(x.shape)).astype(np.float32).view(np.uint32),x.view(np.uint32))

q=mod("quality408_b115118_one_ring",QPATH)
assert sha(QPATH)=="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
area_owner=mod("area_b115118_one_ring",q.DEFAULT_AREA_OWNER)
area_fd=q.verify_area_linearization_layout(area_owner)
with np.load(CURRENT,allow_pickle=False) as archive:
    src,src_receipts=q.compose_source_reference(archive,SOURCE)
srcmm=src*1000.0
bm=x.astype(np.float64)*1000.0
ref=z["area_reference_037_points_world_m_f64"]*1000.0
normals_all,area0,ar,req=q.fixed_reference_area_requirements(bm,ref,faces)

# Build exact-witness regression constraints from the two new bank pairs in the failed flip.
auditmod=mod("audit159_b115118_one_ring",Path("/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py"))
helper=mod("bankhelper_b115118_one_ring",auditmod.BANKHELPER)
E=auditmod.E
bank_dir=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
T,ids,gids,body_rows,M,off,pos,bankkeys,pins,unused=helper.read_bank_inputs(bank_dir,E)
def sat_separator(triA,triB):
    ea=[triA[(i+1)%3]-triA[i] for i in range(3)]
    eb=[triB[(i+1)%3]-triB[i] for i in range(3)]
    axes=[np.cross(ea[0],ea[1]),np.cross(eb[0],eb[1])]
    axes.extend(np.cross(a,b) for a in ea for b in eb)
    best=None
    for axis in axes:
        nrm=float(np.linalg.norm(axis))
        if nrm<1e-12:continue
        n=axis/nrm
        pa=triA@n;pb=triB@n
        gab=float(pb.min()-pa.max()) # A below B
        gba=float(pa.min()-pb.max()) # B below A
        if gab>0 and (best is None or gab>best["gap_mm"]):
            best={"left":"A","n":n.copy(),"gap_mm":gab}
        if gba>0 and (best is None or gba>best["gap_mm"]):
            best={"left":"B","n":n.copy(),"gap_mm":gba}
    return best
pair_rows=[];sep_receipts=[]
# Convert the body 115 / bone 118 exact triangles to the candidate's world frame.
bank_mask=(ids[:,0]==115)&(ids[:,1]==118)
bank_face_ids=ids[bank_mask,3];bank_tri_local=T[bank_mask]
bank_tri_world_mm=((bank_tri_local+off[115])@M[115]+pos[115])*1000.0
for body,bone,sf,bf in sorted(forbid_bank):
    assert (body,bone)==(115,118)
    wi=np.flatnonzero(bank_face_ids==bf);assert len(wi)==1,(bf,len(wi))
    triA=bm[faces[sf]];triB=bank_tri_world_mm[int(wi[0])]
    sep=sat_separator(triA,triB)
    assert sep is not None and sep["gap_mm"]>0,(sf,bf,sep)
    n=sep["n"]
    # Use every vertex-pair inequality along the verified separating axis. Only skin vertices move.
    for va in faces[sf]:
        for vb in triB:
            # Set constraints as n dot (bone point - skin point) >= epsilon if skin is lower,
            # or n dot (skin point - bone point) >= epsilon if bone is lower.
            if sep["left"]=="A":
                coeff=-n; d0=float(np.dot(n,vb-va*0.0)) # replaced below with exact base separation
                base_gap=float(np.dot(n,vb)-np.dot(n,bm[int(va)]))
            else:
                coeff=n
                base_gap=float(np.dot(n,bm[int(va)])-np.dot(n,vb))
            eps=min(0.002,sep["gap_mm"]*0.25)
            pair_rows.append({"nodes":[int(va)],"coefficients":[coeff],
                              "lower":float(eps-base_gap)})
    sep_receipts.append({"kind":"forbidden_bank_pair","typed_bank":[body,bone],
                         "skin_face_row":sf,"bone_face_row":bf,
                         "separating_side":sep["left"],"current_gap_mm":sep["gap_mm"],
                         "margin_mm":min(0.002,sep["gap_mm"]*0.25),
                         "linear_rows":9})

# Add linear non-overlap constraints for failed-flip self pairs whose current triangles
# have no shared topology vertices; shared-vertex pairs remain exact postfit vetoes.
self_pair_linearized=[];self_pair_exact_only=[]
support_set=set(map(int,support))
for fa,fb in sorted(forbid_self):
    va=faces[fa];vb=faces[fb]
    if set(map(int,va))&set(map(int,vb)):
        self_pair_exact_only.append([fa,fb]);continue
    triA=bm[va];triB=bm[vb]
    sep=sat_separator(triA,triB)
    if sep is None or sep["gap_mm"]<=0:
        self_pair_exact_only.append([fa,fb]);continue
    n=sep["n"];eps=min(0.002,sep["gap_mm"]*0.25)
    if sep["left"]=="A":left,right=va,vb
    else:left,right=vb,va
    for lnode in left:
        for rnode in right:
            coeff_nodes=[];coeffs=[]
            if int(lnode) in support_set:
                coeff_nodes.append(int(lnode));coeffs.append(-n)
            if int(rnode) in support_set:
                coeff_nodes.append(int(rnode));coeffs.append(n)
            if not coeff_nodes:continue
            gap=float(np.dot(n,bm[int(rnode)])-np.dot(n,bm[int(lnode)]))
            pair_rows.append({"nodes":coeff_nodes,"coefficients":coeffs,
                              "lower":float(eps-gap)})
    self_pair_linearized.append({"faces":[fa,fb],"gap_mm":sep["gap_mm"],
                                 "margin_mm":eps,"side":sep["left"]})

affected_for_area=np.flatnonzero(np.any(np.isin(faces,support),axis=1))
G,rhs,_,_=area_owner.area_linearization(bm,faces,affected_for_area,
                                         normals_all[affected_for_area],req[affected_for_area],
                                         minimum_ratio=0.1)
Acontact,bcontact=q._owner_matrix(pair_rows,G,rhs,support)
EG,eb,edge_meta=q.build_edge_gradient_constraints(bm,faces,support,0.5)
Aall=sparse.vstack((Acontact,EG),format="csr")
ball=np.r_[bcontact,eb]
lo,hi,source_bounds=q.source_safe_trust_bounds(bm,srcmm,support,anchor_rows,4.0,12.0)
solved=q.solve_sparse_l1(Aall,ball,soft_target,lo,hi,20.0)

report={
 "schema":"numi.human.bank115_118.one_ring_current653_correction.v1",
 "status":None,"current_candidate_path":str(CURRENT),"current_candidate_sha256":CURRENT_SHA,
 "refined_baseline_sha256":BASE_SHA,"current_audit159_sha256":sha(A159),"current_audit130_sha256":sha(A130),
 "failed_flip_incremental_report_sha256":sha(FAILED),"source_reference_sha256":sha(SOURCE),
 "focus_faces":focus.tolist(),"core_nodes":core.tolist(),"one_ring_support_rows":support.tolist(),
 "area_affected_face_count":int(len(affected_for_area)),"target_seed":{"row":21140,"direction_world_unit":direction.tolist(),
 "core_step_mm":target_core_mm,"one_ring_step_mm":target_ring_mm},
 "source_bounds":source_bounds,"area_linearization_fd":area_fd,"edge_gradient":edge_meta,
 "forbidden_new_bank_pairs":[list(p) for p in sorted(forbid_bank)],
 "forbidden_new_self_pairs":[list(p) for p in sorted(forbid_self)],
 "bank_pair_separators":sep_receipts,"self_pair_linearized":self_pair_linearized,
 "self_pair_exact_veto_only":self_pair_exact_only,
 "regression_constraint_count":int(len(pair_rows)),
 "solver":{k:v for k,v in solved.items() if k!="delta"},
 "candidate":None,"candidate_sha256":None,
 "limitations":["No physical or native admission.","Exact incremental postfit checks decide whether this trial is retained or rejected.",
                "Current target direction is reused only after exact start-point parity for row 21140."]}
if not solved["success"]:
    report["status"]="ONE_RING_LP_INFEASIBLE_OR_FAILED"
    report["elapsed_s"]=time.monotonic()-started
    (O/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    raise SystemExit(0)

delta=solved["delta"].reshape(-1,3)
y=x.copy()
y[support]=(x[support].astype(np.float64)+delta/1000.0).astype(np.float32)
actual=(y.astype(np.float64)-x.astype(np.float64))*1000.0
area_gate=q.full_f32_area_gate(y,bm,ref,faces,0.1)
edge_slack=np.asarray(EG@actual[support].reshape(-1)).ravel()-eb
linear_slack=np.asarray(Aall@actual[support].reshape(-1)).ravel()-ball
changed_nodes=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1))
changed_faces=np.flatnonzero(np.any(np.isin(faces,changed_nodes),axis=1))
max_step=float(np.linalg.norm(actual,axis=1).max(initial=0.0))
max_source=float(np.linalg.norm(y.astype(np.float64)*1000.0-srcmm,axis=1).max(initial=0.0))
anchors_ok=bool(np.array_equal(y[anchor_rows].view(np.uint32),x[anchor_rows].view(np.uint32)))
cheap={"area_gate":area_gate,"minimum_edge_slack_mm":float(edge_slack.min(initial=1e30)),
       "minimum_linear_slack":float(linear_slack.min(initial=1e30)),
       "changed_node_count":int(len(changed_nodes)),"changed_nodes":changed_nodes.tolist(),
       "changed_face_count":int(len(changed_faces)),"changed_faces":changed_faces.tolist(),
       "max_step_mm":max_step,"max_source_mm":max_source,"anchors_bitwise_unchanged":anchors_ok}
report["cheap_postfit"]=cheap
cheap_ok=(area_gate["pass"] and cheap["minimum_edge_slack_mm"]>=-1e-4
          and cheap["minimum_linear_slack"]>=-1e-4 and max_step<=4.000001
          and max_source<=12.000001 and anchors_ok)
if not cheap_ok:
    report["status"]="ONE_RING_CHEAP_GATE_FAIL"
    report["elapsed_s"]=time.monotonic()-started
    (O/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
    raise SystemExit(0)

delta_full=np.zeros_like(x,dtype=np.float64)
delta_full[support]=actual[support]/1000.0
receipt={"schema":"numi.human.bank115_118.one_ring_correction_receipt.v1",
         "base_candidate_sha256":CURRENT_SHA,"focus_face_rows":focus.tolist(),
         "core_nodes":core.tolist(),"support_rows":support.tolist(),
         "actual_delta_m_sha256":hashlib.sha256(np.ascontiguousarray(delta_full).view(np.uint8)).hexdigest(),
         "changed_nodes":changed_nodes.tolist(),"changed_faces":changed_faces.tolist(),
         "area_gate_pass":area_gate["pass"],"source_cap_mm":12.0,"trust_mm":4.0,
         "new_bank_pair_regression_constraints":[list(p) for p in sorted(forbid_bank)],
         "new_self_pair_exact_vetoes":[list(p) for p in sorted(forbid_self)]}
payload={k:v.copy() for k,v in z.items()}
payload["points_world_m"]=y
payload["b115118_one_ring_delta_m"]=delta_full
payload["b115118_one_ring_support_rows"]=support
payload["b115118_one_ring_correction_receipt_json"]=np.asarray([json.dumps(receipt,separators=(",",":"))])
candidate=O/"candidate.npz"
with candidate.open("xb") as fp:np.savez_compressed(fp,**payload)
report["candidate"]=str(candidate);report["candidate_sha256"]=sha(candidate)

# Incremental exact bank checks for every affected face against every one of the 185 typed banks.
auditmod=mod("audit159_b115118_final_incremental",Path("/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py"))
exact=auditmod.exact_module()
F=faces
P=y.astype(np.float32).astype(np.float64)
base_pairs=set()
for br in base_report["banks"]:
    if br.get("result") is None:continue
    for sf,bf in br["result"]["triangle_pairs"]:
        base_pairs.add((int(br["body"]),int(br["stable_bone_id"]),int(sf),int(bf)))
local_cache={}
incremental=set();per_bank=[]
for body0,bone0 in bankkeys:
    body,bone=int(body0),int(bone0)
    if body not in local_cache:local_cache[body]=(P-pos[body])@M[body].T-off[body]
    mask=(ids[:,0]==body)&(ids[:,1]==bone);B=T[mask];bid=ids[mask,3]
    if not len(B):
        per_bank.append({"body":body,"stable_bone_id":bone,"new_pair_count":0,"aabb_changed_faces":0,"aabb_bank_faces":0});continue
    ST=local_cache[body][F[changed_faces]]
    blo,bhi=B.min((0,1)),B.max((0,1));slo,shi=ST.min(1),ST.max(1)
    asf=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
    if not len(asf):
        per_bank.append({"body":body,"stable_bone_id":bone,"new_pair_count":0,"aabb_changed_faces":0,"aabb_bank_faces":0});continue
    rlo,rhi=slo[asf].min(0),shi[asf].max(0)
    ab=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
    sr=auditmod.records(ST[asf],changed_faces[asf],exact)
    br=auditmod.records(B[ab],bid[ab],exact)
    result=exact._audit_pair(sr,br,same_surface=False)
    found={(body,bone,int(sf),int(bf)) for sf,bf in result["triangle_pairs"]}
    incremental|=found
    per_bank.append({"body":body,"stable_bone_id":bone,"new_pair_count":int(len(found)),
                     "aabb_changed_faces":int(len(asf)),"aabb_bank_faces":int(len(ab)),
                     "pairs":[[int(sf),int(bf)] for sf,bf in result["triangle_pairs"]]})
post_pairs={p for p in base_pairs if p[2] not in set(map(int,changed_faces))}|incremental
new_pairs=sorted(post_pairs-base_pairs);removed_pairs=sorted(base_pairs-post_pairs)
bank_key=(115,118)
bank_before=sum(p[:2]==bank_key for p in base_pairs)
bank_after=sum(p[:2]==bank_key for p in post_pairs)
# Exact incremental self check: changed faces against all untouched faces with overlapping AABBs and each other.
Ftri=P[F];new_tri=Ftri[changed_faces]
rest_rows=np.setdiff1d(np.arange(len(F),dtype=np.int64),changed_faces,assume_unique=True)
rest_tri=Ftri[rest_rows]
nlo,nhi=new_tri.min(1),new_tri.max(1);rlo,rhi=rest_tri.min(1),rest_tri.max(1)
active=np.zeros(len(rest_rows),dtype=bool)
for l,h in zip(nlo,nhi):active|=np.all(rhi>=l,axis=1)&np.all(rlo<=h,axis=1)
active_rows=rest_rows[active]
newrec=auditmod.records(new_tri,changed_faces,exact)
restrec=auditmod.records(Ftri[active_rows],active_rows,exact)
self_vs=exact._audit_pair(newrec,restrec,same_surface=True)
self_mut=exact._audit_pair(newrec,newrec,same_surface=True)
self_pairs={tuple(sorted(map(int,p))) for p in self_vs["triangle_pairs"]+self_mut["triangle_pairs"]}
forbidden_after=sorted(forbid_bank&post_pairs)
forbidden_self_after=sorted(forbid_self&self_pairs)
report["exact_incremental"]={
 "typed_banks_checked":int(len(bankkeys)),"baseline_child_pair_count":int(len(base_pairs)),
 "candidate_child_pair_count":int(len(post_pairs)),"new_pair_count":int(len(new_pairs)),
 "removed_pair_count":int(len(removed_pairs)),
 "new_pairs_skin_face_bone_face":[list(p) for p in new_pairs],
 "removed_pairs_skin_face_bone_face":[list(p) for p in removed_pairs],
 "bank_115_118_before_after":[int(bank_before),int(bank_after)],
 "forbidden_failed_flip_bank_pairs_present":[list(p) for p in forbidden_after],
 "no_new_bank_pairs":bool(not new_pairs),
 "per_bank_results":per_bank}
report["exact_incremental_self"]={
 "baseline_exact_self_pair_count":int(self_report["exact_PLC_surface_pairs"]),
 "changed_vs_unchanged_aabb_candidates":int(len(active_rows)),
 "changed_vs_unchanged_pair_count":int(self_vs["count"]),
 "changed_vs_unchanged_allowed_shared":int(self_vs["allowed_shared_vertex_or_edge_pairs"]),
 "changed_mutual_pair_count":int(self_mut["count"]),
 "changed_mutual_allowed_shared":int(self_mut["allowed_shared_vertex_or_edge_pairs"]),
 "new_self_pair_count":int(len(self_pairs)),
 "new_self_pairs":[list(p) for p in sorted(self_pairs)],
 "forbidden_failed_flip_self_pairs_present":[list(p) for p in forbidden_self_after],
 "no_new_self_pairs":bool(not self_pairs)}
improved=bank_after<bank_before
if not new_pairs and not self_pairs and improved:
    report["status"]="ONE_RING_EXACT_NO_WORSENING_AND_BANK_REDUCTION"
elif not new_pairs and not self_pairs:
    report["status"]="ONE_RING_EXACT_NO_REGRESSION_NO_BANK_REDUCTION"
else:
    report["status"]="REJECTED_EXACT_REGRESSION"
report["elapsed_s"]=time.monotonic()-started
(O/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:report[k] for k in ("status","current_candidate_sha256","candidate_sha256","cheap_postfit","exact_incremental","exact_incremental_self","elapsed_s")},indent=2,allow_nan=False),flush=True)

