#!/usr/bin/env python3
"""Freeze 037 restart inputs/diagnostics without launching an optimizer."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
from fit_support_037 import (expand_one_ring, discover_quality_rows, layer_floor_mm,
                             parse_all_bank_csv, transform_inside_constraints)

E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"bone-clearance-fit-037-preparation-035"
OUT.mkdir(exist_ok=True)
AP=E/"bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
PP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
CAND=E/"bone-clearance-fit-034/candidate.npz"
CKPT=E/"bone-clearance-fit-034/solver-checkpoint.npz"
PROG=E/"bone-clearance-fit-034/fit-report.json"
DONOR=E/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/donor-frames-fullweights.csv"
POSE=E/"native-candidate020-bone-witness-029/source-preflight-002/registered-skin-poses.json"
BONE_CSV=E/"native-candidate020-all-bank-containment-035/source-preflight-001/all-bank-containment.csv"
Q_FLOOR=1.25e-5
RESTART_FLOOR=Q_FLOOR*1.02

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qmetric(X,T):
    V=np.asarray(X,dtype=np.float64)[T]
    det=np.einsum("ij,ij->i",V[:,1]-V[:,0],np.cross(V[:,2]-V[:,0],V[:,3]-V[:,0]))
    pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edges=np.stack([np.linalg.norm(V[:,i]-V[:,j],axis=1) for i,j in pairs],axis=1)
    return det/np.maximum(edges.max(axis=1)**3,1e-30)
def stats(x):
    x=np.asarray(x,float)
    return {"min":float(np.min(x)),"p01":float(np.percentile(x,1)),"median":float(np.median(x)),"p99":float(np.percentile(x,99)),"max":float(np.max(x))}

A=np.load(AP); P=np.load(PP); C=np.load(CAND); K=np.load(CKPT)
X0=np.asarray(A["points"],dtype=np.float32).astype(np.float64)
T=np.asarray(A["tetrahedra"],dtype=np.int64); mat=np.asarray(P["material_ids"],dtype=np.int64)
assert np.array_equal(T,np.asarray(P["tetrahedra"],dtype=np.int64))
assert np.array_equal(T,np.asarray(C["tetrahedra"],dtype=np.int64))
assert np.array_equal(mat,np.asarray(C["material_ids"],dtype=np.int64))
Xf=np.asarray(C["points_world_f32"],dtype=np.float32).astype(np.float64)
old_Vrows=np.asarray(C["variable_local_rows"],dtype=np.int64)
old_x_mm=np.asarray(K["x_mm"],dtype=np.float64).reshape(-1,3)
assert len(old_Vrows)==842 and old_x_mm.shape==(len(old_Vrows),3) and np.all(np.isfinite(old_x_mm))

# Replace the old 138-node nearest-witness list with the corrected per-bone 035 capture.
inside_rows, unresolved_rows, capture_summary=parse_all_bank_csv(BONE_CSV,node_count=len(X0))
assert not unresolved_rows, "035 contains ambiguous/invalid inside rows; inspect before fit"
pose_doc=json.loads(POSE.read_text())
poses={int(r["body_index"]):r for r in pose_doc["all_bodies"]}
constraint=transform_inside_constraints(inside_rows,X0,poses)
hnode=constraint["constraint_local_node"]
hbody=constraint["constraint_body_index"]
hbone=constraint["constraint_bone_stable_id"]
htri=constraint["constraint_nearest_triangle"]
hdepth=constraint["constraint_inside_depth_m"]
hclosest=constraint["constraint_closest_world_m"]
hnormal=constraint["constraint_outward_normal_world"]
hlay=hnode//2411; hcol=hnode%2411
OF=np.asarray(P["O_faces"],dtype=np.int64)
ring=expand_one_ring(hcol,OF)
seed_nodes=hnode.copy();seed_cols=np.unique(hcol)
ovars=np.unique(hnode[hlay==0])
Vrows=np.unique(np.concatenate([ovars,2411+ring,4822+ring])).astype(np.int64)
assert len(ring)==423 and len(Vrows)==858
Ttouch=np.flatnonzero(np.isin(T,Vrows).any(axis=1))
assert len(Ttouch)==7060
# Warm-start overlapping variables from the retained 034 iterate; newly exposed ring rows start at baseline.
x_mm=X0[Vrows].astype(np.float64)*1000.0
old_lookup={int(r):j for j,r in enumerate(old_Vrows)}
for j,row in enumerate(Vrows):
    if int(row) in old_lookup:
        x_mm[j]=old_x_mm[old_lookup[int(row)]]
Xrestart=X0.copy();Xrestart[Vrows]=x_mm/1000.0
qrestart=qmetric(Xrestart,T)[Ttouch]
qf=qmetric(Xf,T)[Ttouch]
low_global=discover_quality_rows(Ttouch,qrestart,qf,RESTART_FLOOR)
low_double=Ttouch[qrestart<RESTART_FLOOR]
low_f32=Ttouch[qf<RESTART_FLOOR]
bad_global=Ttouch[qf<=0]
assert set(low_global.tolist())==set(low_double.tolist())|set(low_f32.tolist())
varset=set(map(int,Vrows))
# Freeze pairwise plane inputs so the per-bone correspondence is independently replayable.
np.savez_compressed(OUT/"per-bone-constraints-037.npz",constraint_local_node=hnode,
                    constraint_body_index=hbody,constraint_bone_stable_id=hbone,
                    constraint_nearest_triangle=htri,constraint_inside_depth_m=hdepth,
                    constraint_closest_world_m=hclosest,constraint_outward_normal_world=hnormal,
                    unresolved_csv_line=np.asarray([r["line"] for r in unresolved_rows],dtype=np.int64))
rows=[]
for r in bad_global:
    nodes=T[r];free=[int(int(n) in varset) for n in nodes]
    cols=np.unique(nodes%2411)
    rows.append({"tet_row":int(r),"material_id":int(mat[r]),"q_emitted_f32":float(qf[np.searchsorted(Ttouch,r)]),
                 "restart_free_node_count":int(sum(free)),"fixed_node_count":int(4-sum(free)),
                 "node_rows":nodes.astype(int).tolist(),"layers":(nodes//2411).astype(int).tolist(),
                 "columns":cols.astype(int).tolist()})
assert min(x["restart_free_node_count"] for x in rows)>=1

# Donor IDs are carried by the pinned D37 source table; semantic labels are not in this CSV.
import csv
with open(DONOR,newline="") as f: donorrows=list(csv.DictReader(f))
assert len(donorrows)==2411
donor_by_col=np.asarray([int(d["donor"]) for d in donorrows],dtype=np.int64)
for record in rows:
    node=T[record["tet_row"]]
    record["donor_body_ids_by_column"]=[int(donor_by_col[c]) for c in record["columns"]]

# Correct physical units: baseline layer norms are in metres, SLSQP coordinates in mm.
O,I,D=X0[:2411],X0[2411:4822],X0[4822:]
sv=O[ring]-I[ring]; fv=I[ring]-D[ring]
sl=np.linalg.norm(sv,axis=1); fl=np.linalg.norm(fv,axis=1)
sfloor=layer_floor_mm(sl); ffloor=layer_floor_mm(fl)
old_sf=np.minimum(.2*sl,.10);old_ff=np.minimum(.2*fl,.10) # old script mixed m inputs with mm constraint values
sn=sv/sl[:,None];fn=fv/fl[:,None]
sproj=1000.0*np.einsum("ij,ij->i",Xf[ring]-Xf[2411+ring],sn)
fproj=1000.0*np.einsum("ij,ij->i",Xf[2411+ring]-Xf[4822+ring],fn)

rep034=json.loads(PROG.read_text())
min_gap=float(rep034["checks"]["bone_plane_min_gap_m"]); target=float(rep034["fit"]["bone_margin_m"]); tol=2e-7
# 034 checkpoint is the required restart iterate; its stored active rows only include initial seed cells.
np.savez_compressed(OUT/"restart-state-037.npz",x_mm=x_mm,variable_local_rows=Vrows,
                    affected_tet_rows=Ttouch,initial_active_tet_rows=low_global,
                    restart_quality_float64=qrestart.astype(np.float64),
                    restart_quality_emitted_float32=qf.astype(np.float64))
report={
 "schema":"numi.human.d37-bone-clearance-fit-037-preparation.v1",
 "status":"prepared_no_fit_launched_waiting_for_corrected_035_per_bone_capture",
 "pins_sha256":{"033_combined_constraint_npz":sha(AP),"D37_physical_topology_npz":sha(PP),
                 "034_candidate_npz":sha(CAND),"034_solver_checkpoint_npz":sha(CKPT),
                 "034_report_json":sha(PROG),"D37_donor_frame_csv":sha(DONOR),
                 "035_all_bank_capture_csv":sha(BONE_CSV),"registered_pose_json":sha(POSE)},
 "restart":{"source":"034 solver-checkpoint.npz x_mm, same 842 variable local rows",
            "affected_tets":len(Ttouch),"old_034_variable_rows":int(len(old_Vrows)),
            "new_variable_rows":int(len(Vrows)),"old_034_ring_columns":415,"new_035_ring_columns":int(len(ring)),
            "initially_active_seed_rows_in_034":int(len(K["active_tet_rows"])),
            "active_rows_required_from_full_restart_scan":int(len(low_global)),
            "active_tet_rows_global":low_global.astype(int).tolist(),
            "below_q_target_double_rows":low_double.astype(int).tolist(),
            "below_q_target_emitted_f32_rows":low_f32.astype(int).tolist(),
            "restart_threshold":RESTART_FLOOR,"quality_floor":Q_FLOOR,
            "full_scan_checked_all_affected_rows":True,
            "nonpositive_restart_double":int(np.sum(qrestart<=0)),
            "nonpositive_emitted_f32":int(np.sum(qf<=0)),
            "candidate_f32_rows_crossing_1_275e-5_due_to_rounding":sorted(set(low_double.tolist())-set(low_f32.tolist()))},
 "inverted_cell_support":{"count":len(rows),"material_counts":{"1":int(sum(r["material_id"]==1 for r in rows)),"2":int(sum(r["material_id"]==2 for r in rows))},
                          "fixed_only_cells":int(sum(r["restart_free_node_count"]==0 for r in rows)),
                          "minimum_free_nodes_per_bad_cell":min(r["restart_free_node_count"] for r in rows),
                          "bad_cells":rows,
                          "interpretation":"Every current negative cell has at least one variable node under the existing 12-O/415-column ring. This rules out a fixed-only support obstruction, not simultaneous optimization feasibility. Do not expand the ring unless the per-cell all-tet barrier/feasibility check shows a true boundary obstruction."},
 "layer_thickness":{"basis":"reference O-I and I-D vector lengths; units corrected from metres to millimetres before SLSQP constraints",
                    "floor_definition":"min(0.2 * reference thickness_mm, 0.10 mm); construction feasibility bound only, not physiological-thickness qualification",
                    "skin_reference_thickness_m":stats(sl),"fat_reference_thickness_m":stats(fl),
                    "skin_correct_floor_mm":stats(sfloor),"fat_correct_floor_mm":stats(ffloor),
                    "skin_candidate_reference-normal_projection_mm":stats(sproj),"fat_candidate_reference-normal_projection_mm":stats(fproj),
                    "candidate_rows_below_correct_floor":{"skin":int(np.sum(sproj<sfloor)),"fat":int(np.sum(fproj<ffloor))},
                    "candidate_projection_units":"millimetres; baseline projections are converted from metres before comparing with corrected floors",
                    "owner_physiological_thickness_bounds":"pending; no physiological minimum claimed"},
 "previous_margin_gate":{"minimum_gap_mm":min_gap*1000,"target_mm":target*1000,
                          "shortfall_mm":max(0.,target-min_gap)*1000,"shortfall_nm":max(0.,target-min_gap)*1e9,
                          "gate_tolerance_m":tol,"within_tolerance":bool(min_gap>=target-tol),
                          "strict_target_met":bool(min_gap>=target),
                          "interpretation":"034 passed its explicit 0.2 micrometre tolerance but fell 36.66 nm short of the exact 0.1 mm plane target; 037 will use the 035 pair capture and report exact gap separately."},
 "fit_contract":{"per_bone_constraints":"035 frozen; all (node, body, bone) pairs preserved; ambiguous/on-boundary rows unresolved, never silently discarded",
                 "035_capture_summary":capture_summary,"035_unresolved_rows":len(unresolved_rows),
                 "035_inside_pair_count":int(len(hnode)),"035_distinct_inside_nodes":int(len(np.unique(hnode))),
                 "035_distinct_bodies":int(len(np.unique(hbody))),"035_extra_node_vs_034":sorted(set(map(int,hnode))-set(map(int,A["constrained_nodes"]))),
                 "constraint_expansion":"scan every one of all 6959 affected cells at every accepted callback and final candidate; add all newly below qfloor*1.02 rows to monotone persisted active set before a solver restart",
                 "time_budget_s_per_attempt":105,"maximum_attempts":4,
                 "all_7060_per_callback_restart_logic":"requires integrated 037 runner and remains pending; no optimizer was launched by preparation script"}
}
(OUT/"preparation-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT),"inside_pairs":len(hnode),"ring_columns":len(ring),"variable_rows":len(Vrows),"affected_tets":len(Ttouch),"active_rows":len(low_global),"bad_cells":len(rows),"fixed_only_bad_cells":report["inverted_cell_support"]["fixed_only_cells"],"correct_floor_violations":report["layer_thickness"]["candidate_rows_below_correct_floor"],"gap":report["previous_margin_gate"]},indent=2))
