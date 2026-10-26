#!/usr/bin/env python3
"""Cached algebra-only reproduction of pinned 545 coherent-depth groups."""
import hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D/surface-clearance-546"
basej=D/"refined-point-001/baseline.json"
targetj=D/"refined-point-001/point-targets-multi-parent/nearest-union-exits.json"
targetz=targetj.with_suffix(".npz")
depthj=D/"refined-point-001/depth/365-compatible-report.json"
depthz=D/"refined-point-001/depth/point-depth-rows.npz"
candidate_path=Path(json.loads(basej.read_text())["candidate"])
builder=ROOT/"D/contained-clearance-545/build_joint_source_targets545.py"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=json.loads(basej.read_text())
target_report=json.loads(targetj.read_text())
depth_report=json.loads(depthj.read_text())
candidate_sha=sha(candidate_path)
assert candidate_sha==baseline["candidate_sha256"]
assert target_report["inputs"]["candidate_sha256"]==candidate_sha
assert target_report["depth_row_inputs"]["npz_sha256"]==sha(depthz)
assert target_report["depth_row_inputs"]["report_sha256"]==sha(depthj)
with np.load(candidate_path,allow_pickle=False) as z:
    faces=np.asarray(z["source_O_faces"],dtype=np.int64)
    points=np.asarray(z["points_world_m"])
with np.load(targetz,allow_pickle=False) as z:
    target_rows=np.asarray(z["query_rows"],dtype=np.int64)
    target_mask=np.asarray(z["pass_mask"],dtype=bool)
    target_exact=np.asarray(z["target_exactly_outside_all185"],dtype=bool)
    target_depth=np.asarray(z["depth_covector_world_unit"],dtype=np.float64)
with np.load(depthz,allow_pickle=False) as z:
    depth_rows=np.asarray(z["node_rows"],dtype=np.int64)
    depth_normals=np.asarray(z["normals_world_unit_covector"],dtype=np.float64)
assert target_rows.ndim==1 and len(target_rows)==1371 and len(np.unique(target_rows))==len(target_rows)
assert target_mask.shape==target_rows.shape and target_mask.all()
assert target_exact.shape==target_rows.shape and target_exact.all()
targetset=set(map(int,target_rows))
depth_matches={}
for i,n in enumerate(depth_rows):
    n=int(n)
    if n in targetset:
        depth_matches.setdefault(n,[]).append(i)
assert set(depth_matches)==targetset
assert all(len(v)==1 for v in depth_matches.values()), {
    n:ix for n,ix in depth_matches.items() if len(ix)!=1
}
depth_by_node={n:depth_normals[ix[0]] for n,ix in depth_matches.items()}
N=np.stack([depth_by_node[int(n)] for n in target_rows])
assert np.array_equal(target_depth,N), "saved target depth covectors differ from matched 357 rows"
target_set=set(map(int,target_rows))
parent={int(n):int(n) for n in target_rows}
def find(n):
    while parent[n]!=n:
        parent[n]=parent[parent[n]]
        n=parent[n]
    return n
adjacent={n:set() for n in range(len(points))}
for face in faces:
    f=list(map(int,face))
    for n in f:
        adjacent[n].update(f)
for node in map(int,target_rows):
    near=set(adjacent[node])
    for n in tuple(near):
        near.update(adjacent[n])
    for n in sorted(near & target_set):
        parent[find(n)]=find(node)
grouped={}
for node in map(int,target_rows):
    grouped.setdefault(find(node),[]).append(node)
groups=sorted((sorted(g) for g in grouped.values()),key=lambda g:g[0])
fails=[]
solutions=[]
node_idx={int(n):i for i,n in enumerate(target_rows)}
t0=time.time()
for gi,group in enumerate(groups):
    Ng=np.stack([depth_by_node[n] for n in group])
    # Same 545 common-positive-depth LP as build_joint_source_targets545.py.
    result=linprog([0,0,0,-1],A_ub=np.column_stack([-Ng,np.ones(len(Ng))]),
                   b_ub=np.zeros(len(Ng)),bounds=[(-1,1)]*3+[(0,None)],method="highs")
    gamma=float(result.x[3]) if result.x is not None else None
    rec={"group_index":gi,"group_size":len(group),"first_query_row":group[0],
         "last_query_row":group[-1],"linprog_success":bool(result.success),
         "status":str(result.status),"message":str(result.message),
         "maximum_common_depth_progress":gamma}
    if not result.success or gamma is None or gamma<=1e-8:
        rec["query_rows"]=group
        fails.append(rec)
    solutions.append(rec)
elapsed=time.time()-t0
sizes=np.asarray([len(g) for g in groups],dtype=np.int64)
out={
 "schema":"ordinary-545-coherent-depth-algebra-preflight-v1",
 "status":"INFEASIBLE_GROUPS_FOUND" if fails else "ALL_GROUPS_COMMON_DEPTH_DIRECTION_FEASIBLE",
 "geometry_queries_run":False,"exact_predicates_run":False,"native_jobs_run":False,
 "source_owner_path":str(builder),"source_owner_sha256":sha(builder),
 "grouping_semantics":"545 target nodes connect when sharing a face or when connected through one intermediate mesh neighbor; groups formed by the pinned parent-find union order.",
 "lp_semantics":"For each group N, pinned 545 solves linprog([0,0,0,-1], A_ub=[-N,1], b_ub=0, bounds=[(-1,1)]^3 x [0,infinity), method='highs'); failure if unsuccessful or optimal gamma <= 1e-8.",
 "baseline_path":str(candidate_path),"baseline_sha256":candidate_sha,
 "target_report_path":str(targetj),"target_report_sha256":sha(targetj),
 "target_npz_path":str(targetz),"target_npz_sha256":sha(targetz),
 "matched_depth_report_path":str(depthj),"matched_depth_report_sha256":sha(depthj),
 "matched_depth_npz_path":str(depthz),"matched_depth_npz_sha256":sha(depthz),
 "target_count":len(target_rows),"all_first_exit_targets_pass":bool(target_mask.all()),
 "all_targets_saved_exact_outside_all185":bool(target_exact.all()),
 "depth_covectors_match_357_rows_bitwise":True,
 "group_count":len(groups),"feasible_group_count":len(groups)-len(fails),
 "infeasible_group_count":len(fails),"infeasible_target_rows":sum(len(r.get("query_rows",[])) for r in fails),
 "group_size_min":int(sizes.min()),"group_size_median":float(np.median(sizes)),
 "group_size_p95":int(np.quantile(sizes,.95)),"group_size_max":int(sizes.max()),
 "lp_wall_seconds":elapsed,"failures":fails,"group_summaries":solutions,
 "proposal_only":True,"native_admission":False,"accepted_physical_steps":0
}
outdir=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/coherent-depth-targets-546")/"algebra-preflight-1371"
outdir.mkdir(parents=True,exist_ok=False)
(outdir/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
(outdir/"run_algebra_preflight.py").write_text(Path(__file__).read_text())
print(json.dumps({k:out[k] for k in ["status","target_count","group_count","feasible_group_count","infeasible_group_count","infeasible_target_rows","group_size_min","group_size_median","group_size_p95","group_size_max","lp_wall_seconds"]},indent=2))
for f in fails:
    print("INFEASIBLE",json.dumps({k:f[k] for k in ["group_index","group_size","first_query_row","last_query_row","status","message","maximum_common_depth_progress","query_rows"]}))

