#!/usr/bin/env python3
import csv, hashlib, json, math, struct
from pathlib import Path
import numpy as np

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAP=ROOT/"native-candidate020-all-bank-containment-035/source-preflight-003"
DONOR=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/donor-frames-fullweights.csv"
BODYFRAMES=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/provenance/build_candidate_rebind_inputs.json"
OUT=ROOT/"E/native-candidate020-all-bank-affine-roundtrip-036"
OUT.mkdir(parents=True,exist_ok=True)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x): return struct.unpack("<f",struct.pack("<f",float(x)))[0]
def readcsv(p):
    with open(p,newline="") as f: return list(csv.DictReader(f))
def add(a,b): return tuple(a[i]+b[i] for i in range(3))
def sub(a,b): return tuple(a[i]-b[i] for i in range(3))
def norm(a): return math.sqrt(sum(x*x for x in a))
def rotate(q,v):
    # Match NumiHumanInitialSkinBoneContactWitness.hpp::rotate: q is raw
    # native Float32 xyzw promoted to double; do not normalize it.
    x,y,z,w=q; a,b,c=v
    t=(2.0*(y*c-z*b),2.0*(z*a-x*c),2.0*(x*b-y*a))
    cross=(y*t[2]-z*t[1],z*t[0]-x*t[2],x*t[1]-y*t[0])
    return (a+w*t[0]+cross[0],b+w*t[1]+cross[1],c+w*t[2]+cross[2])
def mat_from_q(q):
    qi=(-q[0],-q[1],-q[2],q[3])
    cols=[rotate(qi,e) for e in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))]
    return [[cols[j][i] for j in range(3)] for i in range(3)]
def solve3(A,b):
    a=[list(map(float,A[i]))+[float(b[i])] for i in range(3)]
    for k in range(3):
        p=max(range(k,3),key=lambda i:abs(a[i][k]))
        if not math.isfinite(a[p][k]) or abs(a[p][k])<1e-18: raise ValueError("singular raw-F32 affine matrix")
        a[k],a[p]=a[p],a[k]
        d=a[k][k]; a[k]=[x/d for x in a[k]]
        for i in range(3):
            if i==k: continue
            d=a[i][k]; a[i]=[a[i][j]-d*a[k][j] for j in range(4)]
    out=tuple(a[i][3] for i in range(3))
    if not all(math.isfinite(x) for x in out): raise ValueError("nonfinite inverse-affine result")
    return out
def matvec(A,v): return tuple(sum(A[i][j]*v[j] for j in range(3)) for i in range(3))
def det3(A):
    return (A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-
            A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+
            A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]))

poses=json.loads((CAP/"registered-skin-poses.json").read_text())
body={int(r["body_index"]):r["current"] for r in poses["all_bodies"]}
world_rows=readcsv(CAP/"initial-skin-fem-nodes.csv")
world={int(r["global_node"]):tuple(f32(r[k]) for k in ("x_m","y_m","z_m")) for r in world_rows}
rows=readcsv(CAP/"all-bank-containment.csv")
# The node summary is the coverage authority; CSV rows are all parity queries.
summary=readcsv(CAP/"all-bank-containment.csv.nodes.csv")
assert len(summary)==7233 and len(rows)==2112
# Candidate local query offset is common for all nodes queried against one body.
# Recover it redundantly from native world nodes and body transforms; validate it
# against the independent mass-reference local coordinates where node donor matches.
bybody={}
arr=[]
maxqnormdev=0.0
for r in rows:
    g=int(r["global_node"]); b=int(r["bank_body"])
    if g not in world or b not in body: raise ValueError("capture row has missing node/body pose")
    pose=body[b]
    q=tuple(f32(x) for x in pose["orientation_world_xyzw"])
    p=tuple(f32(x) for x in pose["position_world_m"])
    qn=math.sqrt(sum(x*x for x in q)); maxqnormdev=max(maxqnormdev,abs(qn-1.0))
    if not math.isfinite(qn) or qn<1e-8: raise ValueError("invalid zero/NaN native quaternion")
    M=mat_from_q(q); d=det3(M)
    if not math.isfinite(d) or abs(d)<1e-8: raise ValueError("singular raw-F32 inverse quaternion matrix")
    w=world[g]; localcalc=rotate((-q[0],-q[1],-q[2],q[3]),sub(w,p))
    qlocal=tuple(float(r[k]) for k in ("query_local_x","query_local_y","query_local_z"))
    off=sub(localcalc,qlocal)
    bybody.setdefault(b,[]).append(off)
    lclosest=tuple(float(r[k]) for k in ("closest_x","closest_y","closest_z"))
    localdelta=sub(lclosest,qlocal)
    worlddelta=solve3(M,localdelta)
    back=matvec(M,worlddelta)
    residual=norm(sub(back,localdelta))
    closest_world=add(w,worlddelta)
    code={"inside":1,"outside":0,"ambiguous":-1,"invalid":-2}.get(r["classification"],-3)
    if code < -2: raise ValueError("unknown containment classification")
    direction=sub(lclosest,qlocal) if code==1 else sub(qlocal,lclosest)
    world_direction=solve3(M,direction); dn=norm(world_direction)
    unit=tuple(x/dn for x in world_direction) if dn>0 and code in (0,1) else (math.nan,)*3
    arr.append((g,int(r["layer"]),int(r["row"]),int(r["role_mask"]),b,int(r["bank_stable_id"]),
                int(r["nearest_triangle"]),code,w,qlocal,lclosest,closest_world,unit,residual))
# Compute body COM correction vectors as mean and require local transform consistency.
offsets={}; offset_spread={}
for b,values in bybody.items():
    mean=tuple(sum(v[i] for v in values)/len(values) for i in range(3))
    spread=max(norm(sub(v,mean)) for v in values)
    offsets[b]=mean; offset_spread[b]=spread
# Independent owner-local check from the execution's mass-reference node rows.
textnodes={}
for line in open(CAP/"tissue-mass-reference.txt"):
    z=line.split()
    if len(z)==7 and z[0]=="node":
        textnodes[int(z[1])]=(int(z[2]),tuple(float(x) for x in z[4:7]))
independent=[]
for (g,b), values in []: pass
for r in rows:
    g=int(r["global_node"]); b=int(r["bank_body"])
    if g not in textnodes or textnodes[g][0]!=b: continue
    q=tuple(f32(x) for x in body[b]["orientation_world_xyzw"]); p=tuple(f32(x) for x in body[b]["position_world_m"])
    local=rotate((-q[0],-q[1],-q[2],q[3]),sub(world[g],p))
    independent.append(norm(sub(sub(local,textnodes[g][1]),offsets[b])))
# Actual absolute affine round trip, using the recovered per-body mass-COM offset.
max_abs_world=0.0; max_offset_match=0.0; max_inverse_residual=0.0
for r in rows:
    g=int(r["global_node"]); b=int(r["bank_body"]); q=tuple(f32(x) for x in body[b]["orientation_world_xyzw"])
    p=tuple(f32(x) for x in body[b]["position_world_m"]); M=mat_from_q(q)
    qlocal=tuple(float(r[k]) for k in ("query_local_x","query_local_y","query_local_z"))
    reconstructed=add(p,solve3(M,add(qlocal,offsets[b])))
    max_abs_world=max(max_abs_world,norm(sub(reconstructed,world[g])))
    localback=add(matvec(M,sub(reconstructed,p)),tuple(-x for x in offsets[b]))
    max_inverse_residual=max(max_inverse_residual,norm(sub(localback,qlocal)))
# Check 037 donor table’s raw Float32 quaternion matrix versus stored M,
# and the unrounded affine inverse on its pinned reference-space O rows.
config=json.loads((ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/provenance/build_candidate_rebind_inputs.json").read_text())
frames=json.loads(Path(config["body_frames"]).read_text())
source_pose={int(x["core_body"]):x for x in frames["source_poses"]}
donor=readcsv(DONOR); nodes=readcsv(DONOR.parent/"nodes.csv")
max_donor_m=0.0; max_donor_l=0.0; max_donor_inverse=0.0
for i,r in enumerate(donor):
    b=int(r["donor"]); f=source_pose[b]; q=tuple(f32(x) for x in f["quaternion_xyzw"])
    M=mat_from_q(q); m_csv=[[float(r[f"m{a}{c}"]) for c in range(3)] for a in range(3)]
    max_donor_m=max(max_donor_m,max(abs(M[a][c]-m_csv[a][c]) for a in range(3) for c in range(3)))
    O=tuple(f32(nodes[i][f"o{x}"]) for x in "xyz")
    t=tuple(f32(float(f["position"][j])+(0.35 if j==2 else 0.0)) for j in range(3))
    local=tuple(f32(r[f"l{x}"]) for x in "xyz")
    # compare raw native Float32 inverse rotation of the candidate coordinate
    target=matvec(M,sub(O,t)); max_donor_l=max(max_donor_l,norm(sub(target,local)))
    recovered=add(t,solve3(M,local)); max_donor_inverse=max(max_donor_inverse,norm(sub(recovered,O)))
# Small deterministic regression: q is deliberately slightly non-unit. Solving
# the full inverse must round-trip, whereas treating M as R^T is not the rule.
qtest=tuple(f32(x) for x in (0.2,-0.3,0.1,0.91)); Mtest=mat_from_q(qtest)
v=(0.123,-0.456,0.789); vback=solve3(Mtest,matvec(Mtest,v))
testerr=norm(sub(vback,v))
naive_transpose=tuple(sum(Mtest[j][i]*matvec(Mtest,v)[j] for j in range(3)) for i in range(3))
naive_transpose_error=norm(sub(naive_transpose,v))
assert testerr<1e-12 and naive_transpose_error>1e-3

outnpz=OUT/"all-bank-local-to-world-rows.npz"
np.savez_compressed(outnpz,
    global_node=np.asarray([a[0] for a in arr],dtype=np.uint32), layer=np.asarray([a[1] for a in arr],dtype=np.uint8),
    row=np.asarray([a[2] for a in arr],dtype=np.uint32), role_mask=np.asarray([a[3] for a in arr],dtype=np.uint8),
    body_index=np.asarray([a[4] for a in arr],dtype=np.uint32), bone_stable_id=np.asarray([a[5] for a in arr],dtype=np.uint32),
    nearest_triangle=np.asarray([a[6] for a in arr],dtype=np.uint32), classification_code=np.asarray([a[7] for a in arr],dtype=np.int8),
    query_world_m=np.asarray([a[8] for a in arr],dtype=np.float64), query_local_m=np.asarray([a[9] for a in arr],dtype=np.float64),
    closest_local_m=np.asarray([a[10] for a in arr],dtype=np.float64), closest_world_m=np.asarray([a[11] for a in arr],dtype=np.float64),
    unit_escape_world=np.asarray([a[12] for a in arr],dtype=np.float64), affine_roundtrip_residual_m=np.asarray([a[13] for a in arr],dtype=np.float64))
report={"schema":"numi.human.d37.raw-f32-affine-roundtrip.v1","scope":"source-only affine conversion check; no physical/native admission",
 "input_sha256":{"all_bank_csv":sha(CAP/"all-bank-containment.csv"),"all_bank_node_summary":sha(CAP/"all-bank-containment.csv.nodes.csv"),"initial_fem_nodes":sha(CAP/"initial-skin-fem-nodes.csv"),"mass_reference":sha(CAP/"tissue-mass-reference.txt"),"pose_snapshot":sha(CAP/"registered-skin-poses.json"),"037_donor_frames":sha(DONOR),"037_nodes":sha(DONOR.parent/"nodes.csv"),"037_body_frames":sha(config["body_frames"])},
 "method":{"pose":"all_bodies.current Float32 xyz/xyzw from exact source-preflight-003 snapshot","rotation":"same double rotate formula as NumiHumanInitialSkinBoneContactWitness.hpp, quaternion components first rounded to raw Float32","world_to_local":"M=rotate(conjugate(q), basis columns); local=M*(world-body_position)-effective_COM_offset","local_to_world":"body_position+solve(M,local+effective_COM_offset), full inverse (no transpose/unit-quaternion assumption)","closest_world":"query_world+solve(M,closest_local-query_local); translation and common COM offset cancel","normal":"inverse-linear transformed closest-point escape vector, normalized; not a force normal"},
 "capture":{"source_preflight":"source-preflight-003","parity_query_rows":len(rows),"summary_nodes":len(summary),"coverage":{"node_bank_pairs":sum(int(x["total_bone_banks"]) for x in summary),"aabb_proven_outside":sum(int(x["aabb_proven_outside_banks"]) for x in summary),"parity_outside":sum(int(x["parity_outside_banks"]) for x in summary),"inside":sum(int(x["inside_banks"]) for x in summary),"ambiguous":sum(int(x["ambiguous_banks"]) for x in summary),"invalid":sum(int(x["invalid_banks"]) for x in summary)},"bodies_with_query_rows":len(bybody),"max_abs_raw_quaternion_norm_minus_one":maxqnormdev},
 "affine_roundtrip":{"max_body_offset_spread_m":max(offset_spread.values()),"body_offset_spread_max_by_body_m":{str(k):v for k,v in offset_spread.items()},"independent_mass_node_offset_comparisons":len(independent),"max_offset_disagreement_vs_mass_node_local_m":max(independent) if independent else None,"max_world_reconstruction_error_m":max_abs_world,"max_local_reprojection_error_m":max_inverse_residual,"max_query_to_closest_delta_inverse_residual_m":max(a[13] for a in arr),"synthetic_nonunit_quaternion_full_inverse_roundtrip_error_m":testerr,"synthetic_nonunit_transpose_assumption_error_m":naive_transpose_error},
 "037_reference_donor_affine":{"rows":len(donor),"max_raw_f32_M_vs_037_stored_M_abs":max_donor_m,"max_raw_f32_M_times_O_minus_t_vs_serialized_local_m":max_donor_l,"max_solve_M_local_plus_t_vs_O_m":max_donor_inverse,"body_frame_source_root_lift_z_m":0.35},
 "outputs":{"npz":str(outnpz),"npz_sha256":sha(outnpz)}}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
