#!/usr/bin/env python3
"""Build a diagnostic 039-to-294 source-surface displacement prolongation."""
from __future__ import annotations
import csv, hashlib, json, struct, sys, platform, socket, time
from pathlib import Path
import numpy as np
from surface_displacement_prolongation import prolongate_surface_displacement

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAND = ROOT / "connected-volume-continuation-039/iteration-005/candidate.npz"
AP = ROOT / "bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
EMBED = ROOT / "wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated/embedding.csv"
NODES = ROOT / "wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated/nodes.csv"
FRAME = ROOT / "elbow-refinement-current-main-024-final080-candidate-source-frames-001/native-source-only-024/common-rigid-transfer-row-diagnostic-005.json"
SKIN294 = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
CSV294 = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/native-step-001/registered-skin.csv")
SKIN128 = ROOT / "left-ankle-rest-preserving-weight-repair-128/candidate.nhskin"
OUT = ROOT / "current-294-skin-displacement-prolongation-001/run-002"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def read_csv(path):
    with open(path,newline="") as f: return list(csv.DictReader(f))


def read_abi5_faces(path):
    data=path.read_bytes()
    magic,ver,nb,nv,ni,flags,source_sha=struct.unpack_from("<8s5I32s",data,0)
    if magic.rstrip(b"\0") != b"NHSKIN1" or ver != 5:
        raise ValueError("unexpected NHSKIN ABI/header")
    offset=60 + nb*36 + nv*56
    faces=np.frombuffer(data,dtype="<u4",count=ni,offset=offset).reshape((-1,3)).copy()
    if len(faces)*3 != ni or faces.max(initial=0) >= nv:
        raise ValueError("invalid NHSKIN index payload")
    return data, faces, {"magic":magic.decode("ascii","replace"),"version":ver,"bindings":nb,"vertices":nv,"indices":ni,"faces":len(faces)}


def surface_metrics(points, faces):
    tri=points[faces].astype(np.float64)
    e0=tri[:,1]-tri[:,0]; e1=tri[:,2]-tri[:,1]; e2=tri[:,0]-tri[:,2]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area2=np.linalg.norm(cross,axis=1)
    edges=np.stack((np.linalg.norm(e0,axis=1),np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1)),axis=1)
    alt=np.divide(area2,edges.max(axis=1),out=np.zeros_like(area2),where=edges.max(axis=1)>0)
    return area2*0.5, alt, cross


def main():
    start=time.time(); OUT.mkdir(parents=True,exist_ok=False)
    for path in (CAND,AP,EMBED,NODES,FRAME,SKIN294,CSV294,SKIN128):
        if not path.is_file(): raise FileNotFoundError(path)
    cand=np.load(CAND); ap=np.load(AP)
    cp=np.asarray(cand["points_world_f32"],dtype=np.float32)
    apoints=np.asarray(ap["points"],dtype=np.float64)
    if cp.shape != (7233,3) or apoints.shape != (7233,3): raise ValueError("unexpected O/I/D node layout")
    if "tetrahedra" not in cand or cand["tetrahedra"].shape != (28632,4): raise ValueError("candidate topology changed")
    embed=read_csv(EMBED); nodes=read_csv(NODES)
    if len(embed)!=2411 or len(nodes)!=2411: raise ValueError("D36 anchor table row count mismatch")
    frame=json.loads(FRAME.read_text())
    R=np.asarray(frame["placement_rotation"],dtype=np.float64)
    t=np.asarray(frame["placement_translation_m"],dtype=np.float64)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all(): raise ValueError("invalid placement frame")
    if frame.get("embedding_sha256") != sha(EMBED): raise ValueError("frame diagnostic embedding hash mismatch")
    # Verify that the registered D36 O source points agree with the pinned AP basis.
    coarse_source=np.array([[float(r["ox"]),float(r["oy"]),float(r["oz"])] for r in nodes])
    coarse_ap=coarse_source @ R.T + t
    frame_error=np.linalg.norm(coarse_ap-apoints[:2411],axis=1)
    if frame_error.max(initial=0)>2e-6: raise ValueError("D36-to-AP frame mismatch")

    data294,F,abi294=read_abi5_faces(SKIN294)
    data128,F128,abi128=read_abi5_faces(SKIN128)
    if not np.array_equal(F,F128): raise ValueError("D36 face identity mismatch with current 294 topology")
    if abi294["vertices"]!=61950 or len(F)!=108987: raise ValueError("unexpected current 294 topology size")
    native=read_csv(CSV294)
    if len(native)!=abi294["vertices"]: raise ValueError("registered 294 CSV row count mismatch")
    ids_all=np.asarray([int(r["vertex"]) for r in native],dtype=np.int64)
    if not np.array_equal(ids_all,np.arange(len(native))): raise ValueError("registered CSV vertex identity/order mismatch")
    source_xyz=np.asarray([[float(r["wx"]),float(r["wy"]),float(r["wz"])] for r in native],dtype=np.float64)
    mats=np.asarray([[[float(r[f"m{i}{j}"]) for j in range(3)] for i in range(3)] for r in native],dtype=np.float64)
    if not np.isfinite(source_xyz).all() or not np.isfinite(mats).all(): raise ValueError("nonfinite registered source data")

    coarse_delta_ap=cp[:2411].astype(np.float64)-apoints[:2411]
    coarse_delta_src=coarse_delta_ap @ R
    source_ids=np.asarray([int(r["source_vertex"]) for r in embed],dtype=np.int64)
    sentinel=np.iinfo(np.uint32).max
    is_direct=source_ids!=sentinel
    direct_ids=source_ids[is_direct]
    direct_delta=coarse_delta_src[is_direct]
    bary_rows=np.flatnonzero(~is_direct)
    bary_faces=np.asarray([int(embed[i]["face"]) for i in bary_rows],dtype=np.int64)
    bary_weights=np.asarray([[float(embed[i][f"b{k}"]) for k in range(3)] for i in bary_rows],dtype=np.float64)
    bary_delta=coarse_delta_src[~is_direct]
    if len(np.unique(direct_ids)) != len(direct_ids): raise ValueError("D36 direct source IDs are not unique")
    if direct_ids.min(initial=0)<0 or direct_ids.max(initial=0)>=len(source_xyz): raise ValueError("D36 source ID outside current 294")
    if bary_faces.min(initial=0)<0 or bary_faces.max(initial=0)>=len(F): raise ValueError("D36 barycentric face outside current 294")

    # Position residual diagnoses source correspondence; it does not alter the exact delta constraints.
    direct_pos_error=np.linalg.norm(source_xyz[direct_ids]-coarse_source[is_direct],axis=1)
    bary_pos_error=[]
    for row, face_id in zip(bary_rows,bary_faces):
        w=np.asarray([float(embed[row][f"b{k}"]) for k in range(3)])
        bary_pos_error.append(np.linalg.norm(w @ source_xyz[F[face_id]] - coarse_source[row]))
    bary_pos_error=np.asarray(bary_pos_error)

    result=prolongate_surface_displacement(source_xyz,F,direct_ids,direct_delta,bary_faces,bary_weights,bary_delta,core_rings=2,collar_rings=1)
    delta=result["displacements_m"]
    moved=source_xyz+delta
    moved_f32=moved.astype(np.float32)
    if not np.isfinite(moved_f32).all(): raise ValueError("nonfinite emitted diagnostic positions")
    # The CSV binding matrix is the registered world-to-donor-local linear map used by the mechanical transfer API.
    local_delta=np.einsum("nij,nj->ni",mats,delta)
    reconstructed=np.einsum("nij,nj->ni",np.linalg.inv(mats),local_delta)
    local_roundtrip=float(np.max(np.linalg.norm(reconstructed-delta,axis=1),initial=0.0))
    if local_roundtrip>1e-9: raise ValueError("donor-local delta roundtrip failed")

    area0,alt0,cross0=surface_metrics(source_xyz.astype(np.float32),F)
    area1,alt1,cross1=surface_metrics(moved_f32,F)
    affected=np.any(np.linalg.norm(delta[F],axis=2)>0,axis=1)
    orient=np.einsum("ij,ij->i",cross0,cross1)
    area_ratio=np.divide(area1,area0,out=np.zeros_like(area1),where=area0>0)
    unique_direct_delta=np.linalg.norm(direct_delta,axis=1)
    support_report=result["report"]
    report={
      "schema":"current294-skin-displacement-prolongation-001",
      "classification":"diagnostic_prolongation_only_not_physical_or_native_admission",
      "implementation":{"runner_sha256":sha(Path(__file__)),"helper_sha256":sha(Path(__file__).with_name("surface_displacement_prolongation.py")),"test_sha256":sha(Path(__file__).with_name("test_surface_displacement_prolongation.py"))},
      "inputs":{
        "candidate_npz":{"path":str(CAND),"sha256":sha(CAND)},
        "ap_basis_npz":{"path":str(AP),"sha256":sha(AP)},
        "embedding_csv":{"path":str(EMBED),"sha256":sha(EMBED)},
        "coarse_nodes_csv":{"path":str(NODES),"sha256":sha(NODES)},
        "placement_diagnostic":{"path":str(FRAME),"sha256":sha(FRAME)},
        "skin294_nhskin":{"path":str(SKIN294),"sha256":sha(SKIN294)},
        "skin128_nhskin":{"path":str(SKIN128),"sha256":sha(SKIN128)},
        "registered294_csv":{"path":str(CSV294),"sha256":sha(CSV294)}},
      "abi294":abi294,"abi128":abi128,
      "checks":{
        "same_128_294_face_indices":True,
        "frame_coarse_to_ap_max_residual_m":float(frame_error.max(initial=0.0)),
        "direct_source_anchor_position_residual_m":{"count":int(len(direct_pos_error)),"max":float(direct_pos_error.max(initial=0.0)),"p95":float(np.quantile(direct_pos_error,.95)),"median":float(np.median(direct_pos_error))},
        "bary_source_anchor_position_residual_m":{"count":int(len(bary_pos_error)),"max":float(bary_pos_error.max(initial=0.0)),"p95":float(np.quantile(bary_pos_error,.95)),"median":float(np.median(bary_pos_error))},
        "direct_anchor_displacement_max_mm":float(unique_direct_delta.max(initial=0.0)*1000),
        "direct_anchor_displacement_nonzero_count":int(np.count_nonzero(unique_direct_delta>1e-10)),
        "coarse_O_displacement_nonzero_count":int(np.count_nonzero(np.linalg.norm(coarse_delta_ap,axis=1)>1e-10)),
        "direct_anchor_reproduction_max_m":support_report.get("direct_max_residual_m"),
        "bary_anchor_reproduction_max_m":support_report.get("bary_max_residual_m"),
        "donor_local_world_roundtrip_max_m":local_roundtrip,
        "support":support_report,
        "outside_patch_nonzero_rows":int(np.count_nonzero(np.linalg.norm(delta,axis=1)>0)-np.count_nonzero(np.linalg.norm(delta[np.unique(F[affected])],axis=1)>0)),
        "affected_face_count":int(affected.sum()),
        "affected_face_area_ratio_min":float(area_ratio[affected].min()),
        "affected_face_area_ratio_p01":float(np.quantile(area_ratio[affected],.01)) if affected.any() else 1.0,
        "affected_face_min_altitude_before_mm":float(alt0[affected].min()*1000),
        "affected_face_min_altitude_after_mm":float(alt1[affected].min()*1000),
        "affected_face_f32_degenerate_after_count":int(np.count_nonzero(area1[affected]==0)),
        "affected_face_negative_normal_dot_count":int(np.count_nonzero(orient[affected]<=0)),
        "max_surface_displacement_mm":float(np.linalg.norm(delta,axis=1).max(initial=0.0)*1000),
        "changed_surface_vertex_count":int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-12)),
        "source_payload_mutated":False,
        "weights_bindings_normals_material_thickness_mutated":False,
        "exact_collision_or_material_gate_run":False},
      "solver":"inverse-edge-length graph-harmonic minimizer with exact direct-vertex Dirichlet anchors, exact barycentric equality rows, and fixed one-ring collar",
      "limitations":["Detailed 294 source positions and D36 coarse anchors have large correspondence residuals; displacement transfer is diagnostic only.","No exact bone, soft-tissue, self-intersection, layer, volume, mass, native, or physical acceptance gate was run.","The generated NPZ is an auxiliary displacement map; no NHSKIN payload was modified or emitted."]}
    np.savez_compressed(OUT/"prolongation.npz",source_vertex_ids=np.arange(len(source_xyz),dtype=np.int64),source_faces=F.astype(np.uint32),source_world_positions_m=source_xyz,world_displacements_m=delta,donor_local_displacements_m=local_delta,prolonged_world_positions_f32=moved_f32,direct_anchor_ids=direct_ids,direct_anchor_displacements_m=direct_delta,bary_anchor_face_ids=bary_faces,bary_anchor_weights=bary_weights,bary_anchor_displacements_m=bary_delta,affected_face_ids=np.flatnonzero(affected).astype(np.int64))
    report["outputs"]={"prolongation_npz":{"path":str(OUT/"prolongation.npz"),"sha256":sha(OUT/"prolongation.npz")}}
    report["runtime"]={"seconds":time.time()-start,"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__}
    (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"report":str(OUT/"report.json"),"npz":str(OUT/"prolongation.npz"),"support":support_report,"max_anchor_residuals_m":[support_report.get("direct_max_residual_m"),support_report.get("bary_max_residual_m")],"anchor_residual_m":{"direct_max":float(direct_pos_error.max(initial=0.0)),"bary_max":float(bary_pos_error.max(initial=0.0))},"max_move_mm":float(np.linalg.norm(delta,axis=1).max(initial=0.0)*1000),"seconds":time.time()-start},indent=2))

if __name__=="__main__": main()
