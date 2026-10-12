#!/usr/bin/env python3
import hashlib, importlib.util, json, math, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
R=Path("/Users/n/numi-human-retained-delivery-20261009")
OUT=ROOT/"attempt-094-compensated-kernel-input-fixture-001"
C=ROOT/"attempt-085-interface-refinement-2mm-001/result-001/candidate.pkl"
C_REPORT=C.parent/"report.json"
FIELD=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/route-anchor-reference-field-001.npz"
WIT=ROOT/"attempt-092-late-witness-arithmetic-001/report.json"
VAR=ROOT/"attempt-093-late-witness-accumulation-variants-001/report.json"
VAR_SCRIPT=ROOT/"attempt-093-late-witness-accumulation-variants-001/audit_variants_093.py"
T=A/"temporalis164-native-m6-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
PACK=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011/baseline/native-run/accepted-geometry/step-151999.mrvpack"
RECEIPT=PACK.with_name("step-151999.receipt.json")
ROUTE=ROOT/"attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json"
REG=ROOT/"registration-source-b1b410ad.json"
REG_PROOF=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json"
ARR=ROOT/"attempt-050/arrangement-001/result-001/exact-arrangement.pkl"
ARR_REPORT=ROOT/"attempt-050/arrangement-001/result-001/report.json"
SOURCE_SHADER=Path("/Users/numi/numi-lab-parent-bindings-1353/matter/src/human_respiration.metal")
PATCH_SHADER=Path("/Users/n/numi-lab-parent-precision-1356/matter/src/human_respiration.metal")
PATCH_HEADER=Path("/Users/n/numi-lab-parent-precision-1356/include/metalrobo/numi_human_resting_visual_gpu.h")
OUT.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def req(v,m):
    if not v: raise RuntimeError(m)
def read_json(p): return json.loads(Path(p).read_text())
def main():
    start=time.monotonic()
    require_paths=(C,C_REPORT,FIELD,WIT,VAR,VAR_SCRIPT,T,H,PACK,RECEIPT,ROUTE,REG,REG_PROOF,ARR,ARR_REPORT,SOURCE_SHADER,PATCH_SHADER,PATCH_HEADER)
    for p in require_paths: req(p.is_file(),f"missing input {p}")
    report093=read_json(VAR)
    req(report093["status"]=="completed_local_witness_arithmetic_variants","093 arithmetic witness report is not complete")
    req(report093["inputs"][str(C)]==sha(C) and report093["inputs"][str(FIELD)]==sha(FIELD),"093 geometry/field pins changed")
    helper_spec=importlib.util.spec_from_file_location("accum_variants_093_fixture",VAR_SCRIPT)
    helper=importlib.util.module_from_spec(helper_spec);sys.modules[helper_spec.name]=helper;helper_spec.loader.exec_module(helper)
    fwd_spec=importlib.util.spec_from_file_location("forward_positive_weights_094",H)
    fwd=importlib.util.module_from_spec(fwd_spec);sys.modules[fwd_spec.name]=fwd;fwd_spec.loader.exec_module(fwd)
    tissue=fwd.load_tissue(T); rows={sid:fwd.row_data(tissue,sid) for sid in (132,134)}
    binding={}
    for sid,row in rows.items():
        for li in range(row["bc"]):
            raw=tissue["bindings"][row["fb"]+li];core=int(raw["core"]);val=np.asarray(raw["value"],dtype=np.float32).copy()
            req(core not in binding or np.array_equal(binding[core],val),f"shared core binding differs at {core}")
            binding[core]=val
    z=np.load(FIELD); core_ids=np.asarray(z["core_body_indices"],dtype=np.uint32)
    req(len(core_ids)==21 and all(int(c) in binding for c in core_ids),"core support mismatch")
    meshes={}
    for sid in (132,134):
        ids=np.asarray(z[f"stable{sid}_global_vertex_ids"],dtype=np.uint64)
        pos=np.asarray(z[f"stable{sid}_rest_positions_m"],dtype=np.float32)
        weights=np.asarray(z[f"stable{sid}_core_weights"],dtype=np.float64)
        faces=np.asarray(z[f"stable{sid}_faces"],dtype=np.int64)
        meshes[sid]={"ids":ids,"pos":pos,"weights":weights,"faces":faces,"index":{int(g):i for i,g in enumerate(ids)}}
    witness=read_json(WIT)["witness_faces"];by_union={int(x["union_index"]):x for x in witness}
    pair_ids=[(4772,21817),(4772,21818),(4772,21835)]
    pairs=[(by_union[a],by_union[b]) for a,b in pair_ids]
    vertices=sorted({int(g) for pair in pairs for face in pair for g in meshes[int(face["stable_id"])]["ids"][meshes[int(face["stable_id"])]["faces"][int(face["row_face_index"])]]})
    req(len(vertices)==6,"expected six unique witness vertices")
    source={};weights={};face_data=[]
    for gid in vertices:
        hits=[(sid,m["index"][gid]) for sid,m in meshes.items() if gid in m["index"]]
        req(hits,f"missing gid {gid}")
        sid,i=hits[0];source[gid]=meshes[sid]["pos"][i].copy();weights[gid]=meshes[sid]["weights"][i].copy()
        for ss,j in hits[1:]:
            req(np.array_equal(source[gid],meshes[ss]["pos"][j]) and np.array_equal(weights[gid],meshes[ss]["weights"][j]),f"shared gid field mismatch {gid}")
    ids_index={g:i for i,g in enumerate(vertices)}
    receipt=read_json(RECEIPT)
    req(receipt.get("accepted_step")==151999 and receipt.get("physical_endpoint")=="accepted" and receipt.get("pack_file_sha256")==sha(PACK),"pose receipt is not exact accepted step 151999")
    poses={int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),np.asarray(x["quaternion_xyzw"],dtype=np.float32)) for x in receipt["accepted_registered_body_poses"]}
    world=np.empty((len(vertices),len(core_ids),3),dtype="<f4");w32=np.empty((len(vertices),len(core_ids)),dtype="<f4")
    native=np.empty((len(vertices),3),dtype="<f4");kahan=np.empty((len(vertices),3),dtype="<f4")
    for vi,gid in enumerate(vertices):
        xcore=[]
        for ci,core0 in enumerate(core_ids):
            core=int(core0);b=binding[core];cp,cq=poses[core];p=source[gid]
            local=np.asarray(b[:3]+np.float32(b[7])*helper.rotate_cxx(b[3:7],p),dtype=np.float32)
            world[vi,ci]=np.asarray(cp+helper.rotate_metal(cq,local),dtype=np.float32)
            w32[vi,ci]=np.float32(weights[gid][ci])
            xcore.append(world[vi,ci].copy())
        acc=np.zeros(3,dtype=np.float32)
        for ci in range(len(core_ids)):
            acc=helper.fma32(w32[vi,ci],world[vi,ci],acc)
        native[vi]=acc
        acc=np.zeros(3,dtype=np.float32);correction=np.zeros(3,dtype=np.float32)
        for ci in range(len(core_ids)):
            term=np.asarray(w32[vi,ci]*world[vi,ci],dtype=np.float32)
            adjusted=np.asarray(term-correction,dtype=np.float32)
            nextsum=np.asarray(acc+adjusted,dtype=np.float32)
            correction=np.asarray(np.asarray(nextsum-acc,dtype=np.float32)-adjusted,dtype=np.float32)
            acc=nextsum
        kahan[vi]=acc
    for a,b in pairs:
        fa=meshes[int(a["stable_id"])]["faces"][int(a["row_face_index"])]
        fb=meshes[int(b["stable_id"])]["faces"][int(b["row_face_index"])]
        ga=[int(meshes[int(a["stable_id"])]["ids"][int(k)]) for k in fa]
        gb=[int(meshes[int(b["stable_id"])]["ids"][int(k)]) for k in fb]
        face_data.append({"union_faces":[int(a["union_index"]),int(b["union_index"])],
          "stable_ids":[int(a["stable_id"]),int(b["stable_id"])],
          "row_face_indices":[int(a["row_face_index"]),int(b["row_face_index"])],
          "global_vertex_ids":[ga,gb],"fixture_vertex_indices":[[ids_index[g] for g in ga],[ids_index[g] for g in gb]],
          "topological_shared_global_vertex_ids":sorted(set(ga)&set(gb)),
          "parent054":[a["parent054"],b["parent054"]]})
    records=np.empty((len(vertices),len(core_ids),4),dtype="<f4")
    records[:,:,:3]=world;records[:,:,3]=w32
    bin_path=OUT/"witness-core-world-and-weight-float4-le.bin"
    records.tofile(bin_path)
    npz_path=OUT/"witness-reference-arrays.npz"
    np.savez_compressed(npz_path,global_vertex_ids=np.asarray(vertices,dtype="<u8"),core_body_indices=core_ids,
       source_positions_m=np.asarray([source[g] for g in vertices],dtype="<f4"),records_float4=records,
       native_like_fma_positions_m=native,compensated_kahan_positions_m=kahan,
       weights_float64=np.asarray([weights[g] for g in vertices],dtype="<f8"))
    from datetime import datetime, timezone
    input_paths=[C,C_REPORT,FIELD,WIT,VAR,VAR_SCRIPT,T,H,PACK,RECEIPT,ROUTE,REG,REG_PROOF,ARR,ARR_REPORT,SOURCE_SHADER,PATCH_SHADER,PATCH_HEADER,
      A/"temporalis164-native-m6-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"]
    pins={str(p):sha(p) for p in input_paths}
    meta={"schema":"numi.human.left-fds132-fdp134-compensated-kernel-witness-fixture-094.v1",
      "status":"prepared_cpu_fixture_for_root_gpu_kernel_test",
      "scope":"Six exact source vertices from three 151999 late triangle-pair witnesses; 21 per-core Float32 body-local-to-captured-world results and Float32 route-field weights. This fixture isolates shader accumulation from coordinate-transform arithmetic; no GPU launch or candidate admission.",
      "source_inputs":pins,
      "route_control_source":{"path":str(ROUTE),"sha256":pins[str(ROUTE)],"source_sha256_from_extraction":"91c16b791be3e66f9065a70623d372f5bc7dca12a7f1a302c5cb469f2d1ad304","registration_path":str(REG),"registration_sha256":pins[str(REG)]},
      "accepted_pose":{"step":151999,"pack_path":str(PACK),"pack_sha256":pins[str(PACK)],"receipt_path":str(RECEIPT),"receipt_sha256":pins[str(RECEIPT)],"accepted_body_state_sha256":receipt["accepted_body_state_sha256"]},
      "runtime_kernel":{"baseline_source":str(SOURCE_SHADER),"baseline_source_sha256":pins[str(SOURCE_SHADER)],"root_patch_source":str(PATCH_SHADER),"root_patch_source_sha256":pins[str(PATCH_SHADER)],"root_patch_header":str(PATCH_HEADER),"root_patch_header_sha256":pins[str(PATCH_HEADER)],"test_helper":"restingCompensatedPositionAdd(term, sum, correction); one source record per sorted core per vertex; output in one Metal dispatch; compare sequential FMA control in same dispatch"},
      "layout":{"binary_path":str(bin_path),"binary_sha256":sha(bin_path),"dtype":"little-endian float32","shape":[len(vertices),len(core_ids),4],"byte_length":bin_path.stat().st_size,"record_order":"global_vertex_ids ascending; within vertex core_body_indices ascending; float4=[world_x,world_y,world_z,Float32 route coefficient]","vertex_ids":vertices,"core_body_indices":[int(x) for x in core_ids]},
      "reference_arrays":{"path":str(npz_path),"sha256":sha(npz_path),"native_like_fma_positions":"CPU emulation of 088 sequential Float32 FMA over records","compensated_kahan_positions":"CPU Kahan with each Float32 weight×world term rounded before the patched helper update; expectation only, native test must report actual result"},
      "triangle_pairs":face_data,
      "arithmetic_context":{"093_report_path":str(VAR),"093_report_sha256":pins[str(VAR)],"baseline_f32_pairs":[{"union_faces":r["union_faces"],"nonshared_intersection_point_count":r["variants"]["native_like_f32_fma"]["nonshared_intersection_point_count"]} for r in read_json(VAR)["witness_results"]],"093_kahan_pairs":[{"union_faces":r["union_faces"],"nonshared_intersection_point_count":r["variants"]["f32_products_f32_kahan"]["nonshared_intersection_point_count"]} for r in read_json(VAR)["witness_results"]],"interpretation":"CPU local witness reference only; exact intersections persisted in 093 and must be rerun on captured GPU float32 positions before accepting any conclusion."},
      "candidate":{"path":str(C),"sha256":pins[str(C)],"refinement_report":str(C.parent/"report.json"),"refinement_sha256":pins[str(C.parent/"report.json")]},
      "qualification":{"gpu_executed":False,"native_resting_pipeline_executed":False,"candidate_admission":False},
      "python_runtime":{"executable":sys.executable,"version":sys.version,"numpy":np.__version__},
      "elapsed_seconds":time.monotonic()-start}
    out=OUT/"fixture.json";tmp=OUT/"fixture.json.tmp";tmp.write_text(json.dumps(meta,sort_keys=True,indent=2)+"\n");tmp.replace(out)
    print(json.dumps({"status":meta["status"],"fixture":str(out),"fixture_sha256":sha(out),"binary":str(bin_path),"binary_sha256":sha(bin_path),"records":len(vertices)*len(core_ids)},sort_keys=True))
if __name__=="__main__": main()
