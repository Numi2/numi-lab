#!/usr/bin/env python3
"""One bounded 294 source-reference fit applied uniformly to original AP O/I/D columns."""
from __future__ import annotations
import csv, hashlib, json, socket, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import reference_frame_mapping as ref

AP = ref.AP
DETAIL_SKIN = ref.SKIN294
DETAIL_CURRENT = ref.CSV294
EMBED = ref.EMBED
NODES = ref.NODES
PLACEMENT = ref.FRAME
BODY_FRAMES = ref.BODY_FRAMES
SENTINEL = np.iinfo(np.uint32).max
N = 2411

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_csv(path: Path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def csv_xyz(rows, prefix="w"):
    return np.asarray([[float(r[prefix+c]) for c in "xyz"] for r in rows], dtype=np.float64)

def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return {"count":int(a.size),"min":float(np.min(a)),"median":float(np.median(a)),
            "p95":float(np.quantile(a,.95)),"p99":float(np.quantile(a,.99)),
            "max":float(np.max(a))}

def map_source_anchors(source_world, source_faces, source_vertex_ids, bary_face_ids, bary_weights, R, t):
    source_ap=np.asarray(source_world,dtype=np.float64) @ np.asarray(R,dtype=np.float64).T + np.asarray(t,dtype=np.float64)
    ids=np.asarray(source_vertex_ids,dtype=np.int64)
    fids=np.asarray(bary_face_ids,dtype=np.int64)
    w=np.asarray(bary_weights,dtype=np.float64)
    direct=ids!=SENTINEL
    if len(ids)!=N or w.shape!=(N,3) or len(fids)!=N: raise ValueError("anchor ancestry shape mismatch")
    if np.any(ids[direct]<0) or np.any(ids[direct]>=len(source_ap)): raise ValueError("direct source vertex out of range")
    if np.any(fids[~direct]<0) or np.any(fids[~direct]>=len(source_faces)): raise ValueError("bary source face out of range")
    if np.max(np.abs(w[~direct].sum(axis=1)-1),initial=0)>1e-9: raise ValueError("bary weights do not sum to one")
    targets=np.empty((N,3),dtype=np.float64)
    targets[direct]=source_ap[ids[direct]]
    for i in np.flatnonzero(~direct):
        targets[i]=w[i] @ source_ap[source_faces[fids[i]]]
    return targets,direct,source_ap

def shift_all_layers(points, delta):
    p=np.asarray(points,dtype=np.float64)
    d=np.asarray(delta,dtype=np.float64)
    if p.shape!=(3*N,3) or d.shape!=(N,3): raise ValueError("expected 3 x 2411 layered points and 2411 offsets")
    out=p.copy().reshape(3,N,3)
    out+=d[None,:,:]
    return out.reshape(3*N,3)

def tet_metrics(points, tets):
    p=np.asarray(points,dtype=np.float64)
    t=np.asarray(tets,dtype=np.int64)
    xyz=p[t]
    e1=xyz[:,1]-xyz[:,0];e2=xyz[:,2]-xyz[:,0];e3=xyz[:,3]-xyz[:,0]
    det=np.einsum("ij,ij->i",e1,np.cross(e2,e3))
    volume=det/6.0
    edge2=np.zeros(len(t),dtype=np.float64)
    for a in range(4):
        for b in range(a+1,4):
            edge2+=np.einsum("ij,ij->i",xyz[:,a]-xyz[:,b],xyz[:,a]-xyz[:,b])
    quality=np.zeros(len(t),dtype=np.float64)
    valid=edge2>0
    quality[valid]=12.0*np.power(3.0*np.abs(volume[valid]),2.0/3.0)/edge2[valid]
    return det,volume,quality

def layer_lengths(points):
    p=np.asarray(points,dtype=np.float64).reshape(3,N,3)
    return np.linalg.norm(p[1]-p[0],axis=1),np.linalg.norm(p[2]-p[1],axis=1)

def run(outdir: Path):
    t0=time.time()
    outdir.mkdir(parents=True,exist_ok=False)
    inputs=[AP,DETAIL_SKIN,DETAIL_CURRENT,EMBED,NODES,PLACEMENT,BODY_FRAMES]
    for p in inputs:
        if not p.is_file(): raise FileNotFoundError(p)
    ap=np.load(AP)
    points=np.asarray(ap["points"],dtype=np.float64)
    tets=np.asarray(ap["tetrahedra"],dtype=np.int64)
    materials=np.asarray(ap["material_ids"],dtype=np.int64)
    if points.shape!=(3*N,3) or tets.shape!=(28632,4) or materials.shape!=(len(tets),):
        raise ValueError("pinned AP grid cardinality mismatch")
    if set(np.unique(materials).tolist())!={1,2}: raise ValueError("unexpected material IDs")
    skin=ref.load_skin(DETAIL_SKIN)
    rows=read_csv(DETAIL_CURRENT)
    if len(rows)!=skin["nv"] or len(rows)!=61950: raise ValueError("native detailed row count mismatch")
    if any(int(r["vertex"])!=i for i,r in enumerate(rows)): raise ValueError("native detailed row IDs not contiguous")
    source_world=csv_xyz(rows)
    # Verify current native positions by the full weighted binding/body-pose map,
    # using the exact pinned ABI5 bytes and source body-frame snapshot.
    bodyframes=json.loads(BODY_FRAMES.read_text())
    poses={int(x["core_body"]):x for x in bodyframes["source_poses"]}
    A,c,_,_=ref.effective_binding_affines(skin["bindings"],poses)
    J=ref.weighted_skin_jacobians(skin["weights"],A)
    bias=skin["weights"] @ c
    source_reconstructed=np.einsum("nij,nj->ni",J,skin["positions"],optimize=True)+bias
    lbs_error=np.linalg.norm(source_reconstructed-source_world,axis=1)
    if float(lbs_error.max(initial=0))>20e-6: raise ValueError("full weighted LBS reconstruction exceeds 20um guard")
    embedding=read_csv(EMBED)
    nodes=read_csv(NODES)
    if len(embedding)!=N or len(nodes)!=N: raise ValueError("outer ancestry row count mismatch")
    if any(int(r["row"])!=i for i,r in enumerate(embedding)) or any(int(r["row"])!=i for i,r in enumerate(nodes)):
        raise ValueError("ancestry rows are not identity ordered")
    source_ids=np.asarray([int(r["source_vertex"]) for r in embedding],dtype=np.int64)
    face_ids=np.asarray([int(r["face"]) for r in embedding],dtype=np.int64)
    bary=np.asarray([[float(r[f"b{k}"]) for k in range(3)] for r in embedding],dtype=np.float64)
    targets,direct,source_ap=map_source_anchors(source_world,skin["faces"],source_ids,face_ids,bary,
                                                json.loads(PLACEMENT.read_text())["placement_rotation"],
                                                json.loads(PLACEMENT.read_text())["placement_translation_m"])
    R=np.asarray(json.loads(PLACEMENT.read_text())["placement_rotation"],dtype=np.float64)
    trans=np.asarray(json.loads(PLACEMENT.read_text())["placement_translation_m"],dtype=np.float64)
    node_layers=[]
    for prefix in ("o","i","d"):
        node_layers.append(np.asarray([[float(r[prefix+c]) for c in "xyz"] for r in nodes],dtype=np.float64))
    node_ap=np.stack(node_layers) @ R.T + trans
    ap_layers=points.reshape(3,N,3)
    layer_parity=np.linalg.norm(node_ap-ap_layers,axis=2)
    if float(layer_parity.max(initial=0))>2e-6: raise ValueError("D36 O/I/D table does not match AP rows within 2um")
    original_o=ap_layers[0].copy()
    delta=targets-original_o
    cf_points=shift_all_layers(points,delta)
    # ABI-like Float32 coordinate conversion is reported separately from the
    # primary double-precision geometric counterfactual.
    cf_f32=cf_points.astype(np.float32).astype(np.float64)
    base_det,base_vol,base_q=tet_metrics(points,tets)
    cf_det,cf_vol,cf_q=tet_metrics(cf_points,tets)
    f32_det,f32_vol,f32_q=tet_metrics(cf_f32,tets)
    base_skin,base_fat=layer_lengths(points)
    cf_skin,cf_fat=layer_lengths(cf_points)
    f32_skin,f32_fat=layer_lengths(cf_f32)
    report={
      "schema":"numi.human.current294-whole-layer-reference-fit-counterfactual.v1",
      "classification":"single_geometry_counterfactual_only_no_harmonic_solve_no_render_promotion_no_NHSKIN_mutation",
      "construction":"For every O row, map the current native 294 detailed position through exact direct or barycentric D36 ancestry and the pinned rigid placement into AP frame. Compute delta=target_AP-original_AP_O, then add that same delta to corresponding O, I, and D nodes for all 2411 columns.",
      "source_position_semantics":"native registered-skin.csv world positions are checked against full ABI5 86-binding weighted LBS reconstruction using pinned body poses; reported anchors are composed by source vertex or source face barycentrics, then AP placement.",
      "inputs":{str(p):sha(p) for p in inputs},
      "checks":{
        "detail_vertices":int(len(source_world)),
        "full_weighted_lbs_reconstruction_max_m":float(lbs_error.max(initial=0)),
        "full_weighted_lbs_reconstruction_p95_m":float(np.quantile(lbs_error,.95)),
        "direct_anchors":int(direct.sum()),
        "barycentric_anchors":int((~direct).sum()),
        "unique_bary_source_faces":int(np.unique(face_ids[~direct]).size),
        "barycentric_weight_sum_max_error":float(np.max(np.abs(bary[~direct].sum(axis=1)-1),initial=0)),
        "D36_to_AP_placement":{
          "rotation":R.tolist(),"translation_m":trans.tolist(),
          "max_layer_parity_m":{"O":float(layer_parity[0].max()),"I":float(layer_parity[1].max()),"D":float(layer_parity[2].max())}
        },
        "target_delta_m":stats(np.linalg.norm(delta,axis=1)),
        "direct_target_delta_m":stats(np.linalg.norm(delta[direct],axis=1)),
        "bary_target_delta_m":stats(np.linalg.norm(delta[~direct],axis=1)),
        "changed_OID_nodes":int(np.count_nonzero(np.linalg.norm(cf_points-points,axis=1)>0)),
        "layer_displacement_parity_m":stats(np.linalg.norm((cf_points-points).reshape(3,N,3)-delta[None,:,:],axis=2)),
        "materials":{},
        "thickness_m":{
          "baseline_skin":stats(base_skin),"counterfactual_skin":stats(cf_skin),
          "baseline_fat":stats(base_fat),"counterfactual_fat":stats(cf_fat),
          "float32_counterfactual_skin":stats(f32_skin),"float32_counterfactual_fat":stats(f32_fat),
          "skin_change_m":stats(cf_skin-base_skin),"fat_change_m":stats(cf_fat-base_fat),
          "source_metadata_skin":stats([float(r["skin_thickness_m"]) for r in nodes]),
          "source_metadata_fat":stats([float(r["fat_thickness_m"]) for r in nodes])
        },
        "whole_mesh":{
          "tet_count":int(len(tets)),
          "baseline_signed_det_min":float(base_det.min()),
          "counterfactual_signed_det_min":float(cf_det.min()),
          "counterfactual_nonpositive_det_count":int(np.count_nonzero(cf_det<=0)),
          "float32_counterfactual_nonpositive_det_count":int(np.count_nonzero(f32_det<=0)),
          "baseline_quality":stats(base_q),"counterfactual_quality":stats(cf_q),
          "float32_counterfactual_quality":stats(f32_q),
          "counterfactual_quality_below_1e-5":int(np.count_nonzero(cf_q<1e-5)),
          "float32_quality_below_1e-5":int(np.count_nonzero(f32_q<1e-5))
        }
      },
      "limitations":[
        "This applies one exact-ancestry outer-source correction uniformly to each column's O/I/D nodes; it is a counterfactual, not a fitted or admitted mesh.",
        "The 79 barycentric source anchors use the frozen source-face IDs and weights; their source-to-O residuals can be large and are not hidden.",
        "No bone/tissue collision, boundary continuity, anatomy, mass remap, FEM response, native runtime, or physical acceptance gate was run.",
        "Float32 metrics model coordinate rounding only and do not establish ABI serialization or downstream runtime parity."
      ],
      "runtime":{"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t0}
    }
    for mid in sorted(np.unique(materials)):
        sel=materials==mid
        rel=cf_det[sel]/base_det[sel]
        frel=f32_det[sel]/base_det[sel]
        report["checks"]["materials"][str(int(mid))]={
          "tet_count":int(sel.sum()),
          "baseline_signed_volume_m3":float(base_vol[sel].sum()),
          "counterfactual_signed_volume_m3":float(cf_vol[sel].sum()),
          "float32_counterfactual_signed_volume_m3":float(f32_vol[sel].sum()),
          "baseline_abs_volume_m3":float(np.abs(base_vol[sel]).sum()),
          "counterfactual_abs_volume_m3":float(np.abs(cf_vol[sel]).sum()),
          "float32_counterfactual_abs_volume_m3":float(np.abs(f32_vol[sel]).sum()),
          "baseline_orientation_nonpositive":int(np.count_nonzero(base_det[sel]<=0)),
          "counterfactual_orientation_nonpositive":int(np.count_nonzero(cf_det[sel]<=0)),
          "float32_orientation_nonpositive":int(np.count_nonzero(f32_det[sel]<=0)),
          "determinant_ratio":stats(rel),
          "float32_determinant_ratio":stats(frel),
          "counterfactual_quality":stats(cf_q[sel]),
          "float32_counterfactual_quality":stats(f32_q[sel])
        }
    outnpz=outdir/"whole-layer-reference-fit.npz"
    np.savez_compressed(outnpz,
      target_detail_AP_m=targets,
      original_AP_outer_points_m=original_o,
      source_correction_delta_m=delta,
      direct_anchor_mask=direct,
      direct_source_vertex_ids=source_ids,
      bary_source_face_ids=face_ids,
      bary_source_weights=bary,
      source_current_native_world_m=source_world,
      source_current_AP_world_m=source_ap,
      original_OID_points_m=points,
      counterfactual_OID_points_m=cf_points,
      counterfactual_OID_points_float32_m=cf_f32.astype(np.float32),
      tetrahedra=tets.astype(np.int32),
      material_ids=materials.astype(np.int8),
      baseline_tet_determinants=base_det,
      counterfactual_tet_determinants=cf_det,
      float32_counterfactual_tet_determinants=f32_det,
      baseline_tet_quality=base_q,
      counterfactual_tet_quality=cf_q,
      float32_counterfactual_tet_quality=f32_q,
      baseline_skin_thickness_m=base_skin,
      counterfactual_skin_thickness_m=cf_skin,
      baseline_fat_thickness_m=base_fat,
      counterfactual_fat_thickness_m=cf_fat)
    report["outputs"]={"npz":{"path":str(outnpz),"sha256":sha(outnpz)}}
    (outdir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"report":str(outdir/"report.json"),"outputs":report["outputs"],
                      "checks":report["checks"]},indent=2))

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
