#!/usr/bin/env python3
"""One bounded current-294 outer-target and connected O/I/D displacement diagnostic."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, socket, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsmr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"reference-fit-layer-counterfactual-001"))
import reference_frame_mapping as ref
from reference_fit_layer_counterfactual import map_source_anchors, stats, tet_metrics, layer_lengths, N

E=ref.ROOT
AP=ref.AP
DETAIL_SKIN=ref.SKIN294
DETAIL_CURRENT=ref.CSV294
EMBED=ref.EMBED
NODES=ref.NODES
PLACEMENT=ref.FRAME
BODY_FRAMES=ref.BODY_FRAMES
LAYER_BASELINE=ROOT/"reference-fit-layer-counterfactual-001/run-001/whole-layer-reference-fit.npz"
HELPER=Path("/Users/n/Work/numi-human-connected-surface-069/canonical/matter/tools/human_skin_fat_connected_repair.py")
PRED_SRC=E/"exact-plane-filter-owner-264/src"
PRED=PRED_SRC/"numilab_human/cardiac_cavity_intersections.py"
PRED_SHA="207236eb886e921b21ff281f44637140ea4bf9438b27561954dfe9528be71d27"
HELPER_SHA="3907a29c8363d5f1d0a90e52f637e0f8fce03c23b47296189e4e02884ee9dd2f"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))

def face_area_normals(points,faces):
    x=np.asarray(points,dtype=np.float64)[np.asarray(faces,dtype=np.int64)]
    cross=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
    return .5*np.linalg.norm(cross,axis=1),cross

def load_repair_helper():
    if sha(HELPER)!=HELPER_SHA:raise ValueError("connected repair helper hash changed")
    spec=importlib.util.spec_from_file_location("pinned_connected_repair",HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def load_exact_predicate():
    if not PRED.is_file():raise FileNotFoundError(PRED)
    if sha(PRED)!=PRED_SHA:raise ValueError("exact predicate source hash changed")
    sys.path.insert(0,str(PRED_SRC))
    from numilab_human import cardiac_cavity_intersections as pred
    return pred

def exact_surface_self(points,faces,pred):
    verts=tuple(tuple(Fraction.from_float(float(x)) for x in row) for row in np.asarray(points,dtype=np.float64))
    records=pred._records(verts,np.asarray(faces,dtype=np.int64).tolist())
    return pred._audit_pair(records,records,same_surface=True)

def ancestry_for_face(face_id,faces,stable_source_face,embedding):
    rows=[int(x) for x in faces[int(face_id)]]
    anchors=[]
    for row in rows:
        rec=embedding[row]
        direct=int(rec["source_vertex"])
        ring=int(rec["ring"])
        if direct != np.iinfo(np.uint32).max:
            anchors.append({"outer_row":row,"kind":"direct","source_vertex_id":direct,"ring_id":ring})
        else:
            anchors.append({"outer_row":row,"kind":"barycentric","source_face_id":int(rec["face"]),
                            "barycentric_weights":[float(rec[f"b{k}"]) for k in range(3)],"ring_id":ring})
    return {"outer_face_id":int(face_id),"stable_source_face_id":int(stable_source_face[face_id]),
            "outer_rows":rows,"anchor_vertices":anchors}

def shape_gradients(reference,tets):
    x=np.asarray(reference,dtype=np.float64)[np.asarray(tets,dtype=np.int64)]
    dm=np.stack((x[:,1]-x[:,0],x[:,2]-x[:,0],x[:,3]-x[:,0]),axis=2)
    inv=np.linalg.inv(dm)
    return np.concatenate((-inv.sum(axis=1)[:,None,:],inv),axis=1)

def fixed_boundary_gradient_fit(points,tets,outer_delta_m,repair_helper,*,outer_count=N,iteration_limit=1200):
    X=np.asarray(points,dtype=np.float64)
    T=np.asarray(tets,dtype=np.int64)
    if X.shape!=(3*outer_count,3) or T.ndim!=2 or T.shape[1]!=4:raise ValueError("unexpected AP mesh domain")
    fixed=np.arange(outer_count,dtype=np.int64)
    free=np.arange(outer_count,3*outer_count,dtype=np.int64)
    if not np.array_equal(np.unique(T),np.arange(3*outer_count)):raise ValueError("AP tetrahedra do not use the complete O/I/D node domain")
    outer_delta_m=np.asarray(outer_delta_m,dtype=np.float64)
    if outer_delta_m.shape!=(outer_count,3):raise ValueError("outer boundary delta shape mismatch")
    U=np.zeros_like(X);U[:outer_count]=outer_delta_m
    shape=shape_gradients(X,T)
    fixed_grad=np.einsum("tka,tkb->tab",U[T],shape,optimize=True).reshape(-1)
    D=repair_helper.sparse_displacement_gradient(X,T,free)
    if D.shape!=(9*len(T),3*len(free)):raise ValueError("pinned sparse displacement-gradient shape mismatch")
    solve=lsmr(D,-fixed_grad,atol=1e-10,btol=1e-10,maxiter=iteration_limit)
    u_mm=solve[0].reshape(len(free),3)
    result=X.copy()
    result[:outer_count]=X[:outer_count]+outer_delta_m
    result[free]+=u_mm*1e-3
    return result,{"iterations":int(solve[2]),"stop_code":int(solve[1]),"residual_norm":float(solve[3]),
                   "normal_residual_norm":float(solve[4]),"operator_norm":float(solve[5]),
                   "condition_estimate":float(solve[6]),"free_nodes":int(len(free)),
                   "gradient_rows":int(D.shape[0]),"gradient_nnz":int(D.nnz),
                   "max_free_displacement_m":float(np.linalg.norm(u_mm*1e-3,axis=1).max(initial=0)),
                   "fixed_outer_max_error_m":float(np.linalg.norm(result[:outer_count]-(X[:outer_count]+outer_delta_m),axis=1).max(initial=0)),
                   "displacement_gradient_l2":float(np.linalg.norm(D@u_mm.ravel()+fixed_grad))}

def run(out:Path):
    start=time.time();out.mkdir(parents=True,exist_ok=False)
    inputs=[AP,DETAIL_SKIN,DETAIL_CURRENT,EMBED,NODES,PLACEMENT,BODY_FRAMES,LAYER_BASELINE,HELPER,PRED]
    for p in inputs:
        if not p.is_file():raise FileNotFoundError(p)
    helper=load_repair_helper()
    ap=np.load(AP);baseline=np.load(LAYER_BASELINE);target_data=np.load(LAYER_BASELINE)
    points=np.asarray(ap["points"],dtype=np.float64)
    tets=np.asarray(ap["tetrahedra"],dtype=np.int64)
    material=np.asarray(ap["material_ids"],dtype=np.int64)
    faces_rows=read_csv(ref.D36/"generated/faces.csv")
    faces=np.asarray([[int(r[k]) for k in ("a","b","c")] for r in faces_rows],dtype=np.int64)
    embedding=read_csv(EMBED)
    stable_source_face=np.asarray([int(r["stable_source_face"]) for r in faces_rows],dtype=np.int64)
    source_vertex_ids=np.asarray([int(r["source_vertex"]) for r in embedding],dtype=np.uint32)
    source_face_ids=np.asarray([int(r["face"]) for r in embedding],dtype=np.uint32)
    bary_weights=np.asarray([[float(r[f"b{k}"]) for k in range(3)] for r in embedding],dtype=np.float64)
    ring_ids=np.asarray([int(r["ring"]) for r in embedding],dtype=np.uint32)
    if points.shape!=(3*N,3) or len(tets)!=28632 or faces.shape!=(4777,3):raise ValueError("pinned mesh cardinality mismatch")
    target=np.asarray(target_data["target_detail_AP_m"],dtype=np.float64)
    prior_o=np.asarray(target_data["original_AP_outer_points_m"],dtype=np.float64)
    delta=target-prior_o
    if np.max(np.linalg.norm(prior_o-points[:N],axis=1),initial=0)>1e-14:raise ValueError("prior target run does not match pinned AP O baseline")
    area0,n0=face_area_normals(points[:N],faces)
    area1,n1=face_area_normals(target,faces)
    cosine=np.einsum("ij,ij->i",n0,n1)/(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1))
    face_area_ratio=area1/area0
    # Orientation cosine is diagnostic only: a large rigid rotation can change it
    # without a fold. Exact self-pair and nonzero-area checks decide this gate.
    pred=load_exact_predicate()
    base_self=exact_surface_self(points[:N],faces,pred)
    target_self=exact_surface_self(target,faces,pred)
    source_valid=bool(np.all(area1>0) and target_self["count"]==0)
    report={
      "schema":"numi.human.current294-outer-target-connected-interior-fit.v1",
      "classification":"single_connected_volume_regularized_counterfactual_no_harmonic_solve_no_NHSKIN_mutation_no_admission",
      "inputs":{str(p):sha(p) for p in inputs},
      "outer_target_preflight":{
        "source_target_npz_sha256":sha(LAYER_BASELINE),
        "outer_faces":int(len(faces)),
        "ancestry_domain":"D36 O-face rows with per-vertex direct source vertex IDs or barycentric source face IDs; stable_source_face_id points to detailed294 face ancestry; ring_id is a construction-region label, not an anatomical region name",
        "zero_area_face_witnesses":[ancestry_for_face(i,faces,stable_source_face,embedding) for i in np.flatnonzero(area1==0)[:100]],
        "baseline_exact_self_pairs":int(base_self["count"]),
        "target_exact_self_pairs":int(target_self["count"]),
        "target_self_pair_witnesses":[{"face_pair":[int(pair[0]),int(pair[1])],"source_faces":[ancestry_for_face(pair[0],faces,stable_source_face,embedding),ancestry_for_face(pair[1],faces,stable_source_face,embedding)]} for pair in target_self.get("triangle_pairs",[])[:100]],
        "target_area_zero_faces":int(np.count_nonzero(area1==0)),
        "baseline_area_m2":stats(area0),
        "target_area_m2":stats(area1),
        "target_to_baseline_face_area_ratio":stats(face_area_ratio),
        "face_normal_cosine_vs_baseline":stats(cosine),
        "normal_cosine_negative_count_diagnostic_only":int(np.count_nonzero(cosine<0)),
        "exact_target_surface_valid_for_interior_fit":source_valid
      },
      "fit":None,
      "limitations":[
        "Outer target coordinates are the exact source-ancestry and pinned-placement positions from the preceding bounded reference-fit run.",
        "The least-squares displacement-gradient extension is a construction counterfactual; no material constitutive model, collision gate, or physical acceptance is inferred.",
        "No bone audit, whole-body external-surface audit, native runtime, or NHSKIN mutation was performed."
      ]
    }
    outnpz=out/"outer-target-preflight.npz"
    if not source_valid:
        np.savez_compressed(outnpz,outer_target_AP_m=target,baseline_outer_AP_m=points[:N],
             source_correction_delta_m=delta,outer_faces=faces,
             target_face_area_m2=area1,baseline_face_area_m2=area0,
             target_face_normal_cosine=cosine,source_vertex_ids=source_vertex_ids,source_face_ids=source_face_ids,
             bary_weights=bary_weights,ring_ids=ring_ids,stable_source_face_ids=stable_source_face)
        report["classification"]="stopped_outer_target_surface_failed_exact_geometry_preflight"
        report["outputs"]={"npz":{"path":str(outnpz),"sha256":sha(outnpz)}}
        report["runtime"]={"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-start}
        (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"report":str(out/"report.json"),"outer_target_preflight":report["outer_target_preflight"]},indent=2))
        return
    result,solver=fixed_boundary_gradient_fit(points,tets,delta,helper)
    det0,vol0,q0=tet_metrics(points,tets)
    det1,vol1,q1=tet_metrics(result,tets)
    result_f32=result.astype(np.float32).astype(np.float64)
    det32,vol32,q32=tet_metrics(result_f32,tets)
    base_skin,base_fat=layer_lengths(points)
    fit_skin,fit_fat=layer_lengths(result)
    fit_skin32,fit_fat32=layer_lengths(result_f32)
    baseline_skin,baseline_fat=layer_lengths(points)
    report["fit"]={
      "method":"minimum squared P1 displacement-gradient extension of fixed O boundary displacement into free I/D nodes using pinned sparse_displacement_gradient; no displacement/reference reset",
      "solver":solver,
      "nonlinear_geometry":{
        "tetrahedra":len(tets),
        "baseline_nonpositive_determinants":int(np.count_nonzero(det0<=0)),
        "double_nonpositive_determinants":int(np.count_nonzero(det1<=0)),
        "float32_nonpositive_determinants":int(np.count_nonzero(det32<=0)),
        "baseline_det_min":float(det0.min()),
        "double_det_min":float(det1.min()),
        "float32_det_min":float(det32.min()),
        "baseline_quality":stats(q0),
        "double_quality":stats(q1),
        "float32_quality":stats(q32),
        "double_quality_below_1e-5":int(np.count_nonzero(q1<1e-5)),
        "float32_quality_below_1e-5":int(np.count_nonzero(q32<1e-5))
      },
      "materials":{},
      "thickness_m":{
        "baseline_skin":stats(base_skin),"fit_skin":stats(fit_skin),"float32_fit_skin":stats(fit_skin32),
        "baseline_fat":stats(base_fat),"fit_fat":stats(fit_fat),"float32_fit_fat":stats(fit_fat32),
        "skin_change":stats(fit_skin-base_skin),"fat_change":stats(fit_fat-base_fat),
        "O_target_delta":stats(np.linalg.norm(delta,axis=1)),
        "I_displacement":stats(np.linalg.norm(result[N:2*N]-points[N:2*N],axis=1)),
        "D_displacement":stats(np.linalg.norm(result[2*N:]-points[2*N:],axis=1))
      }
    }
    for mid in sorted(np.unique(material)):
        s=material==mid
        report["fit"]["materials"][str(int(mid))]={
          "tet_count":int(s.sum()),
          "baseline_signed_volume_m3":float(vol0[s].sum()),
          "fit_signed_volume_m3":float(vol1[s].sum()),
          "float32_fit_signed_volume_m3":float(vol32[s].sum()),
          "fit_abs_volume_m3":float(np.abs(vol1[s]).sum()),
          "float32_fit_abs_volume_m3":float(np.abs(vol32[s]).sum()),
          "baseline_nonpositive":int(np.count_nonzero(det0[s]<=0)),
          "fit_nonpositive":int(np.count_nonzero(det1[s]<=0)),
          "float32_fit_nonpositive":int(np.count_nonzero(det32[s]<=0)),
          "baseline_quality":stats(q0[s]),"fit_quality":stats(q1[s]),"float32_fit_quality":stats(q32[s])
        }
    report["fit"]["geometric_orientation_pass"]=bool(np.all(det1>0) and np.all(det32>0))
    report["fit"]["classification"]="diagnostic_only_geometry_pass" if report["fit"]["geometric_orientation_pass"] else "rejected_tetrahedron_orientation_or_degeneracy"
    outnpz=out/"connected-interior-fit.npz"
    skin=ref.load_skin(DETAIL_SKIN)
    emb=embedding
    report["source_ancestry"]={
      "direct_count":sum(int(r["source_vertex"])!=np.iinfo(np.uint32).max for r in emb),
      "barycentric_count":sum(int(r["source_vertex"])==np.iinfo(np.uint32).max for r in emb),
      "direct_source_vertex_ids":[int(r["source_vertex"]) for r in emb],
      "bary_source_face_ids":[int(r["face"]) for r in emb],
      "bary_source_weights":[[float(r[f"b{k}"]) for k in range(3)] for r in emb],
      "detail_skin_sha256":skin["sha256"]
    }
    np.savez_compressed(outnpz,outer_target_AP_m=target,source_correction_delta_m=delta,
         baseline_OID_points_m=points,fit_OID_points_m=result,fit_OID_points_float32_m=result_f32.astype(np.float32),
         outer_faces=faces,tetrahedra=tets.astype(np.int32),material_ids=material.astype(np.int8),
         source_vertex_ids=source_vertex_ids,source_face_ids=source_face_ids,bary_weights=bary_weights,
         ring_ids=ring_ids,stable_source_face_ids=stable_source_face,
         baseline_det=det0,fit_det=det1,fit_float32_det=det32,baseline_quality=q0,fit_quality=q1,fit_float32_quality=q32)
    report["outputs"]={"npz":{"path":str(outnpz),"sha256":sha(outnpz)}}
    report["runtime"]={"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-start}
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"report":str(out/"report.json"),"outer_target_preflight":report["outer_target_preflight"],
      "fit":report["fit"],"output_sha256":sha(outnpz)},indent=2))

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
