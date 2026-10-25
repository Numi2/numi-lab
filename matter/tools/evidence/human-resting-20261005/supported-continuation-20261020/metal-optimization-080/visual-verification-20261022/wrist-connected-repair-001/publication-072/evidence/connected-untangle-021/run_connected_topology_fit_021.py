#!/usr/bin/env python3
"""One bounded connected O/I/D construction fit on topology-017."""
from __future__ import annotations
from collections import defaultdict
import hashlib, importlib.util, json, socket, sys, time
from pathlib import Path
import numpy as np

E=Path("/Users/n/Work/numi-human-connected-surface-069")
FITDIR=E/"experimental-interior-fit-008"
FIT_SOURCE=FITDIR/"human_skin_fat_connected_repair.py"
TOPOLOGY=Path("/Users/n/Work/294-source-frame-audit-001/connected-interior-reference-fit-001/bistellar-internal-017/run-001/candidate-tetrahedra.npz")
OUT=E/"connected-topology-fit-021/run-002"
AUDIT_DEPS=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src")
PREDICATE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
EXPECTED_TOPOLOGY="01ff7b7c0221f6bf6959b0cada4899154d0491ed8190938b0dc02fc7d357c384"
EXPECTED_PREDICATE_SHA="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
EXPECTED_GEOMETRY_SHA="f6e98744dad9e23cd3b505efc02e7faa618fccce81d2ee08b07f948ef182fb72"
N_LAYER=2411
Q_FLOOR=1e-5
J_BOUNDS=(.35,2.5)
MAX_ITERATIONS=24
LINEAR_ITERATIONS=400
STEP_LIMIT_MM=1.0

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def expand_face_rings(tets,seeds,nrings=2):
    face_cells=defaultdict(list)
    for ci,t in enumerate(tets):
        a,b,c,d=map(int,t)
        for f in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            face_cells[tuple(sorted(f))].append(ci)
    active=set(map(int,seeds));front=set(active)
    for _ in range(nrings):
        nxt=set()
        for ci in front:
            a,b,c,d=map(int,tets[ci])
            for f in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
                nxt.update(face_cells[tuple(sorted(f))])
        nxt-=active;active|=nxt;front=nxt
        if not front:break
    ids=np.asarray(sorted(active),dtype=np.int64)
    free=np.unique(np.asarray(tets,dtype=np.int64)[ids].reshape(-1))
    return ids,free

def surface_area2(points,faces):
    tri=np.asarray(points,dtype=np.float64)[np.asarray(faces,dtype=np.int64)]
    return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)

def main():
    t0=time.monotonic()
    if OUT.exists():raise FileExistsError(OUT)
    if sha(TOPOLOGY)!=EXPECTED_TOPOLOGY:raise ValueError("topology-017 candidate hash changed")
    OUT.mkdir(parents=True,exist_ok=False)
    z=np.load(TOPOLOGY)
    Xr64=np.asarray(z["reference_points_AP_m"],dtype=np.float64)
    Xstart=np.asarray(z["target_points_AP_m"],dtype=np.float32).copy()
    T=np.asarray(z["candidate_tetrahedra"],dtype=np.int64)
    mats=np.asarray(z["candidate_material_ids"],dtype=np.int64)
    of=np.asarray(z["candidate_O_faces"],dtype=np.int64)
    inf=np.asarray(z["candidate_I_faces"],dtype=np.int64)
    df=np.asarray(z["candidate_D_faces"],dtype=np.int64)
    if Xr64.shape!=(3*N_LAYER,3) or Xstart.shape!=Xr64.shape or len(T)!=28631:
        raise ValueError("unexpected frozen topology candidate shapes")
    if min(of.min(),inf.min(),df.min())<0 or of.max()>=N_LAYER or inf.min()<N_LAYER or inf.max()>=2*N_LAYER or df.min()<2*N_LAYER or df.max()>=3*N_LAYER:
        raise ValueError("unexpected O/I/D global point-ID layout")
    fit=import_file("connected_fit_008",FIT_SOURCE)
    J0,G0,q0,Q0=fit.cell_values_gradients(Xstart,Xr64,T)
    bad=(J0<=0)|(J0<J_BOUNDS[0])|(J0>J_BOUNDS[1])|(q0<Q_FLOOR)
    seed=np.flatnonzero(bad)
    region_cells,free=expand_face_rings(T,seed,2)
    if len(seed)==0 or len(free)==0:raise ValueError("expected active bad-cell fit domain")
    fixed=np.setdiff1d(np.arange(len(Xstart)),free)
    initial_free=Xstart[free].copy();initial_fixed=Xstart[fixed].copy()
    Xfit,fit_report=fit.fit_connected_layer_interior(
        Xstart,Xr64,T,free,q_min=Q_FLOOR,J_bounds=J_BOUNDS,
        step_limit_mm=STEP_LIMIT_MM,max_iterations=MAX_ITERATIONS,
        linear_iterations=LINEAR_ITERATIONS)
    Xfit=np.asarray(Xfit,dtype=np.float32)
    if not np.array_equal(Xfit[fixed].view(np.uint32),initial_fixed.view(np.uint32)):
        raise AssertionError("helper changed fixed outside-region points")
    if np.array_equal(Xfit.view(np.uint32),Xstart.view(np.uint32)) and not fit_report["history"]:
        raise ValueError("fit emitted no work/history")
    Jf,Gf,qf,Qf=fit.cell_values_gradients(Xfit,Xr64,T)
    det_ref_f32=np.einsum("ij,ij->i",(Xr64.astype(np.float32)[T][:,1]-Xr64.astype(np.float32)[T][:,0]),np.cross(Xr64.astype(np.float32)[T][:,2]-Xr64.astype(np.float32)[T][:,0],Xr64.astype(np.float32)[T][:,3]-Xr64.astype(np.float32)[T][:,0]))
    xr32=Xr64.astype(np.float32).astype(np.float64)
    Jr32,_,qr32,_=fit.cell_values_gradients(Xfit,xr32,T)
    Jstart=J0
    dq=np.linalg.norm((Xfit.astype(np.float64)-Xstart.astype(np.float64)),axis=1)*1000
    o_ids=np.arange(0,N_LAYER)
    i_ids=np.arange(N_LAYER,2*N_LAYER)
    d_ids=np.arange(2*N_LAYER,3*N_LAYER)
    pred=None
    def audit_o(points,label):
        rec,_=pred.float32_surface_records(np.asarray(points[:N_LAYER],dtype=np.float32),of)
        result=pred._audit_self_pairs_touching_faces(rec,set(range(len(of))))
        pairs=np.asarray(result["triangle_pairs"],dtype=np.int64).reshape(-1,2)
        np.save(OUT/f"{label}-o-self-pairs.npy",pairs,allow_pickle=False)
        return {"status":"complete","faces":len(of),"intersection_pairs":int(result["count"]),
                "pair_file":str(OUT/f"{label}-o-self-pairs.npy"),"pair_sha256":sha(OUT/f"{label}-o-self-pairs.npy"),
                "complete_surface_audit":bool(result["complete_surface_audit"]),
                "elapsed_seconds":float(result.get("elapsed_seconds",0.0))}
    candidate_path=OUT/"candidate.npz"
    np.savez_compressed(candidate_path,reference_points_AP_m=Xr64,
        starting_target_points_AP_m=np.asarray(z["target_points_AP_m"]),
        target_points_AP_m=Xfit,candidate_tetrahedra=T.astype(np.int32),
        candidate_material_ids=mats.astype(np.int8),candidate_cell_uids=np.asarray(z["candidate_cell_uids"]),
        candidate_O_faces=of.astype(np.int32),candidate_I_faces=inf.astype(np.int32),candidate_D_faces=df.astype(np.int32),
        free_node_ids=free.astype(np.int32),expanded_cell_ids=region_cells.astype(np.int32))
    try:
        if sha(PREDICATE)!=EXPECTED_PREDICATE_SHA or sha(AUDIT_DEPS/"numilab_human/cardiac_cavity_geometry.py")!=EXPECTED_GEOMETRY_SHA:
            raise ValueError("pinned exact surface audit owner/dependency SHA changed")
        sys.path.insert(0,str(AUDIT_DEPS))
        import numilab_human
        numilab_human.__path__.insert(0,str(PREDICATE.parent))
        pred=importlib.import_module("numilab_human.cardiac_cavity_intersections")
        base_self=audit_o(Xstart,"baseline")
        fit_self=audit_o(Xfit,"candidate")
    except Exception as exc:
        base_self=locals().get("base_self",{"status":"not_completed"})
        fit_self={"status":"failed_closed","error_type":type(exc).__name__,"error":str(exc)}
    report={
      "classification":"unadmitted_connected_construction_fit",
      "source":{"topology_npz":str(TOPOLOGY),"topology_sha256":sha(TOPOLOGY),
                "fit_helper":str(FIT_SOURCE),"fit_helper_sha256":sha(FIT_SOURCE),
                "exact_surface_predicate":str(PREDICATE),"exact_surface_predicate_sha256":sha(PREDICATE)},
      "fit_parameters":{"active_cell_seed":"any current F32 nonpositive cell, J outside [0.35,2.5], or q<1e-5",
         "cell_ring_expansion":"two additional face-adjacent tetrahedron rings",
         "free_nodes_include_all_layers":["O","I","D"],"fixed_nodes_bitwise_unchanged":True,
         "reference_fixed":True,"max_iterations":MAX_ITERATIONS,"linear_iterations":LINEAR_ITERATIONS,
         "step_limit_mm":STEP_LIMIT_MM,"regularization":1e-4},
      "domain":{"seed_bad_cell_count":int(len(seed)),"seed_current_nonpositive":int((J0[seed]<=0).sum()),
         "seed_J_below":int((J0[seed]<J_BOUNDS[0]).sum()),"seed_J_above":int((J0[seed]>J_BOUNDS[1]).sum()),
         "seed_q_below":int((q0[seed]<Q_FLOOR).sum()),"two_ring_cell_count":int(len(region_cells)),
         "free_node_count":int(len(free)),"fixed_node_count":int(len(fixed)),
         "free_O_count":int(np.intersect1d(free,o_ids).size),"free_I_count":int(np.intersect1d(free,i_ids).size),
         "free_D_count":int(np.intersect1d(free,d_ids).size),
         "free_node_ids_sha256":hashlib.sha256(free.astype("<i8").tobytes()).hexdigest(),
         "free_cell_ids_sha256":hashlib.sha256(region_cells.astype("<i8").tobytes()).hexdigest()},
      "starting":{"current_nonpositive":int((Jstart<=0).sum()),"J_below":int((Jstart<J_BOUNDS[0]).sum()),
         "J_above":int((Jstart>J_BOUNDS[1]).sum()),"q_below":int((q0<Q_FLOOR).sum()),
         "J_min":float(Jstart.min()),"J_max":float(Jstart.max()),"q_min":float(q0.min()),
         "reference_F32_nonpositive":int((det_ref_f32<=0).sum()),"reference_F32_q_below":int((qr32<Q_FLOOR).sum())},
      "fit_report":fit_report,
      "emitted_f32":{"current_nonpositive":int((Jf<=0).sum()),"J_below":int((Jf<J_BOUNDS[0]).sum()),
         "J_above":int((Jf>J_BOUNDS[1]).sum()),"q_below":int((qf<Q_FLOOR).sum()),
         "J_min":float(Jf.min()),"J_max":float(Jf.max()),"q_min":float(qf.min()),
         "reference_F32_nonpositive":int((Jr32<=0).sum()),"reference_F32_q_below":int((qr32<Q_FLOOR).sum())},
      "movement":{"max_any_node_mm":float(dq.max()),"max_O_source_node_mm":float(dq[o_ids].max()),
         "max_I_node_mm":float(dq[i_ids].max()),"max_D_node_mm":float(dq[d_ids].max()),
         "changed_nodes":int(np.count_nonzero(np.any(Xfit.view(np.uint32)!=Xstart.view(np.uint32),axis=1))),
         "outside_free_domain_bitwise_unchanged":bool(np.array_equal(Xfit[fixed].view(np.uint32),initial_fixed.view(np.uint32)))},
      "exact_coarse_O_self_audit":{"baseline":base_self,"candidate":fit_self},
      "limits":["The connected static fit is a construction diagnostic, not physical or native admission.",
         "Only coarse O exact self-intersections were checked here; detailed skin, bones, fat, rest, mass, and all-column containment are not qualified.",
         "The input topology is the unadmitted 017 topology-only counterfactual; material ancestry/mass was not revalidated."],
      "runtime":{"hostname":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.monotonic()-t0}}
    report["candidate"]={"path":str(candidate_path),"sha256":sha(candidate_path)}
    (OUT/"fit-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"classification":report["classification"],"starting":report["starting"],
       "emitted_f32":report["emitted_f32"],"movement":report["movement"],
       "O_self":fit_self,"candidate":report["candidate"],"report":str(OUT/"fit-report.json"),
       "elapsed":report["runtime"]["seconds"]},indent=2))
if __name__=="__main__":main()
