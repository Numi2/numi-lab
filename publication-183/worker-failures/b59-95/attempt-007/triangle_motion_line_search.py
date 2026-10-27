from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json, sys, time, traceback
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
BASE=ROOT/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
OUTDIR=ROOT/"per-bank-workers-20261027/b59-95/attempt-007"
C=BASE/"candidate.npz"; A159=BASE/"audit-159/report.json"; A130=BASE/"audit-130/summary.json"
W=ROOT/"partial-midpoint-continuation-root-001"; TARGET_NPZ=W/"targets-001/nearest-union-exits.npz"; TARGET_JSON=W/"targets-001/nearest-union-exits.json"
REFINER=ROOT/"edge-refinement-661-threebank-001/edge_only_refinement585_generation002.py"
QPATH=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
SOURCE=Path("/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz")
PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
TOPO=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/topology/topology-bone-aware-025.py")
OUT=OUTDIR/"candidate-split-face-12184-moved-midpoints-001.npz"
REPORT=OUTDIR/"attempt-report.json"
EXPECTED_C="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
EXPECTED_REFINER="2b15380a43a0a022d4d0d183233ac1107e64e076180e9403f43f124300b62b87"
EXPECTED_Q="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
EXPECTED_SOURCE="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
TARGET_ROW=20912; SOURCE_FACE_ROW=12184; MOVE_MM=1.0
AREA_FLOOR=0.1
MAX_LOCAL_ROTATION_DEG=10.0
EDGE_GRADIENT_CAP_MM=0.5
SOURCE_CAP_MM=12.000001

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def mod(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same_bits(a,b):
    a=np.ascontiguousarray(a);b=np.ascontiguousarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def bank_pairs(rep):
    return {(int(b["body"]),int(b["stable_bone_id"])):{tuple(map(int,v)) for v in (b.get("result") or {}).get("triangle_pairs",[])} for b in rep["banks"]}
def closest_point_triangle(p,tri):
    a,b,c=np.asarray(tri,dtype=np.float64)
    e0=b-a; e1=c-a; n=np.cross(e0,e1); nn=float(n@n)
    if nn<=1e-30: raise ValueError("degenerate bone triangle")
    proj=p-n*(float((p-a)@n)/nn)
    v=proj-a; d00=float(e0@e0); d01=float(e0@e1); d11=float(e1@e1); d20=float(v@e0); d21=float(v@e1)
    den=d00*d11-d01*d01
    u=(d11*d20-d01*d21)/den; w=(d00*d21-d01*d20)/den; bary=np.array([1-u-w,u,w])
    if np.all(bary>=0): return proj,bary
    best=None
    for i,j in ((0,1),(1,2),(2,0)):
        x=tri[i]; y=tri[j]; d=y-x; t=float(np.clip((p-x)@d/max(float(d@d),1e-30),0.,1.)); q=x+t*d; dist=float(np.linalg.norm(p-q))
        if best is None or dist<best[0]: best=(dist,q,i,j,t)
    _,q,i,j,t=best; bary=np.zeros(3);bary[i]=1-t;bary[j]=t
    return q,bary

start=time.monotonic(); report={"schema":"numi.human.skin_bone_bank59_95_conforming_split_trial.v1","status":"CONSTRUCTION_STARTED","classification":"one ancestry-preserving conforming split plus triangle-normal Float32 line-searched midpoint motion; no physical/native qualification","accepted_physical_steps":0,"geometry_admitted":False,"native_admission":False}
try:
    assert not OUT.exists() and not REPORT.exists()
    assert sha(C)==EXPECTED_C
    report["inputs"]={str(p):sha(p) for p in (C,A159,A130,TARGET_NPZ,TARGET_JSON,REFINER,QPATH,SOURCE,PRED,TOPO)}
    base=load(C); X=base["points_world_m"]; F=base["source_O_faces"].astype(np.int64)
    audit=json.loads(A159.read_text()); point_audit=json.loads(A130.read_text()); before=bank_pairs(audit)
    assert audit.get("candidate_sha256")==sha(C) and audit.get("complete") is True and len(before)==185
    assert point_audit.get("candidate_sha256")==sha(C) and point_audit.get("complete_surface_audit") is True and point_audit.get("exact_PLC_surface_pairs")==0
    assert np.load(BASE/"audit-130/surface-self-pairs.npy",allow_pickle=False).size==0
    assert sum(map(len,before.values()))==985
    before_parent_map=np.asarray(base["refinement_face_parent_face_row"],dtype=np.int64)
    before_parent_pairs={k:{(int(before_parent_map[f]),int(t)) for f,t in ps} for k,ps in before.items()}
    assert sum(map(len,before_parent_pairs.values()))==653
    bank=next(b for b in audit["banks"] if (int(b["body"]),int(b["stable_bone_id"]))==(59,95))
    assert len(bank["result"]["triangle_pairs"])==45
    # Load the pinned exact bank geometry once; it supplies both the motion
    # direction derivation and the later exact all-bank patch screen.
    topo=mod("topo_b59_split_line_search",TOPO); counter_owner=mod("counter_b59_split_line_search",topo.OWNER370)
    _,surf,bh,exact,runtime=counter_owner.load_pinned_runtime()
    banks,keys=counter_owner.typed_banks_from_159_inputs(bh.read_bank_inputs(counter_owner.BANK_DIR_DEFAULT,counter_owner.EVIDENCE_ROOT_DEFAULT))
    assert len(keys)==185 and set(keys)==set(before)

    # Confirm the target-backed direction still starts at this candidate's exact local node.
    target_doc=json.loads(TARGET_JSON.read_text()); target_arrays=load(TARGET_NPZ)
    assert target_doc.get("status")=="PASS_PARTIAL_FIRST_UNION_EXIT_TARGETS_ONLY"
    assert target_doc["inputs"]["candidate_sha256"]=="d367af1383d920645c22552569e95614797de8aea8a757b658def68bad4412d8"
    row_rec=next(r for r in target_doc["targets"] if int(r["query_row"])==TARGET_ROW)
    ti=int(np.searchsorted(target_arrays["query_rows"],TARGET_ROW)); assert int(target_arrays["query_rows"][ti])==TARGET_ROW
    assert np.array_equal(X[TARGET_ROW].view(np.uint32),np.asarray(row_rec["start_world_f32_m"],dtype=np.float32).view(np.uint32))
    assert row_rec["selection"]["target_exactly_outside_all_banks"] is True and row_rec["exact_endpoint_bank_count"]==185
    direction=np.asarray(row_rec["proposal_direction_world_unit"],dtype=np.float64); direction/=np.linalg.norm(direction)
    source_normal=np.asarray(row_rec["source_surface_normal_world_unit"],dtype=np.float64)
    source_dot=float(direction@source_normal)
    assert source_dot>=0.05
    target_vec=np.asarray(target_arrays["target_world_f32_m"][ti],dtype=np.float64)-X[TARGET_ROW].astype(np.float64)
    assert np.linalg.norm(target_vec)>0 and float((target_vec/np.linalg.norm(target_vec))@direction)>0.999999
    # Validate the typed bone triangle winding against the exact signed-distance
    # witness at row 20912, then use the actual crossed bone triangles 149/249.
    bank_geom=next(b for b in banks if b.key==(59,95))
    def local_to_world(tris):
        return (np.asarray(tris,dtype=np.float64)+bank_geom.com_offset_m)@bank_geom.world_to_body_m+bank_geom.position_world_m
    ids=np.asarray(bank_geom.stable_triangle_ids,dtype=np.int64)
    probe_ix=int(np.flatnonzero(ids==302)[0]); probe_tri=local_to_world(bank_geom.triangles_local_m[probe_ix:probe_ix+1])[0]
    probe_nr=np.cross(probe_tri[1]-probe_tri[0],probe_tri[2]-probe_tri[0]); probe_nr/=np.linalg.norm(probe_nr)
    probe_w=next(w for w in point_audit["inside_witnesses"] if int(w["output_node"])==TARGET_ROW and int(w["bank_body"])==59 and int(w["bank_stable_id"])==95)
    signed_expected=float(probe_w["signed_face_distance_m"]); signed_raw=float((X[TARGET_ROW].astype(np.float64)-probe_tri[0])@probe_nr)
    orientation_sign=1.0 if signed_expected*signed_raw>=0 else -1.0
    probe_nr*=orientation_sign; signed_aligned=signed_raw*orientation_sign
    assert abs(signed_aligned-signed_expected)<2e-6
    assert float(direction@probe_nr)>0.05
    crossing_ids=sorted(int(t) for f,t in before[(59,95)] if int(f)==SOURCE_FACE_ROW)
    assert crossing_ids==[149,249],crossing_ids
    cross_ix=np.asarray([int(np.flatnonzero(ids==tid)[0]) for tid in crossing_ids],dtype=np.int64)
    bone_cross_world=local_to_world(bank_geom.triangles_local_m[cross_ix])
    bone_normals=np.cross(bone_cross_world[:,1]-bone_cross_world[:,0],bone_cross_world[:,2]-bone_cross_world[:,0])
    bone_normals/=np.linalg.norm(bone_normals,axis=1)[:,None]
    bone_normals*=orientation_sign
    # The basis is tied to the original crossed skin triangle and its winding.
    face=F[SOURCE_FACE_ROW].astype(np.int64)
    pface=X[face].astype(np.float64)*1000.0
    t1=pface[1]-pface[0];t1/=np.linalg.norm(t1)
    nface=np.cross(pface[1]-pface[0],pface[2]-pface[0]);nface/=np.linalg.norm(nface)
    t2=np.cross(nface,t1);t2/=np.linalg.norm(t2)
    basis=np.stack((t1,t2,nface),axis=0)
    bone_direction_records=[]

    # One 1-to-4 split of a 59/95 crossed child triangle. All three edges are
    # conformingly bisected across every incident face.
    face=F[SOURCE_FACE_ROW].astype(np.int64)
    edges=np.unique(np.sort(np.asarray([[face[0],face[1]],[face[1],face[2]],[face[2],face[0]]],dtype=np.int64),axis=1),axis=0)
    incidence=mod("edge_owner_b59_split",REFINER)._edge_incidence(F)
    edge_uses={tuple(map(int,e)):len(incidence[tuple(map(int,e))]) for e in edges}
    assert all(v==2 for v in edge_uses.values()),edge_uses
    # The shipped adapter reserves its old generation-002 archive key. Preserve
    # that history bytewise at a fresh namespace before invoking the pinned owner.
    oldprefix="prior_input_parent_546_generation_002__"
    newprefix="prior_input_parent_546_generation_002_pre_b59_split__"
    old_history={k:v.copy() for k,v in base.items() if k.startswith(oldprefix)}
    assert old_history and not any(k.startswith(newprefix) for k in base)
    staged={k:v.copy() for k,v in base.items()}
    for k,v in old_history.items(): staged[newprefix+k[len(oldprefix):]]=v.copy(); del staged[k]
    area_sha=str(np.asarray(base["area_reference_037_input_sha256"]).reshape(-1)[0])
    owner_mod=mod("edge_refinement585_b59_split",REFINER)
    assert sha(REFINER)==EXPECTED_REFINER
    arrays,build_report=owner_mod.build_refined_arrays585(
        staged,audit,base["area_reference_037_points_world_m_f64"],base["area_reference_037_source_O_faces"],
        candidate_sha256=sha(C),crossing_report_sha256=sha(A159),area_reference_sha256=area_sha,
        selected_edges_override=edges,
        selection_policy={"mode":"three_shared_edges_of_one_exact_crossed_child_face","bank":[59,95],"crossed_skin_face_row":SOURCE_FACE_ROW,"split_edges":edges.tolist(),"movement":"per-new-vertex outward normal of nearest exact crossed 59/95 bone triangle, expressed in local oriented face basis; Float32 area/rotation line search"})
    arrays=dict(arrays); old_n=len(X); Fout=np.asarray(arrays["source_O_faces"],dtype=np.int64)
    mids=np.arange(old_n,len(arrays["points_world_m"]),dtype=np.int64); edges_out=edges.copy()
    assert len(mids)==3 and np.array_equal(edges_out,edges)
    assert np.array_equal(arrays["points_world_m"][:old_n].view(np.uint32),X.view(np.uint32))
    Xpre=arrays["points_world_m"].copy(); assert Xpre.dtype==np.float32
    assert np.array_equal(arrays["refinement_node_parent_edge_rows"][mids],edges_out)
    # Each edge-node follows the outward normal of the closest exact intersecting
    # bone triangle (149/249), expressed and reconstructed in the local oriented
    # face basis. The exact Float32 line search rejects any step that fails the
    # fixed-037 area floor, positive orientation, <=10 degree local rotation,
    # <=0.5 mm edge-displacement gradient, or 12 mm composed-source bound.
    proposed_dirs=[]
    for node,edge in zip(mids,edges_out):
        point=(Xpre[int(node)].astype(np.float64))
        bone_point=point # world metres; bone triangles are world metres here
        near=[]
        for local_ix,(tri,normal,tid) in enumerate(zip(bone_cross_world,bone_normals,crossing_ids)):
            closest,bary=closest_point_triangle(bone_point,tri); dist=float(np.linalg.norm(bone_point-closest))
            signed=float((bone_point-tri[0])@normal); near.append((dist,local_ix,closest,bary,signed,tid))
        dist,local_ix,closest,bary,signed,tid=min(near,key=lambda q:q[0])
        nout=bone_normals[local_ix]
        components=basis@nout
        in_basis=basis.T@components; in_basis/=np.linalg.norm(in_basis)
        proposed_dirs.append(in_basis)
        bone_direction_records.append({"new_node_row":int(node),"input_edge":edge.tolist(),"nearest_exact_crossing_bone_triangle_id":int(tid),"closest_distance_mm":dist*1000.0,"signed_outward_plane_distance_mm":signed*1000.0,"bone_outward_normal_world_unit":nout.tolist(),"dot_source_outward_at_row_20912":float(nout@source_normal),"local_face_basis_components":components.tolist(),"reconstructed_direction_world_unit":in_basis.tolist(),"closest_point_barycentric":bary.tolist()})
    proposed_dirs=np.asarray(proposed_dirs,dtype=np.float64)
    assert proposed_dirs.shape==(3,3) and np.isfinite(proposed_dirs).all()
    # Compose source provenance before searching so every Float32 step is checked against the existing 12 mm cap.
    assert sha(QPATH)==EXPECTED_Q and sha(SOURCE)==EXPECTED_SOURCE
    q=mod("quality408_b59_triangle_motion",QPATH)
    with np.load(C,allow_pickle=False) as z: src_old,src_receipts=q.compose_source_reference(z,SOURCE)
    src_new=np.asarray([(src_old[a]+src_old[b])*0.5 for a,b in edges_out],dtype=np.float64)
    src_out=np.vstack((src_old,src_new)); assert src_out.shape==Xpre.shape
    baseline_source_mm=float(np.linalg.norm((X.astype(np.float64)-src_old)*1000.0,axis=1).max())
    before_source_max=float(np.linalg.norm((Xpre.astype(np.float64)-src_out)*1000.0,axis=1).max())
    # Determine all changed child rows before moving; the output topology is fixed during this line search.
    parent_out=np.asarray(arrays["refinement_face_parent_face_row"],dtype=np.int64)
    assert parent_out.shape==(len(Fout),)
    vals,cnts=np.unique(parent_out,return_counts=True); child_counts={int(k):int(v) for k,v in zip(vals,cnts)}
    affected_old=np.asarray([r for r,n in child_counts.items() if n>1],dtype=np.int64)
    affected_out=np.flatnonzero(np.isin(parent_out,affected_old))
    assert len(affected_out)>0 and len(affected_old)<=4
    RF=np.asarray(arrays["area_reference_037_source_O_faces"],dtype=np.int64); RP=np.asarray(arrays["area_reference_037_points_world_m_f64"],dtype=np.float64)*1000.0
    assert np.array_equal(RF,Fout)
    reftri=RP[RF[affected_out]]; refcross=np.cross(reftri[:,1]-reftri[:,0],reftri[:,2]-reftri[:,0]); refnorm=np.linalg.norm(refcross,axis=1); assert np.all(refnorm>1e-12)
    refunit=refcross/refnorm[:,None]
    base_tri=(Xpre.astype(np.float64)*1000.0)[Fout[affected_out]]
    base_cross=np.cross(base_tri[:,1]-base_tri[:,0],base_tri[:,2]-base_tri[:,0]); base_norm=np.linalg.norm(base_cross,axis=1); assert np.all(base_norm>1e-12)
    base_unit=base_cross/base_norm[:,None]
    all_edges=np.unique(np.sort(np.vstack((Fout[:,[0,1]],Fout[:,[1,2]],Fout[:,[2,0]])),axis=1),axis=0)
    line_trials=[]; selected=None; cos_floor=float(np.cos(np.deg2rad(MAX_LOCAL_ROTATION_DEG)))
    for alpha in (1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625):
        y=Xpre.copy(); proposed=(alpha*MOVE_MM/1000.0)*proposed_dirs
        y[mids]=(Xpre[mids].astype(np.float64)+proposed).astype(np.float32)
        actual=(y[mids].astype(np.float64)-Xpre[mids].astype(np.float64))*1000.0
        changed=bool(np.all(np.any(y[mids].view(np.uint32)!=Xpre[mids].view(np.uint32),axis=1)))
        ym=y.astype(np.float64)*1000.0; tri=ym[Fout[affected_out]]
        cr=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); norm=np.linalg.norm(cr,axis=1)
        area=np.einsum("ij,ij->i",cr,refunit)/refnorm
        orient=np.einsum("ij,ij->i",cr,base_unit)/norm
        cos=np.einsum("ij,ij->i",cr,base_cross)/(np.maximum(norm*base_norm,1e-30))
        angle=float(np.rad2deg(np.arccos(np.clip(cos.min(initial=1.0),-1.0,1.0))))
        disp=np.zeros_like(ym);disp[mids]=actual
        grad=float(np.linalg.norm(disp[all_edges[:,0]]-disp[all_edges[:,1]],axis=1).max(initial=0.0))
        source_max=float(np.linalg.norm((y.astype(np.float64)-src_out)*1000.0,axis=1).max())
        gates={"all_midpoints_change_float32":changed,"positive_non_degenerate_area":bool(np.all(norm>1e-10) and np.all(orient>0)),"fixed037_area_ratio_floor":bool(np.all(area>=AREA_FLOOR)),"local_rotation_limit":bool(cos.min(initial=1.0)>=cos_floor),"edge_displacement_gradient_cap":bool(grad<=EDGE_GRADIENT_CAP_MM+1e-6),"composed_source_deviation_cap":bool(source_max<=SOURCE_CAP_MM)}
        rec={"alpha":alpha,"per_node_displacements_mm":actual.tolist(),"minimum_fixed037_oriented_area_ratio":float(area.min()),"minimum_local_orientation_cosine":float(cos.min()),"maximum_local_rotation_deg":angle,"minimum_positive_orientation_ratio":float(orient.min()),"maximum_edge_displacement_gradient_mm":grad,"maximum_composed_source_deviation_mm":source_max,"gates":gates,"pass":all(gates.values())}
        line_trials.append(rec)
        if rec["pass"]: selected=(y.copy(),rec); break
    if selected is None:
        report["status"]="LINE_SEARCH_INFEASIBLE"
        report["operation"]={"kind":"same ancestry-preserving conforming three-edge split; no candidate move emitted","bank":[59,95],"crossed_input_face":SOURCE_FACE_ROW,"crossing_bone_triangle_ids":crossing_ids,"bone_triangle_winding_probe":{"triangle_id":302,"raw_signed_distance_mm":signed_raw*1000,"aligned_signed_distance_mm":signed_aligned*1000,"audit_signed_distance_mm":signed_expected*1000,"exit_direction_dot_outward_normal":float(direction@probe_nr)},"local_oriented_face_basis_world_rows":basis.tolist(),"new_vertex_directions_from_intersection_triangles":bone_direction_records,"line_search_proposal_magnitude_mm":MOVE_MM,"line_search_rotation_limit_deg":MAX_LOCAL_ROTATION_DEG,"line_search_area_floor":AREA_FLOOR,"line_search_edge_gradient_cap_mm":EDGE_GRADIENT_CAP_MM,"line_search_source_cap_mm":SOURCE_CAP_MM,"line_search_trials":line_trials,"affected_input_face_rows":affected_old.tolist()}
        raise RuntimeError("LINE_SEARCH_INFEASIBLE: no Float32 positive-area/rotation/source-safe midpoint displacement")
    Xout,chosen_trial=selected; arrays["points_world_m"]=Xout
    MOVE_MM_ACTUAL=max(np.linalg.norm(chosen_trial["per_node_displacements_mm"],axis=1).tolist())
    assert np.array_equal(Xout[:old_n].view(np.uint32),X.view(np.uint32))
    anchors=np.asarray(base["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
    assert np.array_equal(Xout[anchors].view(np.uint32),X[anchors].view(np.uint32))
    assert np.array_equal(arrays["area_reference_037_source_O_faces"],Fout)
    assert np.array_equal(arrays["refinement_node_parent_edge_rows"][mids],edges_out)
    assert np.array_equal(arrays["refinement_node_parent_edge_weights"][mids],np.full((3,2),0.5))
    assert np.all(arrays["refinement_node_kind"][mids]==1)
    # The new Q rows still record the exact immediate source edge and weights.
    qip=np.asarray(arrays["immediate_refinement_prolongation_indptr"],dtype=np.int64); qix=np.asarray(arrays["immediate_refinement_prolongation_node_rows"],dtype=np.int64); qw=np.asarray(arrays["immediate_refinement_prolongation_weights"],dtype=np.float64)
    q_rows=[]
    for r,e in zip(mids,edges_out):
        lo,hi=int(qip[r]),int(qip[r+1]); q_rows.append({"row":int(r),"columns":qix[lo:hi].tolist(),"weights":qw[lo:hi].tolist()})
        assert np.array_equal(qix[lo:hi],e) and np.array_equal(qw[lo:hi],np.array([0.5,0.5]))
    # Owner guarantees conforming boundary/lineage; retain exact local Float32 gates from the line search.
    assert int(build_report.get("boundary_edges_before",-1))==int(build_report.get("boundary_edges_after",-2))
    assert int(build_report.get("boundary_components_before",-1))==int(build_report.get("boundary_components_after",-2))
    min_area_ratio=float(chosen_trial["minimum_fixed037_oriented_area_ratio"]); area_gate_pass=bool(chosen_trial["gates"]["fixed037_area_ratio_floor"])
    edge_gradient=float(chosen_trial["maximum_edge_displacement_gradient_mm"]); edge_gradient_pass=bool(chosen_trial["gates"]["edge_displacement_gradient_cap"])
    max_source_mm=float(chosen_trial["maximum_composed_source_deviation_mm"])
    # All old history from the colliding namespace remains bytewise preserved at the renamed key.
    for k,v in old_history.items(): assert same_bits(arrays[newprefix+k[len(oldprefix):]],v)
    # Save exactly one proposed split/movement candidate before its exact geometry screen.
    with OUT.open("xb") as fo: np.savez_compressed(fo,**arrays)
    emitted=load(OUT)
    for k,v in arrays.items(): assert same_bits(emitted[k],v),k
    report.update({"status":"CONSTRUCTION_PASS_SCREEN_PENDING","candidate_path":str(OUT),"candidate_sha256":sha(OUT),"source_candidate_sha256":sha(C),"baseline_audit159_sha256":sha(A159),"baseline_audit130_sha256":sha(A130),"operation":{"kind":"ancestry-preserving conforming 1-to-4 child-face split across all three shared edges","bank":[59,95],"crossed_input_face":SOURCE_FACE_ROW,"input_face_nodes":face.tolist(),"split_edges":edges_out.tolist(),"new_midpoint_rows":mids.tolist(),"new_child_face_count":int(len(affected_out)),"affected_input_face_rows":affected_old.tolist(),"movement_direction_source":"per-midpoint outward normals of its nearest actual intersecting bank-59/95 triangle (149 or 249), expressed/reconstructed in the local oriented face basis","crossing_bone_triangle_ids":crossing_ids,"bone_triangle_winding_probe":{"triangle_id":302,"raw_signed_distance_mm":signed_raw*1000,"aligned_signed_distance_mm":signed_aligned*1000,"audit_signed_distance_mm":signed_expected*1000,"exit_direction_dot_outward_normal":float(direction@probe_nr)},"local_oriented_face_basis_world_rows":basis.tolist(),"new_vertex_directions_from_intersection_triangles":bone_direction_records,"line_search_proposal_magnitude_mm":MOVE_MM,"selected_line_search":chosen_trial,"line_search_trials":line_trials,"movement_distance_mm":MOVE_MM_ACTUAL,"new_midpoint_displacements_mm":((Xout[mids].astype(float)-Xpre[mids].astype(float))*1000).tolist(),"source_outward_dot_at_reference_row":source_dot,"q_rows":q_rows,"renamed_history_prefix":{"from":oldprefix,"to":newprefix,"arrays_preserved_bytewise":len(old_history)},"construction_report":{"boundary_edges_before":build_report.get("boundary_edges_before"),"boundary_edges_after":build_report.get("boundary_edges_after"),"boundary_components_before":build_report.get("boundary_components_before"),"boundary_components_after":build_report.get("boundary_components_after"),"new_immediate_parent_csr_singleton_child_to_input_face":build_report.get("new_immediate_parent_csr_singleton_child_to_input_face"),"new_faces_inherit_full_parent_union":build_report.get("new_faces_inherit_full_parent_union"),"max_child_barycentric_parent_reconstruction_error_m":build_report.get("max_child_barycentric_parent_reconstruction_error_m")},"anchors_bitwise_unchanged":True,"old_point_prefix_bitwise_unchanged":True,"local_min_037_oriented_area_ratio":min_area_ratio,"local_area_gate_pass_at_0_1":area_gate_pass,"max_edge_displacement_gradient_mm":edge_gradient,"edge_gradient_cap_pass_at_0_5_mm":edge_gradient_pass,"baseline_max_composed_source_deviation_mm":baseline_source_mm,"candidate_max_composed_source_deviation_mm":max_source_mm,"source_prolongation_rows_preserved":True},"exact_bank_screen":None,"exact_self_screen":None})

    # Exact all-bank incremental screen: exact-recount every child of changed parent faces,
    # remap every unchanged baseline pair through its singleton child face.
    assert len(keys)==185 and set(keys)==set(before)
    pc=counter_owner.Exact159PatchCounter(Xout,banks,exact,surf.records)
    local=pc.count_faces(affected_out,Fout[affected_out])
    out_by_input=defaultdict(list)
    for oi,pi in enumerate(parent_out): out_by_input[int(pi)].append(int(oi))
    after={k:set() for k in keys}
    for k,ps in before.items():
        for oldface,boneface in ps:
            if int(oldface) in set(map(int,affected_old)): continue
            child=out_by_input[int(oldface)]
            assert len(child)==1,(oldface,child)
            after[k].add((child[0],int(boneface)))
    for k in keys:
        after[k].update(tuple(map(int,p)) for p in local[k]["triangle_pairs"])
    after_counts={f"{k[0]}/{k[1]}":len(after[k]) for k in keys}
    before_counts={f"{k[0]}/{k[1]}":len(before[k]) for k in keys}
    deltas={k:after_counts[k]-before_counts[k] for k in before_counts if after_counts[k]!=before_counts[k]}
    new_parent_map=before_parent_map[parent_out]
    parent_after={k:{(int(new_parent_map[f]),int(t)) for f,t in after[k]} for k in keys}
    parent_deltas={f"{k[0]}/{k[1]}":len(parent_after[k])-len(before_parent_pairs[k]) for k in keys if len(parent_after[k])!=len(before_parent_pairs[k])}

    # Exact incremental self-intersection screen for the new full surface against the pinned zero-pair baseline.
    pred=mod("numilab_human._self_b59_conforming_split",PRED)
    base_records,_=pred.float32_surface_records(X,F,source_seam_groups=())
    selfbase,_=pred._prepare_complete_self_pair_baseline(base_records)
    assert len(selfbase.triangle_pairs)==0
    trial_records,_=pred.float32_surface_records(Xout,Fout,source_seam_groups=())
    _,self_result=pred._update_complete_self_pair_baseline(selfbase,trial_records)
    self_count=int(self_result["count"])
    report["exact_bank_screen"]={
        "bank_count":len(keys),
        "method":"pinned exact-159 triangle counter on every child of all changed input faces against all 185 typed bank surfaces; complete unchanged-face pairs remapped from pinned report",
        "baseline_raw_pair_total":sum(map(len,before.values())),
        "candidate_raw_pair_total":sum(map(len,after.values())),
        "baseline_parent_pair_total":sum(map(len,before_parent_pairs.values())),
        "candidate_parent_pair_total":sum(map(len,parent_after.values())),
        "bank_59_95_baseline":len(before[(59,95)]),
        "bank_59_95_candidate":len(after[(59,95)]),
        "changed_bank_raw_pair_deltas":deltas,
        "changed_bank_parent_pair_deltas":parent_deltas,
        "other_banks_nonworsening":all(after_counts[k]<=before_counts[k] for k in before_counts),
        "affected_output_face_rows":affected_out.tolist(),
        "recounted_pairs_for_59_95_on_patch":sorted([list(p) for p in local[(59,95)]["triangle_pairs"]])
    }
    report["exact_self_screen"]={"method":"pinned exact incremental self-pair owner on full refined surface against zero-pair baseline","baseline_self_pairs":0,"candidate_self_pairs":self_count,"witnesses":self_result.get("triangle_pairs",[])[:32]}
    nonworse=self_count==0 and all(after_counts[k]<=before_counts[k] for k in before_counts)
    improves=after[(59,95)] and len(after[(59,95)])<len(before[(59,95)])
    report["status"]=("EXACT_SCREEN_TARGET_IMPROVED_UNADMITTED" if nonworse and improves and area_gate_pass and edge_gradient_pass and max_source_mm<=12.000001 else ("EXACT_SCREEN_NEUTRAL_OR_CONSTRUCTION_GATE_FAIL" if nonworse else "EXACT_SCREEN_REJECTED_REGRESSION_OR_SELF"))
    report["elapsed_s"]=time.monotonic()-start
except Exception as e:
    if report.get("status")!="LINE_SEARCH_INFEASIBLE": report["status"]="ATTEMPT_STOPPED_WITH_ERROR"
    if OUT.exists(): report.update({"candidate_path":str(OUT),"candidate_sha256":sha(OUT)})
    report["error"]={"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()[-6000:]}
    report["elapsed_s"]=time.monotonic()-start
finally:
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"candidate":report.get("candidate_path"),"candidate_sha256":report.get("candidate_sha256"),"error":report.get("error"),"exact_bank_screen":report.get("exact_bank_screen"),"exact_self_screen":report.get("exact_self_screen"),"elapsed_s":report.get("elapsed_s")},indent=2),flush=True)
