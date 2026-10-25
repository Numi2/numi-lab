from __future__ import annotations
import hashlib, json, pathlib, sys, time, traceback
import numpy as np
from scipy import sparse

ROOT = pathlib.Path('/Users/n/Work/numi-human-connected-surface-069')
WORK = ROOT / 'joint-interface-facet-extraction-164'
TOOLS = WORK / 'canonical/matter/tools'
INPUT = ROOT / 'connected-volume-extraction-157/true-o-foundation-159/run-013-conforming-field-refinement'
CHECKPOINT = INPUT / 'refined-foundation-checkpoint.npz'
FIELDS = INPUT / 'registered-levelsets.npz'
RUNS = WORK / 'run-013-joint-facets'
OUT = RUNS / 'run-003'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()

if OUT.exists():
    raise SystemExit(f'output already exists; refusing overwrite: {OUT}')
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0,str(TOOLS))
from human_skin_fat_layer_reconstruction import (
    extract_joint_zero_level_facets, snap_shared_scalar_interface_vertices)


report={
    'status':'STARTED', 'construction_only':True,
    'physical_or_native_admission':False, 'tetgen_run':False,
    'input_checkpoint':str(CHECKPOINT), 'input_checkpoint_sha256':sha(CHECKPOINT),
    'input_registered_fields':str(FIELDS), 'input_registered_fields_sha256':sha(FIELDS),
    'owner_source':str(TOOLS/'human_skin_fat_layer_reconstruction.py'),
    'owner_source_sha256':sha(TOOLS/'human_skin_fat_layer_reconstruction.py'),
    'test_source_sha256':sha(TOOLS/'test_human_skin_fat_layer_reconstruction.py'),
    'field_semantics':{'phi_inner':'registered_levelsets_m[:,1] (skin_levelset_m)',
                       'phi_deep':'registered_levelsets_m[:,2] (total_levelset_m)',
                       'hfat':'phi_deep - phi_inner, required strictly positive at every foundation vertex',
                       'facet_winding':'outward from the positive scalar half-space, i.e. -gradient(phi)'},
}
(OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
started=time.monotonic()
try:
    with np.load(CHECKPOINT,allow_pickle=False) as z:
        P=np.asarray(z['points_world_m'],dtype=np.float32)
        T=np.asarray(z['tetrahedra_zero_based'],dtype=np.int32)
        parent_foundation=np.asarray(z['parent_foundation_tet_rows'],dtype=np.int32)
        embedded_fields=np.asarray(z['registered_levelsets_m'],dtype=np.float64)
        input_quality_f32=np.asarray(z['quality_emitted_f32'],dtype=np.float64)
    with np.load(FIELDS,allow_pickle=False) as z:
        levels=np.asarray(z['registered_levelsets_m'],dtype=np.float64)
        source_hash=np.asarray(z['source_checkpoint_sha256']).reshape(-1)[0]
    if embedded_fields.shape != levels.shape or not np.array_equal(embedded_fields,levels):
        raise ValueError('registered field table does not match the field table embedded in checkpoint')
    if source_hash != report['input_checkpoint_sha256']:
        raise ValueError('registered-levelsets source_checkpoint_sha256 does not match foundation checkpoint')
    if len(parent_foundation)!=len(T) or len(input_quality_f32)!=len(T):
        raise ValueError('foundation tet ancestry/quality row count mismatch')
    report.update(input_point_count=int(len(P)),input_tetrahedron_count=int(len(T)),
        input_f32_min_quality=float(np.min(input_quality_f32)),
        input_f32_quality_below_1e_5_count=int(np.count_nonzero(input_quality_f32<1e-5)),
        input_f32_nonpositive_quality_count=int(np.count_nonzero(input_quality_f32<=0)),
        parent_foundation_tet_id_unique_count=int(len(np.unique(parent_foundation))))
    (OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    # Apply the existing shared-node scalar snap independently to both fields
    # on the SAME unchanged common foundation. This is a declared construction
    # perturbation bounded by the existing 0.5% local-hskin rule, not a cell
    # positivity waiver and not sequential clipping.
    phiI_original=levels[:,1].copy()
    phiD_original=levels[:,2].copy()
    hskin_local=np.asarray(levels[:,1]+levels[:,0],dtype=np.float64)
    phiI_use, snapI_stats, snapI_ledger=snap_shared_scalar_interface_vertices(
        P,T,phiI_original,hskin_local,fraction=0.005)
    phiD_use, snapD_stats, snapD_ledger=snap_shared_scalar_interface_vertices(
        P,T,phiD_original,hskin_local,fraction=0.005)
    hfat_after=phiD_use-phiI_use
    if not np.isfinite(hfat_after).all() or np.any(hfat_after<=0.0):
        bad=np.flatnonzero(~np.isfinite(hfat_after)|(hfat_after<=0.0))[:32]
        report.update(status='FAILED_JOINT_SNAP_ORDERING',
            hfat_nonpositive_count=int(np.count_nonzero(~np.isfinite(hfat_after)|(hfat_after<=0.0))),
            hfat_bad_rows=bad.astype(int).tolist(),
            snap_inner=snapI_stats,snap_deep=snapD_stats)
        (OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        raise ValueError('joint scalar snaps collapsed or inverted phi_deep - phi_inner ordering')
    if np.any(np.all(phiI_use[T]==0.0,axis=1)) or np.any(np.all(phiD_use[T]==0.0,axis=1)):
        raise ValueError('joint scalar snaps created an all-zero parent tetrahedron')
    for name,ledger in [('inner',snapI_ledger),('deep',snapD_ledger)]:
        np.savez_compressed(OUT/f'{name}-scalar-snap-ledger.npz',**ledger)
    report.update(interface_scalar_snap_policy={
        'method':'existing snap_shared_scalar_interface_vertices, independently on phi_inner and phi_deep, both on the same unchanged points/tets',
        'fraction_of_local_hskin':0.005,
        'local_hskin_definition':'registered_levelsets_m[:,1] + registered_levelsets_m[:,0]',
        'coordinates_changed':False,
        'child_tets_created':False,
        'inner':snapI_stats,'deep':snapD_stats,
        'hfat_before_min_m':float(np.min(phiD_original-phiI_original)),
        'hfat_after_min_m':float(np.min(hfat_after)),
        'hfat_after_nonpositive_count':int(np.count_nonzero(hfat_after<=0.0)),
        'inner_ledger_sha256':sha(OUT/'inner-scalar-snap-ledger.npz'),
        'deep_ledger_sha256':sha(OUT/'deep-scalar-snap-ledger.npz')})
    (OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    result=extract_joint_zero_level_facets(
        P,T,phiI_use,phiD_use,
        global_vertex_ids=np.arange(len(P),dtype=np.int64),
        parent_tet_ids=np.arange(len(T),dtype=np.int64))
    # The extractor's parent CSR is in input foundation-tet row space. Preserve
    # that exact row identity and also expand the immutable TetGen ancestry.
    local_parent=result.facet_parent_local_rows
    mapped_parent=parent_foundation[local_parent]
    np.savez_compressed(OUT/'joint-interface-facets.npz',
        points_world_m_f64=result.points_float64,
        points_world_m_f32=result.points_float32,
        facets_zero_based=result.facets.astype(np.int32),
        interface_ids=result.interface_ids,
        facet_parent_indptr=result.facet_parent_indptr,
        facet_parent_input_tet_rows=result.facet_parent_tet_rows.astype(np.int32),
        facet_parent_foundation_tet_rows=mapped_parent.astype(np.int32),
        facet_parent_local_input_tet_rows=local_parent.astype(np.int32),
        point_interface_ids=result.point_interface_ids,
        point_source_edge_global_ids=result.point_source_edge_global_ids,
        point_edge_parameter_from_low_global_id=result.point_edge_parameter_from_low_global_id,
        point_scalar_values_m=result.point_scalar_values_m,
        phi_inner_original_m=phiI_original,phi_deep_original_m=phiD_original,
        phi_inner_snapped_m=phiI_use,phi_deep_snapped_m=phiD_use,local_hskin_m=hskin_local)
    sparse.save_npz(OUT/'point-interpolation-weights.npz',result.point_interpolation_weights,compressed=True)
    report.update(status='PASS_STATIC_FACET_EXTRACTION', elapsed_seconds=time.monotonic()-started,
        output_npz=str(OUT/'joint-interface-facets.npz'),
        output_npz_sha256=sha(OUT/'joint-interface-facets.npz'),
        point_weights_csr=str(OUT/'point-interpolation-weights.npz'),
        point_weights_csr_sha256=sha(OUT/'point-interpolation-weights.npz'),
        point_weights_shape=list(result.point_interpolation_weights.shape),
        point_weights_nnz=int(result.point_interpolation_weights.nnz),
        parent_foundation_tet_rows_are_not_unique=True,
        parent_foundation_tet_rows_unique_count=int(len(np.unique(parent_foundation))),
        diagnostics=result.diagnostics)
except BaseException as exc:
    # Preserve exact local geometry witnesses when the owner fails closed.
    tb=exc.__traceback__
    owner_locals=None
    while tb is not None:
        if tb.tb_frame.f_code.co_name == 'extract_joint_zero_level_facets':
            owner_locals=tb.tb_frame.f_locals
            break
        tb=tb.tb_next
    if owner_locals is not None and 'collapsed' in owner_locals:
        bad=np.flatnonzero(owner_locals['collapsed'])
        all_facets=owner_locals['facets']
        all_iids=owner_locals['interface_ids']
        face_items=owner_locals['ordered_face_items']
        face_parent_rows=[]
        face_parent_indptr=[0]
        for idx in bad:
            entry=face_items[int(idx)][1]
            rows=sorted(map(int,entry['parents']))
            face_parent_rows.extend(rows)
            face_parent_indptr.append(len(face_parent_rows))
        bad_tri=all_facets[bad]
        np.savez_compressed(OUT/'collapsed-facet-witnesses.npz',
            failed_output_facet_rows=bad.astype(np.int64),
            triangle_vertex_rows=bad_tri.astype(np.int64),
            interface_ids=all_iids[bad].astype(np.int8),
            triangle_points_f32=owner_locals['Pout32'][bad_tri],
            triangle_points_f64=owner_locals['Pout'][bad_tri],
            facet_parent_indptr=np.asarray(face_parent_indptr,dtype=np.int64),
            facet_parent_input_tet_rows=np.asarray(face_parent_rows,dtype=np.int64),
            input_tetrahedra=T[np.asarray(face_parent_rows,dtype=np.int64)] if face_parent_rows else np.empty((0,4),dtype=np.int32),
            input_parent_foundation_tet_rows=parent_foundation[np.asarray(face_parent_rows,dtype=np.int64)] if face_parent_rows else np.empty(0,dtype=np.int32),
            input_phi_inner_m=phiI_use,input_phi_deep_m=phiD_use,
            original_phi_inner_m=phiI_original,original_phi_deep_m=phiD_original,local_hskin_m=hskin_local)
        report['collapsed_facet_witnesses']={
            'path':str(OUT/'collapsed-facet-witnesses.npz'),
            'sha256':sha(OUT/'collapsed-facet-witnesses.npz'),
            'failed_facet_rows':bad.tolist(),
            'interface_ids':all_iids[bad].astype(int).tolist(),
            'input_parent_tet_rows':face_parent_rows,
            'input_parent_foundation_tet_rows':parent_foundation[np.asarray(face_parent_rows,dtype=np.int64)].astype(int).tolist() if face_parent_rows else [],
            'total_facets_before_f32_gate':int(len(all_facets)),
            'facets_interface_1':int(np.count_nonzero(all_iids==1)),
            'facets_interface_2':int(np.count_nonzero(all_iids==2)),
        }
    report.update(status='FAILED_STATIC_FACET_EXTRACTION', elapsed_seconds=time.monotonic()-started,
        error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    (OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise
(OUT/'run-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
