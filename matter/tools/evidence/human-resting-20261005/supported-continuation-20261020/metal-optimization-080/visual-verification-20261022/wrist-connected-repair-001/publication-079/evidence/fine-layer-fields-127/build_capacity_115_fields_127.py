from __future__ import annotations
import csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

E = Path('/Users/n/Work/numi-human-connected-surface-069')
BASE = E / 'whole-reference-tetgen-084-bounded'
W_DIR = BASE / 'connectivity-001/run-001/transfer-map-075-084-002'
COMPACT = BASE / 'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
SOURCE = BASE / 'whole-skin-fat-reference-084-candidate.npz'
WPATH = W_DIR / 'Wnew-detail-to-084-nodes.npz'
SIDECAR = W_DIR / 'transfer-sidecar.npz'
WREPORT = W_DIR / 'transfer-report.json'
TRANSFER_ROWS = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv')
CAP_DIR = E / 'connected-capacity-reference-115'
CAP_CANDIDATE = CAP_DIR / 'candidate.npz'
CAP_REPORT = CAP_DIR / 'report.json'
BASE_FIELDS = E / 'fine-layer-fields-127/baseline-coarse-to-detail-layer-fields.npz'
BASE_REPORT = E / 'fine-layer-fields-127/report.json'
OUT = E / 'fine-layer-fields-127'
ASSET = OUT / 'capacity-115-coarse-to-detail-layer-fields.npz'
REPORT = OUT / 'capacity-115-report.json'
COLUMN_COUNT = 2411
SOURCE_NODE_COUNT = 3 * COLUMN_COUNT

EXPECTED = {
    'compact_084': '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    'source_084': 'd3d0abb590f5858de4a27e0362e98bc9b3088831183ea045096cefc4ebba26e1',
    'Wnew': '55329313bf9c2c3b613fcd7eb52dd9b7b453520dd1b4a14e0e0d6ac457ce8519',
    'sidecar': '050875e760a1b3e3ddbbf18da5aff145b1405eeb4d3aa5d6dc31a09b15908f5d',
    'transfer_report': '5452f7c24b9201734e0823a40a02d7cabcee6f945fd4755ea7acc5ee09fe8413',
    'transfer_rows': '71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
    'capacity_115_candidate': 'abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9',
    'capacity_115_report': 'a36ab99e27a574418b8aa1681644a6c30b7f15e759955312d31acbc978f53d6b',
    'baseline_field_asset': 'ff5d936d9c5f14df7e78a596134a03c5a6fd4f5c09a517203643ded28da8f5c1',
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def stats_mm(v):
    x=np.asarray(v,dtype=np.float64)*1000
    return {'count':int(x.size),'min':float(x.min()),'p01':float(np.quantile(x,.01)),
            'median':float(np.median(x)),'p99':float(np.quantile(x,.99)),
            'max':float(x.max()),'positive_count':int(np.count_nonzero(x>0))}

def neg_stats_m(v):
    x=np.asarray(v,dtype=np.float64)
    return {'negative_count':int(np.count_nonzero(x<0)),'min_m':float(x.min()),'max_m':float(x.max())}

def layer_fields(points_m, normals):
    p=np.asarray(points_m,dtype=np.float64)
    if p.shape!=(SOURCE_NODE_COUNT,3): raise ValueError(f'expected {(SOURCE_NODE_COUNT,3)}, got {p.shape}')
    o,i,d=p[:COLUMN_COUNT],p[COLUMN_COUNT:2*COLUMN_COUNT],p[2*COLUMN_COUNT:]
    skin=i-o; fat=d-i; total=d-o
    hs=np.linalg.norm(skin,axis=1); hf=np.linalg.norm(fat,axis=1)
    if not np.isfinite(hs).all() or not np.isfinite(hf).all() or np.any(hs<=0) or np.any(hf<=0):
        raise ValueError('nonpositive/nonfinite Euclidean thickness')
    n=np.asarray(normals,dtype=np.float64)
    return {'O':o,'I':i,'D':d,'skin_offset':skin,'fat_offset':fat,'total_offset':total,
            'hskin':hs,'hfat':hf,'skin_signed':np.einsum('ij,ij->i',skin,n),
            'fat_signed':np.einsum('ij,ij->i',fat,n)}


def validate_transfer_rows(W, active, expected_active=None):
    active=np.asarray(active,dtype=bool)
    if W.shape[0] != len(active): raise ValueError("active mask row count differs from W")
    rownnz=np.diff(W.indptr)
    rowsum=np.asarray(W.sum(axis=1)).reshape(-1)
    if np.any(active != (rownnz>0)): raise ValueError("active mask differs from W zero-row mask")
    if np.any(rownnz[~active] != 0) or np.any(rowsum[~active] != 0): raise ValueError("inactive rows must be exact zero rows")
    if np.any(rownnz[active] <= 0) or np.any(np.abs(rowsum[active]-1.0)>1e-7): raise ValueError("active W rows must have positive support and unit sum")
    if expected_active is not None and int(active.sum()) != int(expected_active): raise ValueError("unexpected active transfer row count")
    if not np.isfinite(W.data).all() or np.any(W.data < 0): raise ValueError("W contains invalid weights")
    return rownnz,rowsum,np.unique(W.indices)

def apply_W_supported_field(W, nodefield, surface_mask, active):
    x=np.asarray(nodefield,dtype=np.float64)
    mask=np.asarray(surface_mask,dtype=bool)
    active=np.asarray(active,dtype=bool)
    if x.ndim not in (1,2) or x.shape[0] != W.shape[1] or len(mask) != W.shape[1] or len(active) != W.shape[0]:
        raise ValueError("field/mask shape does not match W")
    rownnz=np.diff(W.indptr)
    if np.any(active != (rownnz>0)): raise ValueError("active mask differs from W zero-row mask")
    support=np.unique(W.indices)
    if np.any(~mask[support]): raise ValueError("W support lies outside declared surface nodes")
    if not np.isfinite(x[support]).all(): raise ValueError("W support has undefined/nonfinite field values")
    finite=np.where(mask[:,None],x,0.0) if x.ndim==2 else np.where(mask,x,0.0)
    y=np.asarray(W@finite,dtype=np.float64)
    if x.ndim==1: y=y.reshape(-1)
    y[~active]=np.nan
    return y

def interpolate_parent(tet,bary,field):
    t=np.asarray(tet,dtype=np.int64); b=np.asarray(bary,dtype=np.float64); f=np.asarray(field,dtype=np.float64)
    if t.shape!=(4,) or b.shape!=(4,) or np.any(t<0) or np.any(t>=len(f)): raise ValueError('bad parent')
    if not np.isfinite(b).all() or abs(float(b.sum())-1)>1e-10: raise ValueError('invalid barycentric row')
    return b @ f[t]

def main():
    t0=time.time()
    paths={'compact_084':COMPACT,'source_084':SOURCE,'Wnew':WPATH,'sidecar':SIDECAR,
           'transfer_report':WREPORT,'transfer_rows':TRANSFER_ROWS,
           'capacity_115_candidate':CAP_CANDIDATE,'capacity_115_report':CAP_REPORT,
           'baseline_field_asset':BASE_FIELDS}
    actual={k:sha256(p) for k,p in paths.items()}
    for k,v in EXPECTED.items():
        if actual[k]!=v: raise RuntimeError(f'{k} SHA mismatch {actual[k]} != {v}')
    compact=np.load(COMPACT,allow_pickle=False); source=np.load(SOURCE,allow_pickle=False)
    cap=np.load(CAP_CANDIDATE,allow_pickle=False); base=np.load(BASE_FIELDS,allow_pickle=False)
    w=np.load(WPATH,allow_pickle=False)
    W=csr_matrix((w['data'].astype(np.float64),w['indices'],w['indptr']),shape=tuple(w['shape']))
    W.sort_indices()
    srcR=np.asarray(source['source_reference_points_m'],dtype=np.float64)
    proposed=np.asarray(cap['proposed_reference_points_m'],dtype=np.float64)
    source_rows=np.asarray(cap['source_input_row'],dtype=np.int64)
    modified=np.asarray(cap['modified_D_rows'],dtype=np.int64)
    if srcR.shape!=(SOURCE_NODE_COUNT,3) or proposed.shape!=(7360,3): raise RuntimeError('source/candidate shape changed')
    if not np.array_equal(source_rows[:SOURCE_NODE_COUNT],np.arange(SOURCE_NODE_COUNT)):
        raise RuntimeError('capacity candidate source prefix no longer maps input rows directly')
    p115=proposed[:SOURCE_NODE_COUNT]
    if not np.array_equal(p115[:2*COLUMN_COUNT],srcR[:2*COLUMN_COUNT]):
        raise RuntimeError('capacity candidate changed O/I rows outside its stated D-only scope')
    changed=np.flatnonzero(np.any(p115!=srcR,axis=1))
    expected_changed=np.sort(modified)
    if len(modified)!=176 or not np.array_equal(np.sort(changed),expected_changed):
        raise RuntimeError(f'changed source rows do not equal 176 pinned D rows: {len(changed)} vs {len(modified)}')
    if not np.all((changed>=2*COLUMN_COUNT)&(changed<SOURCE_NODE_COUNT)):
        raise RuntimeError('capacity candidate changed a non-D source row')
    if len(np.asarray(cap['generated_ancestry_disagreement_rows']))!=0:
        raise RuntimeError('capacity candidate has unresolved generated-ancestry disagreements')

    transfer_report=json.loads(WREPORT.read_text())
    if transfer_report.get('075',{}).get('active_rows')!=54551 or transfer_report.get('075',{}).get('inactive_fixed_rows')!=7399 or transfer_report.get('Wnew',{}).get('nnz')!=W.nnz:
        raise RuntimeError('frozen W report contract mismatch')
    row=read_csv(TRANSFER_ROWS); row.sort(key=lambda r:int(r['detail_row']))
    if len(row)!=W.shape[0] or [int(x['detail_row']) for x in row]!=list(range(len(row))):
        raise RuntimeError('transfer row identity/order changed')
    active=np.asarray([int(x['active'])==1 for x in row],dtype=bool)
    rownnz,rowsum,support=validate_transfer_rows(W,active,expected_active=54551)
    if (~active).sum()!=7399: raise RuntimeError('W inactive row count changed')

    normals=np.asarray(base['coarse_source_O_winding_normal_unit'],dtype=np.float64)
    f=layer_fields(p115,normals)
    o_nodes=np.asarray(base['candidate_O_surface_output_node_rows'],dtype=np.int64)
    o_mask=np.zeros(7360,dtype=bool); o_mask[o_nodes]=True
    compact_rows=np.asarray(compact['source_input_row'],dtype=np.int64)
    src_tets=np.asarray(source['source_tetrahedra_zero_based'],dtype=np.int64)
    ancestry_out=np.asarray(compact['newnode_ancestry_output_row'],dtype=np.int64)
    ancestry_tet=np.asarray(compact['newnode_ancestry_source_tet_row'],dtype=np.int64)
    ancestry_bary=np.asarray(compact['newnode_ancestry_barycentric'],dtype=np.float64)
    byrow={}
    for j,r in enumerate(ancestry_out): byrow.setdefault(int(r),[]).append(j)
    source_fields={'hskin':np.tile(f['hskin'],3),'hfat':np.tile(f['hfat'],3),
                   'skin_offset':np.tile(f['skin_offset'],(3,1)),
                   'fat_offset':np.tile(f['fat_offset'],(3,1)),
                   'total_offset':np.tile(f['total_offset'],(3,1)),
                   'skin_signed':np.tile(f['skin_signed'],3),'fat_signed':np.tile(f['fat_signed'],3)}
    nodefields={k:np.full((7360,)+v.shape[1:] if v.ndim>1 else (7360,),np.nan) for k,v in source_fields.items()}
    generated=o_nodes[compact_rows[o_nodes]<0]
    if len(generated)!=53: raise RuntimeError(f'expected 53 generated O field rows, got {len(generated)}')
    max_nono=0.0; max_resid=0.0; max_disagreement=0.0
    for outrow in o_nodes:
        srow=int(compact_rows[outrow])
        if srow>=0:
            if srow>=COLUMN_COUNT: raise RuntimeError(f'O row maps outside O block: {srow}')
            for k,v in source_fields.items(): nodefields[k][outrow]=v[srow]
            continue
        entries=byrow.get(int(outrow),[])
        if not entries: raise RuntimeError(f'generated O row {outrow} lacks parent ancestry')
        pred={k:[] for k in source_fields}
        for j in entries:
            tr=int(ancestry_tet[j]); bary=ancestry_bary[j]
            if tr<0 or tr>=len(src_tets): raise RuntimeError('generated ancestry tet out of range')
            tet=src_tets[tr]
            non_o=float(np.abs(bary[tet>=COLUMN_COUNT]).sum()); max_nono=max(max_nono,non_o)
            if non_o>1e-10: raise RuntimeError(f'generated O row {outrow} is not on source O face: {non_o}')
            pos=float(np.linalg.norm(bary@srcR[tet]-np.asarray(compact['reference_points_m'])[outrow]))
            max_resid=max(max_resid,pos)
            if pos>1e-8: raise RuntimeError(f'generated O parent position residual too large: {pos}')
            for k,v in source_fields.items(): pred[k].append(interpolate_parent(tet,bary,v))
        for k,vals in pred.items():
            ref=np.asarray(vals[0])
            for val in vals[1:]:
                err=float(np.max(np.abs(np.asarray(val)-ref))); max_disagreement=max(max_disagreement,err)
                if err>1e-10: raise RuntimeError(f'generated O parent values disagree for {k}: {err}')
            nodefields[k][outrow]=ref

    if np.any(~o_mask[support]): raise RuntimeError('W support includes non-O surface node')
    fine={k:apply_W_supported_field(W,nv,o_mask,active) for k,nv in nodefields.items()}
    if np.any(fine['hskin'][active]<=0) or np.any(fine['hfat'][active]<=0): raise RuntimeError('fine Euclidean thickness is nonpositive')

    # Shared row/source ancestry, not repeated or re-inferred by nearest geometry.
    shared_names=['coarse_column_ids','coarse_source_global_node_ids_O','coarse_source_global_node_ids_I',
                  'coarse_source_global_node_ids_D','coarse_source_direct_id','coarse_source_virtual_face_vertex_ids',
                  'coarse_source_virtual_face_barycentric','coarse_donor_body','fine_detail_row','fine_detail_source_vertex_id',
                  'fine_active_mask','fine_W_row_nnz','fine_W_row_sum','fine_selected_source_face_row',
                  'fine_selected_source_face_stable_id','fine_selected_source_face_coarse_vertex_rows',
                  'fine_selected_source_face_barycentric','fine_parent_coarse_vertex_source_direct_ids',
                  'candidate_O_surface_output_node_rows','candidate_O_surface_node_field_defined_mask',
                  'generated_O_output_rows','generated_O_source_parent_tet_rows','generated_O_source_parent_barycentric']
    arrays={k:base[k] for k in shared_names}
    arrays.update({
        'capacity115_modified_D_source_rows':modified.astype(np.int32),
        'capacity115_changed_D_source_rows':changed.astype(np.int32),
        'capacity115_coarse_O_reference_m':f['O'],'capacity115_coarse_I_reference_m':f['I'],
        'capacity115_coarse_D_reference_m':f['D'],
        'capacity115_coarse_skin_offset_I_minus_O_m':f['skin_offset'],
        'capacity115_coarse_fat_offset_D_minus_I_m':f['fat_offset'],
        'capacity115_coarse_total_offset_D_minus_O_m':f['total_offset'],
        'capacity115_coarse_hskin_euclidean_m':f['hskin'],'capacity115_coarse_hfat_euclidean_m':f['hfat'],
        'capacity115_coarse_O_winding_normal_unit':normals,
        'capacity115_coarse_skin_signed_projection_on_O_winding_m':f['skin_signed'],
        'capacity115_coarse_fat_signed_projection_on_O_winding_m':f['fat_signed'],
        'capacity115_candidate_O_surface_hskin_m':nodefields['hskin'],
        'capacity115_candidate_O_surface_hfat_m':nodefields['hfat'],
        'capacity115_candidate_O_surface_skin_offset_m':nodefields['skin_offset'],
        'capacity115_candidate_O_surface_fat_offset_m':nodefields['fat_offset'],
        'capacity115_candidate_O_surface_total_offset_m':nodefields['total_offset'],
        'capacity115_candidate_O_surface_skin_signed_projection_m':nodefields['skin_signed'],
        'capacity115_candidate_O_surface_fat_signed_projection_m':nodefields['fat_signed'],
        'capacity115_fine_hskin_euclidean_m':fine['hskin'],
        'capacity115_fine_hfat_euclidean_m':fine['hfat'],
        'capacity115_fine_skin_offset_I_minus_O_m':fine['skin_offset'],
        'capacity115_fine_fat_offset_D_minus_I_m':fine['fat_offset'],
        'capacity115_fine_total_offset_D_minus_O_m':fine['total_offset'],
        'capacity115_fine_skin_signed_projection_on_O_winding_m':fine['skin_signed'],
        'capacity115_fine_fat_signed_projection_on_O_winding_m':fine['fat_signed'],
    })
    np.savez_compressed(ASSET,**arrays)
    report={
      'schema':'capacity_115_column_layer_fields_interpolated_to_075_detail_v1',
      'status':'FIELD_INTERPOLATION_DIAGNOSTIC_ONLY',
      'classification':'The 115 D-shortened values are a geometric construction counterfactual. Euclidean layer lengths are inferred from candidate reference coordinates, not measured anatomy; no geometry, material, native, or physical acceptance is claimed.',
      'field_semantics':{
        'input':'The original 084 O/I/D source coordinates are changed only at the exact 176 modified_D_rows listed in the pinned 115 report; original O and I remain bitwise identical.',
        'thickness':'hskin=norm(I-O), hfat=norm(D-I), Euclidean and inferred, not measured.',
        'offsets':'I-O, D-I, D-O in metres.',
        'fine_interpolation':'The exact pinned Wnew CSR is reused from 075->084 and acts only as a field interpolator over candidate O surface nodes. Generated O nodes use their frozen source-tetra parent barycentrics, checked to lie on the source O layer. W is not used as geometric correspondence.',
        'signed_projection':'Offset dot the area-weighted coarse O face-winding normal. This sign diagnostic is not thickness and is not clamped.',
        'shared_provenance':'Detail-row/source IDs, selected face row/stable ID, source face vertices/barycentrics, and coarse direct/virtual source anchor data are copied from the immutable 084 field asset and remain shared across fieldsets.',
        'inactive_rows':'Inactive W rows remain NaN for interpolated fields; they do not represent zero thickness.'
      },
      'counts':{'source_columns':COLUMN_COUNT,'source_layer_nodes':SOURCE_NODE_COUNT,
                'modified_D_rows':len(modified),'verified_changed_D_rows':len(changed),
                'candidate_O_surface_nodes':len(o_nodes),'generated_O_surface_nodes':len(generated),
                'W_support_nodes':len(support),'active_detail_rows':int(active.sum()),
                'inactive_detail_rows':int((~active).sum()),'fine_detail_rows':len(row)},
      'checks':{'O_and_I_source_rows_bitwise_unchanged':True,
                'modified_D_rows_exactly_match_changed_source_rows':True,
                'all_changed_rows_are_D':True,'candidate115_ancestry_disagreements_zero':True,
                'active_mask_matches_W':True,'W_rows_sum_to_one_active_and_zero_inactive':True,
                'W_support_is_candidate_O_surface_only':True,
                'generated_O_parent_position_max_residual_m':max_resid,
                'generated_O_parent_nonO_barycentric_l1_max':max_nono,
                'generated_O_parent_field_disagreement_max':max_disagreement,
                'W_active_row_sum_error_max':float(np.max(np.abs(np.asarray(W.sum(axis=1)).reshape(-1)[active]-1))),
                'changed_hskin_columns':int(np.count_nonzero(np.abs(f['hskin']-np.asarray(base['coarse_hskin_euclidean_m']))>0)),
                'changed_hfat_columns':int(np.count_nonzero(np.abs(f['hfat']-np.asarray(base['coarse_hfat_euclidean_m']))>0))},
      'ranges_mm':{'coarse_hskin_euclidean':stats_mm(f['hskin']),'coarse_hfat_euclidean':stats_mm(f['hfat']),
                   'fine_active_hskin_euclidean':stats_mm(fine['hskin'][active]),
                   'fine_active_hfat_euclidean':stats_mm(fine['hfat'][active]),
                   'coarse_skin_signed_projection_on_O_winding':neg_stats_m(f['skin_signed']),
                   'coarse_fat_signed_projection_on_O_winding':neg_stats_m(f['fat_signed']),
                   'fine_skin_signed_projection_on_O_winding':neg_stats_m(fine['skin_signed'][active]),
                   'fine_fat_signed_projection_on_O_winding':neg_stats_m(fine['fat_signed'][active])},
      'input_sha256':actual,
      'shared_084_field_asset_sha256':actual['baseline_field_asset'],
      'output':{'asset_path':str(ASSET),'asset_sha256':sha256(ASSET),'asset_bytes':ASSET.stat().st_size},
      'runtime':{'host':platform.node(),'python':sys.version,'numpy':np.__version__,'scipy':__import__('scipy').__version__,
                 'seconds':time.time()-t0,'builder_sha256':sha256(Path(__file__))}
    }
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'asset':str(ASSET),'asset_sha256':report['output']['asset_sha256'],
                      'report':str(REPORT),'counts':report['counts'],'ranges_mm':report['ranges_mm']},indent=2))

if __name__=='__main__': main()
