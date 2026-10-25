from __future__ import annotations
import hashlib, importlib, json, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069')
BASE = ROOT/'connected-volume-extraction-157/true-o-foundation-159'
FOUNDATION = BASE/'run-003-conversion/true-o-tetgen-foundation.npz'
RUN004 = BASE/'run-004-registered-fields/registered-levelsets.npz'
RUN013 = BASE/'run-013-conforming-field-refinement'
CHECKPOINT = RUN013/'refined-foundation-checkpoint.npz'
FIELDS = RUN013/'registered-levelsets.npz'
DAG = RUN013/'point-interpolation-dag.npz'
OUT = BASE/'run-028-closed-volume-sign/run-002'
PRED_IMPORTS = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/exact-audit-coverage-owner-216/src')
EXPECTED = {
    'foundation':'f4f50a725d51b0dd5d0433c6078e6913c64fe0e9cf4ea676aa0f01fe8ce76ddb',
    'run004':'8530e57f83de88862da6f3c258b0ee68befe9b83096316556b808d42750e2e81',
    'checkpoint':'cf51ebb33721a351be149f822f49f0737af136e66eb95a39924f10bbe6e3f179',
    'fields':'67ad75b444382ba864f244dc57b1ac073664379a409963eb6813972fa6091769',
    'dag':'f5b5385e14c26219bba54c189b3a784264cf356f36e8c3e6c1a151d520e68c76',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def edge_hist(faces):
    e=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
    e=np.sort(e,axis=1)
    _,counts=np.unique(e,axis=0,return_counts=True)
    vals,cnts=np.unique(counts,return_counts=True)
    return {str(int(v)):int(c) for v,c in zip(vals,cnts)}

def face_chain_certificate(tets,boundary_faces,points):
    t=np.asarray(tets,dtype=np.int64)
    p=np.asarray(points,dtype=np.float64)
    xyz=p[t]
    det=np.einsum('ij,ij->i',xyz[:,1]-xyz[:,0],np.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,0]))
    positive=bool(np.all(det>0.0))
    # Induced oriented simplex boundary for positively oriented tetrahedra.
    fs=np.concatenate((t[:,[1,2,3]],t[:,[0,2,3]],t[:,[0,1,3]],t[:,[0,1,2]]),axis=0)
    coeff=np.concatenate([np.full(len(t),x,dtype=np.int8) for x in (1,-1,1,-1)])
    inv=((fs[:,0]>fs[:,1]).astype(np.int8)+(fs[:,0]>fs[:,2]).astype(np.int8)+(fs[:,1]>fs[:,2]).astype(np.int8))
    signs=coeff*np.where((inv&1)==0,1,-1).astype(np.int8)
    keys=np.sort(fs,axis=1)
    unique,inv_idx,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    chain=np.bincount(inv_idx,weights=signs,minlength=len(unique)).astype(np.int64)
    bad_count=np.flatnonzero((counts!=1)&(counts!=2))
    interior=np.flatnonzero(counts==2)
    boundary=np.flatnonzero(counts==1)
    bad_orientation=interior[np.abs(chain[interior])!=0]
    expected=np.sort(np.asarray(boundary_faces,dtype=np.int64),axis=1)
    got=unique[boundary]
    boundary_match=len(expected)==len(got) and np.array_equal(np.sort(expected,axis=0),np.sort(got,axis=0))
    # Use row tuples for unambiguous triangle-set comparison; sorted by lexicographic key.
    if len(expected)==len(got):
        eo=np.lexsort((expected[:,2],expected[:,1],expected[:,0]))
        go=np.lexsort((got[:,2],got[:,1],got[:,0]))
        boundary_match=np.array_equal(expected[eo],got[go])
    signed_boundary_vol=np.einsum('ij,ij->i',p[boundary_faces[:,0]],np.cross(p[boundary_faces[:,1]],p[boundary_faces[:,2]])).sum()/6.0
    tet_vol=float(det.sum()/6.0)
    src=np.concatenate((np.asarray(npz_source_O,dtype=np.int64),np.asarray(npz_caps,dtype=np.int64)),axis=0)
    src_keys=np.sort(src,axis=1)
    so=np.lexsort((src_keys[:,2],src_keys[:,1],src_keys[:,0]))
    bo=np.lexsort((expected[:,2],expected[:,1],expected[:,0]))
    source_shell_match=len(src_keys)==len(expected) and np.array_equal(src_keys[so],expected[bo])
    return {
      'tet_count':int(len(t)),'positive_f32_signed_tets':positive,
      'min_signed_det_m3':float(det.min()),'total_positive_tet_volume_m3':tet_vol,
      'signed_boundary_volume_m3':float(signed_boundary_vol),
      'relative_tet_boundary_volume_error':float(abs(tet_vol-abs(signed_boundary_vol))/max(tet_vol,1e-300)),
      'unique_face_count':int(len(unique)),'interior_face_count':int(len(interior)),
      'boundary_face_count_from_chain':int(len(boundary)),'nonmanifold_face_count':int(len(bad_count)),
      'interior_orientation_conflicts':int(len(bad_orientation)),
      'boundary_triangle_set_matches_checkpoint':bool(boundary_match),
      'boundary_triangle_set_matches_source_O_plus_caps':bool(source_shell_match),
      'boundary_edge_incidence_histogram':edge_hist(boundary_faces),
      'source_O_face_count':int(len(npz_source_O)),'temporary_cap_face_count':int(len(npz_caps)),
    }

# Globals used only within the certificate helper to keep the audit inputs explicit in main.
npz_source_O=None
npz_caps=None

def main():
    global npz_source_O,npz_caps
    OUT.mkdir(parents=True,exist_ok=False)
    paths={'foundation':FOUNDATION,'run004':RUN004,'checkpoint':CHECKPOINT,'fields':FIELDS,'dag':DAG}
    pins={k:sha(p) for k,p in paths.items()}
    for k,want in EXPECTED.items():
        if want and pins[k]!=want:
            raise RuntimeError(f'{k} pin mismatch: {pins[k]} != {want}')
    with np.load(FOUNDATION,allow_pickle=False) as z: V={k:z[k] for k in z.files}
    with np.load(RUN004,allow_pickle=False) as z: B={k:z[k] for k in z.files}
    with np.load(CHECKPOINT,allow_pickle=False) as z: C={k:z[k] for k in z.files}
    with np.load(FIELDS,allow_pickle=False) as z: F={k:z[k] for k in z.files}
    with np.load(DAG,allow_pickle=False) as z: D={k:z[k] for k in z.files}
    P=np.asarray(V['points_world_m'],dtype=np.float32)
    T=np.asarray(V['tetrahedra_zero_based'],dtype=np.int32)
    BF=np.asarray(C['boundary_faces_zero_based'],dtype=np.int32)
    npz_source_O=np.asarray(C['source_O_faces'],dtype=np.int32)
    npz_caps=np.asarray(C['temporary_cap_faces'],dtype=np.int32)
    Q=np.asarray(C['points_world_m'],dtype=np.float32)
    lev=np.asarray(F['registered_levelsets_m'],dtype=np.float64)
    if not np.array_equal(Q[:len(P)],P): raise RuntimeError('refined original prefix differs from 159 foundation')
    if not np.array_equal(BF,V['boundary_faces_zero_based']): raise RuntimeError('checkpoint boundary differs from foundation boundary')
    chain=face_chain_certificate(T,BF,P)
    if not all(chain[k] for k in ('positive_f32_signed_tets','boundary_triangle_set_matches_checkpoint','boundary_triangle_set_matches_source_O_plus_caps')) or chain['nonmanifold_face_count'] or chain['interior_orientation_conflicts'] or chain['boundary_edge_incidence_histogram']!={'2':26322}:
        raise RuntimeError('closed positive TetGen chain certificate failed')
    neg=np.flatnonzero(lev[:,0]<0)
    if len(neg)!=670: raise RuntimeError(f'expected 670 negative phiO rows, got {len(neg)}')
    base_face=np.asarray(B['nearest_outer_face_row'],dtype=np.int64)
    base_bary=np.asarray(B['nearest_outer_barycentric'],dtype=np.float64)
    base_amb=np.asarray(B['nearest_open_boundary_feature'],dtype=bool)
    newrows=np.asarray(D['new_node_rows'],dtype=np.int64)
    newface=np.asarray(D['nearest_outer_face_rows'],dtype=np.int64)
    newbary=np.asarray(D['nearest_outer_barycentrics'],dtype=np.float64)
    newamb=np.asarray(D['ambiguous_open_feature'],dtype=bool)
    newlookup={int(r):j for j,r in enumerate(newrows)}
    sys.path.insert(0,str(PRED_IMPORTS))
    import numilab_human
    import numilab_human.cardiac_cavity_intersections as exact
    prepared=exact.prepare_float32_point_location_meters(P,BF)
    records=[]
    t0=time.time()
    for i,r0 in enumerate(neg):
        r=int(r0); q=tuple(Fraction.from_float(float(x)) for x in Q[r])
        loc=exact.point_location_prepared_indexed(q,prepared)
        if r<len(P): frow=int(base_face[r]); bary=base_bary[r]; amb=bool(base_amb[r]); kind='original-foundation-row'
        else:
            j=newlookup.get(r)
            if j is None: raise RuntimeError(f'new row {r} has no interpolation DAG record')
            frow=int(newface[j]); bary=newbary[j]; amb=bool(newamb[j]); kind='refined-interpolation-row'
        if loc['location'] not in ('inside','outside','boundary'):
            raise RuntimeError(f'exact point locator unresolved row={r}: {loc}')
        if r<len(P): parent_tets=[]
        else:
            j=newlookup[r]
            lo=int(D['parent_cell_indptr'][j]); hi=int(D['parent_cell_indptr'][j+1])
            parent_tets=sorted(set(map(int,D['parent_foundation_tet_rows'][lo:hi])))
        records.append({'query_row':r,'xyz_f32_m':Q[r].astype(float).tolist(),
          'phi_outer_original_m':float(lev[r,0]),'closed_O_plus_cap_location':loc['location'],
          'exact_ray':loc.get('ray'),'exact_crossings':loc.get('crossings'),'exact_attempts':loc.get('attempts'),
          'sample_kind':kind,'node_role':int(C['node_role'][r]),'node_source_input_row':int(C['node_source_input_row'][r]),
          'nearest_open_O_face_row':frow,'nearest_open_barycentrics':np.asarray(bary,dtype=float).tolist(),
          'nearest_open_feature_ambiguous':amb,'parent_foundation_tet_rows':parent_tets})
        if (i+1)%100==0: print(json.dumps({'classified':i+1,'total':len(neg),'elapsed_s':time.time()-t0}),flush=True)
    loc_counts={k:sum(x['closed_O_plus_cap_location']==k for x in records) for k in ('inside','outside','boundary')}
    report={'status':'ALL_NEGATIVE_OUTER_FIELD_ROWS_EXACTLY_CLASSIFIED',
      'input_sha256':pins,'query_point_source':'run013 refined-foundation-checkpoint.npz points_world_m, Float32 bits preserved and exact rationalized',
      'closed_volume_certificate':chain,
      'negative_phi_outer_rows':int(len(neg)),'exact_location_counts':loc_counts,
      'all_negative_rows_inside_closed_volume':bool(loc_counts['inside']==len(neg)),
      'elapsed_exact_query_s':time.time()-t0,'records':records,
      'interpretation':'Exact parity is interpreted as solid membership only after the positive tetrahedral chain, orientation cancellation, and exact O+temporary-cap boundary-set certificate pass. Temporary caps remain construction-only.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(OUT/'report.json'),'report_sha256':sha(OUT/'report.json'),
      'topology':chain,'counts':loc_counts,'elapsed_s':report['elapsed_exact_query_s']},indent=2),flush=True)

if __name__=='__main__': main()
