#!/usr/bin/env python3
"""Read-only aligned distance/direction comparison for the 597 and 598 target proposals.

Inputs are immutable completed NPZ/JSONL/report triplets. The script validates hashes,
185-bank completion, pair-ID coverage, source identity and shared inside samples before
measuring F32 emitted endpoint distances and angular caps. It performs no geometry query.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

BASE = Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
OUT = BASE / 'target-comparison-601'
CASES = {
    '598_outward_nearest': BASE / 'connected-fit-598/prepare-001/surface-targets',
    '597_v2_current_objective': BASE / 'source-current-targets-597/targets-1585-v2-001',
    '597_v4_nearest_angular': BASE / 'source-current-targets-597/targets-1585-v4-001',
}
BASE_ARRAYS = ('pair_index','face_rows','face_node_rows','barycentric_weights',
               'inside_sample_world_m','inside_sample_world_f32_m','bank_body',
               'bank_stable_bone_id','bank_triangle_stable_id','nearest_boundary_direction_world_unit')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def quantiles(x):
    a=np.asarray(x,dtype=np.float64)
    return {k:float(np.quantile(a,q)) for k,q in [('min',0),('p10',.1),('median',.5),('p90',.9),('max',1)]}


def load_case(name, folder):
    paths={k:folder/k for k in ('targets.npz','target-records.jsonl','report.json')}
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(p)
    pins={str(p.relative_to(BASE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths.values()}
    with np.load(paths['targets.npz'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    recs=[json.loads(line) for line in paths['target-records.jsonl'].read_text().splitlines() if line]
    report=json.loads(paths['report.json'].read_text())
    ids=[int(r['pair_index']) for r in recs]
    assert len(ids)==len(set(ids))==1585, (name,'record pair ids')
    assert arrays['pair_index'].shape==(1585,) and len(set(map(int,arrays['pair_index'])))==1585
    assert set(ids)==set(map(int,arrays['pair_index']))
    assert report.get('passing_targets')==1585 and report.get('banks')==185 and report.get('matched_159_complete') is True
    assert report.get('failure_counts')=={}
    byid={int(r['pair_index']):r for r in recs}
    # Make every method array row lookup use the common pair-ID order.
    idx={int(v):i for i,v in enumerate(arrays['pair_index'])}
    order=np.array([idx[p] for p in sorted(ids)],dtype=np.int64)
    common=sorted(ids)
    aa={k:v[order] for k,v in arrays.items() if v.ndim and v.shape[0]==1585}
    rr=[byid[p] for p in common]
    return {'name':name,'folder':folder,'paths':paths,'pins':pins,'arrays':aa,'records':rr,'ids':common,'report':report}


def vector(records, key, nested=None):
    vals=[]
    for r in records:
        v=r
        for k in (key if nested is None else nested): v=v[k]
        vals.append(v)
    return np.asarray(vals,dtype=np.float64)


def actual_direction(c):
    a=c['arrays']
    delta=a['target_world_f32_m'].astype(np.float64)-a['inside_sample_world_f32_m'].astype(np.float64)
    lengths=np.linalg.norm(delta,axis=1)
    assert np.all(lengths>0)
    return delta/lengths[:,None], lengths*1000.0


def summarize(c, source, current, raw_nearest):
    a=c['arrays']; proposal=a['proposal_direction_world_unit'].astype(np.float64)
    actual, dist=actual_direction(c)
    sd=np.einsum('ij,ij->i',actual,source)
    cd=np.einsum('ij,ij->i',actual,current)
    pd_s=np.einsum('ij,ij->i',proposal,source)
    pd_c=np.einsum('ij,ij->i',proposal,current)
    rd=np.clip(np.einsum('ij,ij->i',proposal,raw_nearest),-1,1)
    angle=np.degrees(np.arccos(rd))
    faces={}
    for r in c['records']:
        pr=r.get('source_normal_provenance',{})
        if pr.get('kind') is not None and pr.get('row') is not None:
            key=f"{pr['kind']}:{pr['row']}"; faces[key]=faces.get(key,0)+1
    return {
       'classification':c['report']['classification'], 'direction_mode':c['report'].get('direction_mode'),
       'direction_method':c['report'].get('direction_method'),
       'target_count':len(dist), 'actual_f32_endpoint_distance_mm':quantiles(dist),
       'actual_endpoint_gt4mm':int(np.count_nonzero(dist>4.0)),
       'actual_endpoint_gt12mm':int(np.count_nonzero(dist>12.0)),
       'proposal_dot_source':quantiles(pd_s), 'proposal_dot_current':quantiles(pd_c),
       'actual_endpoint_dot_source':quantiles(sd), 'actual_endpoint_dot_current':quantiles(cd),
       'actual_endpoints_both_caps_ge_0_05':int(np.count_nonzero((sd>=.05)&(cd>=.05))),
       'proposal_both_caps_ge_0_05':int(np.count_nonzero((pd_s>=.05)&(pd_c>=.05))),
       'proposal_source_cap_ge_0_05':int(np.count_nonzero(pd_s>=.05)),
       'proposal_current_cap_ge_0_05':int(np.count_nonzero(pd_c>=.05)),
       'proposal_rotation_from_raw_nearest_degrees':quantiles(angle),
       'unique_source_normal_provenance_rows':len(faces),
       'reported_source_only_fold_exception_count':c['report'].get('source_only_fold_exception_count'),
       'reported_joint_outward_direction_count':c['report'].get('joint_outward_direction_count'),
       'reported_minimum_source_hemisphere_dot':c['report'].get('minimum_source_hemisphere_dot'),
       'actual_distance_mm':dist,
       'actual_source_dot':sd, 'actual_current_dot':cd,
    }


def main():
    cases={n:load_case(n,p) for n,p in CASES.items()}
    old=cases['598_outward_nearest']; v2=cases['597_v2_current_objective']; v4=cases['597_v4_nearest_angular']
    ids=v2['ids']
    assert old['ids']==ids==v4['ids'], 'pair-ID coverage differs'
    # Exact shared geometry/identity and inside samples, aligned by pair ID.
    for key in BASE_ARRAYS:
        x=old['arrays'][key]; y=v2['arrays'][key]; z=v4['arrays'][key]
        assert x.dtype==y.dtype==z.dtype and x.shape==y.shape==z.shape, key
        assert x.tobytes()==y.tobytes()==z.tobytes(), f'bitwise alignment mismatch: {key}'
    # Source/current provenance and direction consistency, from the auditable records.
    src2=vector(v2['records'],None,nested=('source_current_outward_selection','source'))
    cur2=vector(v2['records'],None,nested=('source_current_outward_selection','current'))
    src4=vector(v4['records'],None,nested=('source_current_outward_selection','source'))
    cur4=vector(v4['records'],None,nested=('source_current_outward_selection','current'))
    srcold=old['arrays']['source_surface_normal_world_unit'].astype(np.float64)
    assert np.array_equal(src2,src4) and np.allclose(srcold,src2,rtol=0,atol=2e-15)
    assert np.array_equal(cur2,cur4)
    # Independent consistency checks between each record and saved array values.
    for c in (old,v2,v4):
        for arrkey,reckey in [('proposal_direction_world_unit','proposal_direction_world_unit'),
                              ('target_world_f32_m','target_world_f32_m'),
                              ('inside_sample_world_f32_m','inside_sample_world_f32_m')]:
            rec=vector(c['records'],None,nested=(reckey,))
            assert np.array_equal(rec,c['arrays'][arrkey].astype(np.float64)), (c['name'],arrkey)
        assert np.all(c['arrays']['target_exactly_outside_all185'])
    raw=old['arrays']['nearest_boundary_direction_world_unit'].astype(np.float64)
    s_old=summarize(old,src2,cur2,raw)
    s_v2=summarize(v2,src2,cur2,raw)
    s_v4=summarize(v4,src4,cur4,raw)
    d_old=s_old.pop('actual_distance_mm'); s_old.pop('actual_source_dot'); s_old.pop('actual_current_dot')
    d_v2=s_v2.pop('actual_distance_mm'); s_v2.pop('actual_source_dot'); s_v2.pop('actual_current_dot')
    d_v4=s_v4.pop('actual_distance_mm'); s_v4.pop('actual_source_dot'); s_v4.pop('actual_current_dot')
    ratios_v2=d_v2/d_old; ratios_v4=d_v4/d_old
    report={
      'schema':'target-distance-direction-comparison-601-v1',
      'owner':{'path':str(Path(__file__).relative_to(BASE)),'sha256':sha(Path(__file__)),'bytes':Path(__file__).stat().st_size,'python':sys.version.split()[0],'numpy':np.__version__},
      'classification':'read_only_target_comparison_no_geometry_query_no_fit_no_admission',
      'units':{'distance':'mm','coordinates_in_inputs':'m','direction_cosine':'dimensionless','angle':'degrees'},
      'input_hashes_and_sizes':{n:c['pins'] for n,c in cases.items()},
      'candidate_sha256_by_case':{n:c['report'].get('inputs',{}).get('candidate_sha256') for n,c in cases.items()},
      'alignment':{'pair_ids':len(ids),'pair_ids_equal':True,'unique_pair_ids_each':True,
        '185_bank_complete_each':True,'all_failure_counts_empty':True,'all_endpoints_outside_185':True,
        'bitwise_equal_arrays':list(BASE_ARRAYS),'source_normals_equal_v2_v4':True,
        'current_normals_equal_v2_v4':True,'598_saved_source_normals_match597_records_atol':2e-15,
        'distance_definition':'Euclidean norm(target_world_f32_m - inside_sample_world_f32_m), computed from saved F32 endpoints.'},
      'methods':{'598_outward_nearest':s_old,'597_v2_current_objective':s_v2,'597_v4_nearest_angular':s_v4},
      'paired_actual_endpoint_distance_ratio_vs_598':{
        '597_v2_current_objective':quantiles(ratios_v2),'597_v4_nearest_angular':quantiles(ratios_v4)},
      'interpretation_limits':[
        'These are target proposals only; no fit or clearance admission is performed here.',
        'The 597 methods preserve nine source-only fold exceptions where both caps are infeasible; their actual current-normal dot can be negative.',
        'The 597 v4 selector minimizes angular rotation toward the nearest-boundary reference within feasible source/current caps; it does not minimize ray-exit distance.',
        'The 598 result is the prior outward-nearest proposal and is retained for paired comparison, not as an accepted geometry.'
      ]}
    # Add exact source-only face-row accounting from the selected source-normal records.
    ex=[]
    for c in (v2,v4):
        pairs=[]
        for i,r in enumerate(c['records']):
            if not r['source_current_outward_selection']['joint_selection'].get('joint_source_current_feasible',False):
                pairs.append({'pair_index':int(ids[i]),'face_row':int(r['face_row'])})
        ex.append(pairs)
    assert ex[0]==ex[1] and len(ex[0])==9
    report['source_only_exception_rows']={'count':len(ex[0]),'same_v2_v4':True,'pairs':ex[0]}
    # Rehash all inputs immediately before writing, so output pins attest the analyzed bytes.
    after={n:{str(p.relative_to(BASE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in c['paths'].values()} for n,c in cases.items()}
    assert after=={n:c['pins'] for n,c in cases.items()}, 'input changed during analysis'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'comparison.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(OUT/'comparison.json'),'sha256':sha(OUT/'comparison.json'),
      'pair_ids':len(ids),'old598_median_mm':s_old['actual_f32_endpoint_distance_mm']['median'],
      'v2_median_mm':s_v2['actual_f32_endpoint_distance_mm']['median'],
      'v4_median_mm':s_v4['actual_f32_endpoint_distance_mm']['median'],
      'v4_gt4mm':s_v4['actual_endpoint_gt4mm'],'v4_gt12mm':s_v4['actual_endpoint_gt12mm'],
      'v4_both_cap_proposals':s_v4['proposal_both_caps_ge_0_05'],'source_only_exceptions':len(ex[0])},sort_keys=True))

if __name__=='__main__': main()
