#!/usr/bin/env python3
"""Hash-bound postprocessor for result-004 rowwise epsilon pass intervals."""
import argparse, hashlib, json, math, pathlib, sys
import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/source125-fp64-smooth-direction-sweep-006')
INPUT = BASE / 'result-004'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def finite_or_none(x):
    x=float(x)
    return x if math.isfinite(x) else None

def intervals(mask):
    out=[]; start=None
    for i,v in enumerate(mask.tolist()+[False]):
        if v and start is None: start=i
        elif not v and start is not None:
            out.append((start,i-1)); start=None
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=pathlib.Path, default=BASE/'result-005')
    args=ap.parse_args()
    src_report=INPUT/'report.json'; src_npz=INPUT/'rowwise-epsilon.npz'; src_manifest=INPUT/'manifest.json'
    bound={p.name:sha(p) for p in (src_report,src_npz,src_manifest)}
    manifest=json.loads(src_manifest.read_text())
    if bound['report.json'] != manifest['report_sha256'] or bound['rowwise-epsilon.npz'] != manifest['rowwise-epsilon.npz_sha256']:
        raise SystemExit('result-004 manifest/hash mismatch')
    z=np.load(src_npz,allow_pickle=False)
    powers=z['epsilon_power'].astype(int)
    tet=z['tet_id'].astype(int); mat=z['material_index'].astype(int)
    err=z['centered_local_relative_error']
    if err.shape != (len(powers),len(tet)) or len(set(powers.tolist())) != len(powers):
        raise SystemExit('unexpected rowwise array dimensions/grid')
    threshold=1e-6
    stable=(z['plus_projection_ok'] & z['minus_projection_ok'] &
            z['plus_branch_matches_center'] & z['minus_branch_matches_center'])
    passing=stable & np.isfinite(err) & (err <= threshold)
    rows=[]; permat={}; all_nondec=[]
    paper_materials=[0,1]
    excluded_materials={str(int(m)):int(np.count_nonzero(mat==m)) for m in sorted(set(mat.tolist())) if int(m) not in paper_materials}
    for m in paper_materials:
        indices=np.flatnonzero(mat==m)
        mrows=[]; interval_edge_counts={'high_epsilon_grid_edge_power14':0,'low_epsilon_grid_edge_power30':0}
        rows_no_decrease=[]; all_edge_rows=[]; multiple_runs=[]; rows_with_interior_run=[]
        selected_edge_counts={'high_epsilon_grid_edge_power14':0,'low_epsilon_grid_edge_power30':0,'interior':0}
        for j in indices:
            runs=intervals(passing[:,j])
            ir=[]
            for a,b in runs:
                values=err[a:b+1,j]
                dif=np.diff(values)
                nonincrease=int(np.count_nonzero(dif<=0.0))
                strictly_decrease=int(np.count_nonzero(dif<0.0))
                item={
                    'start_epsilon_power':int(powers[a]),
                    'end_epsilon_power':int(powers[b]),
                    'start_epsilon':float(2.0**(-int(powers[a]))),
                    'end_epsilon':float(2.0**(-int(powers[b]))),
                    'grid_points':int(b-a+1),
                    'local_relative_error_by_grid_point':[float(x) for x in values],
                    'minimum_local_relative_error':float(np.min(values)),
                    'maximum_local_relative_error':float(np.max(values)),
                    'adjacent_error_steps_nonincreasing_as_epsilon_shrinks':nonincrease,
                    'adjacent_error_steps_strictly_decreasing_as_epsilon_shrinks':strictly_decrease,
                    'touches_high_epsilon_grid_edge':bool(a==0),
                    'touches_low_epsilon_grid_edge':bool(b==len(powers)-1),
                }
                ir.append(item)
                if a==0: interval_edge_counts['high_epsilon_grid_edge_power14']+=1
                if b==len(powers)-1: interval_edge_counts['low_epsilon_grid_edge_power30']+=1
            if len(ir)>1: multiple_runs.append(int(tet[j]))
            # Descriptive deterministic selection only: widest maximal run, with ties
            # resolved to the run starting at the earliest (largest) epsilon. Never
            # select by minimum observed error.
            selected=None
            if ir:
                selected=max(ir,key=lambda x:(x['grid_points'],-x['start_epsilon_power']))
                selected_idx=(selected['start_epsilon_power']-int(powers[0])) + (selected['grid_points']-1)//2
                seq=err[:,j]
                any_nonincrease=any(np.any(np.diff(seq[a:b+1])<=0.0) for a,b in runs)
                if not any_nonincrease:
                    rows_no_decrease.append(int(tet[j]))
            else:
                selected_idx=None
                any_nonincrease=False
            if ir and all(x['touches_high_epsilon_grid_edge'] or x['touches_low_epsilon_grid_edge'] for x in ir):
                all_edge_rows.append(int(tet[j]))
            if any(not x['touches_high_epsilon_grid_edge'] and not x['touches_low_epsilon_grid_edge'] for x in ir):
                rows_with_interior_run.append(int(tet[j]))
            if selected is not None:
                if selected['touches_high_epsilon_grid_edge']:
                    selected_edge_counts['high_epsilon_grid_edge_power14'] += 1
                elif selected['touches_low_epsilon_grid_edge']:
                    selected_edge_counts['low_epsilon_grid_edge_power30'] += 1
                else:
                    selected_edge_counts['interior'] += 1
            rec={
                'tet_id':int(tet[j]),
                'material_index':int(mat[j]),
                'center_jacobian_condition_2norm':finite_or_none(z['center_jacobian_condition_2norm'][j]),
                'center_scaled_residual':finite_or_none(z['center_scaled_residual'][j]),
                'center_dynamic_correction':finite_or_none(z['center_dynamic_correction'][j]),
                'center_stress_frobenius_Pa':finite_or_none(z['center_stress_frobenius_Pa'][j]),
                'center_tangent_frobenius_Pa_per_s':finite_or_none(z['center_tangent_frobenius_Pa_per_s'][j]),
                'observed_center_branch_signature':{
                    'lambda_active':int(z['center_lambda_active'][j]),
                    'yield_residual_sign':int(z['center_yield_residual_sign'][j]),
                    'fb_rationalized_path':int(z['center_fb_rationalized_path'][j]),
                },
                'passing_intervals':ir,
                'deterministic_descriptive_representative':None if selected_idx is None else {
                    'rule':'longest maximal pass interval; tie by earliest start (largest epsilon); then lower midpoint grid index',
                    'epsilon_power':int(powers[selected_idx]),
                    'epsilon':float(2.0**(-int(powers[selected_idx]))),
                    'local_relative_error':finite_or_none(err[selected_idx,j]),
                    'not_an_independent_gate_or_minimum_error_selection':True,
                },
                'any_adjacent_nonincreasing_error_step_inside_a_passing_interval_as_epsilon_shrinks':bool(any_nonincrease),
            }
            mrows.append(rec)
        runsizes=[x['grid_points'] for r in mrows for x in r['passing_intervals']]
        pass_counts=np.count_nonzero(passing[:,indices],axis=1)
        summary={
            'material_index':int(m),
            'material_name':json.loads(src_report.read_text())['rowwise_resolved_window_audit']['materials'][str(m)]['material_name'],
            'rows':len(indices),
            'rows_with_at_least_one_passing_interval':sum(bool(r['passing_intervals']) for r in mrows),
            'rows_with_at_least_one_interval_of_three_or_more_adjacent_grid_points':sum(any(x['grid_points']>=3 for x in r['passing_intervals']) for r in mrows),
            'minimum_longest_interval_grid_points':min((max((x['grid_points'] for x in r['passing_intervals']),default=0) for r in mrows),default=0),
            'median_longest_interval_grid_points':float(np.median([max((x['grid_points'] for x in r['passing_intervals']),default=0) for r in mrows])),
            'maximum_longest_interval_grid_points':max((max((x['grid_points'] for x in r['passing_intervals']),default=0) for r in mrows),default=0),
            'total_maximal_pass_intervals':len(runsizes),
            'intervals_touching_grid_edges':interval_edge_counts,
            'longest_interval_representative_edge_class_counts':selected_edge_counts,
            'rows_with_multiple_disjoint_pass_intervals':len(multiple_runs),
            'rows_where_every_passing_interval_touches_an_outer_grid_edge':len(all_edge_rows),
            'rows_with_no_adjacent_nonincreasing_error_pair_inside_any_passing_interval':len(rows_no_decrease),
            'tet_ids_with_no_adjacent_nonincreasing_error_pair':rows_no_decrease,
            'tet_ids_with_multiple_disjoint_intervals':multiple_runs,
            'tet_ids_where_every_interval_touches_an_outer_grid_edge':all_edge_rows,
            'rows_with_at_least_one_interior_passing_interval':len(rows_with_interior_run),
            'tet_ids_with_at_least_one_interior_passing_interval':rows_with_interior_run,
            'passing_rows_per_epsilon_power':{str(int(p)):int(c) for p,c in zip(powers,pass_counts)},
            'rows':mrows,
        }
        permat[str(m)]=summary
        rows.extend(mrows)
    all_paper=np.flatnonzero(np.isin(mat,paper_materials))
    shared_counts=np.count_nonzero(passing[:,all_paper],axis=1)
    max_shared=int(np.max(shared_counts))
    report={
        'schema':'numi.cardboard.source125-fp64-smooth-direction-rowwise-interval-addendum.v1',
        'evidence_class':'source-bound CPU-only independent FP64 directional derivative diagnostic; not native or physical qualification',
        'input_bindings':bound,
        'grid':{'epsilon_powers':[int(x) for x in powers], 'epsilon_definition':'2^(-power)', 'predeclared_grid':True},
        'row_pass_rule':{
            'endpoint_projections':'both plus and minus strict diagnostic projections passed the recorded tightened diagnostic gates',
            'branch_predicates':'plus and minus observed lambda-active, yield-residual-sign, and FB-rationalized-path signatures match center',
            'local_comparison':'centered local tangent/secant relative Frobenius error <=1e-6',
            'all_material_rows_reported':True,
        },
        'interval_and_representative_selection':{
            'intervals':'report every maximal contiguous run of grid points meeting the row_pass_rule',
            'descriptive_representative':'longest maximal run; ties by earliest start (largest epsilon); representative is lower midpoint grid index; never chosen by minimum observed error',
            'no_global_epsilon_claim':True,
        },
        'limitations':[
            'Intervals are bounded by the tested grid. A run touching power14 or power30 has no evidence beyond that grid edge.',
            'Rows without an adjacent nonincreasing error step can still remain below the 1e-6 gate across a pass interval; this is reported and not described as a truncation-to-roundoff convergence regime.',
            'This pointwise local comparison does not certify native FP32 finite differences, the full coupon, physical assembly, or calibration.',
            'The separate assembled A-action 1% comparison remains the shared epsilon=2^-22 result in result-004; it is not substituted for local per-row results.',
        ],
        'materials':permat,
        'pointwise_gate_summary':{
            'paper_rows':int(len(all_paper)),
            'rows_with_at_least_three_adjacent_predeclared_grid_points_passing_all_row_gates':int(sum(any(x['grid_points']>=3 for x in row['passing_intervals']) for summary in permat.values() for row in summary['rows'])),
            'all_paper_rows_pass_pointwise_three_adjacent_grid_point_gate':bool(all(any(x['grid_points']>=3 for x in row['passing_intervals']) for summary in permat.values() for row in summary['rows'])),
            'maximum_rows_passing_on_one_shared_epsilon_grid_point':max_shared,
            'epsilon_powers_attaining_maximum_shared_row_count':[int(powers[i]) for i,x in enumerate(shared_counts) if int(x)==max_shared],
            'no_single_shared_epsilon_is_required_for_the_pointwise_gate':True,
        },
        'nonpaper_material_rows_excluded_from_pointwise_1728_row_gate':excluded_materials,
    }
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    (out/'rowwise-window-intervals.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    readme=f'''# Rowwise epsilon interval addendum for source125 result-004\n\nThis postprocessor binds and re-reads the immutable `result-004` report, manifest, and compressed rowwise NPZ. It reports every maximal contiguous interval on the fixed epsilon grid where both endpoint projections pass, the recorded branch predicates match center, and the centered local relative error is at most 1e-6.\n\nThe descriptive representative is the midpoint of the longest passing interval, with ties resolved to the earliest start (largest epsilon); no point is selected by minimizing error. All intervals, including disjoint intervals and grid-edge intervals, remain in the JSON. A window touching power14 or power30 is bounded by an untested edge.\n\nRows with no adjacent nonincreasing error pair are explicitly listed; passing under the threshold alone is not called a resolved truncation-to-roundoff plateau. The separate assembled 1% action result at 2^-22 is inherited from result-004 and remains a different scope. Native FP32 and physical qualification remain open.\n\nReproduce with:\n\n```sh\n/Users/home/.pyenv/versions/3.10.12/bin/python3 {BASE}/intervals.py --output {out}\n```\n'''
    (out/'README.md').write_text(readme)
    manifest_out={
        'schema':'numi.cardboard.source125-fp64-smooth-direction-rowwise-interval-addendum-manifest.v1',
        'input_result004':str(INPUT),
        'input_sha256':bound,
        'script_path':str(BASE/'intervals.py'),
        'script_sha256':sha(BASE/'intervals.py'),
        'outputs':{
            'README.md_sha256':sha(out/'README.md'),
            'rowwise-window-intervals.json_sha256':sha(out/'rowwise-window-intervals.json'),
        },
        'command':['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(BASE/'intervals.py'),'--output',str(out)],
        'execution':'CPU-only postprocessing; source125 result-004 outputs immutable',
    }
    (out/'manifest.json').write_text(json.dumps(manifest_out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(out),'report_sha256':sha(out/'rowwise-window-intervals.json'),'manifest_sha256':sha(out/'manifest.json'),'summary':{k:{x:v for x,v in q.items() if x not in ('rows','tet_ids_with_no_adjacent_nonincreasing_error','tet_ids_with_multiple_disjoint_intervals','tet_ids_where_every_interval_touches_an_outer_grid_edge','passing_rows_per_epsilon_power')} for k,q in permat.items()}},indent=2,allow_nan=False))

if __name__=='__main__': main()
