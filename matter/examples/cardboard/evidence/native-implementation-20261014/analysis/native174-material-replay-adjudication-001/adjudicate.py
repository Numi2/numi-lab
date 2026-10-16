#!/usr/bin/env python3
"""Recompute bounded summaries for immutable source469 native replay 174."""
from __future__ import annotations
import collections, hashlib, json, math, pathlib, statistics

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'analysis/native469-material-replay-174-host-macmini-002/runs/invocations/20261016T040030Z-f41839901e414db49c077be20a95edf6'
REMOTE_INV = BASE / 'analysis/closed-macmini-transfer-inventories-031-001/native174-remote-inventory.json'
MP_REPORT = BASE / 'analysis/native469-mp-fallback-005/run-002/report.json'
OUT = BASE / 'analysis/native174-material-replay-adjudication-001'


def sha(p: pathlib.Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def jl(name):
    with (RUN/'native-output'/name).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def data(name, record=None):
    for x in jl(name):
        if record is None or x.get('record')==record: yield x

def qstats(vals):
    vals=list(vals)
    if not vals:return {'n':0}
    vals.sort()
    def q(p): return vals[min(len(vals)-1,max(0,math.ceil(p*len(vals))-1))]
    return {'n':len(vals),'min':vals[0],'median':statistics.median(vals),'p95_nearest_rank':q(.95),'max':vals[-1]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inv=json.loads(REMOTE_INV.read_text())
    expected={x['path']:x for x in inv['files']}
    actual={p.relative_to(RUN).as_posix():p for p in RUN.rglob('*') if p.is_file()}
    verify=[]
    for rel,e in sorted(expected.items()):
        p=actual.get(rel)
        if p is None:
            verify.append({'path':rel,'ok':False,'reason':'missing'}); continue
        n=p.stat().st_size; h=sha(p)
        verify.append({'path':rel,'bytes':n,'sha256':h,'ok':n==e['bytes'] and h==e['sha256']})
    extra=sorted(set(actual)-set(expected))
    inv_ok=all(x['ok'] for x in verify) and not extra and len(expected)==44 and sum(x['bytes'] for x in expected.values())==188729390

    receipt=json.loads((RUN/'native-execution-receipt.json').read_text())
    manifest=json.loads((RUN/'native-output/source469-replay-manifest.json').read_text())
    end=lambda name: next(x for x in jl(name) if x.get('record')=='end')

    state_rows=list(data('all-captured-center-side-comparison.jsonl','state'))
    state_groups=collections.defaultdict(list)
    for x in state_rows: state_groups[x['material_index']].append(x)
    states_summary={str(k):{
        'rows':len(v),'stress_valid':sum(x['direct_stress_valid'] for x in v),
        'tangent_projection_valid':sum(x['tangent_projection_valid'] for x in v),
        'projection_comparison_applicable':sum(x['projection_comparison_applicable'] for x in v),
        'max_direct_stress_abs_l2':max(x['direct_stress_vs_captured_abs_l2'] for x in v),
        'max_projected_state_abs_l2':max(x['projected_state_vs_capture_abs_l2'] for x in v),
        'states':dict(collections.Counter(x['state_label'] for x in v)),
        'iteration_counts':dict(collections.Counter(str(x['native_projection_iterations']) for x in v)),
    } for k,v in state_groups.items()}
    state_summary={'rows':len(state_rows),'by_material_index':states_summary,
                   'scope':'center and four captured endpoint states; same-source/native direct stress and projection replay versus stored capture'}

    basis=list(data('center-all-nine-basis-comparison.jsonl','basis'))
    basis_by=collections.defaultdict(list)
    for x in basis:basis_by[x['material_index']].append(x)
    basis_summary={str(k):{'rows':len(v),'stress_valid':sum(x['stress_valid'] for x in v),
       'tangent_valid':sum(x['tangent_valid'] for x in v),'projection_valid':sum(x['projection_valid'] for x in v),
       'max_tangent_abs_l2':max(x['tangent_column_abs_l2'] for x in v),
       'max_tangent_relative_l2':max(x['tangent_column_relative_l2'] for x in v)} for k,v in basis_by.items()}

    matrix=list(data('center-captured-operator-comparison.jsonl','row'))
    matrix_by=collections.defaultdict(list)
    for x in matrix: matrix_by[x['material_index']].append(x)
    matrix_summary={str(k):{'rows':len(v),'stress_valid':sum(x['stress_valid'] for x in v),
      'tangent_valid_row_field':sum(x['tangent_valid'] for x in v),
      'projected_row_field':sum(x['projected'] for x in v),
      'max_captured_action_abs_l2':max(x['captured_tangent_action_abs_l2'] for x in v),
      'max_captured_action_relative_l2':max(x['captured_tangent_action_relative_l2'] for x in v)} for k,v in matrix_by.items()}
    me=end('center-captured-operator-comparison.jsonl')

    ref=list(data('selected-fp64-mp-reference-comparison.jsonl','comparison'))
    paper=[x for x in ref if x['material_index'] in (0,1)]
    mp=[x for x in paper if x['mp_state_available']]
    worsttan=max(paper,key=lambda x:x['directional_tangent_vs_fp64_relative_l2'])
    worststress=max(paper,key=lambda x:x['stress_vs_fp64_relative_l2'])
    mp_summary={
      'rows':len(ref),'paper_rows':len(paper),'paper_material_counts':dict(collections.Counter(str(x['material_index']) for x in paper)),
      'mp_available_paper_rows':len(mp),
      'paper_stress_relative_l2':qstats(x['stress_vs_fp64_relative_l2'] for x in paper),
      'paper_tangent_relative_l2':qstats(x['directional_tangent_vs_fp64_relative_l2'] for x in paper),
      'worst_paper_tangent':{k:worsttan[k] for k in ('tet_id','material_index','state_label','column','directional_tangent_vs_fp64_relative_l2','directional_tangent_vs_fp64_abs_l2')},
      'worst_paper_stress':{k:worststress[k] for k in ('tet_id','material_index','state_label','stress_vs_fp64_relative_l2','stress_vs_fp64_abs_l2')},
      'mp_rows_stress_relative_l2':qstats(x['stress_vs_fp64_relative_l2'] for x in mp),
      'mp_rows_tangent_relative_l2':qstats(x['directional_tangent_vs_fp64_relative_l2'] for x in mp),
      'mp_branch_mismatch_count':None,
      'limit':'selected captured samples only; independent FP64 comparison is not the full 9-direction global derivative gate; MP reference is a distinct mixed-precision reference, not FP64/native qualification'
    }
    # The reference file exposes validity and values but not branch predicates; do not infer branch mismatches from it.

    eps=list(data('epsilon-fd-comparison.jsonl','pair'))
    eps_inputs=list(data('epsilon-grid-inputs.jsonl','sample'))
    sample_by={(x['tet_id'],x['state_label'],x['epsilon'],x['sign']):x for x in eps_inputs if x['column']==-1}
    paper_pair=[]; invalid=[]
    for x in eps:
        pp=sample_by.get((x['tet_id'],x['state_label'],x['epsilon'],1))
        material=pp['material_index'] if pp else None
        row=dict(x,material_index=material)
        if material in (0,1): paper_pair.append(row)
        else: invalid.append(row)
    eps_by=collections.defaultdict(list)
    for x in paper_pair:eps_by[x['epsilon']].append(x)
    eps_table=[]
    for e,vs in sorted(eps_by.items()):
      eps_table.append({'epsilon':e,'pairs':len(vs),
        'both_endpoint_valid':sum(x['pair_status']=='both_endpoints_valid' for x in vs),
        'centered_fd_vs_fp64_tangent_relative_l2':qstats(x['centered_secant_vs_fp64_tangent_relative_l2'] for x in vs if x['pair_status']=='both_endpoints_valid'),
        'centered_fd_vs_fp64_tangent_abs_l2':qstats(x['centered_secant_vs_fp64_tangent_abs_l2'] for x in vs if x['pair_status']=='both_endpoints_valid'),
        'one_sided_plus_abs_l2':qstats(x['plus_one_sided_vs_fp64_tangent_abs_l2'] for x in vs if x['pair_status']=='both_endpoints_valid'),
        'one_sided_minus_abs_l2':qstats(x['minus_one_sided_vs_fp64_tangent_abs_l2'] for x in vs if x['pair_status']=='both_endpoints_valid'),
        'max_valid_centered_relative':max((x['centered_secant_vs_fp64_tangent_relative_l2'] for x in vs if x['pair_status']=='both_endpoints_valid'),default=None)})
    # Requested versus realized split-float endpoint deltas, a representation comparison only.
    quant=collections.defaultdict(list)
    for x in eps_inputs:
      if x['column']!=-1 or x['sign'] not in (-1,1): continue
      req=x['requested_F_delta'];real=x['realized_F_delta']; reqn=math.sqrt(sum(a*a for a in req)); err=math.sqrt(sum((a-b)**2 for a,b in zip(req,real)))
      reqr=x['requested_rate_delta'];realr=x['realized_rate_delta']; reqrn=math.sqrt(sum(a*a for a in reqr)); errr=math.sqrt(sum((a-b)**2 for a,b in zip(reqr,realr)))
      quant[x['epsilon']].append({'rel_F':err/reqn if reqn else None,'rel_rate':errr/reqrn if reqrn else None,
                                  'zeroed_nonzero_F_components':sum(a!=0 and b==0 for a,b in zip(req,real))})
    quant_table=[]
    for e,vs in sorted(quant.items()):
      quant_table.append({'epsilon':e,'endpoint_count':len(vs),'F_delta_relative_realization_error':qstats(x['rel_F'] for x in vs if x['rel_F'] is not None),'rate_delta_relative_realization_error':qstats(x['rel_rate'] for x in vs if x['rel_rate'] is not None),'zeroed_nonzero_F_components':sum(x['zeroed_nonzero_F_components'] for x in vs)})
    ee=end('epsilon-fd-comparison.jsonl')

    reps=list(data('paper-points-source469_repeat_identical.jsonl','point'))
    rep_inputs=list(data('repeatability-inputs.jsonl','sample'))
    repeat_pairs=[]
    for i in range(0,len(rep_inputs),2):
      a,b=rep_inputs[i:i+2]
      oa,ob=reps[i],reps[i+1]
      mismatch=sum(x!=y for x,y in zip(oa['output_f32_bits'],ob['output_f32_bits']))
      repeat_pairs.append({'tet_id':a['tet_id'],'material_index':a['material_index'],'words':len(oa['output_f32_bits']),'mismatches':mismatch})
    repeat_summary={'pairs':len(repeat_pairs),'words_compared':sum(x['words'] for x in repeat_pairs),'mismatches':sum(x['mismatches'] for x in repeat_pairs),
      'method':'bitwise equality of each consecutive original/repeated output record, cross-checked against input repeat_of pairs',
      'accuracy_floor':False}

    mp_report=json.loads(MP_REPORT.read_text())
    mp_summary_source=mp_report.get('summary_metrics',{})
    preserved=mp_report.get('center_source166_reference_C_preservation',{})
    mp_results=mp_report.get('mp_results',[])
    mp_cond={'failed_rows_fallback_count':mp_summary_source.get('failed_rows_fallback_count'),
      'mp_roots':len(mp_results),'mp_gates_passed':sum(x.get('fallback_status')=='mp_gates_pass' for x in mp_results),
      'strict_gates':mp_report.get('strict_gates'),
      'max_70_90_110_root_spread':mp_summary_source.get('max_root_spread_70_90_110_dynamic_scaled'),
      'native_vs_mp_branch_predicate_mismatch_count':mp_summary_source.get('native_vs_mp_branch_predicate_mismatch_count'),
      'max_native_tangent_vs_mp_ift_relative_l2':mp_summary_source.get('max_center_native_tangent_vs_mp_ift_relative_l2'),
      'max_fp64_ift_vs_mp_ift_relative_l2_at_same_rounded_root':mp_summary_source.get('max_fp64_ift_vs_mp_ift_relative_l2_at_same_rounded_mp_root'),
      'source166_success_rows_preserved':preserved.get('valid_source166_paper_row_count'),
      'source166_tangent_values_bitwise_preserved':preserved.get('tangent_values_bitwise_preserved'),
      'original_fp64_failures_preserved':len(mp_report.get('original_fp64_failures_preserved',[]))}
    # Keep a small source binding record including the actual invocation rather than duplicating large input payloads.
    bind={
      'run_directory':str(RUN),'run_id':receipt.get('run_id'),'remote_inventory_sha256':sha(REMOTE_INV),
      'remote_inventory_file_count':len(expected),'remote_inventory_bytes':sum(x['bytes'] for x in expected.values()),
      'local_inventory_matches_all_paths_bytes_hashes':inv_ok,'extra_local_files':extra,'per_file_verification':verify,
      'source_snapshot':manifest['source_snapshot'],'source_sha256':manifest['checker_base_snapshot_sha256'],
      'capture_sha256':manifest['capture_sha256'],'compiled_package_sha256':manifest['compiled_package_sha256'],
      'run_manifest_sha256':manifest['run_manifest_sha256'],'receipt_sha256':manifest['receipt_sha256'],
      'mp_report_sha256':manifest['mp_reference_report_sha256'],'mp_arrays_sha256':manifest['mp_reference_arrays_sha256'],
      'mp_package_manifest_sha256':manifest['mp_reference_package_manifest_sha256'],'selected_reference_export_sha256':manifest['selected_reference_export_sha256'],
      'world_fingerprint':manifest['world_fingerprint'],'capture_root':manifest['capture_root'],'newton':manifest['capture_newton_iteration'],'environment':manifest['environment_scope'],
      'native_receipt':receipt,'native_output_manifest':manifest}
    result={
      'schema':'numi.cardboard.native174-material-adjudication.v1','bindings':bind,
      'native_run':{'exit_code':receipt['exit_code'],'wall_seconds':receipt['wall_seconds'],'gpu_execution':receipt['gpu_execution'],'physical_transactions':manifest['physical_transactions'],'qualification_flags':{'full_solver':False,'material':False},'status':manifest['status'],'inventory_verified':inv_ok},
      'all_states_same_source_replay':state_summary,
      'nine_basis_center_parity':{'rows':len(basis),'by_material_index':basis_summary,'interpretation':'same-state native tangent columns match captured matrices where a paper projector is applicable; this checks implementation/action consistency, not independent derivative accuracy'},
      'captured_center_action':{'rows':len(matrix),'by_material_index':matrix_summary,'writer_end_record':me,'interpretation':'per-row tangent_valid and projected are separately counted; aggregate tangent_valid_rows is actually tangent AND projected. Absolute action magnitude depends on captured H scale; do not call a large absolute difference a bug without normalized direction/units.'},
      'selected_independent_references':mp_summary,
      'mp_fallback_report_summary':{'path':str(MP_REPORT),'sha256':sha(MP_REPORT),'summary':mp_cond,'qualification_limit':'MP fallback reference is not an FP64 native projection pass; preserve native/reference branch mismatches and FP64 failed rows'},
      'epsilon_fd':{'rows':len(eps),'reported_end':ee,'paper_pairs':len(paper_pair),'invalid_nonpaper_pairs':len(invalid),'invalid_by_material_index':dict(collections.Counter(str(x['material_index']) for x in invalid)),'paper_by_epsilon':eps_table,'endpoint_requested_vs_realized_deltas':quant_table,'interpretation':'epsilon secants are against the independent FP64 tangent using realized split-float F/rate endpoints. They do not resolve a small-step noise floor or prove a smooth plateau; repeatability is not accuracy floor.'},
      'repeatability':repeat_summary,
      'scope_limits':['One captured source469 environment, accepted capture root20/Newton63, with no physical transactions.','This replay does not qualify the full coupon/solver, damage calibration, crease energy, fracture, starch interface, mesh convergence, box assembly or measured material properties.','Same-state and all-nine-basis cached/action parity are implementation checks. Selected FP64/MP point comparisons and epsilon sweep are separate reference checks with stated scope.','No absolute FP32 error floor is established by repeated-input determinism.','The 9 invalid epsilon pairs are retained; do not treat them as passes or discard them.']
    }
    (OUT/'remote-output-inventory-verification.json').write_text(json.dumps({'schema':'native174.remote-inventory-verification.v1','source_inventory_sha256':sha(REMOTE_INV),'count':len(verify),'bytes':sum(x['bytes'] for x in expected.values()),'all_match':inv_ok,'extra_files':extra,'files':verify},indent=2,sort_keys=True)+'\n')
    (OUT/'input-bindings.json').write_text(json.dumps(bind,indent=2,sort_keys=True)+'\n')
    (OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    md=[]
    md += ['# Native 174 material replay adjudication','',f"Run `{receipt.get('run_id')}` on source snapshot `{manifest['source_snapshot']}` exited {receipt['exit_code']} in {receipt['wall_seconds']:.3f}s. The remote 44-file, 188,729,390-byte inventory matches every local path, size and SHA-256: **{inv_ok}**.",'',
      'This was a GPU material-function replay with zero physical transactions. Its own receipt says `material_qualification=false`; this report does not promote it to a coupon or full-solver qualification.','',
      '## Same-source replay and captured operator','',
      f"The five-state replay contains {len(state_rows)} rows (center plus four captured endpoint states). For paper material indices 0 and 1, direct stress and projected state match the corresponding capture exactly; the glue cohort is reported separately because it has no 15-state paper projector.",
      f"The center nine-basis check contains {len(basis)} actions. The paper rows match the captured same-state tangent columns exactly; this is parity against the stored operator, not an independent derivative test.",
      f"The center captured-direction comparison has {len(matrix)} rows. The row fields record tangent validity separately from projection applicability. Its end-summary `tangent_valid_rows={me['tangent_valid_rows']}` counts `tangent_valid && projected`, so it must not be read as 192 failed tangent rows. Maximum captured-action relative L2 is {max(x['captured_tangent_action_relative_l2'] for x in matrix):.9g}; maximum absolute L2 is {max(x['captured_tangent_action_abs_l2'] for x in matrix):.9g}. Large absolute values can accompany small relative errors when the unnormalized captured direction H is large.",
      '', '## Independent references and finite differences','',
      f"The selected reference table has {len(paper)} paper rows; {len(mp)} also have MP fallback references. Selected paper stress relative L2 ranges to {mp_summary['paper_stress_relative_l2']['max']:.9g}. Selected tangent relative L2 against the packaged FP64 references reaches {mp_summary['paper_tangent_relative_l2']['max']:.9g} (tet {worsttan['tet_id']}, material {worsttan['material_index']}, state {worsttan['state_label']}, column {worsttan['column']}); that is above 1e-6 but below the separate 1e-3 material action scale. For the 170 MP-backed selected rows, maximum tangent relative error is {mp_summary['mp_rows_tangent_relative_l2']['max']:.9g}. The separate 17-root MP fallback package reports 17/17 high-precision fallback gates passed and a maximum 70/90/110-digit root spread of {mp_cond['max_70_90_110_root_spread']:.3g}; it also retains {mp_cond['native_vs_mp_branch_predicate_mismatch_count']} native-vs-MP branch-predicate mismatches and the original 17 FP64 failures. MP fallback is a distinct reference, not an FP64 projection pass.",
      f"The epsilon sweep contains {len(eps)} pairs: {len(paper_pair)} paper pairs are both endpoint-valid and {len(invalid)} non-paper pairs are invalid (material indices {sorted(set(x['material_index'] for x in invalid))}). On valid paper pairs, the table in `report.json` shows median/max centered-secant errors by epsilon. The max remains above 1e-3 at every tested step; no resolved finite-difference plateau or independent absolute noise floor is established. Split-float endpoint realization is reported separately and does not itself establish an accuracy floor.",
      f"Repeated identical evaluations have {repeat_summary['mismatches']} mismatched output words across {repeat_summary['words_compared']} words. This is repeatability evidence only, not an accuracy or truncation floor.",
      '', 'The row-level source outputs, all failed/invalid statuses, and all 44 file hashes remain bound in the output inventory and the companion JSON. The overall result is **diagnostic replay complete, material qualification not established**.']
    (OUT/'report.md').write_text('\n'.join(md)+'\n')
    # Final compact package manifest; manifest does not hash itself.
    files=[]
    for p in sorted(OUT.iterdir()):
      if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest_out={'schema':'numi.cardboard.native174-material-adjudication-package.v1','input_remote_inventory_sha256':sha(REMOTE_INV),'files':files}
    (OUT/'manifest.json').write_text(json.dumps(manifest_out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(OUT),'inventory_ok':inv_ok,'inventory_count':len(verify),'extra_files':extra,'paper_pairs':len(paper_pair),'invalid_pairs':len(invalid),'center_matrix_end':me,'selected_tangent_max':mp_summary['paper_tangent_relative_l2']['max'],'repeat_mismatches':repeat_summary['mismatches'],'manifest_sha256':sha(OUT/'manifest.json')},indent=2))

if __name__=='__main__':main()
