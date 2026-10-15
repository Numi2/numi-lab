#!/usr/bin/env python3
"""Strict CPU-only comparison of source103 pair-CCD results 367 vs 372."""
import argparse, hashlib, json, math, pathlib

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_bundle(d):
    m=json.loads((d/'manifest.json').read_text())
    for entry in m['files']:
        p=d/entry['path']
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError(f"bundle binding mismatch: {p}")
    result=json.loads((d/'native-result.json').read_text())
    receipt=json.loads((d/'receipt.json').read_text())
    if receipt['compiled_package_unchanged'] is not True:
        raise ValueError('compiled package changed during run')
    if receipt['physical_transactions'] != 0 or result['physical_transactions_executed'] != 0:
        raise ValueError('diagnostic unexpectedly executed physical transactions')
    if any(v['exit_code'] != 0 or v['timed_out'] for v in receipt['commands'].values()):
        raise ValueError('run receipt is not a complete successful diagnostic')
    return m,result,receipt

def flatten_max_abs(v):
    if isinstance(v,list): return max((flatten_max_abs(x) for x in v),default=0.0)
    return abs(float(v))

def compare(base_dir, cont_dir):
    bm,a,br=verify_bundle(base_dir)
    cm,b,cr=verify_bundle(cont_dir)
    for key in ('cases_bin_sha256','source_snapshot','source_contact_sha256',
                'source_main_shader_sha256','source_common_sha256','prepared_manifest_sha256'):
        if a.get(key) != b.get(key): raise ValueError(f'run binding differs: {key}')
    if a.get('cases_bin_sha256') != 'e116911335b4fbf88683535cde08a97b26ca99bc1975989d67532196c2ff996f':
        raise ValueError('unexpected captured input corpus')
    if b.get('edge_edge_iteration_budget') != 33:
        raise ValueError('continuation run is not budget 33')
    if a.get('production_vs_trace_clone_all_bitwise_equal') is not True or b.get('production_vs_trace_clone_all_bitwise_equal') is not True:
        raise ValueError('bundle aggregate clone parity failed')
    ac,bc=a['cases'],b['cases']
    if len(ac)!=30 or len(bc)!=30: raise ValueError('expected exactly 30 rays')
    diffs=[]; unchanged_certified=0; old_uncertified=[]
    for i,(x,y) in enumerate(zip(ac,bc)):
        for k in ('name','kind','kind_id','feature_a','feature_b','primitive_pair','primitive_nodes',
                  'candidate_identity','candidate_slot','points_start_f32_bits','points_finish_f32_bits',
                  'dt_f32_bits','record_f32_bits'):
            if x.get(k)!=y.get(k): raise ValueError(f'case {i} input/identity changed: {k}')
        if x.get('production_and_trace_clone_bitwise_equal') is not True or y.get('production_and_trace_clone_bitwise_equal') is not True:
            raise ValueError(f'case {i} production/trace clone mismatch')
        px,py=x['production_result'],y['production_result']
        tx,ty=x['native_conservative_advance_trace'],y['native_conservative_advance_trace']
        samep=px==py
        samet=tx==ty
        if px.get('certified'):
            unchanged_certified += 1
            if not samep or not samet: raise ValueError(f'previously certified case changed: {i}')
        else:
            old_uncertified.append(i)
            diffs.append({'index':i,'name':x['name'],'baseline_certified':px.get('certified'),
                          'continued_certified':py.get('certified'),'production_equal':samep,
                          'trace_equal':samet,'baseline_trace_rows':len(tx),'continued_trace_rows':len(ty)})
            if tx != ty[:len(tx)]: raise ValueError(f'uncertified ray trace did not continue from baseline prefix: {i}')
    if old_uncertified != [21] or len(diffs)!=1: raise ValueError('unexpected uncertified case set')
    x,y=ac[21],bc[21]
    px,py=x['production_result'],y['production_result']
    tx,ty=x['native_conservative_advance_trace'],y['native_conservative_advance_trace']
    if x['name']!='trial358_ee_edge0_edge0' or x['kind']!='edge_edge' or x['feature_a']!=0 or x['feature_b']!=0:
        raise ValueError('target ray identity mismatch')
    if px['certified'] or px['hit'] or not py['certified'] or not py['hit']:
        raise ValueError('expected uncertified-miss to certified-hit transition absent')
    if len(tx)!=17 or len(ty)!=18 or ty[-1]['iteration']!=17 or ty[-1]['terminal_code']!=1:
        raise ValueError('expected one terminal evaluation at iteration 17')
    if tx[-1]['iteration']!=16 or tx[-1]['terminal_code']!=0:
        raise ValueError('baseline did not exhaust after iteration 16')
    if ty[-2] != tx[-1]: raise ValueError('continuation did not resume at exact prior state')
    thickness=float(y['thickness_m'])
    scale=max(thickness,flatten_max_abs(y['points_start_m']),flatten_max_abs(y['points_finish_m']),1e-12)
    roundoff=8.0*2.0**-23*scale
    effective=thickness+roundoff
    separation=float(py['separation_m'])
    if not (thickness < separation <= effective):
        raise ValueError('reported hit does not lie within the source roundoff-expanded thickness')
    return {
      'schema':'numi.cardboard.source103_pair_ccd_continuation_adjudication.v1',
      'baseline_directory':str(base_dir),'continuation_directory':str(cont_dir),
      'baseline_result_sha256':sha(base_dir/'native-result.json'),
      'continuation_result_sha256':sha(cont_dir/'native-result.json'),
      'input_corpus_sha256':a['cases_bin_sha256'],'case_count':30,
      'previously_certified_cases_unchanged':unchanged_certified,
      'production_trace_clone_bitwise_equal_all_cases':True,
      'newly_certified_case':{
        'index':21,'name':y['name'],'kind':y['kind'],'feature_a':y['feature_a'],'feature_b':y['feature_b'],
        'primitive_pair':y['primitive_pair'],'primitive_nodes':y['primitive_nodes'],
        'candidate_identity':y['candidate_identity'],
        'baseline':{'certified':False,'hit':False,'terminal_iteration':16,'trace_rows':17,
                    'time':tx[-1]['time'],'separation_m':px['separation_m'],'last_step':tx[-1]['step']},
        'continued':{'certified':True,'hit':True,'terminal_iteration':17,'trace_rows':18,
                     'time':ty[-1]['time'],'separation_m':separation,'toi':py['toi'],
                     'normal':py['normal'],'witness_point':py['witness_point'],
                     'feature_parameter0':py['feature_parameter0'],'feature_parameter1':py['feature_parameter1'],
                     'last_step':ty[-1]['step'],'terminal_code':ty[-1]['terminal_code']},
        'trace_prefix_exact':True,
        'iteration16_to17':{'start_time':ty[-2]['time'],'step':ty[-2]['step'],
                            'next_evaluation_time':ty[-1]['time'],
                            'distance_at_new_evaluation_m':ty[-1]['distance_m'],
                            'terminal_code':ty[-1]['terminal_code']},
        'nominal_thickness_m':thickness,'distance_roundoff_m':roundoff,
        'effective_thickness_m':effective,'separation_minus_nominal_thickness_m':separation-thickness,
        'interpretation':'The returned hit is within the implementation roundoff-expanded band; its measured separation remains slightly above nominal thickness.'
      },
      'all_other_cases':{'count':29,'production_result_equal':True,'trace_equal':True},
      'limitations':['This is a 30-ray native CCD diagnostic, not a physical transaction or full-contact qualification.',
                     'Only one captured EE ray needed an additional sample; this does not prove all global EE rays complete within 33 iterations.',
                     'The existing direct EE regression corpus includes endpoint-clear crossing and long-tangent certified miss, but lacks a matching near-parallel exhaustion ray.',
                     'The available direct regression runs use source049 contact code whose EE loop bound is 16; source103 bound is 17. The direct EE routine body is otherwise identical, but rerun the direct suite against the continuation metallib.',
                     'No full-contact transaction with continuation bound 33 is present in this comparison. Keep global adoption pending direct-suite and full-contact same-build runs.']
    }

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',type=pathlib.Path,required=True)
    ap.add_argument('--continued',type=pathlib.Path,required=True)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    a=ap.parse_args()
    result=compare(a.baseline,a.continued)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
