#!/usr/bin/env python3
"""Replay the strict v2 projection API on the bound 1728+576 source173 inputs."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np

SOURCE_SHA = "a910339573822bf32c7ea895c295a0a51b8b5a99d901b1e3ab1c788c094f6995"
CENTER_SHA = "320004cabcb3bf327fdd8c87ad1f82b42e0531d7e311e75fef207daf0dc5346a"  # filled from exact run016 input archive
ENDPOINT_SHA = "44fdd524f6af2c7e848ad3e2c77eb73a74e91de61b53d3b953de077fc8bc064c"  # filled from exact run016 input archive
SCALE = np.array([1.0e4]*6 + [1.0]*6 + [1.0e4]*3)

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-reference',required=True,type=pathlib.Path)
    ap.add_argument('--portable-tools',required=True,type=pathlib.Path)
    ap.add_argument('--run016',required=True,type=pathlib.Path)
    ap.add_argument('--output',required=True,type=pathlib.Path)
    a=ap.parse_args()
    if a.output.exists(): raise SystemExit(f'refusing overwrite: {a.output}')
    if sha(a.source_reference)!=SOURCE_SHA: raise SystemExit('source reference hash mismatch')
    center=a.run016/'projection-center.npz'; endpoint=a.run016/'exact-endpoints-projection.npz'
    if sha(center)!=CENTER_SHA or sha(endpoint)!=ENDPOINT_SHA: raise SystemExit('run016 input archive hash mismatch')
    tool=a.portable_tools.resolve()
    sys.path.insert(0,str(tool))
    v1=load('cardboard_compensated_fp64_base',tool/'cardboard_compensated_fp64_base.py')
    load('cardboard_dd_fp64',tool/'cardboard_dd_fp64.py')
    v2=load('cardboard_compensated_fp64_reference',tool/'cardboard_compensated_fp64_reference.py')
    base=v1.import_reference(a.source_reference); ref=v2.Reference(base)
    a.output.mkdir(parents=True)
    result={'schema':'numi.cardboard.strict-v2-projection-api-run016-replay.v1',
            'source_sha256':sha(a.source_reference),'portable_hashes':{p.name:sha(p) for p in sorted(tool.glob('cardboard_*.py'))},
            'run016_inputs':{},'strict_gates':{'scaled_residual_max':2e-12,'dynamic_correction_max':2e-11},'cases':[]}
    for label,path,oldkey in [('centers',center,'old'),('realized_endpoints',endpoint,'accepted')]:
        data=np.load(path); p=data['parameters']; f=data['F']; old=data[oldkey]
        t=time.monotonic(); z,it,rn,cn,ok,reasons,meta=ref.project_with_metrics(p,f,old); elapsed=time.monotonic()-t
        expected=data['state']; expected_it=data['iterations']; expected_r=data['residual']; expected_c=data['correction']; expected_ok=data['valid']
        api_z,api_it,api_r=ref.project(p,f,old)
        state_delta=np.abs(z-expected)
        case={'name':label,'input_sha256':sha(path),'rows':int(len(p)),'elapsed_seconds':elapsed,
              'strict_pass_rows':int(ok.sum()),'strict_fail_rows':int((~ok).sum()),'expected_valid_rows':int(expected_ok.sum()),
              'max_scaled_residual':float(np.max(rn)),'max_dynamic_correction':float(np.max(cn)),
              'strict_residual_gate_pass':bool(np.all(ok) and np.max(rn)<=2e-12),
              'strict_correction_gate_pass':bool(np.all(ok) and np.max(cn)<=2e-11),
              'exact_state_match_rows':int(np.sum(np.all(z==expected,axis=1))),
              'max_state_abs_delta':float(np.max(state_delta)),
              'exact_iteration_match':bool(np.array_equal(it,expected_it)),
              'exact_residual_match':bool(np.array_equal(rn,expected_r)),
              'exact_correction_match':bool(np.array_equal(cn,expected_c)),
              'public_project_api_matches_detailed':bool(np.array_equal(api_z,z) and np.array_equal(api_it,it) and np.array_equal(api_r,rn)),
              'all_reasons':sorted(set(str(x) for x in reasons if x))}
        result['run016_inputs'][label]={'sha256':sha(path),'expected_valid_rows':int(expected_ok.sum()),'rows':int(len(p))}
        result['cases'].append(case)
        if not case['strict_residual_gate_pass'] or not case['strict_correction_gate_pass'] or not case['public_project_api_matches_detailed']:
            raise SystemExit(f'strict API case failed: {label}: {case}')
    result['all_api_gates_pass']=all(c['strict_residual_gate_pass'] and c['strict_correction_gate_pass'] and c['public_project_api_matches_detailed'] for c in result['cases'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()
