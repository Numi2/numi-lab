#!/usr/bin/env python3
"""Bounded FP64 directional epsilon sweep for tet1186 from source101 capture360."""
from __future__ import annotations
import csv,hashlib,importlib.util,json,pathlib,sys,time,warnings
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
HERE=pathlib.Path(__file__).resolve().parent
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
DROWS=BASE/'analysis/source101-capture360-directional-fd-003/output/all_1728_row_metrics.csv'
HELPER=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
RUNNER=BASE/'analysis/source101-capture360-small-eps-34-005/replay_continuation.py'
OUT=HERE/'output-001'
TARGETS=(1.0e-6,3.0e-7,1.0e-7,3.0e-8,1.0e-8)
EXPECTED={'capture':'548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33',
'directional_rows':'226a12e13a5dae5cf17f4189a65d71ba6a5e26c94d14d11407e5712a76979180',
'helper':'742b0920c4f543e284e41ed18f8dbc78253915a31f39b3ef8d32ff7dec01b5e8',
'reference':'c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3',
'runner':'36716c3a41f39805f718dc37d47349e99156f4492afc013586b9a530534df8f6'}
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def parse_int(t): return -0.0 if t=='-0' else int(t)
def main():
 if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
 OUT.mkdir(parents=True)
 paths={'capture':CAP,'directional_rows':DROWS,'helper':HELPER,'reference':REF,'runner':RUNNER}
 hashes={k:sha(v) for k,v in paths.items()}
 if hashes!=EXPECTED: raise RuntimeError(f'input binding mismatch: {hashes}')
 fd=load(HELPER,'tet1186_direction_helper');cont=load(RUNNER,'tet1186_v3_runner');ref=load(REF,'tet1186_independent_reference')
 cont.TARGETS=TARGETS
 cap=[json.loads(x,parse_int=parse_int) for x in CAP.read_text().splitlines() if x.strip()]
 header=cap[0]
 if (header.get('root'),header.get('newton_iteration'),header.get('world_fingerprint_hex'))!=(2,23,'6de3475316b7e8b9'):
  raise RuntimeError('wrong captured Newton state')
 if cap[-1].get('record')!='integrity' or cap[-1].get('production_inputs_bitwise_unchanged') is not True:
  raise RuntimeError('missing capture integrity record')
 native=[r for r in cap if r.get('record')=='tetrahedron' and r.get('environment')==0 and int(r.get('tet_id',-1))==1186]
 if len(native)!=1: raise RuntimeError('capture row 1186 is not unique')
 with DROWS.open(newline='') as f: direction=[r for r in csv.DictReader(f) if int(r['tet_id'])==1186]
 if len(direction)!=1: raise RuntimeError('direction row 1186 is not unique')
 pt=fd.f32(header['parameters_by_environment'][0])
 prior=dict(direction[0])
 t0=time.monotonic()
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  result=cont.replay_target_row(fd,ref,prior,direction[0],native[0],pt)
 warning_records=[{'category':w.category.__name__,'message':str(w.message),'filename':w.filename,'line':w.lineno} for w in caught]
 elapsed=time.monotonic()-t0
 payload={'schema':'numi.cardboard.source101-tet-epsilon-diagnostic.v1','classification':'One-row CPU FP64 diagnostic; no native, physical, or universal noise-floor qualification.',
  'bindings':{k:{'path':str(paths[k]),'sha256':hashes[k]} for k in paths},
  'capture_selection':{'root':header['root'],'newton_iteration':header['newton_iteration'],'world_fingerprint_hex':header['world_fingerprint_hex'],'environment':0,'tet_id':1186},
  'accepted_history_policy':'same captured FP32 accepted 15-state history at center and every plus/minus endpoint; center candidate is only an initial numerical seed for endpoint continuation.',
  'projection_policy':{'residual_tolerance':cont.RESIDUAL_TOL,'dynamic_correction_tolerance':cont.V3_DYNAMIC_CORRECTION_TOL,'max_newton_updates':cont.MAX_CONTINUATION_UPDATES,'max_backtracks':cont.MAX_BACKTRACKS,'dynamic_scale':'max(abs(old),abs(candidate),parameter[26]) componentwise'},
  'branch_predicates':'multiplier-increment sign, yield-gap sign, elastic_smoothed/plastic_smoothed FB selector; continuous Hill q and hardening factor retained; no separate discrete hardening predicate exposed.',
  'targets_deltaF_frobenius':list(TARGETS),'row':result,'warnings':warning_records,'elapsed_seconds':elapsed,
  'limits':['Perturbation sweep is only for captured row1186 and direction.','No universal FP64 or native FP32 noise floor is inferred.','A target-specific pass does not change the full1728 one-epsilon result.']}
 (OUT/'row1186.json').write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'output':str(OUT/'row1186.json'),'elapsed_seconds':elapsed,'warnings':len(warning_records),
  'epsilon_results':[{'target':x['target_deltaF_frobenius'],'error':x['central_fd_vs_fp64_tangent']['relative_l2'],
  'status':x['strict_1e-6_diagnostic'],'projection_ok':x['center_and_both_endpoints_v3_converged'],
  'branch_stable':x['branch_signature_stable']} for x in result['continuation_v3']['epsilon_cases']]},indent=2,allow_nan=False))
if __name__=='__main__': main()
