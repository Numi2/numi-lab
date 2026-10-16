from pathlib import Path
import json,hashlib,datetime,subprocess,os,time,importlib.util,sys,argparse
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=Q/'analysis/native521-fresh-damped-coupon-command-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
s=json.loads((P/'command-spec.json').read_text());S=Path(s['source_snapshot']);D=Path(s['runtime_dir'])
assert sha(S/'frozen-source.json')==s['source_frozen_sha256']
assert sha(D/'source-build.json')==s['source_build_sha256']
f=json.loads((S/'frozen-source.json').read_text());b=json.loads((D/'source-build.json').read_text())
assert b['success'] and b['source_sha256']==f['source_sha256']
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
for n,h in b['artifacts_sha256'].items():assert sha(n)==h,n
for m in s['material_bindings']:assert sha(m['path'])==m['sha256']
manifest=json.loads((P/'package-manifest.json').read_text())
for n,h in manifest['files_sha256'].items():assert sha(P/n)==h,n
H=Q/s['required_prior_native_helper'];helper={}
for opt in ['O3','O0']:
 receipt=H/opt/'receipt.json';r=json.loads(receipt.read_text())
 assert r['exit_code']==0 and not r['timed_out']
 helper[opt]={'receipt_sha256':sha(receipt),'stdout_sha256':sha(H/opt/'stdout.json')}
for opt in ['O3','O0']:
 report=json.loads((H/opt/'stdout.json').read_text())
 damped=[case for case in report['cases'] if case['mu_kg']>0]
 assert len(damped)==4 and {case['requested_cap'] for case in damped}=={32,64}
 for case in damped:
  assert case['error_bits']==0 and case['status'] in ('iteration_cap','zero_normal_residual')
  assert case['recursive_residual_mismatch_l2']<=case['recursive_residual_consistency_bound']
  assert case['reference_error_l2']<=1e-3*sum(x*x for x in case['cpu_normal_equation_reference'])**0.5
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(k in x for k in ('numi-matter-','geometry-replay','contact_barrier_trial_replay','damped_cgls_helper_replay')) and not x.split()[1].startswith('Z')]
assert not workers,workers
capacity=importlib.util.spec_from_file_location('capacity521',P/'capacity-helper.py');g=importlib.util.module_from_spec(capacity);capacity.loader.exec_module(g);g.B=Q
assert g.available()>=g.MINIMUM
cmd=[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(S/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(S),'--runtime-dir',str(D),'--build-provenance',str(D/'source-build.json'),'--case',s['case'],'--timeout',str(s['timeout_s']),'--question','Can the exact source198 bounded native full-from-zero damped direction complete the full60control8x2x1 coupon with unchanged physical residual certificates?','--prediction',s['prediction'],'--diagnostic-run']
for k,v in sorted(s['environment'].items()):cmd+=['--env',k+'='+v]
inputs=[P/'command-spec.json',P/'host-validation-binding.json',S/'frozen-source.json']+[Path(m['path']) for m in s['material_bindings']]
names=[p.name for p in inputs]+['source-build.json'];assert len(names)==len(set(names))
for p in inputs:cmd+=['--input-file',str(p)]
cmd+=['--']+s['native_argv']
pre={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files_verified':len(f['source_sha256']),'build_artifacts_verified':len(b['artifacts_sha256']),'helper_replay':helper,'gpu_workers':workers,'available_bytes':g.available(),'argv':cmd,'execute_requested':args.execute}
print(json.dumps(pre),flush=True)
if not args.execute:sys.exit(0)
C=Path(s['case']);O=P/'execution-521';assert not C.exists() and not O.exists();O.mkdir()
(O/'launch.json').write_text(json.dumps(pre,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
reserve=O/'emergency-reserve.bin';proc=None;errors=[];timed=False;space=False;minimum=g.available();start=time.monotonic()
try:
 g.write_reserve_file(reserve)
 with (O/'wrapper.log').open('xb') as out:
  proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  while proc.poll() is None:
   minimum=min(minimum,g.available());space=minimum<g.ABORT_FLOOR;timed=time.monotonic()-start>s['timeout_s']+120
   if space or timed:
    reserve.unlink(missing_ok=True);errors.extend(g.terminate_owned_native(proc,s['native_argv'][0])['cleanup_errors']);break
   time.sleep(2)
except BaseException as e:
 errors.append(repr(e))
 if proc is not None and proc.poll() is None:errors.extend(g.terminate_owned_native(proc,s['native_argv'][0])['cleanup_errors'])
finally:
 reserve.unlink(missing_ok=True)
 code=proc.poll() if proc else None
 unchanged=all(sha(S/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in b['artifacts_sha256'].items())
 result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'timed_out':timed,'space_abort':space,'errors':errors,'wall_seconds':time.monotonic()-start,'minimum_available_bytes':minimum,'source_and_artifacts_unchanged':unchanged,'full_protocol_qualification':False,'source_build_sha256':sha(D/'source-build.json'),'wrapper_sha256':sha(O/'wrapper.log')}
 (O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
sys.exit(code if type(code)is int and code!=0 else int(bool(errors) or not unchanged or timed or space or code is None))
