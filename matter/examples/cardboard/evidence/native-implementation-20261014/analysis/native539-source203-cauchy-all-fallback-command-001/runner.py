from pathlib import Path
import json,hashlib,datetime,subprocess,os,time,importlib.util,sys,argparse,struct
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=Q/'analysis/native539-source203-cauchy-all-fallback-command-001'
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
binding=json.loads((P/'host-validation-binding.json').read_text())
assert binding['source_frozen_sha256']==s['source_frozen_sha256']
assert binding['source_build_sha256']==s['source_build_sha256']
assert binding['helper_production_cgls_identical'] is True

# Historical source201 E2 helper (actual Metal O0/O3; synthetic only, carried for context).
helper534=json.loads((P/'native534-binding.json').read_text())
assert sha(helper534['receipt_path'])==helper534['receipt_sha256']
r534=json.loads(Path(helper534['receipt_path']).read_text())
assert r534['status']=='pass' and r534['physical_transactions']==0
assert r534['binding_check']['frozen_source_sha256']==helper534['source201_frozen_sha256']
assert r534['binding_check']['source_count']==817 and r534['exact_output_and_exit_parity'] is True
assert helper534['historical_only'] is True
for row in r534['runs']:
    root=Path(helper534['receipt_path']).parent/row['label']
    assert row['exit_code']==0 and sha(root/'stdout.bin')==row['stdout_sha256'] and sha(root/'stderr.bin')==row['stderr_sha256']
assert (Path(helper534['receipt_path']).parent/'metal-o0/stdout.bin').read_bytes()==(Path(helper534['receipt_path']).parent/'metal-o3/stdout.bin').read_bytes()
# The source198 baseline is a failed 17-control prefix, not a completion target.
baseline=json.loads((P/'baseline-521-binding.json').read_text())
oldout=Path(baseline['case'])/'output'
assert sha(oldout/'result.json')==baseline['result_sha256'] and sha(oldout/'manifest.json')==baseline['output_manifest_sha256']
oldresult=json.loads((oldout/'result.json').read_text())
assert oldresult['status']=='failed' and oldresult['accepted_steps']==17 and oldresult['physical_validation'] is False
actual={}
for fp in sorted(oldout.rglob('*')):
    if fp.is_symlink(): raise RuntimeError('baseline symlink '+str(fp))
    if fp.is_file(): actual[fp.relative_to(oldout).as_posix()]={'bytes':fp.stat().st_size,'sha256':sha(fp)}
assert actual==baseline['output_inventory'] and len(actual)==baseline['output_file_count']
parity=Path(baseline['native530_report']['path'])
assert sha(parity)==baseline['native530_report']['sha256']
parity_report=json.loads(parity.read_text())
assert parity_report['verdict']=='PASS' and parity_report['accepted_controls']==17 and parity_report['full_trajectory_qualified'] is False
# Source203 CGLS recurrence bytes remain identical to source198.
src198=Q/'snapshots/source-contact-damped-memory-198-001'
f198=json.loads((src198/'frozen-source.json').read_text())
assert f198['source_sha256']['matter/src/metal/base_cgls.metalinc']==f['source_sha256']['matter/src/metal/base_cgls.metalinc']
assert f198['source_sha256']['matter/src/metal/common.metalinc']==f['source_sha256']['matter/src/metal/common.metalinc']
H=Q/s['required_prior_native_helper'];helper={}
expected_damped_mu=float('0.001000000047')
expected_tuples={(scale,mu,cap) for scale in (1,2) for mu in (0.0,expected_damped_mu) for cap in (32,64)}
helper_reports={}
for opt in ['O3','O0']:
 receipt=H/opt/'receipt.json';stdout=H/opt/'stdout.json'
 r=json.loads(receipt.read_text());expected=binding['kernel_helper'][opt]
 assert r['exit_code']==0 and not r['timed_out']
 assert sha(receipt)==expected['receipt_sha256'],(opt,'receipt hash mismatch')
 assert sha(stdout)==expected['stdout_sha256'],(opt,'stdout hash mismatch')
 helper[opt]={'receipt_sha256':sha(receipt),'stdout_sha256':sha(stdout)}
 report=json.loads(stdout.read_text());helper_reports[opt]=report
 assert report['native_metal_executed'] is True
 assert report['compiled_production_recurrence'] is True
 assert report['full_coupon_qualified'] is False
 assert report['production_kernels']==['nm_fgmres_base_cgls_form_alpha','nm_fgmres_base_cgls_form_beta','nm_fgmres_base_cgls_update_xr','nm_fgmres_base_cgls_update_p','nm_fgmres_base_cgls_complete_iteration']
 actual_tuples={(int(c['scale']),float(c['mu_kg']),int(c['requested_cap'])) for c in report['cases']}
 assert len(report['cases'])==8 and actual_tuples==expected_tuples,(opt,actual_tuples)
 for case in report['cases']:
  assert case['recursive_residual_mismatch_l2']<=case['recursive_residual_consistency_bound']
  if case['mu_kg']==expected_damped_mu:
   assert case['error_bits']==0 and case['status']=='iteration_cap'
   assert case['iterations']==case['requested_cap']
   assert case['reference_error_l2']<=1e-3*sum(x*x for x in case['cpu_normal_equation_reference'])**0.5
  else:
   assert case['status']=='breakdown' and case['iterations']==6 and case['error_bits']==32
assert (H/'O0/stdout.json').read_bytes()==(H/'O3/stdout.json').read_bytes()
assert helper_reports['O0']==helper_reports['O3']
strict=json.loads((P/'strict199-audit-binding.json').read_text())
assert sha(strict['audit_report_path'])==strict['audit_report_sha256']
assert sha(strict['audit_manifest_path'])==strict['audit_manifest_sha256']
strict_report=json.loads(Path(strict['audit_report_path']).read_text())
assert strict_report['verdict']=='PASS' and strict_report['negative_control_rejected_for_intended_reason'] is True
for strict_case in strict['cases']:
    for rel,h in strict_case['files'].items(): assert sha(Q/'native'/strict_case['name']/rel)==h,(strict_case['name'],rel)
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(k in x for k in ('numi-matter-','geometry-replay','contact_barrier_trial_replay','damped_cgls_helper_replay')) and not x.split()[1].startswith('Z')]
assert not workers,workers
capacity=importlib.util.spec_from_file_location('capacity521',P/'capacity-helper.py');g=importlib.util.module_from_spec(capacity);capacity.loader.exec_module(g);g.B=Q
assert g.available()>=g.MINIMUM
cmd=[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(S/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(S),'--runtime-dir',str(D),'--build-provenance',str(D/'source-build.json'),'--case',s['case'],'--timeout',str(s['timeout_s']),'--question','Can the exact source203 fresh-from-zero damped CGLS configuration with all:all base-Cauchy fallback complete the full 60-control 8x2x1 coupon under the unchanged physical residual certificate?','--prediction',s['prediction'],'--diagnostic-run']
for k,v in sorted(s['environment'].items()):cmd+=['--env',k+'='+v]
inputs=[P/'command-spec.json',P/'host-validation-binding.json',P/'baseline-521-binding.json',P/'source203-build-binding.json',P/'strict199-audit-binding.json',P/'native534-binding.json',S/'frozen-source.json']+[Path(m['path']) for m in s['material_bindings']]
names=[p.name for p in inputs]+['source-build.json'];assert len(names)==len(set(names))
for p in inputs:cmd+=['--input-file',str(p)]
cmd+=['--']+s['native_argv']
pre={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_files_verified':len(f['source_sha256']),'build_artifacts_verified':len(b['artifacts_sha256']),'helper_replay':helper,'gpu_workers':workers,'available_bytes':g.available(),'argv':cmd,'execute_requested':args.execute,'full_schedule_controls':60,'resume':False,'diagnostic_capture_selectors':'off'}
print(json.dumps(pre),flush=True)
if not args.execute:sys.exit(0)
ready_path=Path(s['predecessor_readiness_binding'])
assert ready_path.is_file(), '537/538 predecessor readiness binding is not present; refusing 539 launch'
ready=json.loads(ready_path.read_text())
assert ready.get('schema')=='numi.cardboard.native539.predecessor-readiness.v1'
assert ready.get('decision')=='approve_for_539'
assert {x.get('id') for x in ready.get('predecessors',[])}=={'native537','native538'}
for item in ready['predecessors']:
    assert item.get('admit_539') is True and item.get('reviewed') is True, item
    evidence=Path(item['evidence_path'])
    assert sha(evidence)==item['evidence_sha256'], item
C=Path(s['case']);O=P/'execution-539';assert not C.exists() and not O.exists();O.mkdir()
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
