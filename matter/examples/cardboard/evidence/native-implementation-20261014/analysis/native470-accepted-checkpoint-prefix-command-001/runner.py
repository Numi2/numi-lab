#!/usr/bin/env python3
"""Fail-closed source/build/capacity guard for native470; preflight is default."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
PACKAGE=Path(__file__).resolve().parent; SPEC_PATH=PACKAGE/'native-470-command.json'; MANIFEST_PATH=PACKAGE/'package-manifest.json'
B=Path('/Users/home/cardboard-implementation-20261014'); EXECUTIONS=PACKAGE/'executions'
MIB=1024**2; GIB=1024*MIB; MINIMUM=2*GIB; ABORT_FLOOR=512*MIB; RESERVE_BYTES=128*MIB
TIMING_KEYS={'NUMI_CARDBOARD_STEP_TIMING','NUMI_CARDBOARD_NATIVE_TIMING','NUMI_MATTER_GPU_TIMING_DENSE45'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def available():
 s=os.statvfs(B); return s.f_bavail*s.f_frsize
def verify():
 spec=json.loads(SPEC_PATH.read_text()); manifest=json.loads(MANIFEST_PATH.read_text())
 assert manifest['schema']=='numi.cardboard.native470-package-manifest.v1'
 for rel,digest in manifest['files_sha256'].items():
  p=PACKAGE/rel; assert not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(PACKAGE), f'unsafe package file: {rel}'
  assert sha(p)==digest, f'package file changed: {rel}'
 source=Path(spec['source_snapshot']).resolve(strict=True); builddir=Path(spec['runtime_dir']).resolve(strict=True)
 assert source==Path('/Users/home/cardboard-implementation-20261014/snapshots/source-diagnostic-and-resume-167-final').resolve()
 assert builddir==Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-167-final').resolve()
 frozen_path=Path(spec['frozen_source_file']).resolve(strict=True); assert sha(frozen_path)==spec['frozen_source_sha256']
 frozen=json.loads(frozen_path.read_text()); assert frozen.get('schema')=='numi.cardboard.frozen-source.v1'
 assert len(frozen['source_sha256'])==spec['source_file_count']
 for rel,digest in frozen['source_sha256'].items():
  p=(source/rel).resolve(strict=True); assert p.is_relative_to(source) and sha(p)==digest, f'source drift: {rel}'
 provenance=Path(spec['source_build_provenance']).resolve(strict=True)
 assert provenance==builddir/'source-build.json' and sha(provenance)==spec['source_build_sha256']
 built=json.loads(provenance.read_text()); assert built.get('schema')=='numi.cardboard.source-build.v1' and built.get('success') is True
 assert Path(built['source_snapshot']).resolve()==source and Path(built['build_dir']).resolve()==builddir
 assert built['source_sha256']==frozen['source_sha256'] and built['targets']==spec['build_targets'] and built['artifacts_sha256']==spec['artifact_sha256']
 native=spec['native_argv']; assert native[0]==str((builddir/'numi-matter-cardboard-probe').resolve())
 assert native[native.index('--metallib')+1]==str((builddir/'shaders/NumiMatter.metallib').resolve())
 assert native[native.index('--output')+1]==str(Path(spec['case'])/'output')
 assert native.count('--coupon-checkpoint-after-control')==1 and native[native.index('--coupon-checkpoint-after-control')+1]=='19'
 assert not any(x in native for x in ('--coupon-resume-checkpoint','--coupon-resume-metadata'))
 for filename,digest in built['artifacts_sha256'].items():
  p=Path(filename).resolve(strict=True); assert p.is_relative_to(builddir) and sha(p)==digest, f'artifact drift: {filename}'
 env=spec['environment_overrides']; assert env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR']=='20:0:63:1:0:0'
 assert len(env)==12 and not any(k.startswith('NM_FEM_OPERATOR_FD') for k in env)
 assert 'NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH' not in env
 assert spec['physical_arguments_equal_to_469_after_executable_metallib_output_checkpoint_exclusions'] is True
 baseline_path=PACKAGE/'inputs/source469-prediction.json'; baseline=json.loads(baseline_path.read_text())
 assert sha(baseline_path)==spec['baseline_469_prediction_sha256']
 def physical_tokens(values):
  out=[]; i=0
  while i<len(values):
   if i==0: i+=1; continue
   if values[i] in ('--metallib','--output','--coupon-checkpoint-after-control'): i+=2; continue
   out.append(values[i]); i+=1
  return out
 assert physical_tokens(native)==physical_tokens(baseline['argv']), 'physical argv differs from469 beyond declared path/stop additions'
 expected_env=dict(baseline['environment'])
 expected_env.pop('NM_FEM_OPERATOR_FD_SELECTOR',None); expected_env.pop('NM_FEM_OPERATOR_FD_CAPTURE_PATH',None)
 expected_env['NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR']='20:0:63:1:0:0'
 assert env==expected_env, 'environment controls differ from declared469 baseline plus regional selector'
 input_by_alias={item['alias']:item for item in spec['input_bindings']}
 assert 'source167-build-cpu-validation-report.json' in input_by_alias
 assert 'source167-build-cpu-validation-manifest.json' in input_by_alias
 assert json.loads((PACKAGE/'inputs/source167-build-cpu-validation-report.json').read_text())['source_build_sha256']==spec['source_build_sha256']
 assert (PACKAGE/'inputs/source167-ctest-registration-report.json').is_file()
 assert json.loads((PACKAGE/'inputs/source167-ctest-registration-report.json').read_text()).get('status') in ('pass','passed','success',None)
 for material_flag,alias in (('--liner-material','source157-liner-material.nmatter'),('--medium-material','source157-medium-material.nmatter')):
  assert native[native.index(material_flag)+1]==input_by_alias[alias]['original_path']
 names=[]
 for item in spec['input_bindings']:
  names.append(Path(item['alias']).name); copy=PACKAGE/'inputs'/item['alias']
  assert copy.is_file() and sha(copy)==item['package_sha256']
  original=Path(item['original_path']).resolve(strict=True); assert sha(original)==item['original_sha256']
 assert len(set(names))==len(names)
 assert len(spec['input_bindings'])==12
 wrapper=spec['wrapper_argv']; assert Path(wrapper[1]).resolve()==source/'matter/tools/cardboard_source_bound_run.py'
 assert wrapper[wrapper.index('--source-snapshot')+1]==str(source)
 assert wrapper[wrapper.index('--runtime-dir')+1]==str(builddir)
 assert wrapper[wrapper.index('--build-provenance')+1]==str(provenance)
 assert wrapper[wrapper.index('--case')+1]==spec['case'] and '--diagnostic-run' in wrapper
 assert wrapper[wrapper.index('--diagnostic-capture')+1]=='NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH'
 wrapper_inputs=[wrapper[i+1] for i,value in enumerate(wrapper[:-1]) if value=='--input-file']
 assert sorted(wrapper_inputs)==sorted(str(PACKAGE/'inputs'/item['alias']) for item in spec['input_bindings'])
 wrapper_env=[wrapper[i+1].split('=',1) for i,value in enumerate(wrapper[:-1]) if value=='--env']
 assert dict(wrapper_env)==env
 assert wrapper[wrapper.index('--')+1:]==native
 case=Path(spec['case']); assert case.parent.is_dir() and not case.parent.is_symlink() and case.parent.resolve()==B/'solver'
 assert not case.exists() and not case.is_symlink(), 'fresh case required'
 assert not EXECUTIONS.exists(), 'fresh execution-record directory required'
 active=[]
 for row in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines():
  parts=row.split(None,2)
  if len(parts)==3 and parts[2].startswith('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-') and parts[2].endswith('/numi-matter-cardboard-probe') and not parts[1].startswith('T'):
   active.append({'pid':int(parts[0]),'stat':parts[1],'executable':parts[2]})
 assert not active, f'one-GPU-worker guard found active probes: {active}'
 free=available(); assert free>=MINIMUM, f'launch needs 2 GiB free; observed {free}'
 return spec,{'available_bytes':free,'active_probe_processes':active,'verified_source_files':len(frozen['source_sha256']),'verified_build_artifacts':len(built['artifacts_sha256']),'verified_inputs':len(names),'source_build_sha256':sha(provenance),'frozen_source_sha256':sha(frozen_path)}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
 spec,checks=verify(); print(json.dumps({'status':'preflight_pass_no_launch','checks':checks,'execute_requested':args.execute},sort_keys=True))
 if not args.execute: return 0
 EXECUTIONS.mkdir(parents=True,exist_ok=False)
 reserve=EXECUTIONS/'emergency-reserve.bin'; proc=None; code=None; space_abort=False; signalled=[]; minfree=available(); start=time.monotonic(); errors=[]
 parent={k:v for k,v in os.environ.items() if not k.startswith('NM_FEM_') and k not in TIMING_KEYS}
 removed=sorted(set(os.environ)-set(parent))
 launch={'schema':'numi.cardboard.native470-capacity-launch.v1','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command_sha256':sha(SPEC_PATH),'package_manifest_sha256':sha(MANIFEST_PATH),'source_build_sha256':checks['source_build_sha256'],'frozen_source_sha256':checks['frozen_source_sha256'],'parent_environment_removed_keys':removed,'parent_NM_FEM_values_recorded':False,'minimum_launch_available_bytes':MINIMUM,'abort_floor_bytes':ABORT_FLOOR,'emergency_reserve_bytes':RESERVE_BYTES,'available_before_reserve':checks['available_bytes'],'diagnostic_only':True,'checkpoint_control_index':19}
 try:
  with reserve.open('xb') as f:
   os.posix_fallocate(f.fileno(),0,RESERVE_BYTES); f.flush(); os.fsync(f.fileno())
   reserve_stat=os.fstat(f.fileno())
   if reserve_stat.st_size<RESERVE_BYTES: raise RuntimeError('emergency reserve allocation is short')
  launch['emergency_reserve_allocated_blocks_bytes']=reserve_stat.st_blocks*512
  launch['available_after_reserve']=available()
  (EXECUTIONS/'launch.json').write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
  minfree=available()
  with (EXECUTIONS/'wrapper.log').open('x') as log:
   proc=subprocess.Popen(spec['wrapper_argv'],stdout=log,stderr=subprocess.STDOUT,env=parent,start_new_session=True)
   while proc.poll() is None:
    free=available(); minfree=min(minfree,free)
    if free<ABORT_FLOOR and not space_abort:
     space_abort=True; reserve.unlink(missing_ok=True)
     rows=subprocess.check_output(['ps','-axo','pid=,ppid=,pgid=,comm='],text=True).splitlines(); expected=spec['native_argv'][0]
     for row in rows:
      p=row.split(None,3)
      if len(p)==4 and int(p[1])==proc.pid and Path(p[3]).resolve()==Path(expected).resolve():
       child,pgid=int(p[0]),int(p[2])
       if child==pgid: os.killpg(pgid,signal.SIGTERM); signalled.append(pgid)
    time.sleep(2)
  code=proc.returncode
 except BaseException as exc:
  errors.append(type(exc).__name__+': '+str(exc))
 finally:
  reserve.unlink(missing_ok=True)
  receipt={'schema':'numi.cardboard.native470-capacity-guard-receipt.v1','execution_started':proc is not None,'wrapper_exit_code':proc.returncode if proc else None,'space_abort':space_abort,'signalled_native_child_process_groups':signalled,'child_group_not_observed_on_abort':space_abort and not signalled,'errors':errors,'minimum_observed_available_bytes':minfree,'available_after_run':available(),'wall_seconds':time.monotonic()-start,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_receipt_path':str(Path(spec['case'])/'receipt.json'),'diagnostic_only':True,'full_protocol_complete':False,'checkpoint_restore_parity_established':False}
  (EXECUTIONS/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,sort_keys=True))
 return code if type(code) is int else (0 if not errors else 1)
if __name__=='__main__': raise SystemExit(main())
