from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=Q/'analysis/native531-root17-newton34-replay-command-001';M=1024**2;G=1024*M
def H(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def kv(p):
 d={}
 for l in Path(p).read_text().splitlines():
  if not l or l.startswith('#'):continue
  k,s,v=l.partition('=')
  if not s or k in d:raise ValueError('malformed metadata')
  d[k]=v
 return d
def prefix(p,root):
 d={};rs=[]
 for l in Path(p).read_text().splitlines():
  if l.startswith('file='):
   a=l[5:].split('|')
   if len(a)!=4:raise ValueError('malformed prefix row')
   rel,n,h,o=a
   if Path(rel).is_absolute() or '..' in Path(rel).parts or chr(92) in rel or not n.isdecimal() or o not in ('0','1') or len(h)!=64:raise ValueError('unsafe prefix row')
   rs.append((rel,int(n),h,o=='1'))
  else:
   k,s,v=l.partition('=')
   if not s or k in d:raise ValueError('malformed prefix header')
   d[k]=v
 if d.get('schema')!='numi.cardboard.accepted-coupon-prefix.v1' or d.get('source_run_directory')!=str(root) or d.get('accepted_control_index')!='16':raise ValueError('prefix identity mismatch')
 ns=[x[0] for x in rs]
 if ns!=sorted(ns) or len(ns)!=len(set(ns)) or sum(x[3] for x in rs)!=1 or not any(x[0]=='observations.csv' and x[3] for x in rs):raise ValueError('prefix inventory invalid')
 total=0
 for rel,n,h,_ in rs:
  f=root/rel
  if f.is_symlink() or not f.is_file() or f.resolve()!=f or f.stat().st_size!=n or H(f)!=h:raise ValueError('changed prefix file '+rel)
  total+=n
  if total>2*1024**3:raise ValueError('prefix exceeds 2GiB')
 return len(rs),total
def workers():
 rows=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines()
 return [r.strip() for r in rows if len(r.split(None,2))==3 and not r.split(None,2)[1].startswith('Z') and (Path(r.split(None,2)[2]).name.startswith('numi-matter-') or Path(r.split(None,2)[2]).name in ('sparse-scan-checker','source469-material-replay-check'))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');x=ap.parse_args()
 s=json.loads((P/'command-spec.json').read_text());man=json.loads((P/'package-manifest.json').read_text())
 actual={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='package-manifest.json' and 'execution-531' not in p.parts and '__pycache__' not in p.parts}
 if actual!=set(man['files_sha256']):raise ValueError('package inventory mismatch')
 for n,h in man['files_sha256'].items():
  if H(P/n)!=h:raise ValueError('package file hash mismatch '+n)
 S=Path(s['source_snapshot']);D=Path(s['runtime_dir']);f=json.loads((S/'frozen-source.json').read_text());b=json.loads((D/'source-build.json').read_text())
 if H(S/'frozen-source.json')!=s['source_frozen_sha256'] or len(f['source_sha256'])!=816:raise ValueError('frozen source mismatch')
 for n,h in f['source_sha256'].items():
  if H(S/n)!=h:raise ValueError('source mismatch '+n)
 if H(D/'source-build.json')!=s['source_build_sha256'] or not b['success'] or b['source_sha256']!=f['source_sha256'] or b['artifacts_sha256']!=s['artifacts_sha256']:raise ValueError('build map mismatch')
 for n,h in b['artifacts_sha256'].items():
  if H(n)!=h:raise ValueError('build artifact mismatch')
 for z in s['material_bindings']:
  if H(z['path'])!=z['sha256']:raise ValueError('material hash mismatch')
 a=json.loads((P/'parent-521-command-spec.json').read_text());z=json.loads((P/'parent-530-command-spec.json').read_text())
 if a['environment']!=s['baseline_environment'] or z['environment']!=s['baseline_environment'] or a['source_frozen_sha256']!=s['source_frozen_sha256'] or z['source_frozen_sha256']!=s['source_frozen_sha256'] or a['source_build_sha256']!=s['source_build_sha256'] or z['source_build_sha256']!=s['source_build_sha256']:raise ValueError('parent source/environment mismatch')
 if H(P/'parent-521-command-spec.json')!=s['parent_521_spec_sha256'] or H(P/'parent-530-command-spec.json')!=s['parent_530_spec_sha256'] or H(Path(s['parent_530_package'])/'package-manifest.json')!=s['parent_530_package_manifest_sha256'] or z['parent_command']['manifest_sha256']!=s['parent_521_package_manifest_sha256']:raise ValueError('parent chain mismatch')
 if s['baseline_environment'].get('NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR')!='all:3' or s['environment']!={**s['baseline_environment'],**s['diagnostic_environment']} or any(k.startswith('NM_FEM_BASE_CGLS_CAPTURE') or k=='NM_FEM_BASE_CGLS_STATE_CAPTURE' for k in s['environment']):raise ValueError('numeric policy changed')
 exp=list(a['native_argv']);exp[exp.index('--output')+1]=str(Path(s['case'])/'output');exp+=['--coupon-resume-checkpoint',s['checkpoint_path'],'--coupon-resume-metadata',s['resume_metadata_path'],'--coupon-resume-diagnostic-replay']
 if exp!=s['native_argv'] or '--coupon-checkpoint-after-control' in exp:raise ValueError('resume argv mismatch')
 if s['diagnostic_environment']!={'NM_FEM_OPERATOR_FD_SELECTOR':'17:34','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT':'17'} or s['diagnostic_capture_keys']!=['NM_FEM_OPERATOR_FD_CAPTURE_PATH']:raise ValueError('diagnostic scope mismatch')
 if s['native_argv'][0]!=str(D/'numi-matter-cardboard-probe') or s['native_argv'][s['native_argv'].index('--metallib')+1]!=str(D/'shaders/NumiMatter.metallib') or Path(s['case']).exists():raise ValueError('native executable/metallib/case mismatch')
 if len([Path(p).name for p in s['input_files']])!=len(set(Path(p).name for p in s['input_files'])):raise ValueError('duplicate external input basenames')
 C=Path(s['checkpoint_source_case']);O=C/'output';PP=Path(s['parent_530_package'])
 ck=O/'checkpoints/coupon-control-000016.nmcb';md=O/'checkpoints/coupon-control-000016.resume';pm=O/'checkpoints/coupon-control-000016.prefix'
 req=[PP/'execution-530/receipt.json',C/'receipt.json',C/'prediction.json',C/'native-environment.json',O/'result.json',O/'manifest.json',O/'observations.csv',O/'coupon-checkpoint-prefix.json',O/'coupon-checkpoint-saved.txt',ck,md,pm]
 missing=[str(p) for p in req if not p.is_file()]
 cp={'status':'waiting_for_530_checkpoint','missing':missing}
 if not missing:
  outer=json.loads(req[0].read_text());nr=json.loads(req[1].read_text());pred=json.loads(req[2].read_text());res=json.loads(req[4].read_text());pr=json.loads(req[7].read_text())
  if outer.get('exit_code')!=0 or outer.get('timed_out') or outer.get('space_abort') or outer.get('source_and_artifacts_unchanged') is not True:raise ValueError('530 outer receipt invalid')
  if nr.get('execution_started') is not True or nr.get('exit_code')!=0 or nr.get('timed_out') or nr.get('source_inputs_unchanged') is not True:raise ValueError('530 native receipt invalid')
  if pred.get('source_snapshot')!=s['source_snapshot'] or pred.get('runtime_dir')!=s['runtime_dir'] or res.get('status')!='accepted_checkpoint_prefix_saved' or res.get('accepted_steps')!=17 or res.get('physical_validation') is not False:raise ValueError('530 accepted run status invalid')
  if pr.get('schema')!='numi.cardboard.coupon-checkpoint-prefix.v1' or pr.get('status')!='accepted_checkpoint_prefix_saved' or pr.get('accepted_control_index')!=16 or pr.get('next_control_index')!=17 or any(pr.get(k) is not False for k in ('full_protocol_complete','trajectory_parity_established','physical_validation')):raise ValueError('530 prefix status/index invalid')
  if pr.get('checkpoint_file')!='checkpoints/coupon-control-000016.nmcb' or pr.get('metadata_file')!='checkpoints/coupon-control-000016.resume':raise ValueError('530 checkpoint names invalid')
  m=kv(md)
  if H(ck)!=pr.get('checkpoint_sha256') or H(md)!=pr.get('metadata_sha256') or H(pm)!=m.get('prefix_manifest_sha256') or H(O/'manifest.json')!=m.get('protocol_manifest_sha256'):raise ValueError('530 checkpoint digest mismatch')
  if m.get('schema')!='numi.cardboard.accepted-coupon-resume.v2' or m.get('accepted_control_index')!='16' or m.get('next_control_index')!='17' or m.get('source_prefix_control_count')!='17' or m.get('environment_count')!='2' or m.get('both_environments_accepted')!='true' or m.get('source_run_directory')!=str(O) or m.get('prefix_manifest_file')!='coupon-control-000016.prefix' or any(m.get(k)!='none' for k in ('source_runner_case_directory','source_runner_environment_record_sha256','source_runner_prediction_sha256','source_runner_non_output_settings_sha256','source_runner_contract_sha256')):raise ValueError('530 native metadata invalid')
  n,total=prefix(pm,O)
  # Mirror native diagnostic-replay source proof: metadata source_runner fields
  # may be none, but source-runner records must still reproduce runtime settings.
  source_pred=json.loads((C/'prediction.json').read_text());source_env=json.loads((C/'native-environment.json').read_text())
  contract=source_pred.get('effective_native_environment_contract');record_contract=source_env.get('contract')
  if source_pred.get('schema')!='numi.cardboard.native-diagnostic-prediction.v1' or source_pred.get('run_mode')!='diagnostic' or source_env.get('schema')!='numi.cardboard.native-diagnostic-environment-run.v1' or source_env.get('producer')!='source-bound-diagnostic-runner' or source_env.get('mode')!='diagnostic' or source_env.get('launch',{}).get('native_controls_scrubbed') is not True:raise ValueError('530 source runner mode/scrub proof invalid')
  if contract!=record_contract or contract.get('case_directory')!=str(C) or contract.get('schema')!='numi.cardboard.effective-native-diagnostic-controls.v1':raise ValueError('530 source runner contract identity mismatch')
  payload={k:contract[k] for k in ('schema','keys','effective','case_directory','capture_prefixes')}
  canon=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False)
  ch=hashlib.sha256(canon.encode()).hexdigest()
  if contract.get('sha256')!=ch or source_pred.get('effective_native_environment_contract_sha256')!=ch or source_env.get('contract_sha256')!=ch or source_env.get('prediction_sha256')!=H(C/'prediction.json'):raise ValueError('530 source runner contract digest mismatch')
  setvars={k:v['value'] for k,v in contract['effective'].items() if v.get('set') is True}
  if source_pred.get('environment')!=setvars or setvars!=z['environment']:raise ValueError('530 source runner effective overrides differ from bound 530 command')
  settings={k:v for k,v in setvars.items() if k.startswith('NM_') or k.startswith('NUMI_MATTER_')}
  setting_bytes='numi.cardboard.coupon-runtime-settings.v2\n'+''.join(f'{len(k)}:{k}{len(v)}:{v}\n' for k,v in sorted(settings.items()))
  if hashlib.sha256(setting_bytes.encode()).hexdigest()!=m.get('runtime_environment_sha256'):raise ValueError('530 source runner settings do not match checkpoint runtime fingerprint')
  if any(v.get('set') for k,v in contract['effective'].items() if 'CAPTURE_PATH' in k or k.endswith('_CAPTURE_PATH')):raise ValueError('530 source runner has unexpected output-path capture')
  if any(k not in contract['keys'] for k in setvars):raise ValueError('530 source runner has unregistered settings')
  cp={'status':'checkpoint_ready','checkpoint_sha256':H(ck),'metadata_sha256':H(md),'prefix_manifest_sha256':H(pm),'protocol_manifest_sha256':H(O/'manifest.json'),'prefix_files_verified':n,'prefix_bytes_verified':total,'accepted_control_index':16,'next_control_index':17,'environment_count':2}
 active=workers();available=os.statvfs(Q).f_bavail*os.statvfs(Q).f_frsize
 report={'schema':'numi.cardboard.native531-preflight.v1','time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execute_requested':x.execute,'launch_allowed':cp['status']=='checkpoint_ready' and not active and available>=2*G and not Path(s['case']).exists(),'active_known_native_workers':active,'available_bytes':available,'source_snapshot_sha256':s['source_frozen_sha256'],'source_build_sha256':s['source_build_sha256'],'artifact_count':len(b['artifacts_sha256']),'checkpoint':cp,'native_argv':s['native_argv'],'environment':s['environment'],'diagnostic_capture_keys':s['diagnostic_capture_keys'],'physical_qualification':False}
 if cp['status']!='checkpoint_ready' or not x.execute:print(json.dumps(report,sort_keys=True));return 0
 if workers():raise ValueError('another native worker active')
 avail=lambda:os.statvfs(Q).f_bavail*os.statvfs(Q).f_frsize
 if avail()<2*G:raise ValueError('free-space below2GiB launch floor')
 E=P/'execution-531'
 if E.exists():raise ValueError('531 execution directory exists')
 E.mkdir();lock=(E/'runner.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (E/'launch.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 sys.path.insert(0,str(P));import capacity_helper_531 as guard;guard.B=Q;reserve=E/'emergency-reserve.bin';guard.write_reserve_file(reserve,128*M)
 env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
 t=time.monotonic();minimum=avail();timed=space=False;proc=None;errors=[]
 try:
  with (E/'wrapper.log').open('xb') as log:
   proc=subprocess.Popen(s['wrapper_argv'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   while proc.poll() is None:
    minimum=min(minimum,avail());timed=time.monotonic()-t>s['timeout_s']+120;space=avail()<512*M
    if timed or space:reserve.unlink(missing_ok=True);errors.extend(guard.terminate_owned_native(proc,s['native_argv'][0])['cleanup_errors']);break
    time.sleep(2)
 except BaseException as e:
  errors.append(repr(e))
  if proc and proc.poll() is None:errors.extend(guard.terminate_owned_native(proc,s['native_argv'][0])['cleanup_errors'])
 finally:reserve.unlink(missing_ok=True)
 code=proc.poll() if proc else None
 rec={'schema':'numi.cardboard.native531-runner-receipt.v1','exit_code':code,'timed_out':timed,'space_abort':space,'minimum_free_bytes':minimum,'wall_seconds':time.monotonic()-t,'errors':errors,'source_build_sha256':s['source_build_sha256'],'checkpoint':cp,'physical_qualification':False}
 (E/'receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n');print(json.dumps(rec,sort_keys=True))
 return code if isinstance(code,int) and code else int(bool(errors) or timed or space or code is None)
if __name__=='__main__':raise SystemExit(main())
