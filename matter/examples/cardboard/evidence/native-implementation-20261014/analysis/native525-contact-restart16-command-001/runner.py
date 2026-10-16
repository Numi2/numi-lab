from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys,time
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
P=B/'analysis/native525-contact-restart16-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--phase',choices=['525'],required=True);a.add_argument('--execute',action='store_true');args=a.parse_args()
s=json.loads((P/'command-spec.json').read_text());binding=json.loads((P/'package-manifest.json').read_text())
for name,digest in binding['files_sha256'].items():assert sha(P/name)==digest,name
S=Path(s['source_snapshot']);D=Path(s['build_directory']);assert sha(S/'frozen-source.json')==s['source_frozen_sha256']
f=json.loads((S/'frozen-source.json').read_text())
for name,digest in f['source_sha256'].items():assert sha(S/name)==digest,name
for name,digest in s['files_sha256'].items():assert sha(D/name)==digest,name
material=S/s['material_relative_path'];assert sha(material)==s['material_sha256']
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(k in x for k in ('numi-matter-','sparse-scan-checker','source469-material-replay-check')) and not x.split()[1].startswith('Z')]
assert not workers,workers
sp=importlib.util.spec_from_file_location('capacity',P/'capacity-helper.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g);g.B=B
assert g.available()>=g.MINIMUM
cmd=[str(D/'numi-matter-cardboard-full-contact-armijo-check'),str(material),*s['cases'][args.phase]]
pre={'schema':'numi.cardboard.native525-preflight.v1','phase':args.phase,'source_files_verified':len(f['source_sha256']),'bound_build_files':s['files_sha256'],'native_argv':cmd,'available_bytes':g.available(),'execute_requested':args.execute}
print(json.dumps(pre),flush=True)
if not args.execute:sys.exit(0)
O=P/('execution-'+args.phase);assert not O.exists();O.mkdir();(O/'launch.json').write_text(json.dumps(dict(pre,started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_CARDBOARD_','NUMI_MATTER_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
env.update(s['environment'][args.phase])
C=Path(s['case_outputs'][args.phase]);assert not C.exists()
start=time.monotonic();proc=None;errors=[];abort=timeout=False;reserve=O/'emergency-reserve.bin';low=g.available()
try:
 g.write_reserve_file(reserve)
 with (O/'stdout.log').open('x') as out,(O/'stderr.log').open('x') as err:
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
  while proc.poll() is None:
   free=g.available();low=min(low,free);abort=free<g.ABORT_FLOOR;timeout=time.monotonic()-start>s['timeout_s']
   if abort or timeout:
    reserve.unlink(missing_ok=True);cleanup=g.terminate_owned_native(proc,cmd[0]);errors.extend(cleanup['cleanup_errors']);break
   time.sleep(0.5)
except BaseException as e:
 errors.append(repr(e))
 if proc is not None and proc.poll() is None:errors.extend(g.terminate_owned_native(proc,cmd[0])['cleanup_errors'])
finally:
 reserve.unlink(missing_ok=True)
 code=proc.poll() if proc else None
 receipt={'schema':'numi.cardboard.native525-receipt.v1','phase':args.phase,'exit_code':code,'timed_out':timeout,'space_abort':abort,'errors':errors,'wall_seconds':time.monotonic()-start,'minimum_available_bytes':low,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scrubbed_parent_key_names':sorted(set(os.environ)-set(env)),'stdout_sha256':sha(O/'stdout.log') if (O/'stdout.log').exists() else None,'stderr_sha256':sha(O/'stderr.log') if (O/'stderr.log').exists() else None,'shipping_or_physical_validation':False,'case_files':{str(p.relative_to(C)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(C.rglob('*')) if p.is_file()} if C.exists() else {}}
 (O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
sys.exit(code if isinstance(code,int) and code!=0 else int(bool(errors) or timeout or abort or code is None))
