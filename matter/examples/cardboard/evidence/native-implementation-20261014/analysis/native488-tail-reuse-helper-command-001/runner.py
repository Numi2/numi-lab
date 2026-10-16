from pathlib import Path
import datetime,hashlib,importlib.util,json,os,subprocess,sys,time
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'analysis/native488-tail-reuse-helper-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in json.loads((P/'package-manifest.json').read_text())['files_sha256'].items():assert sha(P/name)==digest,name
s=json.loads((P/'command-spec.json').read_text());O=P/'execution-488';assert not O.exists();C=Path(s['case_output']);assert not C.exists()
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(k in x for k in ('numi-matter-','sparse-scan-checker','source469-material-replay-check','tail-reuse-check')) and not x.split()[1].startswith('Z')];assert not workers,workers
sp=importlib.util.spec_from_file_location('capacity',P/'capacity-helper.py');g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g);g.B=B
assert g.available()>=g.MINIMUM
cmd=[str(P/'inputs/tail-reuse-check'),str(P/'inputs/base.metallib'),str(P/'inputs/overlay.metallib'),str(C)]
O.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_CARDBOARD_','NUMI_MATTER_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
(O/'launch.json').write_text(json.dumps({'argv':cmd,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_package_manifest_sha256':sha(P/'package-manifest.json'),'available_bytes':g.available(),'host':subprocess.check_output(['hostname'],text=True).strip()},indent=2)+'\n')
start=time.monotonic();proc=None;errors=[];abort=timeout=False;reserve=O/'emergency-reserve.bin';low=g.available()
try:
 g.write_reserve_file(reserve)
 with (O/'stdout.log').open('x') as out,(O/'stderr.log').open('x') as err:
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
  while proc.poll() is None:
   free=g.available();low=min(low,free);abort=free<g.ABORT_FLOOR;timeout=time.monotonic()-start>s['timeout_s']
   if abort or timeout:
    reserve.unlink(missing_ok=True);errors.extend(g.terminate_owned_native(proc,cmd[0])['cleanup_errors']);break
   time.sleep(0.25)
except BaseException as e:
 errors.append(repr(e))
 if proc is not None and proc.poll() is None:errors.extend(g.terminate_owned_native(proc,cmd[0])['cleanup_errors'])
finally:
 reserve.unlink(missing_ok=True);code=proc.poll() if proc else None
 files={str(p.relative_to(C)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(C.rglob('*')) if p.is_file()} if C.exists() else {}
 result={'schema':'numi.cardboard.native488-tail-reuse-helper-receipt.v1','exit_code':code,'errors':errors,'timed_out':timeout,'space_abort':abort,'wall_seconds':time.monotonic()-start,'minimum_available_bytes':low,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stdout_sha256':sha(O/'stdout.log') if (O/'stdout.log').exists() else None,'stderr_sha256':sha(O/'stderr.log') if (O/'stderr.log').exists() else None,'case_files':files,'physical_qualification':False}
 (O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
sys.exit(code if isinstance(code,int) and code else int(bool(errors) or timeout or abort or code is None))
