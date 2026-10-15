from pathlib import Path
import datetime,hashlib,json,subprocess,time,os
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');A=B/'analysis/cgls-right-preconditioning-study-146-002';P=B/'analysis/native-residual-direction-replay-419-001';O=B/'analysis/native-block-inverse-cgls-residual-replay-457';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (B/'solver/cgls-state-sidecar-observer-455/receipt.json').is_file(),'previousGPUworker must finish'
assert not O.exists();v=os.statvfs(B);assert v.f_bavail*v.f_frsize>=1024**3
manifest=json.loads((A/'manifest.json').read_text());protected={A/p:r['sha256'] for p,r in manifest['files'].items()}
report=json.loads((A/'report.json').read_text());program=json.loads((P/'compiled-program.json').read_text());protected.update({P/p:h for p,h in program['files'].items()});buildpath=R/'build-cardboard-provenance-125/source-build.json';build=json.loads(buildpath.read_text());protected.update({Path(p):h for p,h in build['artifacts_sha256'].items()});protected[buildpath]=sha(buildpath);protected[A/'manifest.json']=sha(A/'manifest.json')
for p,h in protected.items():assert sha(p)==h,p
O.mkdir();commands={};helper=report['replay_contract']['helper'];pack=report['replay_contract']['compiled_package']
for name,row in report['native_replay_payloads'].items():commands[name]=[helper,pack,row['payload_path'],str(O/name)]
(O/'prediction.json').write_text(json.dumps({'schema':'numi.cardboard.native_preconditioned_CGLS_residual_replay.v1','question':'Do right-scaled CGLS proposals improve the actual nonlinear residual of the captured419 state?','prediction':'Offline principal3x3block scaling reduces true linear residual at64 iterations compared with identity. Native residual, finite material/geometry feasibility, useful tested alpha and relative benefit remain unknown. Evaluate each full FP32-exported candidate through the same native residual, hold accepted history fixed, preserve nooperator/noparameter/nothresholdchanges. This is zero physicaltransactions and no Newton or nonlinearproductioncertificate.','package_manifest_sha256':sha(A/'manifest.json'),'helper_program_sha256':sha(P/'compiled-program.json'),'source125_build_sha256':sha(buildpath),'commands':commands,'semantic_erratum_to146_first_report':'In A=M+dt^2 K, K=-df_internal/dx, not -df_internal/dv; captured matrix and recurrence unchanged.'},indent=2)+'\n')
results={};start=time.monotonic()
for name,cmd in commands.items():
 for mode,argv in [('preflight',cmd+['--preflight-only']),('native',cmd)]:
  with (O/(name+'.'+mode+'.stdout.log')).open('x') as out,(O/(name+'.'+mode+'.stderr.log')).open('x') as err:
   try:q=subprocess.run(argv,stdout=out,stderr=err,timeout=180);r={'exit_code':q.returncode,'timed_out':False}
   except subprocess.TimeoutExpired:r={'exit_code':None,'timed_out':True}
  results[name+'.'+mode]=r
  if r['exit_code']!=0:break
 for p,h in protected.items():assert sha(p)==h,p
(O/'receipt.json').write_text(json.dumps({'commands':results,'wall_seconds':time.monotonic()-start,'physical_transactions':0,'source_and_helper_inputs_unchanged':True,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n');(O/'runner.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(O)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()}},indent=2)+'\n');print('457',results,flush=True);raise SystemExit(0 if all(x['exit_code']==0 for x in results.values()) else 2)
