import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
source=root/'tests/contact/fem_p2_n4_order18_trajectory.py';model=root/'tests/contact/fem_p2_dynamics.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
m={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'model_sha256':sha(model),'python':sys.executable,'runs':[]}
for variant,timeout in (('coarse',480),('fine',900)):
 cmd=['/usr/bin/time','-l',sys.executable,str(source),variant];start=time.monotonic()
 with (p/(variant+'.json')).open('w') as out,(p/(variant+'.stderr')).open('w') as err:
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
  try:status=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   status='timeout'
 m['runs'].append({'variant':variant,'argv':cmd,'exit':status,'timeout_s':timeout,'elapsed_s':time.monotonic()-start,'output_sha256':sha(p/(variant+'.json'))})
 (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m['runs'][-1],flush=True)
 if status!=0:raise SystemExit('failure retained; stopping')
