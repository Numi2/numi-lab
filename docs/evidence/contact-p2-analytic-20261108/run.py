import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2];source=root/'tests/contact/fem_p2_analytic_diagnostic.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
m={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'python':sys.executable,'timeout_seconds':180,
 'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),'runs':[]}
for n in (2,4,8,16):
 now=time.monotonic();cmd=['/usr/bin/time','-l',sys.executable,str(source),str(n)]
 with (p/f'n{n}.json').open('w') as out,(p/f'n{n}.stderr').open('w') as err:
  proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
  try:status=proc.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   status='timeout'
 m['runs'].append({'n':n,'argv':cmd,'exit':status,'elapsed_seconds':time.monotonic()-now,'output_sha256':sha(p/f'n{n}.json')})
 (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(m['runs'][-1],flush=True)
 if status!=0:raise SystemExit('failure retained; stopping')
