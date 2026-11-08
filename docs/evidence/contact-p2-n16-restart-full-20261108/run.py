import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
source=root/'tests/contact/fem_p2_full_restart.py';model=root/'tests/contact/fem_p2_dynamics.py'
reference=root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json'
prerequisite=root/'docs/evidence/contact-p2-n8-restart-full-20261108/analysis.json'
pr=json.loads(prerequisite.read_text())
assert pr['prerequisite_pass'], 'n8 full-reconstruction prerequisite failed'
assert pr['reference_sha256']==hashlib.sha256(reference.read_bytes()).hexdigest(), 'n8 reference changed'
assert pr['restart_sha256']==hashlib.sha256((prerequisite.parent/'fine.json').read_bytes()).hexdigest(), 'n8 restart changed'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
m={'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'model_sha256':sha(model),'reference_sha256':sha(reference),'prerequisite_sha256':sha(prerequisite),
 'python_executable':sys.executable,'python_version':sys.version,
 'numerical_versions':subprocess.check_output([sys.executable,'-c','import numpy,scipy,skfem; print(numpy.__version__,scipy.__version__,skfem.__version__)'],text=True).strip(),'runs':[]}
def host(label):
 d={}
 for name,cmd in [('uptime',['uptime']),('memory_pressure',['memory_pressure'])]:
  r=subprocess.run(cmd,text=True,capture_output=True);d[name]={'exit':r.returncode,'output':r.stdout}
 (p/(label+'-host.json')).write_text(json.dumps(d,indent=2)+'\n')
for v,timeout in (('coarse',5400),('fine',10800)):
 host(v+'-before');cmd=[sys.executable,str(source),'16',v];start=time.monotonic();peak=0;reason=None
 with (p/(v+'.json')).open('w') as out,(p/(v+'.stderr')).open('w') as err,(p/(v+'-rss.csv')).open('w') as log:
  log.write('elapsed_seconds,rss_kib\n');proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
  while proc.poll() is None:
   elapsed=time.monotonic()-start;r=subprocess.run(['ps','-o','rss=','-p',str(proc.pid)],text=True,capture_output=True)
   try:rss=int(r.stdout.strip())
   except ValueError:rss=0
   peak=max(peak,rss);log.write(f'{elapsed:.6f},{rss}\n');log.flush()
   if rss>4*1024*1024:reason='rss_limit';break
   if elapsed>timeout:reason='timeout';break
   time.sleep(1)
  if reason:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  status=proc.wait()
 host(v+'-after')
 m['runs'].append({'variant':v,'cmd':cmd,'exit':status,'limit_reason':reason,'timeout_s':timeout,'rss_limit_kib':4*1024*1024,'peak_rss_kib':peak,'elapsed_s':time.monotonic()-start,'output_sha256':sha(p/(v+'.json'))})
 (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m['runs'][-1],flush=True)
 if reason or status!=0:raise SystemExit('failure retained; stopping')
