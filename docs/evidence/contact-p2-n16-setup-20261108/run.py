import hashlib,json,os,pathlib,signal,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
source=root/'tests/contact/fem_p2_n16_setup.py';model=root/'tests/contact/fem_p2_dynamics.py'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def host(label):
 d={}
 for name,cmd in [('uptime',['uptime']),('memory_pressure',['memory_pressure'])]:
  x=subprocess.run(cmd,text=True,capture_output=True);d[name]={'exit':x.returncode,'output':x.stdout}
 (p/(label+'-host.json')).write_text(json.dumps(d,indent=2)+'\n')
host('before');start=time.monotonic();cmd=[sys.executable,str(source)];limit_kib=4*1024*1024;peak=0;reason=None
with (p/'setup.json').open('w') as out,(p/'setup.stderr').open('w') as err,(p/'rss.csv').open('w') as log:
 log.write('elapsed_seconds,rss_kib\n');proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
 while proc.poll() is None:
  elapsed=time.monotonic()-start
  s=subprocess.run(['ps','-o','rss=','-p',str(proc.pid)],text=True,capture_output=True)
  try:rss=int(s.stdout.strip())
  except ValueError:rss=0
  peak=max(peak,rss);log.write(f'{elapsed:.6f},{rss}\n');log.flush()
  if rss>limit_kib:reason='rss_limit';break
  if elapsed>180:reason='timeout';break
  time.sleep(1)
 if reason is not None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 status=proc.wait()
host('after')
m={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'source_sha256':sha(source),'model_sha256':sha(model),'cmd':cmd,'exit':status,'limit_reason':reason,
 'wall_seconds':time.monotonic()-start,'sampled_peak_rss_kib':peak,'rss_limit_kib':limit_kib,'output_sha256':sha(p/'setup.json')}
if reason is None and status==0:
 d=json.loads((p/'setup.json').read_text());m['setup_check_pass']=d['dofs']==35937 and abs(d['total_mass_kg']-.0088)/.0088<=1e-10 and d['factor_nonzeros']>0
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m,flush=True)
if reason is not None or status!=0 or not m.get('setup_check_pass'):raise SystemExit('failure retained')
