import hashlib,json,os,pathlib,re,signal,subprocess,sys,time
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2]
if len(sys.argv)!=2 or sys.argv[1] not in ('coarse','fine'):raise SystemExit('usage: run.py coarse|fine')
variant=sys.argv[1];timeout={'coarse':5400,'fine':10800}[variant]
source=root/'tests/contact/fem_p2_full_restart.py';model=root/'tests/contact/fem_p2_dynamics.py'
reference=root/'docs/evidence/contact-p2-n8-order18-20261108/fine.json'
prerequisite=root/'docs/evidence/contact-p2-n8-restart-full-20261108/analysis.json'
sha=lambda f:hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
pr=json.loads(prerequisite.read_text())
assert pr['prerequisite_pass'] and pr['reference_sha256']==sha(reference)
assert pr['restart_sha256']==sha(prerequisite.parent/'fine.json')
identity={'source_sha256':sha(source),'model_sha256':sha(model),
 'reference_sha256':sha(reference),'prerequisite_sha256':sha(prerequisite),
 'prerequisite_restart_sha256':pr['restart_sha256']}
if variant=='fine':
 prev=json.loads((p/'coarse-manifest.json').read_text())
 assert prev['exit']==0 and prev['limit_reason'] is None, 'coarse did not complete'
 assert all(prev[k]==v for k,v in identity.items()), 'phase inputs changed'
 assert prev['output_sha256']==sha(p/'coarse.json'), 'coarse output changed'
for suffix in ('.json','.stderr','-rss.csv','-pressure.csv','-disk.csv','-manifest.json'):
 if (p/(variant+suffix)).exists():raise SystemExit('phase artifact already exists; preserve it')
def diskfree():
 st=os.statvfs(root);return st.f_bavail*st.f_frsize
def sample():
 m=subprocess.run(['memory_pressure'],capture_output=True,text=True,check=True)
 s=subprocess.run(['sysctl','-n','vm.swapusage'],capture_output=True,text=True,check=True)
 a=re.search(r'System-wide memory free percentage:\s*(\d+)%',m.stdout)
 b=re.search(r'used\s*=\s*([0-9.]+)([MG])',s.stdout)
 if not a or not b:raise RuntimeError('memory monitor output not parseable')
 return int(a.group(1)),float(b.group(1))*(1024 if b.group(2)=='G' else 1),s.stdout.strip()
initial_disk=diskfree()
if initial_disk<5*1024**3:raise SystemExit(f'preflight refused: only {initial_disk} available bytes')
before=sample()
cmd=['/usr/bin/nice','-n','10',sys.executable,str(source),'16',variant]
m={**identity,'prediction_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'variant':variant,'cmd':cmd,'baseline_free_percent':before[0],'baseline_swap_used_MiB':before[1],
 'baseline_swap_text':before[2],'initial_disk_available_bytes':initial_disk,
 'python_executable':sys.executable,'python_version':sys.version,
 'numerical_versions':subprocess.check_output([sys.executable,'-c','import numpy,scipy,skfem; print(numpy.__version__,scipy.__version__,skfem.__version__)'],text=True).strip()}
start=time.monotonic();peak=0;reason=None;bad=0;last_pressure=-5
with (p/(variant+'.json')).open('w') as out,(p/(variant+'.stderr')).open('w') as err,(p/(variant+'-rss.csv')).open('w') as log,(p/(variant+'-pressure.csv')).open('w') as plog,(p/(variant+'-disk.csv')).open('w') as dlog:
 log.write('elapsed_seconds,rss_kib\n');plog.write('elapsed_seconds,free_percent,swap_used_MiB,swap_total_report\n');dlog.write('elapsed_seconds,available_bytes\n')
 proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
 while proc.poll() is None:
  elapsed=time.monotonic()-start
  try:
   r=subprocess.run(['ps','-o','rss=','-p',str(proc.pid)],text=True,capture_output=True,check=True);rss=int(r.stdout.strip())
   available=diskfree()
  except Exception as e:
   if proc.poll() is not None:break
   reason='resource_monitor_failure: '+str(e);break
  peak=max(peak,rss);log.write(f'{elapsed:.6f},{rss}\n');log.flush()
  dlog.write(f'{elapsed:.6f},{available}\n');dlog.flush()
  if available<2*1024**3:reason='disk_available_limit';break
  if rss>4*1024*1024:reason='rss_limit';break
  if elapsed>timeout:reason='timeout';break
  if elapsed-last_pressure>=5:
   try:free,swap,text=sample()
   except Exception as e:reason='pressure_monitor_failure: '+str(e);break
   plog.write(f'{elapsed:.6f},{free},{swap:.2f},"{text}"\n');plog.flush();last_pressure=elapsed
   bad=bad+1 if free<20 or swap>before[1]+128 else 0
   if bad>=2:reason='sustained_memory_pressure_or_swap_growth';break
  time.sleep(1)
 if reason:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 status=proc.wait()
m.update({'exit':status,'limit_reason':reason,'timeout_s':timeout,'rss_limit_kib':4*1024*1024,
 'disk_start_minimum_bytes':5*1024**3,'disk_stop_minimum_bytes':2*1024**3,
 'peak_rss_kib':peak,'elapsed_s':time.monotonic()-start,'output_sha256':sha(p/(variant+'.json'))})
(p/(variant+'-manifest.json')).write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2),flush=True)
if reason or status!=0:raise SystemExit('failure retained; stopping')
