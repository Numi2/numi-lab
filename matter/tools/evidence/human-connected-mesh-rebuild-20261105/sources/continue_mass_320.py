from pathlib import Path
import json,subprocess,time,shutil,hashlib,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'joint-boundary-bone-admitted-320';O=W/'compiler-rebuild-320'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert json.loads((D/'full-context/report.json').read_text())['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
stages=[('mass-support-320',[W/'rebind_joint64_mass_support.py',D]),('mass-targets-320',[W/'prepare_prescribed_local_mass_315.py','--candidate-root',D,'--output',D/'compiler/mass-targets','--halo-rings','0','--all-proposed-cell-masses']),('mass-allocation-320',[W/'allocate_joint64_mass.py',D]),('source-qualification-320',[W/'qualify_joint64_source.py',D]),('mass-admission-prepare-320',[W/'prepare_joint64_mass_admission.py',D])]
save(O/'mass-continuation-plan.json',{'stages':[{'name':name,'argv':[sys.executable,*map(str,args)],'source_sha256':sha(args[0])} for name,args in stages],'minimum_free_bytes':12*2**30,'maximum_rss_kib':8*2**20,'maximum_stage_wall_s':900,'physical_admission':False,'native_steps':0})
archive_receipt=W/'cold-mass-preparation-320-recovery/human-cold-mass-preparation-320-verification.json'
wait_start=time.time()
while not archive_receipt.exists():
 alive=subprocess.run(['pgrep','-f','[a]rchive_cold_mass_preparation_320.py'],capture_output=True).returncode==0
 if not alive:raise RuntimeError('archive terminated without verified restoration')
 if time.time()-wait_start>1800:raise RuntimeError('archive still live; observation budget reached, inspect same job')
 time.sleep(5)
receipt=json.loads(archive_receipt.read_text())
assert receipt['status']=='PASS_INDEPENDENT_DOWNLOAD_ALL_MEMBERS_AND_RESTORE'
assert shutil.disk_usage(W).free>=15*2**30,'need at least 15 GiB for mass preparation output'
results=[]
for name,args in stages:
 assert shutil.disk_usage(W).free>=12*2**30
 start=time.time();stop=None
 with (O/(name+'.stdout.log')).open('w') as a,(O/(name+'.stderr.log')).open('w') as b,(O/(name+'.resources.jsonl')).open('w') as mon:
  p=subprocess.Popen([sys.executable,*map(str,args)],stdout=a,stderr=b)
  save(O/'compiler-active.json',{'pid':p.pid,'stage':name,'started_epoch':start})
  while p.poll() is None:
   # Include direct child compiler processes in the working-set guard.
   rows=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True).stdout.splitlines()
   values=[tuple(map(int,r.split())) for r in rows]
   descendants={p.pid}
   for _ in range(8):
    added={pid for pid,parent,rss in values if parent in descendants};old=len(descendants);descendants|=added
    if len(descendants)==old:break
   rss=sum(rss for pid,parent,rss in values if pid in descendants)
   free=shutil.disk_usage(W).free;elapsed=time.time()-start
   mon.write(json.dumps({'elapsed_s':elapsed,'rss_kib':rss,'disk_free_bytes':free})+'\n');mon.flush()
   if free<12*2**30 or rss>8*2**20 or elapsed>900:
    stop='disk' if free<12*2**30 else 'rss' if rss>8*2**20 else 'time'
    for pid in sorted(descendants,reverse=True):
     try:__import__('os').kill(pid,15)
     except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill()
    break
   time.sleep(5)
  code=p.wait()
 result={'name':name,'exit_code':code,'guard_stop':stop,'wall_s':time.time()-start,'stdout_sha256':sha(O/(name+'.stdout.log')),'stderr_sha256':sha(O/(name+'.stderr.log'))}
 results.append(result);save(O/'mass-continuation-execution.json',{'stages':results,'physical_admission':False,'native_steps':0})
 print(json.dumps(result),flush=True)
 if code or stop:
  print((O/(name+'.stderr.log')).read_text()[-3000:],flush=True);sys.exit(code or 1)
print('EXISTING_COMPILER_AND_P1_PASS',flush=True)
