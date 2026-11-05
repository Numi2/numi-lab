from pathlib import Path
import json,subprocess,time,shutil,hashlib,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=W/'compiler-interior-rebuild-326'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((O/'compiler-plan-003.json').read_text())
for stage in plan['stages']:assert sha(stage['argv'][1])==stage['source_sha256']
stages=[(s['name'],s['argv'][1:]) for s in plan['stages']]
archive=W/'cold320-compiler-recovery-326/execution.json'
while True:
 if archive.exists():
  rows=json.loads(archive.read_text())['batches']
  if len(rows)==10:
   assert all(r['exit_code']==0 and Path(r['receipt']).is_file() for r in rows)
   break
 live=subprocess.run(['pgrep','-f','[a]rchive_cold320_compiler_326.py'],capture_output=True).returncode==0
 if not live:raise RuntimeError('archive stopped; no compiler started')
 time.sleep(5)
assert shutil.disk_usage(W).free>=17*2**30,'need at least 17 GiB before complete compiler'
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
 results.append(result);save(O/'compiler-execution-003.json',{'stages':results,'physical_admission':False,'native_steps':0})
 print(json.dumps(result),flush=True)
 if code or stop:
  print((O/(name+'.stderr.log')).read_text()[-3000:],flush=True);sys.exit(code or 1)
print('INTERIOR_COMPILER_PREPARATION_PASS_NATIVE_PENDING',flush=True)
