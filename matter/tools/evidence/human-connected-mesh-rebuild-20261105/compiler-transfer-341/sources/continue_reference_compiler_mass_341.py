"""Resume only unexecuted compiler stages, retaining the current guard budgets."""
from pathlib import Path
import json,subprocess,time,shutil,hashlib,sys,os
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');O=W/'compiler-reference-screened-341';D=W/'reference-screened-interior-341'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
prefix=json.loads((O/'compiler-prefix-execution.json').read_text())
assert len(prefix['stages'])==7 and all(s['exit_code']==0 and s['guard_stop'] is None for s in prefix['stages']),'prefix incomplete'
assert not (O/'compiler-execution.json').exists(),'do not duplicate or overwrite a prior continuation'
assert shutil.disk_usage(W).free>=16*2**30
results=list(prefix['stages'])
def runstage(s):
 name=s['name'];argv=s['argv']
 assert sha(argv[1])==s['source_sha256'],argv[1]
 assert shutil.disk_usage(W).free>=12*2**30
 start=time.time();stop=None
 with (O/(name+'.stdout.log')).open('x') as a,(O/(name+'.stderr.log')).open('x') as b,(O/(name+'.resources.jsonl')).open('x') as mon:
  p=subprocess.Popen(argv,stdout=a,stderr=b)
  save(O/'compiler-active.json',{'pid':p.pid,'stage':name,'started_epoch':start})
  while p.poll() is None:
   rows=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True).stdout.splitlines()
   values=[tuple(map(int,r.split())) for r in rows];descendants={p.pid}
   for _ in range(8):
    old=len(descendants);descendants|={pid for pid,parent,rss in values if parent in descendants}
    if len(descendants)==old:break
   rss=sum(rss for pid,parent,rss in values if pid in descendants)
   free=shutil.disk_usage(W).free;elapsed=time.time()-start
   mon.write(json.dumps({'elapsed_s':elapsed,'rss_kib':rss,'disk_free_bytes':free})+'\n');mon.flush()
   if free<12*2**30 or rss>8*2**20 or elapsed>900:
    stop='disk' if free<12*2**30 else 'rss' if rss>8*2**20 else 'time'
    for pid in sorted(descendants,reverse=True):
     try:os.kill(pid,15)
     except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill()
    break
   time.sleep(5)
  code=p.wait()
 result={'name':name,'exit_code':code,'guard_stop':stop,'wall_s':time.time()-start,'stdout_sha256':sha(O/(name+'.stdout.log')),'stderr_sha256':sha(O/(name+'.stderr.log'))}
 results.append(result);save(O/'compiler-execution.json',{'stages':results,'physical_admission':False,'native_steps':0})
 print(json.dumps(result),flush=True)
 if code or stop:
  print((O/(name+'.stderr.log')).read_text()[-3000:],flush=True);sys.exit(code or 1)
plan=json.loads((O/'compiler-mass-plan.json').read_text())
for s in plan['stages']:runstage(s)
oldroot=str(W/'interior-preserved-boundary-326')
def remap(v):
 if isinstance(v,str):return v.replace(oldroot,str(D)).replace(str(W/'interior-preserved-native-327'),str(W/'reference-screened-native-344'))
 if isinstance(v,list):return [remap(x) for x in v]
 if isinstance(v,dict):return {k:remap(x) for k,x in v.items()}
 return v
post=remap(json.loads((W/'compiler-interior-rebuild-326/admission-conversion-plan.json').read_text()))
for s in post['stages']:
 s['name']=s['name'].replace('326','341').replace('327','344')
 if s['name']=='mass-target-parity-341':s['argv'].append(str(D))
 s['source_sha256']=sha(s['argv'][1])
post['stages'].append({'name':'current-runtime-pins-344','argv':[post['stages'][0]['argv'][0],str(W/'prepare_current_runtime_344.py')],'source_sha256':sha(W/'prepare_current_runtime_344.py')})
save(O/'admission-conversion-plan.json',post)
for s in post['stages']:runstage(s)
save(O/'compiler-active.json',{'pid':None,'stage':'complete-native-trial-pending','native_steps':0})
print('SCREENED_COMPILER_AND_MASS_ADMISSION_COMPLETE_NATIVE344_READY',flush=True)
