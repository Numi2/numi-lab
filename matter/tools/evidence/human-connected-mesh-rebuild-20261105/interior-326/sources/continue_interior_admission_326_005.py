from pathlib import Path
import json,subprocess,time,shutil,hashlib,sys
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'interior-preserved-boundary-326';O=W/'compiler-interior-rebuild-326'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
stages=[('mass-receipt-326',[D/'compiler/mass-admission/build_current_fractional_mass_receipt_1061.py','--run-manifest',D/'compiler/mass-admission/run-manifest.json','--execution',D/'compiler/mass-admission/execution.json','--stdout-log',D/'compiler/mass-allocation/stdout.log','--identity-config',D/'compiler/mass-admission/identity.json','--control-manifest',D/'compiler/mass-allocation/source-control.json','--output',D/'compiler/mass-admission/mass-receipt.json','--source-projection-report',D/'compiler/mass-admission/source-projection.json','--failure-report',D/'compiler/mass-admission/failure.json']),('mass-target-parity-326',[W/'check_mass_targets_326.py']),('mass-converter-326',[W/'convert_joint64_mass.py',D]),('native-plan-327',[W/'prepare_joint64_native.py',D,W/'interior-preserved-native-327'])]
wait_start=time.time()
while True:
 upstream=O/'compiler-execution-004.json'
 record=json.loads(upstream.read_text()) if upstream.exists() else {'stages':[]}
 assert not any(x['exit_code'] or x['guard_stop'] for x in record['stages']), 'upstream stopped; inspect same attempt'
 if len(record['stages'])==12:break
 assert subprocess.run(['pgrep','-f','[c]ontinue_interior_compiler_326_004.py'],capture_output=True).returncode==0, 'upstream not live'
 if time.time()-wait_start>3000:raise RuntimeError('upstream observation budget reached; inspect same attempt')
 time.sleep(5)
assert json.loads((D/'full-context/report.json').read_text())['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
save(O/'admission-conversion-plan.json',{'stages':[{'name':name,'argv':[sys.executable,*map(str,args)],'source_sha256':sha(args[0])} for name,args in stages],'minimum_free_bytes':12*2**30,'maximum_rss_kib':8*2**20,'maximum_stage_wall_s':900,'physical_admission':False,'native_steps':0})
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
 results.append(result);save(O/'admission-conversion-execution.json',{'stages':results,'physical_admission':False,'native_steps':0})
 print(json.dumps(result),flush=True)
 if code or stop:
  print((O/(name+'.stderr.log')).read_text()[-3000:],flush=True);sys.exit(code or 1)
print('CURRENT_NATIVE_INPUTS_READY',flush=True)
