from pathlib import Path
import json,subprocess,time,hashlib,os,sys,re,shutil
D=Path(__file__).resolve().parent;stage_name=sys.argv[1];p=D/(stage_name+'.plan.json');plan=json.loads(p.read_text());stage=plan['stages'][0]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(plan['native_binary']['path'])==plan['native_binary']['sha256']
for raw,ref in plan['runtime_dependencies'].items():assert sha(raw)==ref['sha256'],raw
root=Path(stage['argv'][4]);assert not root.exists();(root/'diagnostics').mkdir(parents=True)
env={k:v for k,v in os.environ.items() if not k.startswith('NUMI_')};env.update(stage['environment']);t=time.time();target=D/(stage_name+'.execution.json');assert not target.exists()
with (D/(stage_name+'.stdout.log')).open('w') as o,(D/(stage_name+'.stderr.log')).open('w') as e:
 process=subprocess.Popen(stage['argv'],env=env,cwd=stage['cwd'],stdout=o,stderr=e)
 (D/(stage_name+'.active.json')).write_text(json.dumps({'pid':process.pid,'start_epoch':t,'plan_path':str(p),'plan_sha256':sha(p)},indent=2)+'\n')
 stop_reason=None
 with (D/"resource-monitor.jsonl").open("w") as monitor:
  while process.poll() is None:
   vm=subprocess.run(["sysctl","vm.swapusage"],capture_output=True,text=True).stdout.strip()
   rss=subprocess.run(["ps","-p",str(process.pid),"-o","rss="],capture_output=True,text=True).stdout.strip()
   swap=re.search(r"used = ([0-9.]+)M",vm);swap_mib=float(swap.group(1)) if swap else 0.0
   free=shutil.disk_usage(D).free;elapsed=time.time()-t
   monitor.write(json.dumps({"elapsed_s":elapsed,"rss_kib":int(rss) if rss else None,"swap_used_mib":swap_mib,"disk_available_bytes":free})+"\n");monitor.flush()
   if swap_mib>3072 or free<2*1024**3 or elapsed>900:
    stop_reason="swap_above_3GiB" if swap_mib>3072 else "disk_below_2GiB" if free<2*1024**3 else "15_minute_diagnostic_budget"
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill()
    break
   time.sleep(5)
 code=process.wait()
log=(D/(stage_name+'.stdout.log')).read_text()
result={'exit_code':code,'elapsed_s':time.time()-t,'plan_path':str(p),'plan_sha256':sha(p),'stdout_sha256':sha(D/(stage_name+'.stdout.log')),'stderr_sha256':sha(D/(stage_name+'.stderr.log')),'physical_admission':False,'resource_stop_reason':stop_reason,'source_import_pass':code==0 and 'registered_volume_source_only=PASS' in log}
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));print(log[-6500:]);print((D/(stage_name+'.stderr.log')).read_text()[-2500:])
