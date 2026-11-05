from pathlib import Path
import json,subprocess,time,hashlib,os,sys,re,shutil,gzip
D=Path(__file__).resolve().parent;stage_name=sys.argv[1];p=D/(stage_name+'.plan.json');plan=json.loads(p.read_text());stage=plan['stages'][0]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(plan['native_binary']['path'])==plan['native_binary']['sha256']
for raw,ref in plan['runtime_dependencies'].items():assert sha(raw)==ref['sha256'],raw
for ref in plan['required_transitive_input_pins']:assert sha(ref['path'])==ref['sha256'],ref['path']
root=Path(stage['argv'][4]);assert not root.exists()
for output_key in ("NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP", "NUMI_HUMAN_SKIN_REFERENCE_DUMP", "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT", "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV"):
 if output_key in stage['environment']:
  assert Path(stage['environment'][output_key]).is_relative_to(root), output_key
(root/'diagnostics').mkdir(parents=True)
reuse=plan["reused_immutable_preparation_dump"];assert reuse["producer_disabled"] and "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP" not in stage["environment"]
assert sha(reuse["source"]["path"])==reuse["source"]["sha256"]
os.link(reuse["source"]["path"],root/reuse["target_relative"])
vm0=subprocess.run(["sysctl","vm.swapusage"],capture_output=True,text=True).stdout
baseline_swap_mib=float(re.search(r"used = ([0-9.]+)M",vm0).group(1))
(D/"resource-policy.json").write_text(json.dumps({"baseline_swap_used_mib":baseline_swap_mib,"maximum_additional_swap_mib":3072,"minimum_available_disk_bytes":12*1024**3,"maximum_wall_s":900,"reason":"post-restart after older-storage cleanup: permit up to3GiB paging of inactive CPU preparation data; retain device working-set guard and at least12GiB disk; physical acceptance unchanged"},indent=2)+"\n")
env={k:v for k,v in os.environ.items() if not k.startswith('NUMI_')};env.update(stage['environment']);t=time.time();target=D/(stage_name+'.execution.json');assert not target.exists()
stdout_path=D/(stage_name+'.stdout.log.gz')
with stdout_path.open('wb') as o,(D/(stage_name+'.stderr.log')).open('w') as e:
 compressor=subprocess.Popen(['/usr/bin/gzip','-1c'],stdin=subprocess.PIPE,stdout=o,stderr=e)
 try:process=subprocess.Popen(stage['argv'],env=env,cwd=stage['cwd'],stdout=compressor.stdin,stderr=e)
 except BaseException:
  compressor.stdin.close();compressor.wait();raise
 compressor.stdin.close()
 (D/(stage_name+'.active.json')).write_text(json.dumps({'pid':process.pid,'start_epoch':t,'plan_path':str(p),'plan_sha256':sha(p)},indent=2)+'\n')
 stop_reason=None
 sampled_gpu_entry=False
 early_sample=None
 with (D/"resource-monitor.jsonl").open("w") as monitor:
  while process.poll() is None:
   vm=subprocess.run(["sysctl","vm.swapusage"],capture_output=True,text=True).stdout.strip()
   rss=subprocess.run(["ps","-p",str(process.pid),"-o","rss="],capture_output=True,text=True).stdout.strip()
   swap=re.search(r"used = ([0-9.]+)M",vm);swap_mib=float(swap.group(1)) if swap else 0.0
   if False and not sampled_gpu_entry and "newton_tail_dispatch_root=" in (D/(stage_name+'.stderr.log')).read_text():
    sampled_gpu_entry=True
    for name,args in [("gpu-entry-vmmap",["vmmap","-summary",str(process.pid)]),("gpu-entry-pressure",["memory_pressure","-Q"])]:
     try:
      diagnostic=subprocess.run(args,capture_output=True,text=True,timeout=8)
      (D/(name+".stdout.log")).write_text(diagnostic.stdout);(D/(name+".stderr.log")).write_text(diagnostic.stderr)
     except subprocess.TimeoutExpired:(D/(name+".timeout.txt")).write_text("8 second timeout\n")
   free=shutil.disk_usage(D).free;elapsed=time.time()-t
   if False and early_sample is None and elapsed>20:
    early_sample=subprocess.Popen(["sample",str(process.pid),"3","-file",str(D/"early-cpu-sample.txt")],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   monitor.write(json.dumps({"elapsed_s":elapsed,"rss_kib":int(rss) if rss else None,"swap_used_mib":swap_mib,"disk_available_bytes":free})+"\n");monitor.flush()
   if swap_mib>baseline_swap_mib+3072 or free<12*1024**3 or elapsed>900:
    stop_reason="swap_growth_above_3GiB" if swap_mib>baseline_swap_mib+3072 else "disk_below_12GiB" if free<12*1024**3 else "10_minute_one_root_diagnostic_budget"
    process.terminate()
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill()
    break
   time.sleep(5)
 code=process.wait()
 compression_code=compressor.wait()
 if compression_code!=0:
  if code==0:code=1
  stop_reason=stop_reason or 'stdout_compression_failed'
from collections import deque
log_tail=deque(maxlen=200); source_import_pass=False
uncompressed_hash=hashlib.sha256()
try:
 with gzip.open(stdout_path,'rb') as streamed:
  for raw in streamed:
   uncompressed_hash.update(raw);line=raw.decode('utf-8',errors='strict')
   source_import_pass=source_import_pass or ('whole_body_skin_fat_registered_volume_install=PASS' in line or 'registered_volume_source_only=PASS' in line or 'registered_volume_source_preflight=PASS' in line)
   log_tail.append(line)
except (OSError,EOFError,UnicodeError):
 if code==0:code=1
 stop_reason=stop_reason or 'stdout_integrity_failed'
log=''.join(log_tail)[-6500:]
result={'exit_code':code,'elapsed_s':time.time()-t,'plan_path':str(p),'plan_sha256':sha(p),'stdout_path':str(stdout_path),'stdout_encoding':'gzip','stdout_compressor_exit_code':compression_code,'stdout_sha256':sha(stdout_path),'stdout_uncompressed_sha256':uncompressed_hash.hexdigest(),'stderr_sha256':sha(D/(stage_name+'.stderr.log')),'physical_admission':False,'resource_stop_reason':stop_reason,'source_import_pass':source_import_pass, 'accepted_steps_are_separately_recorded_in_native_report':True}
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));print(log[-6500:]);print((D/(stage_name+'.stderr.log')).read_text()[-2500:])

sys.exit(code if code>=0 else 128-code)
