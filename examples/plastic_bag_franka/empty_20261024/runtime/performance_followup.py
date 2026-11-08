"""Exclusive, paired native timing after the full scene has completed."""
from pathlib import Path
import json,subprocess,time,statistics,hashlib
ROOT=Path('/Users/home/Documents/Codex/2026-10-21/i-wa');O=ROOT/'outputs/plastic_bag_empty_franka_20261024';REMOTE='/Users/n/numi-plastic-bag-empty-franka-20261024'
def run(a,**kw):return subprocess.run(a,check=True,**kw)
while subprocess.run(['ssh','macmini',f'test -f {REMOTE}/combined/run_summary.json'],capture_output=True).returncode:time.sleep(45)
active=subprocess.run(['ssh','macmini','pgrep -fl numi_classic_bag_drop'],capture_output=True,text=True).stdout.strip();assert not active,'Another native bag process owns GPU: '+active
pairs=[]
for step in [240,960]:
 cp='first_drop/checkpoint_240.bin' if step==240 else 'combined/checkpoint_960.bin'
 for rep in range(3):
  order=['baseline','skip'] if rep%2==0 else ['skip','baseline'];summaries={}
  for variant in order:
   name=f'performance/cp{step}_r{rep}_{variant}';flags='--retain-empty-work ' if variant=='baseline' else ''
   cmd=f'cd {REMOTE} && ./build/numi_classic_bag_drop --empty-bag {flags}--output {name} --resume {cp} --steps {step+8} --checkpoint-every 8 > {name.replace("/","_")}.log 2>&1'
   run(['ssh','macmini',cmd]);dest=O/name;dest.mkdir(parents=True,exist_ok=True);run(['rsync','-a',f'macmini:{REMOTE}/{name}/',str(dest)+'/']);summaries[variant]=json.loads((dest/'run_summary.json').read_text())
  a=O/f'performance/cp{step}_r{rep}_baseline';b=O/f'performance/cp{step}_r{rep}_skip';fields=['trajectory.json','latest_checkpoint.bin','solver_accuracy.csv','energy_balance.csv','grip_history.csv']
  equivalence={f:(a/f).read_bytes()==(b/f).read_bytes() for f in fields};assert all(equivalence.values()),equivalence
  base=summaries['baseline'];skip=summaries['skip'];r={'checkpoint_step':step,'repetition':rep,'order':order,'byte_equivalence':equivalence,'baseline_gpu_ms':base['gpu_time_ms'],'skip_gpu_ms':skip['gpu_time_ms'],'baseline_wall_s':base['wall_time_s'],'skip_wall_s':skip['wall_time_s'],'gpu_reduction_percent':100*(1-skip['gpu_time_ms']/base['gpu_time_ms'])};pairs.append(r);print(json.dumps(r),flush=True)
  (O/'performance_followup_progress.json').write_text(json.dumps({'pairs':pairs},indent=2))
result={'scope':'Six paired eight-interval native measurements: settled-table approach and held-bag lift, exclusive GPU execution; no full-scene throughput claim','pairs':pairs,'all_saved_state_and_checkpoint_bytes_equal':True,'median_gpu_reduction_percent':statistics.median(x['gpu_reduction_percent'] for x in pairs),'minimum_gpu_reduction_percent':min(x['gpu_reduction_percent'] for x in pairs),'maximum_gpu_reduction_percent':max(x['gpu_reduction_percent'] for x in pairs),'unchanged':'Material/contact equations, active film/self/table/grip passes, tolerances and adaptive budgets','change':'Skip coin-only Metal dispatches when active coin count is zero'}
(O/'performance_followup.json').write_text(json.dumps(result,indent=2));print('PERFORMANCE COMPLETE '+json.dumps(result),flush=True)
