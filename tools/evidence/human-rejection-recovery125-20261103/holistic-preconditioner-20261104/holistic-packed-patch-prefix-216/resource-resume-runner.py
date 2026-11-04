from pathlib import Path
import json,shutil,sys,subprocess
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
p=D/'holistic-packed-patch-prefix-216'
assert subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).returncode!=0
assert not (p/'install-motion.execution.json').exists()
old=p/'prelaunch-empty-override';old.mkdir()
for name in ['install-motion.plan.json','runner.log','executed-runner.py']:shutil.copy2(p/name,old/name)
plan=json.loads((p/'install-motion.plan.json').read_text())
assert plan['stages'][0]['environment'].pop('NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP')==''
root=Path(plan['stages'][0]['argv'][4]);(root/'diagnostics').rmdir();root.rmdir()
# Existing runner removes all inherited NUMI_* values, and its reused-dump
# contract requires the producer key to be absent. The empty override was
# redundant and was rejected before any physical process launched.
(p/'install-motion.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
assert shutil.disk_usage(D).free>=15.8*2**30
shutil.copy2(__file__,p/'resource-resume-runner.py')
sys.path.insert(0,str(D));from trial_tools import run
sys.exit(run(p))
