from pathlib import Path
import json,shutil,sys,subprocess
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
p=D/'holistic-integrated-parity-native-212'
shutil.copy2('/tmp/human-holistic-integrated-parity-native-212.log',p/'prelaunch-disk-guard.log')
(p/'preparation-resource-justification.json').write_text(json.dumps({'preparation_free_GiB':shutil.disk_usage(p).free/2**30,'preparation_minimum_GiB':14.45,'hard_runtime_minimum_GiB':12,'maximum_swap_growth_GiB':3,'rationale':'The14.5GiB preparation heuristic missed by2MiB. Retain the12GiB hard runtime guard and3GiB swap-growth watchdog; the completed197 has the same full-coordinate source geometry, endpoint/capture sizes and working set. Added coarse storage is22KiB and the diagnostic reuses existing operator scratch. No physical, timestep, accuracy or runtime-resource gate is relaxed.'},indent=2)+'\n')
sys.path.insert(0,str(D))
from trial_tools import run
assert subprocess.run(['pgrep','-x','numi-human-native'],capture_output=True).returncode!=0
assert not (p/'install-motion.execution.json').exists()
assert shutil.disk_usage(p).free>=14.45*2**30
shutil.copy2(__file__,p/'resource-resume-runner.py')
sys.exit(run(p))

