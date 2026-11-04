from pathlib import Path
import sys,shutil,json
D=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/regional-compensation-1061-root/memory-performance-123/rejection-recovery-125')
p=D/'holistic-coarse-rebuild-native-205'
sys.path.insert(0,str(D))
from trial_tools import run
assert not (p/'install-motion.execution.json').exists()
assert shutil.disk_usage(p).free>=14.5*2**30
shutil.copy2(__file__,p/'resource-resume-runner.py')
sys.exit(run(p))

