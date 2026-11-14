"""Extract only the original lane 5 terminal measurements, preserving lines."""
import gzip,json
from pathlib import Path
r=Path('/Users/n/Developer/numi-crow-rl-20261107')
src=r/'artifacts/crow-v15-uniform-handoff-second-seed-20261114/push-20261127-rebound-uniform/run.stderr.gz'
dest=Path('/tmp/crow-48s-landing-terminal-observations.json')
keep=[]
with gzip.open(src,'rt') as f:
    for line in f:
        if line.startswith('CROW_SUPPORT_SAMPLE '):
            obj=json.loads(line.split(' ',1)[1])
            if obj.get('lane')==5 and obj.get('step') in (2397,2398,2399,2400): keep.append(line.rstrip())
        elif line.startswith(('CROW_DRIVE_TARGET ','CROW_TWO_TOE_SUBSTEP ','CROW_TOE_V_SUBSTEP ','CROW_PRE_AERO_WRENCH ')):
            x=line.split()
            if x[0]=='CROW_DRIVE_TARGET': step,lane=int(x[1]),int(x[2])
            else: step,lane=int(x[1]),int(x[3])
            if lane==5 and step in (2399,2400): keep.append(line.rstrip())
dest.write_text(json.dumps({'lines':keep,'scope':'Original observer log, lane 5 final intervals'},indent=2)+'\n')
print('retained',len(keep),'lines',dest)
