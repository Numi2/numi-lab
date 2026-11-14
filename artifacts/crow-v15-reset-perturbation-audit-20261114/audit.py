from pathlib import Path
import gzip,json,hashlib
A=Path('artifacts/crow-v15-reset-perturbation-audit-20261114');F=Path('artifacts/crow-v15-flight-release-latch-20261114/turn.metal');s=F.read_text(); marker='// Initial-state perturbation only; no mid-episode state edits or injected force.'
starts=[];i=0
while True:
 i=s.find(marker,i)
 if i<0:break
 starts.append(i);i+=len(marker)
blocks=[]
for i in starts:
 j=s.index('\n        controllerParameters[environment].z',i) if i==starts[-1] else starts[starts.index(i)+1]
 blocks.append(s[i:j].strip())
assert len(blocks)==2 and blocks[0]==blocks[1]
seeds={}
for seed,folder in [('20261126','crow-v15-upward-trigger-stroke-20261114'),('20261127','crow-v15-upward-trigger-seed27-20261114')]:
 data=json.load(gzip.open(Path('artifacts')/folder/'push-window.json.gz','rt'))['state']; seeds[seed]={str(x['lane']):{'q':x['q'],'v':x['v']} for x in data if x['lane'] in (6,7) and x['step']==1}
diff={str(e):{'max_abs_q':max(abs(a-b) for a,b in zip(seeds['20261126'][str(e)]['q'],seeds['20261127'][str(e)]['q'])),'max_abs_v':max(abs(a-b) for a,b in zip(seeds['20261126'][str(e)]['v'],seeds['20261127'][str(e)]['v']))} for e in (6,7)}
o={'source_sha256':hashlib.sha256(F.read_bytes()).hexdigest(),'duplicate_blocks':len(blocks),'byte_identical':True,'condition':'hierarchical navigation flag and curriculum 10','random_channels_reused':True,'first_observer_state_by_seed':seeds,'first_observed_state_differences':diff,'limitations':['State at observer step 1 is a recorded state, not an independently observed pre-reset snapshot.','Source shows a potential double perturbation; runtime program flag and reset-stage execution have not been independently instrumented.','Different seeds and first observed states alone do not establish a specified perturbed-start distribution or recovery test.']};(A/'audit.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'duplicate_blocks':len(blocks),'byte_identical':True,'differences':diff},indent=2))
