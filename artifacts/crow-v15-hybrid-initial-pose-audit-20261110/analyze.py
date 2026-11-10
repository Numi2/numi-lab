from pathlib import Path
import importlib.util,gzip,json
p='artifacts/crow-v15-hybrid-individual-toe-feedback-20261110/analyze.py';sp=importlib.util.spec_from_file_location('prior',p);x=importlib.util.module_from_spec(sp);sp.loader.exec_module(x)
D=Path('artifacts/crow-v15-hybrid-initial-pose-audit-20261110');pos={};ground=None;trace={}
with gzip.open(D/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0] in ('initialpose','pose'):pos.setdefault(int(v[3]),{})[v[4]]=[float(z) for z in v[5:]]
  elif v[0]=='groundpose':ground=[float(z) for z in v[3:]]
  elif v[0]=='trace':trace[int(v[3])]=v[4:]
assert len(trace)==12 and ground and set(pos)==set(range(-1,12))
parity=all(trace[t]==x.trace[(1,0,t)] for t in range(12))
gaps={str(t):{name:x.gap(pose,ground) for name,pose in values.items()} for t,values in sorted(pos.items())}
out={'baseline_state_parity':parity,'gap_m':gaps};(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for t in [-1,0,5,10,11]:
 row=gaps[str(t)]; print('step',t,'worst gap mm',round(1000*min(row.values()),4),'rear gap mm',[(name,round(1000*g,4)) for name,g in row.items() if 'rear_toe' in name])
print('baseline parity',parity)
