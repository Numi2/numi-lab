from pathlib import Path
import gzip,json
p=Path('artifacts/crow-v15-ground-balance-offset-screen-20261112')
def load(d):
 targets={};states={}
 with gzip.open(d/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_DRIVE_TARGET '):
    v=l.split();targets[int(v[1]),int(v[2])]=[float(x) for x in v[3:]]
   elif l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);states[x['step'],x['lane']]=x
 return targets,states
a,sa=load(p/'candidate-smoke-20261127-target110');b,sb=load(p/'candidate-smoke-20261127-target-baseline110');_,refa=load(p/'candidate-20261127');_,refb=load(Path('artifacts/crow-v15-shank-proxy-trim-screen-20261112/candidate-20261127'))
rows=[]
for t in range(98,107):
 for e in range(8):
  d=[{'dof':i,'difference_rad':x-y} for i,(x,y) in enumerate(zip(a[t,e],b[t,e])) if x!=y];rows.append({'step':t,'lane':e,'target_changes':d})
def phys_same(x,y):return all(x[k]==y[k] for k in ('q','v','contacts'))
parity={'candidate_vs_retained':sum(not phys_same(sa[t,e],refa[t,e]) for t in range(1,111) for e in range(8)), 'baseline_vs_retained':sum(not phys_same(sb[t,e],refb[t,e]) for t in range(1,111) for e in range(8)), 'lane3_between_instrumented':sum(not phys_same(sa[t,3],sb[t,3]) for t in range(1,111))}
out={'rows':rows,'physical_parity_mismatches':parity,'scope':'Used-seed post-hoc instrumented target timing audit; targets are positions, not applied motor forces. No new stability qualification.'};(p/'target110-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print('physical parity mismatches',parity)
for t in range(98,107):
 print('step',t,'lane0',rows[(t-98)*8]['target_changes'],'lane7',rows[(t-98)*8+7]['target_changes'])
