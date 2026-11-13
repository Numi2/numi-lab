from pathlib import Path
import gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-early-low-wing-screen-20261113'
paths=[R/'crow-v15-low-wing-effort-screen-20261113/push-wing-20261126-rebound-low/run.stderr.gz',A/'push-wing-20261126-rebound-early/run.stderr.gz']
def load(p):
 d={};c={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]); e=x['lane'];t=x['step']
    if 1<=t<=345:
     assert (e,t) not in d;d[e,t]=[x[k] for k in ('q','v','policy_action','contacts')]
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if 335<=t<=345:
     assert (e,t,k) not in c;c[e,t,k]=z[4:]
 assert len(d)==8*345 and len(c)==8*11*16
 return d,c
x,y=map(load,paths);out={'state_mismatches':sum(x[0][k]!=y[0][k] for k in x[0]),'contact_mismatches':sum(x[1][k]!=y[1][k] for k in x[1]),'scope':'Exact logged push-arm state/action/contact parity over first 345 intervals and contact substeps 335–345. Does not establish equality of hidden state.'}
(A/'prior-prefix-parity.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if out['state_mismatches'] or out['contact_mismatches']:raise SystemExit('Prior prefix mismatch')
