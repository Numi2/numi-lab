from pathlib import Path
import gzip,json
A=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-strong-winghold-leg-isolation-20261113');out=[]
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-stronghold/run.stderr.gz');d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   z=line.split()
   if not z or z[0]!='CROW_WING_TARGET_PAIR':continue
   t,e=map(int,z[1:3]);assert (e,t) not in d;v=list(map(float,z[3:]));assert len(v)==4;d[e,t]=v
 for e in range(2,8):
  missing=[t for t in range(335,401) if (e,t) not in d];bad=[]
  if not missing:
   for t in range(336,401):
    now=d[e,t];prev=d[e,t-1]
    if now[1]!=prev[0] or now[3]!=prev[2]:bad.append(t)
  out.append({'arm':arm,'lane':e,'missing':missing,'previous_target_mismatches':bad})
valid=all(not x['missing'] and not x['previous_target_mismatches'] for x in out);(A/'target-pair-check.json').write_text(json.dumps({'valid':valid,'rows':out,'scope':'Observer compares stored previous wing target against preceding control-step target for the declared instrumented window; this does not prove tracking or launch.'},indent=2)+'\n');print('valid target history',valid,out)
if not valid:raise SystemExit('target-history test invalid')
