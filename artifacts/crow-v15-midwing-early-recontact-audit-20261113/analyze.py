from pathlib import Path
import collections,gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-midwing-early-recontact-audit-20261113'
paths={'20261127':R/'crow-v15-midwing-transition-20261113','20261126':R/'crow-v15-midwing-second-seed-20261113'}
rows=[]
for seed,p in paths.items():
 analysis=json.loads((p/'analysis.json').read_text());records={}
 name='push-wing-'+seed+('-rebound-midwing' if seed=='20261127' else '-rebound-second')
 with gzip.open(p/name/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 348<=t<=380:records[e,t,k]=list(map(float,z[6:11]))
 assert len(records)==8*33*16
 for e in range(2,8):
  release=analysis['rows'][e]['push']['first_three_unloaded']
  support=lambda t:any(sum(records[e,t,k][2:4])>1e-7 for k in range(16))
  interruptions=[t for t in range(348,release-1) if not support(t) and support(t+1)] if release else []
  reimp=[{'resumption_step':t+1,'feet_upward_impulse_ns':sum(sum(records[e,t+1,k][:2]) for k in range(16))} for t in interruptions]
  rows.append({'seed':seed,'lane':e,'sustained_release_step':release,'early_unloaded_then_recontact':reimp})
out={'scope':'Posthoc control-interval analysis; a fully unloaded interval followed by support before the first three-unloaded-interval event. This is an engineering diagnostic, not a claim of biological slipping or isolated force.','rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['seed'],x['lane'],'release',x['sustained_release_step'],'early recontact',x['early_unloaded_then_recontact'])
