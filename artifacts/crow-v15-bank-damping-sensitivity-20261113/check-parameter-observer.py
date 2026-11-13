from pathlib import Path
import collections,gzip,json,sys
A=Path('artifacts/crow-v15-bank-damping-sensitivity-20261113');ref=A/sys.argv[1];obs=A/sys.argv[2]
def read(path,observed):
 states={};params={}
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);k=(x['lane'],x['step']);assert k not in states;states[k]=x
   if l.startswith('CROW_TORSO_PARAMETERS '):
    z=l.split();t,sub,e=map(int,z[1:4]);assert sub==0 and (e,t) not in params;params[e,t]=list(map(float,z[4:]))
 return states,params
old,_=read(ref/'run.stderr.gz',False);new,par=read(obs/'run.stderr.gz',True);m0=json.loads((ref/'run.stdout').read_text())['first_episode_navigation'];m1=json.loads((obs/'run.stdout').read_text())['first_episode_navigation'];rows=[]
for e in range(8):
 stop=m0['steps_by_environment'][e];bad=[];missing=[]
 for t in range(1,stop+1):
  k=e,t
  if k not in old or k not in new:missing.append(t)
  elif any(old[k][f]!=new[k][f] for f in ('q','v','policy_action','contacts')):bad.append(t)
 rows.append({'lane':e,'stop':stop,'reference_end_matches':stop==m1['steps_by_environment'][e],'missing':missing,'different':bad})
# Readback is sampled only over ticks 650–1150. Ignore post-first-episode samples.
reads=[]
for e in (6,7):
 end=min(1150,m1['steps_by_environment'][e]);samples=[(t,par[e,t]) for t in range(650,end+1,10) if (e,t) in par];reads.append({'lane':e,'sample_count':len(samples),'expected_count':len(range(650,end+1,10)),'scale_values':sorted(set(v[3] for t,v in samples)),'all_half':bool(samples) and len(samples)==len(range(650,end+1,10)) and all(abs(v[3]-.5)<1e-6 for t,v in samples)})
out={'parity':rows,'parity_passed':all(r['reference_end_matches'] and not r['missing'] and not r['different'] for r in rows),'torso_parameter_readback':reads,'scope':'Observer-host repeat of original damping candidate; sampled GPU torso material parameters and exact first-episode parity. Readback alone does not biologically calibrate damping.'};(obs/'parameter-observer-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
