from pathlib import Path
import gzip,json,sys
R=Path.cwd();A=R/'artifacts/crow-v15-mirrored-launch-phase-20261113';B=R/'artifacts/crow-v15-mirrored-launch-transition-20261113';rows=[]
def read(p):
 d={'s':{},'target':{}}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);key=(x['lane'],x['step']);
    if key[0]<4 and key[1]<=500:d['s'][key]=x
   elif l.startswith('CROW_DRIVE_TARGET '):
    z=l.split();t,e=map(int,z[1:3]);
    if e<4 and 335<=t<=500:d['target'][e,t]=list(map(float,z[3:]))
 return d
for arm in ('push','hold'):
 p=A/(arm+'-wing-20261126-rebound-phase');q=B/(arm+'-wing-20261126-rebound-transition');x=read(p);y=read(q);xi=json.loads((p/'inputs.json').read_text());yi=json.loads((q/'inputs.json').read_text());same=all(xi[k]==yi[k] for k in ('host_sha256','landing_policy_sha256','route_policy_sha256'))
 for e in range(4):
  bad=[t for t in range(1,501) if (e,t) not in x['s'] or (e,t) not in y['s'] or any(x['s'][e,t][k]!=y['s'][e,t][k] for k in ('q','v','policy_action','contacts'))];targets=[t for t in range(335,501) if x['target'].get((e,t))!=y['target'].get((e,t))];rows.append({'arm':arm,'lane':e,'state_differences':bad,'target_differences':targets,'host_and_packs_match':same})
a=json.loads((A/'analysis.json').read_text());valid=all(not r['state_differences'] and not r['target_differences'] and r['host_and_packs_match'] for r in rows);out={'reference_valid':valid,'reference_rows':rows,'pairs':{k:(v and valid) for k,v in a['pairs'].items()},'scope':'Paired same-lane first-episode comparison; no flight qualification.'};(A/'combined-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));sys.exit(0 if valid else 1)
