from pathlib import Path
import gzip,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-long-walk-failure-replay-20261113';P=R/'artifacts/crow-v15-walk-hold-slower-cycle-20261111/autonomous-20261126';D=A/'autonomous-20261126-lane1'
def measure(path):
 hs={x:hashlib.sha256() for x in ('state','contact','toe_velocity')};n={x:0 for x in hs};errors=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];kind='state'
   elif line.startswith(('CROW_TWO_TOE_SUBSTEP ','CROW_TOE_V_SUBSTEP ')):
    z=line.split(None,4);t,e=int(z[1]),int(z[3]);kind='contact' if z[0]=='CROW_TWO_TOE_SUBSTEP' else 'toe_velocity'
   elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip());continue
   else:continue
   if e==1 and 1<=t<=600:hs[kind].update(line.encode());n[kind]+=1
 return {'counts':n,'sha256':{x:hs[x].hexdigest() for x in hs},'errors':errors}
a=measure(P/'run.stderr.gz.part000');b=measure(D/'run.stderr.gz')
with gzip.open(D/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
rv={'environment':replay['environment'],'frames':len(replay['frames']),'first_step':replay['frames'][0]['step'],'last_step':replay['frames'][-1]['step']}
out={'original':a,'repeat':b,'replay':rv,'exact_records_match':a==b,'scope':'Lane 1 recorded endpoint state/contact and toe substep aggregate/velocity text, not every simulator bit or policy memory. Both runs use the same previously exposed seed and source; this is reproduction, not a held-out trial.'}
(A/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert a==b and a['counts']=={'state':600,'contact':9600,'toe_velocity':9600} and not a['errors'] and rv['environment']==1 and rv['frames']==600
