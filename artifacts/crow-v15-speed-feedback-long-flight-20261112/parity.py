from pathlib import Path
import gzip,json
R=Path('artifacts');A=R/'crow-v15-speed-feedback-long-flight-20261112';B=R/'crow-v15-smooth-launch-open-flight-20261112';p='candidate-20261127-rebound-flexed/run.stderr.gz'
def get(path):
 states={};contacts=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);
    if x['lane']==5 and x['step']<=800:states[x['step']]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if e==5 and t<=800:contacts.append(tuple(z[1:]))
 return states,contacts
s,c=get(B/p);t,d=get(A/p);diff=[]
for i in range(1,801):
 for k in ('q','v','policy_action','contacts'):
  if s[i][k]!=t[i][k]:diff.append([i,k])
if c!=d:diff.append(['substep_contacts'])
out={'lane':5,'through_steps':800,'state_samples':len(s),'contact_substeps':len(c),'differences':diff};(A/'parity.json').write_text(json.dumps(out,indent=2)+'\n');print('lane 5 parity',out['state_samples'],out['contact_substeps'],'differences',len(diff));assert not diff,diff[:10]
