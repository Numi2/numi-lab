from pathlib import Path
import collections,gzip,hashlib,json,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
paths={(s,'candidate'):root/'artifacts/crow-v15-front-toe-settling-screen-20261112'/f'candidate-{s}-rebound-phase'/'run.stderr.gz' for s in (20261126,20261127)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def persistent(samples):
 stable=list(samples[0]);pending=[0,0];states=[]
 for pair in samples:
  for k,x in enumerate(pair):
   if x==stable[k]:pending[k]=0
   else:
    pending[k]+=1
    if pending[k]>=3:stable[k]=x;pending[k]=0
  states.append(int(stable[0])+2*int(stable[1]))
 return states
def events(states):
 changes=[(0,states[0])]+[(i,x) for i,x in enumerate(states[1:],1) if x!=states[i-1]];found=[]
 for i in range(1,len(changes)):
  start,kind=changes[i]
  if kind not in (0,3):continue
  target=3-kind;seq=[]
  for j in range(i+1,len(changes)):
   state=changes[j][1]
   if state==kind:break
   seq.append(state)
   if state==target:
    typ='touchdown' if kind==0 else 'liftoff'
    if len(seq)==1:order='simultaneous'
    elif len(seq)==2 and seq[0] in (1,2):
     # During touchdown state 1 means hallux present first; during lift-off state 2 means hallux absent first.
     order='hallux_first' if (typ=='touchdown' and seq[0]==1) or (typ=='liftoff' and seq[0]==2) else 'digit3_first'
    else:order='complex'
    found.append({'type':typ,'order':order,'from_sample':start,'to_sample':changes[j][0],'gap_ms':(changes[j][0]-changes[i+1][0])*1.25,'states':seq});break
 return found
# A loaded first sample is left-censored; it must not manufacture a landing.
assert events(persistent([(True,True)]*10+[(False,True)]*10+[(False,False)]*10))==[]
synthetic=([(False,False)]*8+[(True,False)]*8+[(True,True)]*8+[(False,True)]*8+[(False,False)]*8+[(True,False)]*8+[(True,True)]*8)
assert [(x['type'],x['order']) for x in events(persistent(synthetic))]==[('liftoff','hallux_first'),('touchdown','hallux_first')]
results={}
for (seed,arm),p in paths.items():
 values={(e,side):{} for e in range(8) for side in (0,1)}
 with gzip.open(p,'rt') as f:
  for line in f:
   if not line.startswith('CROW_TWO_TOE_SUBSTEP '):continue
   z=line.split();t,k,e=map(int,z[1:4]);
   if not 201<=t<=300:continue
   toes=list(map(float,z[11:27]));assert len(toes)==16
   for side in (0,1):
    b=8*side;idx=(t-201)*16+k;assert idx not in values[e,side]
    values[e,side][idx]=(any(toes[b+i]>1e-7 for i in (6,7)),any(toes[b+i]>1e-7 for i in (2,3)))
 rows=[];counts={k:collections.Counter() for k in ('touchdown','liftoff')}
 for (e,side),v in sorted(values.items()):
  assert len(v)==1600 and set(v)==set(range(1600)),(seed,arm,e,side,len(v))
  ev=events(persistent([v[i] for i in range(1600)]))
  for x in ev:counts[x['type']][x['order']]+=1
  rows.append({'lane':e,'side':['left','right'][side],'events':ev})
 results[f'{seed}-{arm}']={'input':str(p.relative_to(root)),'sha256':sha(p),'groups':{k:dict(v) for k,v in counts.items()},'rows':rows}
(out/'toe-sequence.json').write_text(json.dumps(results,indent=2)+'\n')
for key,v in results.items():print(key,v['groups'])
