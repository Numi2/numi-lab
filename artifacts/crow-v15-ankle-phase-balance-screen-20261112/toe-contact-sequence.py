from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-ankle-phase-balance-screen-20261112');D=A/'candidate-20261127-rebound-hip';events=collections.defaultdict(set);seen=collections.Counter()
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_LOADED_POINT '):continue
  z=line.split();t,k,e,b=map(int,z[1:5]);v=list(map(float,z[5:]));assert len(v)==7
  if not 201<=t<=300 or v[-1]<=1e-7:continue
  if b in (11,15):side=0 if b==11 else 1;toe='sole'
  elif 19<=b<=34:
   side=0 if b<=26 else 1;idx=(b-(19 if side==0 else 27))//2;toe=('front1','front2','front3','hallux')[idx]
  else:continue
  i=(t-201)*16+k;events[e,side,i].add(toe);seen[e,side,toe]+=1
rows=[]
for e in range(8):
 for side in range(2):
  active=[i for i in range(1600) if events[e,side,i]];groups=[]
  for i in active:
   if not groups or i-groups[-1][-1]>3:groups.append([i])
   else:groups[-1].append(i)
  for g in groups:
   start,end=g[0],g[-1]
   if start==0 or end==1599 or end-start+1<32:continue
   times={toe:[i for i in g if toe in events[e,side,i]] for toe in ('sole','front1','front2','front3','hallux')}
   on={toe:min(a) if a else None for toe,a in times.items()};off={toe:max(a) if a else None for toe,a in times.items()}
   front_on=[on[k] for k in ('front1','front2','front3') if on[k] is not None];front_off=[off[k] for k in ('front1','front2','front3') if off[k] is not None]
   rows.append({'lane':e,'side':'left' if side==0 else 'right','start_index':start,'end_index':end,'onsets':on,'offsets':off,'rear_before_any_front':on['hallux']<min(front_on) if on['hallux'] is not None and front_on else None,'rear_before_or_same_substep_as_any_front':on['hallux']<=min(front_on) if on['hallux'] is not None and front_on else None,'front3_last_or_tied_among_toes':off['front3']>=max([off['hallux']]+front_off) if off['front3'] is not None and off['hallux'] is not None and front_off else None,'front3_strictly_last':off['front3']>max(off[k] for k in ('hallux','front1','front2') if off[k] is not None) if off['front3'] is not None and off['hallux'] is not None else None,'rear_minus_first_front_ms':(on['hallux']-min(front_on))*1.25 if on['hallux'] is not None and front_on else None})
r={'scope':'V15 ground contact detector, normal impulse >1e-7 N s; 1.25 ms sampling and loaded point grouping. Intervals separated by <=2 missing substeps are merged, partial and <40 ms intervals excluded. A 2000 black-billed magpie study described hallux touchdown first and third digit lift-off last, but species, speed, surface, toe geometry, contact threshold and marker methods differ. Normal contact first/last is not geometric sole contact or crow ground truth.','coverage':'Loaded point observations only from 4–6 s window, 8 closely related starts; no claim for unobserved windows.','rows':rows};(A/'toe-contact-sequence.json').write_text(json.dumps(r,indent=2)+'\n')
for k in ('rear_before_any_front','rear_before_or_same_substep_as_any_front','front3_last_or_tied_among_toes','front3_strictly_last'):
 valid=[x[k] for x in rows if x[k] is not None];print(k,'yes',sum(valid),'valid',len(valid),'missing',len(rows)-len(valid))
