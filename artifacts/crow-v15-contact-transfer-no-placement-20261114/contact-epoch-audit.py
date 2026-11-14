from pathlib import Path
import gzip,json,math
p=Path('artifacts/crow-v15-contact-transfer-no-placement-20261114');events={};sub={}
with gzip.open(p/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='event':events[int(x[1]),int(x[2])]=list(map(float,x[5:11]))
with gzip.open(p/'substeps.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='CROW_TWO_TOE_SUBSTEP' and int(x[2])==31:
   tick,e=int(x[1]),int(x[3]);v=list(map(float,x[6:]));sub[e,tick]=[v[0],v[1],sum(v[21:27]),sum(v[27:29]),sum(v[29:35]),sum(v[35:37])]
channel_names=['left_up_normal_vs_total','right_up_normal_vs_total','left_front_normal_x','left_rear_normal_x','right_front_normal_x','right_rear_normal_x']
rows=[]
for lag in (-2,-1,0,1,2):
 errs=[];channels=[[] for _ in range(6)];n=0
 for (e,t),v in events.items():
  y=sub.get((e,t+lag))
  if y is not None:
   n+=1
   for j,(a,b) in enumerate(zip(v,y)):
    d=abs(a-b);errs.append(d);channels[j].append(d)
 rows.append({'substep_tick_minus_event_tick':lag,'pairs':n,'max_abs_difference_Ns':max(errs,default=None),'mean_abs_difference_Ns':sum(errs)/len(errs) if errs else None,'fraction_values_within_1e-8':sum(x<=1e-8 for x in errs)/len(errs) if errs else None,'channels':[{'name':channel_names[j],'max_abs_difference_Ns':max(c,default=None),'fraction_within_1e-8':sum(x<=1e-8 for x in c)/len(c) if c else None} for j,c in enumerate(channels)]})
out={'rows':rows,'scope':'Endpoint up fields include normal impulse only; substep up fields include normal and tangential impulses, so those differ by definition. Four signed toe-normal-x groups are directly comparable. Exact contact-to-q epoch alignment remains unproven.'};(p/'contact-epoch-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
