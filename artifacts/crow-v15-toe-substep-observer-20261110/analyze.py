from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-toe-substep-observer-20261110');name=sys.argv[1];release=int(sys.argv[2]);start=int(sys.argv[3]);end=int(sys.argv[4]);status={};contact={}
with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
 for line in f:
  if line.startswith('CROW_SUBSTEP_STATUS '):
   _,step,sub,lane,count,phys=line.split();key=(int(step),int(sub),int(lane));status[key]={'count':int(count),'physics_substep':int(phys)}
  elif line.startswith('CROW_SUBSTEP_TOE '):
   _,step,sub,lane,a,b,imp,sep,nz,nx,ny=line.split();key=(int(step),int(sub),int(lane));a=int(a);b=int(b);toe=a if 17<=a<=24 else b;normal=float(nx) if toe==a else -float(nx);side='left' if toe<=20 else 'right';digit='rear' if toe in (20,24) else 'front';d=contact.setdefault(key,{'left':{'front':0.,'rear':0.},'right':{'front':0.,'rear':0.},'any':0.});p=abs(float(imp));d['any']+=p
   if (digit=='front' and normal<-.5) or (digit=='rear' and normal>.5):d[side][digit]+=p
expected={(t,s,0) for t in range(start,end+1) for s in range(4)}
missing=expected-set(status)
def summary(lo,hi):
 keys=sorted(k for k in expected if lo<=k[0]<=hi and k in status);op=[];anys=[]
 for k in keys:
  c=contact.get(k,{'left':{'front':0.,'rear':0.},'right':{'front':0.,'rear':0.},'any':0.});op.append(all(c[side][digit]>.0000125 for side in ('left','right') for digit in ('front','rear')));anys.append(c['any']>.0000125)
 runs=[];n=0
 for x in op+[False]:
  if x:n+=1
  elif n:runs.append(n);n=0
 return {'recorded_substeps':len(keys),'opposed_bilateral_substeps':sum(op),'longest_opposed_run_substeps':max(runs,default=0),'any_toe_contact_substeps':sum(anys)}
r={'trace_start_submission':start,'trace_end_submission':end,'release_step_endpoint':release,'expected_substeps':len(expected),'observed_substeps':len(expected & set(status)),'missing_count':len(missing),'before_release_100_control_steps':summary(max(start,release-100),min(end,release-1)),'after_release_early':summary(max(start,release+20),min(end,release+119)),'note':'Control step and submission association must be checked by replay parity. This records contact constraints/normal impulses only; zero impulse is not automatically positive geometric separation. Threshold is 0.0000125 Ns per substep, one quarter of the prior endpoint threshold; not biologically calibrated.'}
(A/(name+'-substep-analysis.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
