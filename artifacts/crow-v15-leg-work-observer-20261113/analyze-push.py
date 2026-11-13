import collections,gzip,hashlib,json,math
from pathlib import Path
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-leg-work-observer-20261113';reference=Path('/tmp/crow-leg-work-reference.stderr.gz');run=A/'push-nowings-20261127-rebound-work/run.stderr.gz'
def load(p):
 d={'support':{},'contact':{},'effort':{},'errors':[]}
 with gzip.open(p,'rt',errors='strict') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=(x['step'],x['lane']);v=json.dumps(x,sort_keys=True,separators=(',',':'));d['support'].setdefault(k,[]).append(v)
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=line.split();t,sub,lane=map(int,a[1:4]);
    if 335<=t<=380:d['contact'][(t,sub,lane)]=a[4:]
   elif line.startswith('CROW_LEG_EFFORT '):
    a=line.split();t,sub,lane=map(int,a[1:4]);mode=tuple(map(int,a[4:7]));v=list(map(float,a[7:]));assert len(v)==30;d['effort'][(t,sub,lane)]=(mode,[v[i:i+5] for i in range(0,30,5)])
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 return d
old=load(reference);new=load(run)
def cmp(a,b):
 shared=a.keys()&b.keys();diff=[k for k in sorted(shared) if a[k]!=b[k]];return {'reference_keys':len(a),'new_keys':len(b),'shared_keys':len(shared),'mismatches':len(diff),'first_mismatch':list(diff[0]) if diff else None,'missing_reference':len(a.keys()-b.keys()),'missing_new':len(b.keys()-a.keys())}
parity={'support':cmp(old['support'],new['support']),'contact_substeps':cmp(old['contact'],new['contact']),'old_errors':old['errors'],'new_errors':new['errors'],'effort_samples':len(new['effort'])}
(A/'parity.json').write_text(json.dumps(parity,indent=2)+'\n');print('Parity:',json.dumps(parity))
if parity['support']['mismatches'] or parity['support']['missing_new'] or parity['support']['missing_reference'] or parity['contact_substeps']['mismatches'] or parity['contact_substeps']['missing_new'] or parity['contact_substeps']['missing_reference'] or old['errors'] or new['errors'] or len(new['effort'])!=46*16*8:raise SystemExit('Observer parity failed; no effort interpretation')
modes=collections.Counter(str(m) for m,_ in new['effort'].values());assert set(modes)=={'(1, 0, 0)'},modes
rows=[];continuity=[]
for lane in range(8):
 for t in range(335,381):
  for sub in range(16):
   _,joints=new['effort'][(t,sub,lane)]
   if sub>0:
    prev=new['effort'][(t,sub-1,lane)][1]
    continuity.extend(abs(j[1]-q[2]) for j,q in zip(joints,prev))
 for lo,hi in ((335,347),(348,365),(366,380)):
  per=[{'positive':0.,'negative':0.,'net':0.,'max_abs_effort':0.} for _ in range(6)];imp=0.
  for t in range(lo,hi+1):
   for sub in range(16):
    _,joints=new['effort'][(t,sub,lane)]
    for j,(tau,q0,q1,v0,v1) in enumerate(joints):
     w=tau*(q1-q0);per[j]['positive']+=max(0.,w);per[j]['negative']+=min(0.,w);per[j]['net']+=w;per[j]['max_abs_effort']=max(per[j]['max_abs_effort'],abs(tau))
    c=new['contact'][(t,sub,lane)];imp+=float(c[2])+float(c[3])
  rows.append({'lane':lane,'steps':[lo,hi],'joint_indices':[13,14,15,16,17,18],'joint_work_proxy':per,'summed_positive_j':sum(x['positive'] for x in per),'summed_negative_j':sum(x['negative'] for x in per),'summed_net_j':sum(x['net'] for x in per),'upward_foot_impulse_ns':imp})
result={'parity':parity,'actuation_modes':modes,'max_within_step_q_continuity_error':max(continuity),'rows':rows,'scope':'Generalized working-effort times observed coordinate increment, not isolated muscle work. Mode 1 implicit drive, bounded/coupled disabled. Inputs are post-contact recorder samples; mapping assumes a 7-q/6-v free root. Contact and passive terms remain, 8 correlated lanes one used seed. No takeoff qualification.'}
(A/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print('Modes:',modes,'maximum within-step q continuity error:',max(continuity))
for r in rows:
 if r['lane']>=2 and r['steps']==[348,365]:print('lane',r['lane'],'positive',round(r['summed_positive_j'],6),'negative',round(r['summed_negative_j'],6),'net',round(r['summed_net_j'],6),'foot impulse',round(r['upward_foot_impulse_ns'],6))
