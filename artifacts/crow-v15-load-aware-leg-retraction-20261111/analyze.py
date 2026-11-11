from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-load-aware-leg-retraction-20261111');R=Path('artifacts/crow-v15-repaired-toe-takeoff-control-20261111')
def read(p):
 sup={};sub={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=270:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=270:sub[lane,t,k]=list(map(float,v[6:]))
 assert len(sup)==8*270 and len(sub)==8*270*16,(len(sup),len(sub))
 return sup,sub
base,_=read(R/'push.stderr.gz')
def longest(x):
 best=cur=0
 for a in x:cur=cur+1 if a else 0;best=max(best,cur)
 return best
results=[]
for height in ('166','170','174'):
 for kind in ('crouch','moderate'):
  name='h'+height+'-'+kind;sup,sub=read(A/(name+'.stderr.gz'));diff=[]
  for lane in range(8):
   for t in range(1,271):
    if any(sup[lane,t][f]!=base[lane,t][f] for f in ('q','v')):diff.append((lane,t))
  assert all(t>250 for lane,t in diff),(name,diff[:10]);rows=[]
  for lane in range(8):
   z0=sup[lane,247]['q'][2];flags=[];impulse=[];up=[];other=[]
   for t in range(248,271):
    s=[sub[lane,t,k] for k in range(16)];z=[sum(q[i] for q in s) for i in range(7)];impulse.append(z[2]+z[3]);up.append(z[0]+z[1]);other.append(z[6]);flags.append(z[2]+z[3]<=1e-7 and sup[lane,t]['v'][2]>.05)
   rise=max(sup[lane,t]['q'][2]-z0 for t in range(248,271));n=longest(flags)
   rows.append({'lane':lane,'first_difference':next((t for l,t in diff if l==lane),None),'unloaded_upward_samples':sum(flags),'longest_unloaded_upward_run':n,'first_unloaded_upward_step':next((248+i for i,v in enumerate(flags) if v),None),'maximum_height_gain_m':rise,'max_upward_speed_m_s':max(sup[lane,t]['v'][2] for t in range(248,271)),'total_foot_normal_impulse_Ns':sum(impulse),'total_foot_upward_impulse_Ns':sum(up),'nonfoot_contact_intervals':sum(x>1e-7 for x in other),'lane_pass':n>=2 and rise>=.02 and all(x<=1e-7 for x in other)})
  summary=json.loads((A/(name+'.stdout')).read_text());passed=all(r['lane_pass'] for r in rows) and summary['failed_environment_steps']==0 and summary['termination_count']==0;results.append({'variant':name,'failed_environment_steps':summary['failed_environment_steps'],'termination_count':summary['termination_count'],'pass':passed,'lanes':rows})
out={'matched_reference_qv_through_step':250,'results':results,'scope':'development test, no active flapping during window but passive aerodynamics remain; no holdout or route qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
