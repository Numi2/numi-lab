from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-latched-leg-retraction-20261111');R=Path('artifacts/crow-v15-repaired-toe-takeoff-control-20261111')
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
base,_=read(R/'push.stderr.gz');results=[]
for height in ('166','170','174'):
 for kind in ('fast','base'):
  name='h'+height+'-'+kind;sup,sub=read(A/(name+'.stderr.gz'));diff=[]
  for lane in range(8):
   for t in range(1,271):
    if any(sup[lane,t][f]!=base[lane,t][f] for f in ('q','v')):diff.append((lane,t))
  assert all(t>248 for lane,t in diff),(name,diff[:10]);rows=[]
  for lane in range(8):
   z0=sup[lane,247]['q'][2];samples=[]
   for t in range(251,271):
    s=[sub[lane,t,k] for k in range(16)];z=[sum(q[i] for q in s) for i in range(7)];samples.append({'step':t,'foot_normal':z[2]+z[3],'other':z[6],'vz':sup[lane,t]['v'][2],'up':z[0]+z[1]})
   first=next((i for i,v in enumerate(samples) if v['foot_normal']<=1e-7),None);rising=0
   if first is not None:
    for x in samples[first:]:
     if x['foot_normal']<=1e-7 and x['vz']>.05:rising+=1
     else:break
   rise=max(sup[lane,t]['q'][2]-z0 for t in range(248,271));no_other=all(x['other']<=1e-7 for x in samples);passed=first is not None and rising>=2 and rise>=.02 and no_other
   rows.append({'lane':lane,'first_difference':next((t for l,t in diff if l==lane),None),'first_unloaded_step':samples[first]['step'] if first is not None else None,'first_unloaded_vz_m_s':samples[first]['vz'] if first is not None else None,'first_unloaded_upward_run':rising,'maximum_height_gain_m':rise,'nonfoot_contact_intervals':sum(x['other']>1e-7 for x in samples),'total_foot_upward_impulse_Ns':sum(x['up'] for x in samples),'lane_pass':passed})
  summary=json.loads((A/(name+'.stdout')).read_text());passed=all(x['lane_pass'] for x in rows) and summary['failed_environment_steps']==0 and summary['termination_count']==0;results.append({'variant':name,'failed_environment_steps':summary['failed_environment_steps'],'termination_count':summary['termination_count'],'pass':passed,'lanes':rows})
out={'matched_reference_qv_through_step':248,'results':results,'scope':'development test, first unloading must be upward; passive aerodynamics retained, no independent geometry or flight qualification'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
