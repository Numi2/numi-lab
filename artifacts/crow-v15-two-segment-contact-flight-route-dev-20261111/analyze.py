from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-two-segment-contact-flight-route-dev-20261111');B=Path('artifacts/crow-v15-two-segment-contact-flight-handoff-20261111');T=1600
def read(path,withsub):
 state={};ground=collections.defaultdict(float);coverage=collections.Counter();errors=[]
 with gzip.open(path,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif withsub and line.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=line.split();t,k,e=map(int,a[1:4]);coverage[e,t,k]+=1;ground[e,t]+=float(a[8])+float(a[9])+float(a[10])
   elif withsub and ('CROW_SUBSTEP_ERROR' in line or 'failed' in line.lower()):errors.append(line.strip())
 return state,ground,coverage,errors
s,g,c,errors=read(A/'stderr.gz',True);ref,_,_,_=read(B/'stderr.gz',False)
try:data=json.loads((A/'stdout').read_text());failed=data['failed_environment_steps'];terms=data['termination_count'];waypoints=data['maximum_navigation_waypoints_by_environment'];teacher=data['birdflow_journey_teacher'];resets=data['host_requested_resets']
except Exception as ex:data={};failed=terms=teacher=resets=None;waypoints=[None]*8;errors.append(str(ex))
rows=[]
for e in range(8):
 complete=all((e,t) in s for t in range(1,T+1));pre=sum(s.get((e,t))!=ref.get((e,t)) for t in range(1,601));missing=sum(c[e,t,k]!=1 for t in range(1,T+1) for k in range(16));groundstep=next((t for t in range(356,T+1) if g[e,t]>1e-7),None);first_contact=None
 for t in range(356,T+1):
  if (e,t) not in s:continue
  hits=[z for z in s[e,t]['contacts'] if z.get('impulses',[0])[0]>1e-7 and 'locomotion_ground' not in (z.get('a_name'),z.get('b_name'))]
  if hits:first_contact={'step':t,'pairs':[[z.get('a_name'),z.get('b_name')] for z in hits],'root_position_m':s[e,t]['q'][:3],'root_velocity_m_s':s[e,t]['v'][:3]};break
 toe=[t for t in range(356,T+1) if (e,t) in s and any(abs(x)>=11.999 for x in s[e,t]['v'][22:38])];last=max((t for lane,t in s if lane==e),default=0)
 rows.append({'lane':e,'complete':complete,'prefix_different_samples':pre,'missing_substeps':missing,'max_waypoints_reported':waypoints[e],'first_ground_impulse_step':groundstep,'first_observed_nonground_contact':first_contact,'post_launch_endpoint_toe_cap_steps':toe,'last_observed_step':last,'last_root_position_m':s[e,last]['q'][:3] if last else None})
report={'lanes':rows,'failed_environment_steps':failed,'termination_count':terms,'teacher':teacher,'host_requested_resets':resets,'errors':errors[-8:],'scope':'development training layout only; non-ground contacts are accepted endpoint observations, not a complete substep first-contact audit; no supported-landing qualification'};(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
