from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-full-sequence-no-body-moments-20261112');arms={}
for name in ('baseline','flexed'):
 p=Path('artifacts/crow-v15-unassisted-push-wing-screen-20261112/candidate-20261127-rebound-flexed') if name=='baseline' else A/'candidate-20261127-rebound-flexed';meta=json.loads((p/'run.stdout').read_text())['first_episode_navigation'];state={};sub={};caps=collections.Counter();errors=[]
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if t<=600:state[e,t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=600:
     assert (e,t,k) not in sub;sub[e,t,k]=z[4:]
     if int(z[4])>int(z[5]):errors.append(['overflow',e,t,k])
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4])
    if t<=600 and max(abs(float(v)) for v in z[4:])>=11.99:caps[e,t]+=1
   elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
 rows=[]
 for e in range(8):
  end=min(meta['steps_by_environment'][e],600);ss=[state[e,t] for t in range(1,end+1) if (e,t) in state]; assert len(ss)==end
  def quat_angle(q):
   x,y,z,w=q[3:7];return {'yaw':math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)),'pitch':math.asin(max(-1,min(1,2*(w*y-z*x)))),'roll':math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))}
  contacts=[]
  for x in ss:
   for c in x['contacts']:
    if c['impulses'][0]>1e-7 and not any(('foot' in c[k] or 'toe' in c[k]) for k in ('a_name','b_name')):contacts.append({'step':x['step'],'a':c['a_name'],'b':c['b_name'],'impulse':c['impulses'][0]})
  rows.append({'lane':e,'end_step':end,'unloaded_substeps_4_to_6s':sum(float(sub[e,t,k][4])+float(sub[e,t,k][5])<=1e-7 for t in range(201,301) for k in range(16) if (e,t,k) in sub),'toe_caps_4_to_6s':sum(caps[e,t] for t in range(201,301)), 'combined_upward_impulse_348_to_353_Ns':sum(float(sub[e,t,k][2])+float(sub[e,t,k][3]) for t in range(348,354) for k in range(16) if (e,t,k) in sub),'failed':meta['physical_failure_by_environment'][e], 'reason':meta['termination_reason_by_environment'][e], 'waypoints':meta['maximum_waypoints_by_environment'][e],'height_min_max_after_8s':[min((x['q'][2] for x in ss if x['step']>=400),default=None),max((x['q'][2] for x in ss if x['step']>=400),default=None)], 'final_position':ss[-1]['q'][:3],'final_velocity':ss[-1]['v'][:3],'final_attitude_rad':quat_angle(ss[-1]['q']),'toe_caps_after_7_4s':sum(caps[e,t] for t in range(371,end+1)), 'foot_loaded_after_8s':sum(float(sub[e,t,k][4])+float(sub[e,t,k][5])>1e-7 for t in range(401,end+1) for k in range(16) if (e,t,k) in sub),'first_nonfoot_contact':contacts[:1]})
 arms[name]={'meta':meta,'state':state,'sub':sub,'errors':errors,'rows':rows}
b=arms['baseline'];c=arms['flexed'];prefix={}
for e in range(8):
 common=sorted(set(k[1] for k in b['state'] if k[0]==e)&set(k[1] for k in c['state'] if k[0]==e));joint=[(t,k) for t in common for k in range(16) if (e,t,k) in b['sub'] and (e,t,k) in c['sub']]
 prefix[str(e)]={'first_endpoint_divergence':next((t for t in common if b['state'][e,t]!=c['state'][e,t]),None),'first_substep_divergence':next(([t,k] for t,k in joint if b['sub'][e,t,k]!=c['sub'][e,t,k]),None),'matched_endpoints_before_100':sum(b['state'][e,t]==c['state'][e,t] for t in common if t<=100),'matched_substeps_before_100':sum(b['sub'][e,t,k]==c['sub'][e,t,k] for t,k in joint if t<=100)}
out={'prefix':prefix,'arms':{name:{'meta':x['meta'],'errors':x['errors'],'rows':x['rows']} for name,x in arms.items()},'scope':'Compare body moments 13 and 14 suppressed from the start versus after 6.30 s, with otherwise identical early wing and leg commands. Other aerodynamic closures remain. Endpoint contacts do not provide a guaranteed fatal collider audit.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
