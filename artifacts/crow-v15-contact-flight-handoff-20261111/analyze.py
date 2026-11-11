from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-contact-flight-handoff-20261111');R=Path('artifacts/crow-v15-latched-leg-retraction-20261111')
def read(p,maxstep):
 sup={};sub={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if d['step']<=maxstep:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if t<=maxstep:sub[lane,t,k]=list(map(float,v[6:]))
 return sup,sub
base,bs=read(R/'h170-fast.stderr.gz',270);sup,sub=read(A/'handoff.stderr.gz',355);rows=[]
for lane in range(8):
 def ints(db,t):
  assert all((lane,t,k) in db for k in range(16)),(lane,t)
  return [sum(db[lane,t,k][i] for k in range(16)) for i in range(7)]
 bfirst=next((t for t in range(251,271) if sum(ints(bs,t)[2:4])<=1e-7),None);first=next((t for t in range(251,271) if sum(ints(sub,t)[2:4])<=1e-7),None);differences=[t for t in range(1,271) if any(base[lane,t][f]!=sup[lane,t][f] for f in ('q','v'))];firstdiff=min(differences) if differences else None;matches=first is not None and bfirst==first and all(t>first for t in differences)
 grounded=None;aircount=0
 if first is not None:
  for t in range(first,356):
   z=ints(sub,t)
   if z[2]+z[3]>1e-7 or z[6]>1e-7:grounded={'step':t,'foot_normal_impulse_Ns':z[2]+z[3],'nonfoot_impulse_Ns':z[6],'root_height_m':sup[lane,t]['q'][2],'root_velocity_m_s':sup[lane,t]['v'][:3]};break
   aircount+=1
 q=sup[lane,353]['q'];v=sup[lane,353]['v'];tilt=math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2))));rise=q[2]-sup[lane,247]['q'][2];passlane=matches and aircount>=100 and rise>=.15 and v[0]>.2 and tilt<=.35
 rows.append({'lane':lane,'baseline_first_unload_step':bfirst,'first_unload_step':first,'first_qv_divergence':firstdiff,'leg_jump_matches_through_unload':matches,'continuous_airborne_intervals':aircount,'first_ground_recontact':grounded,'step353_height_gain_m':rise,'step353_forward_speed_m_s':v[0],'step353_tilt_rad':tilt,'lane_pass':passlane})
data=json.loads((A/'handoff.stdout').read_text());out={'failed_environment_steps':data['failed_environment_steps'],'termination_count':data['termination_count'],'lanes':rows,'pass':all(r['lane_pass'] for r in rows) and data['failed_environment_steps']==0 and data['termination_count']==0,'scope':'eight development starts, full substep contact through 355, no qualification beyond short handoff'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
