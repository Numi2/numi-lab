from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-two-segment-contact-flight-handoff-20261111');P=Path('artifacts/crow-v15-two-segment-leg-takeoff-comparison-20261111');T=600

def read(path,withsub):
 state={};sub=collections.defaultdict(lambda:[0.]*5);cover=collections.Counter();errors=[]
 with gzip.open(path,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif withsub and l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);cover[e,t,k]+=1
    for i in range(5):sub[e,t][i]+=float(a[6+i])
   elif withsub and 'failed' in l.lower():errors.append(l.strip())
 return state,sub,cover,errors
s,sub,cov,errors=read(A/'stderr.gz',True);ref,_,_,_=read(P/'push.stderr.gz',False)
try:
 data=json.loads((A/'stdout').read_text());failed=data['failed_environment_steps'];terms=data['termination_count']
except Exception as ex:failed=terms=None;errors.append(str(ex))
rows=[]
for e in range(8):
 complete=all((e,t) in s for t in range(1,T+1));pre=sum(s.get((e,t))!=ref.get((e,t)) for t in range(1,351));missing=sum(cov[e,t,k]!=1 for t in range(1,T+1) for k in range(16));eligible=None
 if complete:
  for t in range(350,T+1):
   x=s[e,t]
   if all(sub[e,u][2]+sub[e,u][3]<=1e-7 for u in range(t-2,t+1)) and x['q'][2]>.195 and x['v'][2]>.05:eligible=t;break
 def tilt(t):
  q=s[e,t]['q'];return math.acos(max(-1.,min(1.,1.-2*(q[3]**2+q[4]**2))))
 end=eligible+149 if eligible is not None else None;window=range(eligible,end+1) if eligible is not None and end<=T else []
 good=bool(window and all(s[e,t]['q'][2]>.2 and tilt(t)<=.5 and sub[e,t][2]+sub[e,t][3]<=1e-7 and abs(sub[e,t][4])<=1e-7 and not any(abs(v)>=11.999 for v in s[e,t]['v'][22:38]) for t in window))
 first_contact=next((t for t in range(eligible+1,T+1) if sub[e,t][2]+sub[e,t][3]>1e-7 or abs(sub[e,t][4])>1e-7),None) if eligible else None
 wing=next((t for t in range(eligible,T+1) if max(abs(s[e,t]['q'][9]-s[e,350]['q'][9]),abs(s[e,t]['q'][10]-s[e,350]['q'][10]))>.02),None) if eligible and complete else None
 toe_hits=[t for t in range(1,T+1) if any(abs(v)>=11.999 for v in s[e,t]['v'][22:38])] if complete else []
 rows.append({'lane':e,'complete':complete,'pre_handoff_different_samples':pre,'missing_substeps':missing,'first_conservative_eligibility_step':eligible,'first_observed_wing_motion_step':wing,'first_later_ground_impulse_step':first_contact,'minimum_height_150_m':min(s[e,t]['q'][2] for t in window) if window else None,'maximum_tilt_150_rad':max(tilt(t) for t in window) if window else None,'toe_cap_steps':toe_hits,'post_eligibility_toe_cap_steps':[t for t in toe_hits if eligible is not None and t>=eligible],'strict_150_interval_window':good,'pass':bool(complete and pre==0 and missing==0 and eligible is not None and wing is not None and good and failed==terms==0 and not errors)})
report={'lanes':rows,'failed_environment_steps':failed,'termination_count':terms,'errors':errors[-8:],'scope':'development contact-triggered handoff; eligibility inferred conservatively from full-substep unloaded intervals and observed state, not direct latch telemetry; no clearance or landing qualification'};(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
