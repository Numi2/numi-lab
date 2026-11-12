from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-walk-takeoff-integration-20261112')
references={20261127:Path('artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low'),20261126:Path('artifacts/crow-v15-toe-preload-second-seed-20261112/candidate-20261126-rebound-low')}
def load(p):
 state={};sub={};caps=collections.Counter();errors=[]; toe_count=collections.Counter()
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);assert (e,t,k) not in sub;sub[e,t,k]=[float(v) for v in z[4:]]
    if int(z[4])>int(z[5]):errors.append('overflow')
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);toe_count[e,t,k]+=1
    if max(abs(float(v)) for v in z[4:])>=11.99:caps[e,t]+=1
   elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
 return state,sub,caps,errors,toe_count
results={}
for seed,ref in references.items():
 p=A/f'candidate-{seed}-rebound-integrated';s,sub,caps,err,tc=load(p);rs,rsub,_,re,_=load(ref)
 s={k:v for k,v in s.items() if k[1]<=400};sub={k:v for k,v in sub.items() if k[1]<=400};tc={k:v for k,v in tc.items() if k[1]<=400};rs={k:v for k,v in rs.items() if k[1]<=300};rsub={k:v for k,v in rsub.items() if k[1]<=300}
 assert not err and not re and len(s)==3200 and len(sub)==51200 and len(tc)==51200 and all(c==1 for c in tc.values())
 prefix={'endpoints_compared':len(rs),'endpoint_mismatches':sum(rs[k]!=s[k] for k in rs),'substeps_compared':len(rsub),'substep_mismatches':sum(rsub[k]!=sub[k] for k in rsub)}
 rows=[]
 for e in range(8):
  def window(lo,hi):return [sub[e,t,k] for t in range(lo,hi+1) for k in range(16)]
  launch=window(341,370);up=[sum(v[2+i] for v in launch) for i in (0,1)];normal=[sum(v[4+i] for v in launch) for i in (0,1)]
  supported=[(t,k) for t in range(341,401) for k in range(16) if sum(sub[e,t,k][4:6])>1e-7]
  post=[(t,k) for t in range(371,401) for k in range(16) if sum(sub[e,t,k][4:6])>1e-7]
  rows.append({'lane':e,'upward_foot_impulse_348_to_353_Ns':[sum(v[2+i] for v in window(348,353)) for i in (0,1)],'peak_root_z_348_to_370_m':max(s[e,t]['q'][2] for t in range(348,371)),'toe_cap_6_to_8s':sum(caps[e,t] for t in range(301,401)),'upward_foot_impulse_6_8_to_7_4_Ns':up,'normal_foot_impulse_6_8_to_7_4_Ns':normal,'last_loaded_6_8_to_8s':supported[-1] if supported else None,'loaded_substeps_7_4_to_8s':len(post),'root_z_6_8_7_4_8s':[s[e,t]['q'][2] for t in (340,370,400)],'root_vz_6_8_7_4_8s':[s[e,t]['v'][2] for t in (340,370,400)],'nonfoot_ground_impulse_6_to_8_Ns':sum(x[6] for x in window(301,400))})
 out=json.loads((p/'run.stdout').read_text())['first_episode_navigation']
 results[str(seed)]={'prefix':prefix,'steps':out['steps_by_environment'],'physical_failure':out['physical_failure_by_environment'],'rows':rows}
(A/'analysis.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
