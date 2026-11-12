from pathlib import Path
import gzip,json,math,statistics,subprocess,os
A=Path('artifacts/crow-v15-standing-bounce-audit-20261112');path=Path('artifacts/crow-v15-rear-toe-matched-trial-20261112/candidate-20261127-rebound-control');st={};sub={}
with gzip.open(path/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if 1<=x['step']<=151:st[x['lane'],x['step']]=x
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if 1<=t<=151:sub[e,t,k]=list(map(float,z[6:]))
assert len(st)==1208 and len(sub)==19328
inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};r=subprocess.run(['build/crow-v15-standing-bounce-audit-20261112/com'],input=inp,text=True,capture_output=True,check=True,env=env);kin={}
for line in r.stdout.splitlines():
 z=line.split();e,t=map(int,z[:2]);v=list(map(float,z[2:]));assert len(v)==7;kin[e,t]=v
assert len(kin)==1208
rows=[];residuals={str(i):[] for i in (-1,0,1)};residual_no_duplicate=[]
for e in range(8):
 zero=[];other=[];vz=[];records=[]
 for t in range(2,151):
  j=sum(sub[e,t,k][0]+sub[e,t,k][1] for k in range(16));no=sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for k in range(16));oth=sum(sub[e,t,k][4] for k in range(16));zero.append(no);other.append(oth);vz.append(kin[e,t][6]);dv=kin[e,t][6]-kin[e,t-1][6];rec={'step':t,'com_z_m':kin[e,t][3],'com_vz_m_s':kin[e,t][6],'delta_com_vz_m_s':dv,'foot_up_impulse_Ns':j,'unloaded_substeps':no,'other_normal_impulse_Ns':oth};records.append(rec)
  if t>2: residual_no_duplicate.append(dv-(j/kin[e,t][0]-9.81*.02))
  for shift in (-1,0,1):
   tt=t+shift;jj=sum(sub[e,tt,k][0]+sub[e,tt,k][1] for k in range(16));pred=jj/kin[e,t][0]-9.81*.02;residuals[str(shift)].append(dv-pred)
 rows.append({'lane':e,'zero_foot_substeps':sum(zero),'other_normal_sum_Ns':sum(other),'max_abs_com_vz_m_s':max(map(abs,vz)),'min_com_z_m':min(kin[e,t][3] for t in range(2,151)),'max_com_z_m':max(kin[e,t][3] for t in range(2,151)),'steps':records})
out={'rows':rows,'model_mass_kg':kin[0,2][0],'same_interval_residual_excluding_duplicate_first_endpoint':{'n':len(residual_no_duplicate),'rms_m_s':math.sqrt(statistics.mean(x*x for x in residual_no_duplicate)),'max_abs_m_s':max(map(abs,residual_no_duplicate))},'shifted_impulse_residuals':{s:{'n':len(v),'mean_m_s':statistics.mean(v),'rms_m_s':math.sqrt(statistics.mean(x*x for x in v)),'max_abs_m_s':max(map(abs,v))} for s,v in residuals.items()},'scope':'Retained used-seed zero-preload crouch run, 0–3s. Native FK mass-weighted COM velocity and summed accepted foot upward impulses. Residual compares successive endpoint COM velocities with foot impulse/mass and gravity; other ground, scene, aerodynamic, motor external wrenches and observer ordering are not accounted for. Three timestep shifts explore alignment but choosing the lowest error post hoc does not validate timing or prove an energy source. Zero normal foot load is not automatically a failure or a solver defect. COM reconstruction is separately validated against the lane-5 recorded replay; this does not establish energy origin or external force timing.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['lane'],'unloaded',x['zero_foot_substeps'],'max_vz',round(x['max_abs_com_vz_m_s'],4),'height_span',round(x['max_com_z_m']-x['min_com_z_m'],4),'otherN',round(x['other_normal_sum_Ns'],5))
print(out['shifted_impulse_residuals'])
