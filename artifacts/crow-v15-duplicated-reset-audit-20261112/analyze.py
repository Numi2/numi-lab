from pathlib import Path
import contextlib,io,runpy,json,math,subprocess,os,collections,gzip
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path('artifacts/crow-v15-standing-restitution-control-20261112/analyze.py')
A=Path('artifacts/crow-v15-duplicated-reset-audit-20261112');refs={20261127:Path('artifacts/crow-v15-toe-preload-matched-trial-20261112/candidate-20261127-rebound-low'),20261126:Path('artifacts/crow-v15-toe-preload-second-seed-20261112/candidate-20261126-rebound-low')};out={}
for seed in (20261127,20261126):
 paths={'double':refs[seed],'single':A/('candidate-'+str(seed)+'-rebound-single')};arms={n:d['read'](p) for n,p in paths.items()};out[str(seed)]={}
 for label,(st,sub,caps,errors) in arms.items():
  assert not errors
  inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};proc=subprocess.run(['build/crow-v15-standing-bounce-audit-20261112/com'],input=inp,text=True,capture_output=True,check=True,env=env);kin={}
  for line in proc.stdout.splitlines():
   z=line.split();kin[int(z[0]),int(z[1])]=list(map(float,z[2:]))
  rows=[]
  for e in range(8):
   def unloaded(lo,hi):return sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(lo,hi+1) for k in range(16))
   q=st[e,1]['q'];x,y,z,w=q[3:7];roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));rows.append({'lane':e,'first_recorded_roll_rad':roll,'first_recorded_q_legs':q[14:20],'first_recorded_horizontal_velocity_m_s':st[e,1]['v'][:2],'unloaded_0_2':unloaded(1,100),'unloaded_2_3':unloaded(101,150),'toe_cap_substeps_0_3':caps[e],'peak_abs_com_vz_0_3_m_s':max(abs(kin[e,t][6]) for t in range(2,151))})
  episode=json.loads((paths[label]/'run.stdout').read_text())['first_episode_navigation'];assert not any(episode['physical_failure_by_environment']) and all(x>=150 for x in episode['steps_by_environment']);out[str(seed)][label]=rows
out['scope']='Different initialized states by design. First recorded endpoint is after the first physical interval and is duplicated in this observer, so it is not the pre-integration reset state. Foot load counts and velocity caps do not diagnose contact penetration, realism or stability. Both seeds are exposed development cases.'
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
