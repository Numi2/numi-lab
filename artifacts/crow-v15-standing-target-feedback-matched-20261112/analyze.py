import contextlib,io,runpy,json,subprocess,os,collections,gzip
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path('artifacts/crow-v15-standing-restitution-control-20261112/analyze.py')
A=Path('artifacts/crow-v15-standing-target-feedback-matched-20261112');s=d['read'](Path('artifacts/crow-v15-standing-target-feedback-screen-20261112/candidate-20261127-rebound-screen'));v={'control':d['v']['control']}
for name in ('minus','plus'):v[name]=d['read'](A/('candidate-20261127-rebound-'+name))
parity={}
for name,lane in [('minus',1),('plus',5)]:
 a=v[name];parity[name]={'endpoint_differences':sum(a[0][x]!=s[0][x] for x in a[0] if x[0]==lane),'substep_differences':sum(a[1][x]!=s[1][x] for x in a[1] if x[0]==lane)}
assert all(not any(q.values()) for q in parity.values()),parity
res={'screen_parity':parity,'arms':{}}
for name,(st,sub,caps,errors) in v.items():
 assert not errors
 inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};proc=subprocess.run(['build/crow-v15-standing-bounce-audit-20261112/com'],input=inp,text=True,capture_output=True,env=env,check=True);kin={}
 for l in proc.stdout.splitlines():
  z=l.split();kin[int(z[0]),int(z[1])]=list(map(float,z[2:]))
 rows=[]
 for e in range(8):
  def unloaded(lo,hi):return sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(lo,hi+1) for k in range(16))
  rows.append({'lane':e,'unloaded_0_2':unloaded(1,100),'unloaded_2_3':unloaded(101,150),'peak_abs_com_vz_0_2':max(abs(kin[e,t][6]) for t in range(2,101)),'peak_abs_com_vz_2_3':max(abs(kin[e,t][6]) for t in range(101,151)),'toe_cap_substeps_0_3':caps[e]})
 res['arms'][name]=rows
 if name!='control':
  episode=json.loads((A/('candidate-20261127-rebound-'+name)/'run.stdout').read_text())['first_episode_navigation'];res[name+'_episode']=episode;assert episode['steps_by_environment']==[150]*8 and not any(episode['physical_failure_by_environment'])
(A/'analysis.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
