import contextlib,io,runpy,json,subprocess,os
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):d=runpy.run_path('artifacts/crow-v15-standing-restitution-control-20261112/analyze.py')
A=Path('artifacts/crow-v15-standing-target-feedback-screen-20261112');p=A/'candidate-20261127-rebound-screen';v=d['read'](p);c=d['v']['control'];st,sub,caps,errors=v
inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};proc=subprocess.run(['build/crow-v15-standing-bounce-audit-20261112/com'],input=inp,text=True,capture_output=True,env=env,check=True);kin={}
for l in proc.stdout.splitlines():
 s=l.split();kin[int(s[0]),int(s[1])]=list(map(float,s[2:]))
base=json.loads(Path('artifacts/crow-v15-standing-restitution-control-20261112/analysis.json').read_text())['arms']['control']; gains=[-1,-.5,-.25,0,.25,.5,1,2];rows=[]
for e in range(8):
 row={'lane':e,'gain':gains[e],'control_same_lane':base[e],'first_q_or_v_difference_step':next((t for t in range(1,152) if st[e,t]['q']!=c[0][e,t]['q'] or st[e,t]['v']!=c[0][e,t]['v']),None),'first_substep_difference':next(([t,k] for t in range(1,152) for k in range(16) if sub[e,t,k]!=c[1][e,t,k]),None),'endpoint_differences_vs_control':sum(st[x]!=c[0][x] for x in st if x[0]==e),'substep_differences_vs_control':sum(sub[x]!=c[1][x] for x in sub if x[0]==e),'unloaded_substeps_0_to_2s':sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(1,101) for k in range(16)),'unloaded_substeps_2_to_3s':sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(101,151) for k in range(16)),'max_abs_com_vertical_m_s':max(abs(kin[e,t][6]) for t in range(2,151)),'toe_speed_cap_substeps':caps[e]};rows.append(row)
out={'rows':rows,'errors':errors,'episode':json.loads((p/'run.stdout').read_text())['first_episode_navigation']};assert rows[3]['endpoint_differences_vs_control']==0 and rows[3]['substep_differences_vs_control']==0; (A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
