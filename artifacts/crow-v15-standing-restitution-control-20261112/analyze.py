from pathlib import Path
import gzip,json,collections,statistics,subprocess,os,math
A=Path('artifacts/crow-v15-standing-restitution-control-20261112');paths={'control':A/'candidate-20261127-rebound-control','zero':A/'candidate-20261127-rebound-zero','prior':Path('artifacts/crow-v15-rear-toe-matched-trial-20261112/candidate-20261127-rebound-control')}
def read(path):
 st={};sub={};caps=collections.Counter();errors=[]
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if 'CROW_SUBSTEP_ERROR' in line:errors.append(line.strip())
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);t=x['step'];
    if 1<=t<=151:st[x['lane'],t]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=151:
     sub[e,t,k]=z[4:]
     if int(z[4])>int(z[5]):errors.append('overflow')
   elif line.startswith('CROW_TOE_V_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);
    if 1<=t<=150 and max(map(abs,map(float,z[4:])))>=11.99:caps[e]+=1
 assert len(st)==1208 and len(sub)==19328
 return st,sub,caps,errors
v={k:read(p) for k,p in paths.items()};c,z,r=v['control'],v['zero'],v['prior'];par={'control_endpoint_mismatches':sum(c[0][x]!=r[0][x] for x in c[0]),'control_substep_mismatches':sum(c[1][x]!=r[1][x] for x in c[1])};assert not any(par.values()),par
out={'parity':par,'zero_vs_control_endpoint_differences':sum(z[0][x]!=c[0][x] for x in c[0]),'arms':{}}
for label in ('control','zero'):
 st,sub,caps,err=v[label];ep=json.loads((paths[label]/'run.stdout').read_text())['first_episode_navigation'];assert ep['steps_by_environment']==[150]*8 and not any(ep['physical_failure_by_environment']) and not err
 inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())};p=subprocess.run(['build/crow-v15-standing-bounce-audit-20261112/com'],input=inp,text=True,capture_output=True,env=env,check=True);kin={}
 for l in p.stdout.splitlines():
  s=l.split();kin[int(s[0]),int(s[1])]=list(map(float,s[2:]))
 rows=[]
 for e in range(8):
  def zcount(lo,hi):return sum(float(sub[e,t,k][4])<=1e-7 and float(sub[e,t,k][5])<=1e-7 for t in range(lo,hi+1) for k in range(16))
  def other(lo,hi):return sum(float(sub[e,t,k][6]) for t in range(lo,hi+1) for k in range(16))
  rows.append({'lane':e,'unloaded_substeps_0_to_2s':zcount(1,100),'unloaded_substeps_2_to_3s':zcount(101,150),'other_ground_normal_Ns':other(1,150),'max_abs_com_vertical_m_s':max(abs(kin[e,t][6]) for t in range(2,151)),'com_height_span_m':max(kin[e,t][3] for t in range(2,151))-min(kin[e,t][3] for t in range(2,151)),'toe_speed_cap_substeps':caps[e]})
 out['arms'][label]=rows
out['scope']='Used development starts; diagnostic shader forces restitution contribution to zero in two new-impact branches, including nonfoot contacts. No proof all runtime paths are covered unless an independent positive control demonstrates sensitivity. COM reconstruction previously verified on the same source model and lane-5 replay. A zero foot normal is not proof of a solver defect or an unphysical takeoff. Other scene contacts and energy transfer must be audited separately.'
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
