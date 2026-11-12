from pathlib import Path
import collections,gzip,json,math,os,statistics,subprocess,sys
A=Path('artifacts/crow-v15-contact-sensor-discrepancy-audit-20261112');D=Path('artifacts/crow-v15-knee-contact-feedback-screen-20261112/candidate-20261127-rebound-phase');q={};v={};pts=collections.defaultdict(list);sensor={};ref={};wt={};status={}
with gzip.open(D/'run.stderr.gz','rt') as f:
 for line in f:
  z=line.split()
  if not z:continue
  if z[0] in ('CROW_Q_SUBSTEP','CROW_V_SUBSTEP'):
   t,k,e=map(int,z[1:4]);
   if 201<=t<300 and k==15:(q if z[0]=='CROW_Q_SUBSTEP' else v)[e,t]=list(map(float,z[4:]))
  elif z[0]=='CROW_LOADED_POINT':
   t,k,e,b=map(int,z[1:5]);
   if 201<=t<300 and k==15:pts[e,t].append((b,list(map(float,z[5:]))))
  elif z[0]=='CROW_PHASE_SAMPLE':
   t,e,flags=map(int,z[1:4]);
   if 201<=t<300:status[e,t]=flags
  elif z[0]=='CROW_SENSOR_VALUE':
   t,e=map(int,z[1:3]);
   if 201<=t<300:sensor[e,t]=int(float(z[3]))
assert len(q)==len(v)==len(sensor)==len(status)==8*99,(len(q),len(v),len(sensor),len(status))
keys=sorted(q);inp=''.join(f'{e} {t} '+' '.join(map(str,q[e,t]+v[e,t]))+'\n' for e,t in keys)
env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
p=subprocess.run(['build/crow-v15-gait-slip-observer-20261112/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
bodies={};names={11:'crow_left_foot',15:'crow_right_foot'}
with gzip.open(D/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
names={i:s for i,s in enumerate(replay['body_names'])}
for line in p.stdout.splitlines():
 z=line.split()
 if z and z[0]=='body':
  e,t=int(z[1]),int(z[2]);bodies[e,t,z[3]]=list(map(float,z[4:]))
for e,t in keys:
 rot=q[e,t][3:7];xx,yy,zz,ww=rot;h=[1-2*(yy*yy+zz*zz),2*(xx*yy+ww*zz)];norm=math.hypot(*h);h=[a/norm for a in h]
 sums=[0.,0.];weights=[0.,0.]
 for bi,a in pts[e,t]:
  side=0 if bi==11 or 19<=bi<=26 else 1 if bi==15 or 27<=bi<=34 else None
  if side is None:continue
  b=bodies[e,t,names[bi]];rx,ry,rz=[a[j]-b[j] for j in range(3)];wx,wy,wz=b[10:13];vx=b[7]+wy*rz-wz*ry;vy=b[8]+wz*rx-wx*rz;w=a[-1];sums[side]+=(vx*h[0]+vy*h[1])*w;weights[side]+=w
 ref[e,t]=[sums[i]/weights[i] if weights[i] else 0. for i in range(2)];wt[e,t]=weights
rows=[]
for e in range(2,8):
 for t in range(202,299):
  packed=sensor[e,t];decoded=[((packed//1024)-512)/511,((packed%1024)-512)/511]
  for side in range(2):
   value=decoded[side];target=ref[e,t][side];rows.append({"lane":e,"step":t,"side":"left" if side==0 else "right","sensor_m_s":value,"reference_m_s":target,"absolute_error_m_s":abs(value-target),"previous_error_m_s":abs(value-ref[e,t-1][side]),"next_error_m_s":abs(value-ref[e,t+1][side]),"loaded_impulse_ns":wt[e,t][side],"same_step_support_flag":bool(status[e,t]&(1<<(20 if side==0 else 21)))})
errors=sorted(x["absolute_error_m_s"] for x in rows);big=[x for x in rows if x["absolute_error_m_s"]>.02];top=sorted(rows,key=lambda x:-x["absolute_error_m_s"])[:30]
def subgroup(flag):
 vals=sorted(x['absolute_error_m_s'] for x in rows if x['same_step_support_flag']==flag)
 return {'samples':len(vals),'median_error_m_s':statistics.median(vals),'p95_error_m_s':vals[int(.95*(len(vals)-1))],'max_error_m_s':max(vals)}
out={"support_groups":{'flagged':subgroup(True),'unflagged':subgroup(False)},"seed":20261127,"lanes":list(range(2,8)),"interval_steps":[202,298],"samples":len(rows),"median_error_m_s":statistics.median(errors),"p95_error_m_s":errors[int(.95*(len(errors)-1))],"count_error_above_0_02_m_s":len(big),"big_error_impulse_below_1e_5_ns":sum(x["loaded_impulse_ns"]<1e-5 for x in big),"big_error_same_step_support_flags":sum(x["same_step_support_flag"] for x in big),"big_error_prev_step_closer":sum(x["previous_error_m_s"]<x["absolute_error_m_s"] for x in big),"big_error_next_step_closer":sum(x["next_error_m_s"]<x["absolute_error_m_s"] for x in big),"top_errors":top,"records":rows,"limits":"Retained trajectories; final-substep reference cutoff differs from shader; delayed and changing contact state, quantization and possible frame timing mismatch. No causation inferred."};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ("records","top_errors")},indent=2));[print(x) for x in top[:10]]
