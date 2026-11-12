from pathlib import Path
import gzip,json,math,os,subprocess,sys
A=Path('artifacts/crow-v15-whole-gait-audit-20261111')
ROOT=Path('artifacts/crow-v15-coordinated-foot-lift-screen-20261111')
BASE={20261126:Path('artifacts/crow-v15-walk-hold-localized-route-20261111/autonomous-20261126'),20261127:Path('artifacts/crow-v15-walk-seed27-reference-20261111/autonomous-20261127')}
N=16; LO=201; HI=300; DT=.02; EPS=1e-7; SUPPORT=.5*.45006*9.81*DT

def require(ok,msg):
 if not ok: raise ValueError(msg)

def intervals(flags):
 out=[];start=None
 for i,x in enumerate(flags+[False]):
  if x and start is None:start=i
  if not x and start is not None:
   if i-start>=3:out.append([LO+start,LO+i-1])
   start=None
 return out

def qualified(gaps,normals,other):
 require(len(gaps)==len(normals)==len(other)==100,'incomplete gait window')
 out=[]
 for side in range(2):
  flags=[]
  for gap,n,o in zip(gaps,normals,other):
   require(len(gap)==2 and len(n)==2 and all(math.isfinite(x) for x in (*gap,*n,o)),'invalid input')
   flags.append(gap[side]>.001 and n[side]<=EPS and n[1-side]>=SUPPORT and o<=EPS)
  out.append(intervals(flags))
 return out

def selftest():
 gap=[[0.,0.] for _ in range(100)]; norm=[[.1,.1] for _ in range(100)];other=[0.]*100
 for i in range(4,9):gap[i][0]=.002;norm[i][0]=0
 assert qualified(gap,norm,other)[0]==[[205,209]]
 # A partial final interval is a candidate only, never a complete stride.
 truncated=[[0.,0.] for _ in range(100)];tn=[[.1,.1] for _ in range(100)]
 for i in range(97,100):truncated[i][0]=.002;tn[i][0]=0.
 assert qualified(truncated,tn,[0.]*100)[0]==[[298,300]]
 gap[6][0]=0;assert qualified(gap,norm,other)[0]==[]
 gap[6][0]=.002;norm[6][1]=0;assert qualified(gap,norm,other)[0]==[]
 norm[6][1]=.1;other[6]=1.;assert qualified(gap,norm,other)[0]==[]
 try:qualified(gap[:-1],norm,other);raise AssertionError('missing input passed')
 except ValueError:pass
 print('synthetic supported swing, dragging, insufficient support, nonfoot and missing input: passed')

def load(path):
 sup={};sub={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    z=json.loads(line.split(' ',1)[1]);t=z['step'];e=z['lane']
    if LO<=t<=HI:
     require((e,t) not in sup,'duplicate endpoint');sup[e,t]=z
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    x=line.split();t,k,e=map(int,x[1:4])
    if LO<=t<=HI:
     require((e,t,k) not in sub,'duplicate substep');v=list(map(float,x[6:]));require(len(v)==37,'contact schema');sub[e,t,k]=v
 require(len(sup)==800 and len(sub)==12800,'incomplete logged samples')
 inp=''.join(str(e)+' '+str(t)+' '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(sup.items()))
 env={**os.environ,'DYLD_LIBRARY_PATH':str(Path('build/crow-v15-hip-roll-journey-integration-20261111').resolve())}
 p=subprocess.run(['build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics'],input=inp,text=True,capture_output=True,check=True,env=env)
 bodies={};com={}
 for line in p.stdout.splitlines():
  z=line.split(); key=int(z[1]),int(z[2]);v=list(map(float,z[4:]))
  require(all(math.isfinite(a) for a in v),'nonfinite kinematics')
  if z[0]=='body':bodies[(*key,z[3])]=v
  elif z[0]=='com':com[key]=v
  else:raise ValueError('unknown FK row')
 require(len(com)==800,'missing COM');return sup,sub,bodies,com

def names(s):return ['crow_'+s+'_foot']+['crow_'+s+'_'+d+'_'+seg for d in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for seg in ('proximal','distal')]
def span(x):return max(x)-min(x)
def rotate_inv(q,v):
 x,y,z,w=q;u=(x,y,z);dot=sum(u[i]*v[i] for i in range(3));uv=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(a*a for a in u)
 return [s*v[i]+2*dot*u[i]-2*w*uv[i] for i in range(3)]

def analyze(path,ground,expected=None):
 sup,sub,body,com=load(path);rows=[]
 # Compare the logged endpoint state and FK to the independently recorded lane-5 replay.
 with gzip.open(path/'replay.json.gz','rt') as f: replay=json.load(f)['payload']
 require(replay['environment']==5 and abs(replay['timestep_seconds']-DT)<1e-9,'replay identity or timestep mismatch')
 frames={v['step']:v for v in replay['frames'] if 200<=v['step']<=299}
 require(len(frames)==100,'incomplete replay validation interval')
 qerr=max(abs(a-b) for t,f in frames.items() for a,b in zip(sup[5,t+1]['q'],f['q']))
 require(qerr<1e-6,'replay endpoint mismatch')
 poserr=0.;roterr=0.;replay_ground=[];bn=replay['body_names']
 for t,f in frames.items():
  replay_ground.append(f['body_states'][bn.index('locomotion_ground')*13+2])
  for name in names('left')+names('right')+['crow_head']:
   k=body[5,t+1,name];i=bn.index(name)*13;v=f['body_states'][i:i+13]
   poserr=max(poserr,math.dist(k[:3],v[:3]));roterr=max(roterr,min(math.dist(k[3:7],v[3:7]),math.dist(k[3:7],[-a for a in v[3:7]])))
 require(poserr<1e-5 and roterr<1e-5,'replay body kinematics mismatch')
 require(max(abs(g-ground) for g in replay_ground)<1e-6,'replay ground mismatch')
 print(str(path),'q_error',qerr,'position_error_m',poserr,'rotation_error',roterr)

 for e in range(8):
  gaps=[];norm=[];other=[];loaded=[0,0];doubles=0;neither=0;toe_drag=[0,0];head_world=[];head_torso=[];roll=[];pitch=[];height=[];joints=[[] for _ in range(8)];comz=[];footmotion=[0.,0.];previous=[None,None]
  for t in range(LO,HI+1):
   x=sup[e,t];q=x['q'];xq,yq,zq,wq=q[3:7];roll.append(math.atan2(2*(wq*xq+yq*zq),1-2*(xq*xq+yq*yq)));pitch.append(math.asin(max(-1,min(1,2*(wq*yq-zq*xq)))));height.append(q[2]);comz.append(com[e,t][2])
   for arr,i in zip(joints,(14,15,16,17,18,19,20,21)):arr.append(q[i])
   torso=body[e,t,'crow_body'] if (e,t,'crow_body') in body else None
   # The root q is the actual torso COM, with body-to-world quaternion xyzw.
   head=body[e,t,'crow_head'];head_world.append(head[:3]);head_torso.append(rotate_inv(q[3:7],[head[i]-q[i] for i in range(3)]))
   counts=[];groups=[]
   for k in range(N):
    z=sub[e,t,k];l=z[2]>EPS;r=z[3]>EPS;loaded[0]+=l;loaded[1]+=r;doubles+=l and r;neither+=not(l or r)
   summ=[sum(sub[e,t,k][2+i] for k in range(N)) for i in range(2)];norm.append(summ);other.append(sum(sub[e,t,k][4] for k in range(N)))
   for si,side in enumerate(('left','right')):
    clearance=[]
    for name in names(side):
     z=body[e,t,name];xq,yq,zq,wq=z[3:7];row=(2*(xq*zq-yq*wq),2*(yq*zq+xq*wq),1-2*(xq*xq+yq*yq));h=(.010,.01725,.0092) if name.endswith('_foot') else (.009,.0025,.0092);clearance.append(z[2]-.0067*row[2]-ground-.0065*abs(row[0])-.0025 if 'candidate-20261127' in str(path) and 'rear_toe' in name else z[2]-ground-sum(abs(a)*b for a,b in zip(row,h)))
    groups.append(min(clearance));toe_drag[si]+=clearance[0]>.001 and min(clearance[1:])<=.001
    foot=body[e,t,'crow_'+side+'_foot'][:3]
    if previous[si] is not None and summ[si]>EPS and previous[si][1]:footmotion[si]+=math.dist(foot,previous[si][0])
    previous[si]=(foot,summ[si]>EPS)
   gaps.append(groups)
  bouts=qualified(gaps,norm,other)
  if expected is not None:require(bouts==[expected[e]['left_bouts'],expected[e]['right_bouts']],f'prior comparison mismatch lane {e}')
  q0=sup[e,LO]['q'];q1=sup[e,HI]['q'];headspans=[span([p[i] for p in head_world]) for i in range(3)];relspans=[span([p[i] for p in head_torso]) for i in range(3)]
  rows.append({'lane':e,'window_seconds':(HI-LO+1)*DT,'forward_root_displacement_m':q1[0]-q0[0], 'root_height_span_m':span(height),'whole_body_com_height_span_m':span(comz),'root_roll_span_rad':span(roll),'root_pitch_span_rad':span(pitch),'joint_spans_rad':dict(zip(('left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle','neck_yaw','head_pitch'),map(span,joints))),'head_world_axis_spans_m':headspans,'head_relative_to_torso_axis_spans_m':relspans,'foot_loaded_fraction_window':[x/(100*N) for x in loaded],'both_loaded_fraction_window':doubles/(100*N),'neither_loaded_fraction_window':neither/(100*N),'sole_clear_but_toes_below_clearance_intervals':toe_drag,'foot_COM_motion_while_loaded_proxy_m':footmotion,'support_qualified_swing_bouts':bouts,'stride_metrics_status':'not calculated: touchdown/stride correspondence not yet validated','stride_cadence_hz':None,'stride_duty_factor':None,'stride_length_m':None,'stride_phase_offset':None})
 return rows

def main():
 selftest();out=[]
 for seed in (20261126,20261127):
  s=ROOT/f'autonomous-{seed}';legacy=json.loads((ROOT/f'analysis-{seed}.json').read_text());ground=legacy['replay_validation']['replay_ground_z_m'];expected=legacy['rows'];require(legacy['replay_validation']['foot_toe_position_max_error_m']<1e-5,'prior replay validation failed')
  out.append({'seed':seed,'ground_z_m':ground,'candidate':analyze(s,ground,expected),'baseline':analyze(BASE[seed],ground)})
 result={'scope':'Retained development data; 201-300 window; native FK; no footage or biological calibration. Whole-window contact fractions are not duty factors; loaded foot COM motion is not measured contact-point slip. No continuous clearance claim. Camera/world axes not stride-normalized.','rows':out}
 (A/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in out:
  for group in ('baseline','candidate'):
   z=r[group];print(r['seed'],group,'lanes',len(z),'bilateral swings',sum(bool(x['support_qualified_swing_bouts'][0]) and bool(x['support_qualified_swing_bouts'][1]) for x in z),'forward_m range',min(x['forward_root_displacement_m'] for x in z),max(x['forward_root_displacement_m'] for x in z),'drag endpoint range',min(sum(x['sole_clear_but_toes_below_clearance_intervals']) for x in z),max(sum(x['sole_clear_but_toes_below_clearance_intervals']) for x in z))
if __name__=='__main__':main()
