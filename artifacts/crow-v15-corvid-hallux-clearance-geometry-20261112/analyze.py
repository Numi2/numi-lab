from pathlib import Path
import gzip,json,math,statistics,os,subprocess
R=Path.cwd();p=R/'artifacts/crow-v15-corvid-hallux-clearance-geometry-20261112';src=R/'artifacts/crow-v15-corvid-hallux-amplitude-screen-20261112';events=json.loads((src/'target-timing.json').read_text())['own_or_opposite_candidate']['events'];q={};v={}
with gzip.open(src/'candidate-20261127-rebound/run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_Q_SUBSTEP '):
   z=l.split();t,k,e=map(int,z[1:4]);q[e,t,k]=list(map(float,z[4:]))
  elif l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);v[x['lane'],x['step']]=x['v']
ground=next(x['ground_z_m'] for x in json.loads((R/'artifacts/crow-v15-whole-gait-audit-20261111/audit.json').read_text())['rows'] if x['seed']==20261127)
inputs=[];meta={};offsets=[0,-.05,-.1,-.2]
for i,event in enumerate(events):
 e,side=event['lane'],event['side'];t,k=event['last_front'];
 for j,d in enumerate(offsets):
  index=i*4+j;state=q[e,t,k].copy();qi=29+side*8;state[qi]=max(-.3,state[qi]+d);meta[index]={'lane':e,'side':side,'t':t,'substep':k,'offset_requested_rad':d,'actual_q_rad':state[qi]};inputs.append(f'{index} 1 '+' '.join(map(str,state+v[e,t]))+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str((R/'build/crow-v15-hip-roll-journey-integration-20261111').resolve())};cmd=[str(R/'build/crow-v15-gait-slip-observer-20261112/kinematics')];run=subprocess.run(cmd,input=''.join(inputs),capture_output=True,text=True,check=True,env=env);b={}
for line in run.stdout.splitlines():
 z=line.split()
 if z[0]=='body' and 'rear_toe' in z[3]:b[int(z[1]),z[3]]=list(map(float,z[4:]))
rows=[]
for i,event in enumerate(events):
 side='left' if event['side']==0 else 'right'; vals=[]
 for j,d in enumerate(offsets):
  index=i*4+j;gaps=[]
  for segment in ['proximal','distal']:
   x=b[index,'crow_'+side+'_rear_toe_'+segment];qx,qy,qz,qw=x[3:7];rot=(2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy));gaps.append(x[2]-ground-sum(abs(z)*h for z,h in zip(rot,(.009,.0025,.0092))))
  vals.append({**meta[index],'proximal_bottom_gap_m':gaps[0],'distal_bottom_gap_m':gaps[1],'minimum_bottom_gap_m':min(gaps)})
 rows.append({'lane':event['lane'],'side':side,'last_front':event['last_front'],'poses':vals})
def st(a):return {'min':min(a),'median':statistics.median(a),'max':max(a)}
summs=[]
for j,d in enumerate(offsets):
 a=[x['poses'][j] for x in rows];delta=[x['poses'][j]['minimum_bottom_gap_m']-x['poses'][0]['minimum_bottom_gap_m'] for x in rows];summs.append({'requested_offset_rad':d,'proximal_bottom_gap_m':st([x['proximal_bottom_gap_m'] for x in a]),'distal_bottom_gap_m':st([x['distal_bottom_gap_m'] for x in a]),'min_bottom_gap_m':st([x['minimum_bottom_gap_m'] for x in a]),'min_gap_change_m':st(delta),'min_gap_over_1mm':sum(x['minimum_bottom_gap_m']>.001 for x in a),'at_lower_limit':sum(x['actual_q_rad']<=-.3+1e-9 for x in a)})
out={'ground_z_m':ground,'summary':summs,'events':rows,'scope':'Static native FK at retained last-front-contact joint state and counterfactual proximal angles, with endpoint velocities supplied only to satisfy input shape. No dynamic contact response, actuator feasibility or biological inference. Gap is collision-box geometric bottom minus ground, not exact solver separation. Toe box dimensions +/- (9, 2.5, 9.2) mm. q at substep and ground reference sampling may have rounding differences.'};(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summs,indent=2))
