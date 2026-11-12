from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-takeoff-substep-comparison-20261112');C=A/'autonomous-20261126-rebound';R=Path('artifacts/crow-v15-coordinated-takeoff-screen-20261112/autonomous-20261126')
rows={};full={}
for n,path in [(16,R),(32,C)]:
 sub={};state={};phase={};errors=[];overflow=[];episode=json.loads((path/"run.stdout").read_text()).get("first_episode_navigation");assert episode is not None;first_steps=episode["steps_by_environment"]
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if 'CROW_SUBSTEP_ERROR' in line:errors.append(line.strip())
   if line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);required,capacity=map(int,z[4:6]);
    if required>capacity:overflow.append([t,k,e,required,capacity])
    if 1<=t<=min(360,first_steps[e]):sub[e,t,k]=list(map(float,z[6:]))
   elif line.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(line.split(' ',1)[1]);t=d['step'];e=d['lane']
    if 1<=t<=min(360,first_steps[e]):state[e,t]=d
   elif line.startswith('CROW_PHASE_SAMPLE '):
    z=line.split();t,e,status=map(int,z[1:4]);
    if 1<=t<=min(360,first_steps[e]):phase[e,t]=status
 expected=8*360*n;bw=.45006*9.81;dt=.02/n;lane=[]
 for e in range(8):
  samples=[(t,k,sub[e,t,k]) for t in range(340,361) for k in range(n) if (e,t,k) in sub];raw=[[z[j]/dt/bw for _,_,z in samples] for j in (0,1)];width=round(.025/dt)
  def peak25(v):return max((sum(v[i:i+width])/width for i in range(len(v)-width+1)),default=None)
  loaded=[(t,k) for t,k,z in samples if z[2]+z[3]>1e-7];scores={'lane':e,'window_samples':len(samples),'last_loaded_step_substep':loaded[-1] if loaded else None,'upward_impulse_340_360_Ns':sum(z[0]+z[1] for _,_,z in samples),'per_foot_peak_vertical_BW':[max(v,default=None) for v in raw],'per_foot_peak_25ms_mean_vertical_BW':[peak25(v) for v in raw],'nonfoot_impulse_340_360':sum(z[4] for _,_,z in samples),'phase_sequence_nonzero_count':sum(bool(((w>>16)&15)|(((w>>28)&15)<<4)) for (i,t),w in phase.items() if i==e)};lane.append(scores)
 rows[str(n)]={'lanes':lane,'all_substep_rows':len(sub),'expected_all_substep_rows':expected,'endpoint_rows':len(state),'phase_rows':len(phase),'errors':errors,'overflows':overflow,'run_result':json.loads((path/'run-result.json').read_text()),'first_episode':episode};full[n]=state
parity={}
for e in range(8):
 for t in (1,50,100,200,300,340,352,360):
  a,b=full[16].get((e,t)),full[32].get((e,t))
  if a and b:parity[f'{e}:{t}']={'root_position_distance_m':math.dist(a['q'][:3],b['q'][:3]),'max_abs_q_delta':max(abs(x-y) for x,y in zip(a['q'],b['q']))}
rows['state_difference']=parity;rows['scope']='Different substep dynamics from initialization; not a matched-state intervention. 25ms box average is not source paper filter. Ground impulses include simulated toes and sole. Samples outside first episode cannot validate a flight or landing. Grip threshold is not admitted for 32 substeps.'
(A/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
