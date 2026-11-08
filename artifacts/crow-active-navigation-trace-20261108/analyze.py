from pathlib import Path
import gzip,json,math,hashlib
A=Path('artifacts/crow-active-navigation-trace-20261108')
report=json.loads((A/'rollout.json').read_text())
end=report['first_episode_navigation']['steps_by_environment'][0]
assert end==1411 and report['first_episode_navigation']['termination_reason_by_environment'][0]==3
samples=[]
with gzip.open(A/'state.tsv.gz','rt') as f:
 header=next(f).strip();assert 'nq=23' in header and 'scene_bodies=6' in header
 for line in f:
  v=list(map(float,line.split()));step=int(v[0])
  if step not in (1,200,400,800,1200,end-1):continue
  q=v[1:24];scene=v[24:];left=scene[13:16];right=scene[26:29]
  midpoint=[(a+b)/2 for a,b in zip(left,right)]
  samples.append({'step':step,'root_xyz':q[:3],'gate_midpoint_xyz':midpoint,'distance_to_gate_midpoint_m':math.dist(q[:3],midpoint)})
result={'seed':20261109,'environment':0,'course':'held-out-a','difficulty_band':10,'physics_substeps':4,
        'first_episode_contact_termination_step':end,'failed_simulation_steps':report['failed_environment_steps'],
        'trace_sha256':hashlib.sha256((A/'state.tsv.gz').read_bytes()).hexdigest(),'samples':samples,
        'interpretation':'Sampled root position moves away from the initial gate midpoint after takeoff. Contact type and causal reason for wrong steering are not identified by this trace.'}
(A/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
