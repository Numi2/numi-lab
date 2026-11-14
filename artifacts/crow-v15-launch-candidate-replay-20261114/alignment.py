from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-launch-candidate-replay-20261114');P=A/'push-20261126-rebound-lane7'
payload=json.load(gzip.open(P/'replay.json.gz','rt'))['payload'];frames=payload['frames'];states={}
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=x['lane'],x['step'];assert k not in states;states[k]=x
trace={}
with gzip.open(P/'trace.tsv.gz','rt') as f:
 for line in f:
  if line.startswith('#'):continue
  x=line.split();trace[int(x[0])]=list(map(float,x[1:42]))
maxq=0.;maxshift=0.;missing=[];wrong_same=0
for frame in frames:
 t=frame['step'];q=trace.get(t)
 if q is None:missing.append(t)
 else:maxq=max(maxq,max(abs(a-b) for a,b in zip(q,frame['q'])))
 if t<=499:
  src=states.get((7,t+1))
  if src is None:missing.append(('observer',t+1));continue
  maxshift=max(maxshift,max(abs(a-b) for a,b in zip(frame['q']+frame['v'],src['q']+src['v'])))
  same=states[7,t];wrong_same+=max(abs(a-b) for a,b in zip(frame['q']+frame['v'],same['q']+same['v']))>1e-6
duplicated=all(states[e,1]['q']==states[e,2]['q'] and states[e,1]['v']==states[e,2]['v'] for e in range(8))
o={'environment':payload['environment'],'frame_count':len(frames),'trace_q_max_abs_difference':maxq,'observer_shifted_qv_max_abs_difference_steps_1_to_499':maxshift,'observer_same_label_mismatch_count_steps_1_to_499':wrong_same,'first_two_observer_states_duplicated_all_lanes':duplicated,'missing':missing,'last_frame_q_matches_state_trace':max(abs(a-b) for a,b in zip(frames[-1]['q'],trace[500]))<1e-6,'last_frame_v_independently_observed':False,'pass_for_physical_pose_presentation':payload['environment']==7 and len(frames)==500 and not missing and maxq<1e-6 and maxshift<1e-6 and duplicated,'scope':'native replay q matches same-run state trace; shifted observer q/v checked through frame 499 only; timing convention requires separate audit'}
(A/'alignment.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
