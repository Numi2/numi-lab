from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-candidate-replay-fk-20261114');P=Path('artifacts/crow-v15-flight-release-latch-20261114/turn-20261127-rebound-latch27/replay.json.gz');x=json.load(gzip.open(P,'rt'))['payload'];frames={f['step']:f for f in x['frames']};rows=[]
for line in (A/'output.tsv').read_text().splitlines():
 a=line.split();t,b=map(int,a[:2]);z=list(map(float,a[2:]));n=x['body_state_stride'];record=frames[t]['body_states'][n*b:n*(b+1)];pos=math.sqrt(sum((z[i]-record[i])**2 for i in range(3))); d1=math.sqrt(sum((z[3+i]-record[3+i])**2 for i in range(4)));d2=math.sqrt(sum((z[3+i]+record[3+i])**2 for i in range(4)));rows.append({'frame':t,'body':b,'name':x['body_names'][b],'position_error_m':pos,'quaternion_error_sign_invariant':min(d1,d2)})
assert len(rows)==9*35
r={'scope':'CPU forward kinematics using separately constructed experimental hip-roll model at nine candidate replay frames; a mismatch can indicate model or epoch differences','samples':len(rows),'max_position_error':max(rows,key=lambda y:y['position_error_m']),'max_quaternion_error':max(rows,key=lambda y:y['quaternion_error_sign_invariant']),'rows':rows};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
