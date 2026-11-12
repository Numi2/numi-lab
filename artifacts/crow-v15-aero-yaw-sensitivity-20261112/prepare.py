from pathlib import Path
import hashlib,json
A=Path('artifacts/crow-v15-aero-yaw-sensitivity-20261112');P=Path('artifacts/crow-v15-aerodynamic-load-replay-20261112/input.txt')
records=[];counts={'baseline':0,'sweep_pairs':0,'pronation_pairs':0};delta=.03
for line in P.read_text().splitlines():
 z=line.split();arm,lane,step=map(int,z[:3])
 if arm!=0:continue
 q=list(map(float,z[3:44]));v=list(map(float,z[44:]));assert len(q)==41 and len(v)==40
 def write(variant,qq):
  records.append(' '.join(map(str,[variant,lane,step]+qq+v)))
 write(0,q);counts['baseline']+=1
 for name,idx,lim,vid in [('sweep',(7,8),.2,1),('pronation',(11,12),.35,3)]:
  if not all(abs(q[i])+delta<=lim for i in idx):continue
  counts[name+'_pairs']+=1
  for sign in (1,-1):
   qq=q.copy();qq[idx[0]]+=sign*delta;qq[idx[1]]-=sign*delta;write(vid+(sign==-1),qq)
assert counts['baseline']==72 and counts['sweep_pairs'] and counts['pronation_pairs']
(A/'input.txt').write_text('\n'.join(records)+'\n')
(A/'input-metadata.json').write_text(json.dumps({'source_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'counts':counts,'delta_rad':delta,'mapping':{'baseline':0,'sweep_plus':1,'sweep_minus':2,'pronation_plus':3,'pronation_minus':4},'limits':'Only state pairs with both positive and negative counterfactual coordinate perturbations within authored position limits. Other q and all v unchanged.'},indent=2)+'\n')
print(counts)
