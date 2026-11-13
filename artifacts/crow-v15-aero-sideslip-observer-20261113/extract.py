from pathlib import Path
import gzip,hashlib,json,math
A=Path('artifacts/crow-v15-aero-sideslip-observer-20261113');src=Path('/tmp/crow-bank-audit-20261113.gz');state={};wrench={};kin={}
with gzip.open(src,'rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0] not in ('CROW_AERO_INPUT','CROW_KIN_STATE','CROW_AERO_WRENCH'):continue
  t,k,e=map(int,z[1:4]);key=e,t,k
  if e not in (6,7) or not (650<=t<=1150) or (t-650)%4:continue
  if z[0]=='CROW_AERO_INPUT':
   assert key not in state;v=list(map(float,z[4:]));assert len(v)==81 and all(map(math.isfinite,v));state[key]=v
  elif z[0]=='CROW_KIN_STATE':
   assert z[4:6]==['41','40'];kin[key]=list(map(float,z[6:]))
  else:
   body=int(z[4]);assert body in (0,5,6,7) and key+(body,) not in wrench;v=list(map(float,z[5:]));assert len(v)==6;wrench[key+(body,)]=v
keys=sorted(state);assert len(keys)==2*126*16 and len(kin)==len(keys) and len(wrench)==4*len(keys)
with gzip.open(A/'samples.tsv.gz','wt') as out:
 for key in keys:
  assert max(abs(x-y) for x,y in zip(state[key],kin[key]))<1e-7
  vals=[*key,*state[key]]
  for b in (0,5,6,7):vals.extend(wrench[key+(b,)])
  out.write(' '.join(map(str,vals))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(A/'sample-index.json').write_text(json.dumps({'source_sha256':sha(src),'samples_sha256':sha(A/'samples.tsv.gz'),'count':len(keys),'lanes':[6,7],'ticks':[650,1150,4],'substeps':16,'fields':'lane tick substep q[41] v[40] four body wrenches [force xyz, torque xyz]'},indent=2)+'\n');print('extracted',len(keys))
