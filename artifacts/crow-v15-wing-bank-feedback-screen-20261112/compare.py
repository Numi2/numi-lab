from pathlib import Path
import gzip,json,statistics,math
A=Path('artifacts/crow-v15-wing-bank-feedback-screen-20261112');B=Path('artifacts/crow-v15-wing-steering-without-body-moments-20261112')
def read(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d[x['lane'],x['step']]=x
 return d
r=read(B/'baseline.stderr.gz');x=read(A/'bank.stderr.gz');pre=sorted(k for k in x if k[1]<450);mismatch=[k for k in pre if k not in r or any(r[k][z]!=x[k][z] for z in ('q','v','policy_action'))];assert len(pre)==48 and not mismatch,(len(pre),mismatch)
rows=[]
for e in range(6):
 ts=sorted(t for i,t in x if i==e and t>=500 and (e,t) in r)
 ds=[[x[e,t]['q'][9+j]-r[e,t]['q'][9+j] for j in range(2)] for t in ts]
 def yaw(q):a,b,c,w=q[3:7];return math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))
 vals=[x[e,t]['q'] for t in ts];last=x[e,ts[-1]]['q'];base=r[e,ts[-1]]['q'];rows.append({'lane':e,'target_sign':1 if e%2==0 else -1,'target_roll_rad':[.08,.16,.24][e//2],'matched_sample_steps':ts,'mean_left_flap_delta_rad':statistics.mean(d[0] for d in ds),'mean_right_flap_delta_rad':statistics.mean(d[1] for d in ds),'max_abs_flap_delta_rad':max(abs(v) for d in ds for v in d),'final_yaw_rad':yaw(last),'prior_final_yaw_rad':yaw(base),'sample_height_range_m':[min(q[2] for q in vals),max(q[2] for q in vals)]})
out={'prefix_samples':len(pre),'prefix_mismatches':mismatch,'rows':rows,'scope':'Matched baseline source until 9 s. 0/1 retain same targets after 9 s, remaining modes have different target commands. Sparse flap samples cannot measure stroke amplitude, aerodynamic forces, continuous extremes, or physical wing-generated turning. Previously used seed, two known direct body moments disabled only from 9 s onward.'};(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
