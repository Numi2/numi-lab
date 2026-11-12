from pathlib import Path
import gzip,json,statistics,math
A=Path('artifacts/crow-v15-strong-pronation-turn-screen-20261112');B=Path('artifacts/crow-v15-wing-bank-feedback-screen-20261112')
def read(p):
 d={}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d[x['lane'],x['step']]=x
 return d
r=read(B/'bank.stderr.gz');x=read(A/'bank.stderr.gz');pre=sorted(k for k in x if k[1]<450);mismatch=[k for k in pre if k not in r or any(r[k][z]!=x[k][z] for z in ('q','v','policy_action'))];assert len(pre)==48 and not mismatch,(len(pre),mismatch)
rows=[]
for e in range(6):
 ts=sorted(t for i,t in x if i==e and t>=500 and (e,t) in r);ds=[[x[e,t]['q'][11+j]-r[e,t]['q'][11+j] for j in range(2)] for t in ts];vals=[x[e,t]['q'] for t in ts]
 rows.append({'lane':e,'residual_normalized':[0.,.60,1.00][e//2],'matched_sample_steps':ts,'mean_left_pronation_delta_vs_bank_rad':statistics.mean(d[0] for d in ds),'mean_right_pronation_delta_vs_bank_rad':statistics.mean(d[1] for d in ds),'sample_height_range_m':[min(q[2] for q in vals),max(q[2] for q in vals)]})
out={'prefix_samples':len(pre),'prefix_mismatches':mismatch,'rows':rows,'scope':'Compared prior bank-feedback screen before 9 s; after 9 s the other screen uses a different bank intervention, so sampled pronation differences are not a controlled actuator-effect estimate. Sparse samples do not measure stroke loads or continuous extremes.'};(A/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
