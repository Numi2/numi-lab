from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-front-toe-perturbation-20261111');B=Path('artifacts/crow-v15-front-toe-long-replay-20261111');C=Path('artifacts/crow-v15-bilateral-perturbation-screen-20261110')
def records(p,kind,lane):
 with gzip.open(p/'results.txt.gz','rt') as f:return [v.split() for v in f if v.startswith(kind+' ') and int(v.split()[1])==lane]
a,b=records(A,'trace',1),records(B,'trace',1);diff=[i for i in range(1000) if i>=len(a) or i>=len(b) or a[i][4:]!=b[i][4:]]
initial={}
for lane in range(1,17):
 case=((lane-1)//2+1)%8;x=records(A,'initialshape',lane);y=[z for z in records(C,'initialshape',case) if int(z[2])==2];initial[lane]={'records':len(x),'reference_records':len(y),'differences':sum(p[3:]!=q[3:] for p,q in zip(x,y))+abs(len(x)-len(y))}
r={'exact_reference_samples':len(a),'reference_differences':len(diff),'first_differences':diff[:20],'initial_shapes':initial};(A/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert len(a)==len(b)==1000 and not diff and all(x['records']==x['reference_records']==21 and x['differences']==0 for x in initial.values())
