from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-anticipatory-foot-placement-20261111');B=Path('artifacts/crow-v15-front-toe-perturbation-20261111')
def records(p,kind,lane):
 with gzip.open(p/'results.txt.gz','rt') as f:return [v.split() for v in f if v.startswith(kind+' ') and int(v.split()[1])==lane]
reports=[]
for lane,ref in ((1,1),(5,15),(9,3),(13,5)):
 a,b=records(A,'trace',lane),records(B,'trace',ref);d=[t for t in range(500) if t>=len(a) or t>=len(b) or a[t][4:]!=b[t][4:]];reports.append({'lane':lane,'reference_lane':ref,'samples':len(a),'differences':len(d),'first_differences':d[:10]})
initial=[]
for lane in range(1,17):
 ref=(1,15,3,5)[(lane-1)//4];a,b=records(A,'initialshape',lane),records(B,'initialshape',ref);initial.append({'lane':lane,'differences':sum(x[3:]!=y[3:] for x,y in zip(a,b))+abs(len(a)-len(b)),'records':len(a)})
differences=[x for x in reports+initial if x['differences'] or x.get('samples',500)!=500 or x.get('records',21)!=21];r={'references':reports,'initial':initial,'differences':differences};(A/'parity.json').write_text(json.dumps(r,indent=2)+'\n');print(r);assert not differences
