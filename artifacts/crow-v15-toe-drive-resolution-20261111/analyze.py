from pathlib import Path
import json
A=Path('artifacts/crow-v15-toe-drive-resolution-20261111')
def read(p):
 r=[[float(x) for x in l.split()] for l in p.read_text().splitlines()];assert len(r)==80 and all(x[-1]==0 for x in r);return r
a=read(A/'steps16.txt');b=read(A/'steps32.txt');diff=[max(abs(x[2]-y[2]),abs(x[3]-y[3])) for x,y in zip(a,b)];r={'samples':len(a),'max_toe_angle_difference_rad':max(diff),'time_of_max_difference':diff.index(max(diff)),'toe_angles_at_tick_50':[a[50][2],b[50][2]],'tolerance_rad':.05,'comparable_within_tolerance':max(diff)<=.05,'ground_impulses':0};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
